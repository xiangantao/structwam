#!/usr/bin/env python3
"""Select direct-K regions using a complete instruction, without noun lists."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

from evaluate_vace_prompt_k_guidance import (
    CASE_PHRASES, FONT, ROOT, encode_prompts, evaluation_masks,
)
from search_vace_tracking_heads import CAMERAS, ControlTower, load_vae, read_video, write_video
from train_vace_k_vector_lora import install_k_lora
from vace_vae_warmup import observation_control, observation_window_indices

NEUTRAL = "A robot is in a scene with several objects."
SWAP = "pick up the butter and place it in the basket"


def k_regions(k, neighbors=16, seed=42):
    partitions, stats = [], []
    for values in k:
        values = values.astype(np.float64)
        values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
        distances, indices = cKDTree(values).query(values, k=neighbors + 1)
        # A sparse feature-space graph: no spatial/RGB cues or full KK matrix.
        rows = np.repeat(np.arange(len(values)), neighbors)
        cols = indices[:, 1:].ravel()
        scale = np.maximum(distances[:, -1], 1e-8)
        weights = np.exp(-distances[:, 1:].ravel() ** 2 / (scale[rows] * scale[cols]))
        graph = sparse.coo_matrix((weights, (rows, cols)), shape=(len(values), len(values))).tocsr()
        graph = graph.maximum(graph.T)
        communities = nx.community.louvain_communities(nx.from_scipy_sparse_array(graph),
                                                       weight="weight", resolution=1, seed=seed)
        labels = np.empty(len(values), dtype=np.int64)
        for label, nodes in enumerate(sorted(communities, key=lambda group: min(group))):
            labels[list(nodes)] = label
        partitions.append(labels)
        stats.append({"groups": len(communities), "sizes": np.bincount(labels).tolist()})
    return np.stack(partitions), stats


@torch.inference_mode()
def capture_instruction(tower, regions, positions, noise, control, text):
    group_maps, pixel_maps = {}, {}
    device, frames, cells = noise.device, *regions.shape
    partitions = [torch.as_tensor(row, dtype=torch.long, device=device) for row in regions]
    positions = torch.as_tensor(positions, dtype=torch.long, device=device)

    def callback(layer):
        def capture(query, key, attention_mask, is_causal, **_):
            if attention_mask is not None or is_causal:
                raise RuntimeError("Expected unmasked native tower attn2")
            q = query[0].float().reshape(12, frames, cells, -1)
            k = key[0, :, positions].float()
            if k.shape[0] != 12:
                raise RuntimeError("Text key indexing changed the head dimension")
            gm, pm = [], []
            for ti, labels in enumerate(partitions):
                count = torch.bincount(labels)
                grouped = torch.zeros((12, len(count), q.shape[-1]), dtype=torch.float32, device=device)
                grouped.index_add_(1, labels, q[:, ti])
                grouped /= count[None, :, None]
                logits = grouped @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
                # Normalize each text token across K regions, not across text.
                # Group means prevent large image regions winning by area alone.
                density = logits.softmax(dim=1).mean(-1) * len(count)
                gm.append(density[:, labels].cpu().numpy())
                pixel_logits = q[:, ti] @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
                pm.append((pixel_logits.softmax(dim=1).mean(-1) * cells).cpu().numpy())
            group_maps[layer], pixel_maps[layer] = np.stack(gm, 1), np.stack(pm, 1)
        return capture

    for layer, block in enumerate(tower.vace_blocks):
        block.attn2._heft_capture = callback(layer)
    try:
        tower(noise, control, text, torch.tensor([1000], device=device), lambda *_: None)
    finally:
        for block in tower.vace_blocks:
            del block.attn2._heft_capture
    return np.stack([group_maps[i] for i in range(15)]), np.stack([pixel_maps[i] for i in range(15)])


def relevance_metrics(heatmap, masks, relevant):
    distractors = {label: mask for label, mask in masks.items() if label not in relevant and mask.any()}
    rows = {}
    for label in relevant:
        positive = heatmap[masks[label]]
        if not len(positive) or not distractors:
            rows[label] = {"auc": None, "visible_cells": len(positive)}
            continue
        aucs = []
        for negative_mask in distractors.values():
            negative = heatmap[negative_mask]
            aucs.append(roc_auc_score(np.r_[np.ones(len(positive)), np.zeros(len(negative))],
                                     np.r_[positive, negative]))
        rows[label] = {"auc": float(np.mean(aucs)), "visible_cells": len(positive),
                       "mean_score": float(positive.mean())}
    valid = [row["auc"] for row in rows.values() if row["auc"] is not None]
    return {"mean_auc": float(np.mean(valid)) if valid else None, "objects": rows,
            "distractors": list(distractors),
            "gripper_mean_score": float(heatmap[masks["robot gripper"]].mean()) if masks["robot gripper"].any() else None}


def grouping_metrics(regions, masks):
    same, different = [], []
    names = list(masks)
    for ti, partition in enumerate(regions):
        counts = {}
        for name in names:
            labels = partition[masks[name][ti].ravel()]
            if len(labels):
                counts[name] = np.bincount(labels, minlength=int(partition.max()) + 1)
                if len(labels) > 1:
                    same.append(float((counts[name] * (counts[name] - 1)).sum() / (len(labels) * (len(labels) - 1))))
        for i, a in enumerate(counts):
            for b in list(counts)[i + 1:]:
                different.append(float((counts[a] * counts[b]).sum() / (counts[a].sum() * counts[b].sum())))
    return {"same_object_coassignment": float(np.mean(same)) if same else None,
            "different_object_coassignment": float(np.mean(different)) if different else None}


def select_heads(group_maps, masks, relevant):
    ranking = []
    for layer in range(15):
        for head in range(12):
            metric = relevance_metrics(group_maps[layer, head].reshape(-1, 32, 32), masks, relevant)
            ranking.append({"layer": layer, "head": head, **metric})
    ranking.sort(key=lambda row: row["mean_auc"] if row["mean_auc"] is not None else -1, reverse=True)
    return [(row["layer"], row["head"]) for row in ranking[:3]], ranking


def region_image(labels):
    palette = np.array([cv2.cvtColor(np.array([[[int(i * 67 % 180), 180, 240]]], np.uint8),
                                    cv2.COLOR_HSV2RGB)[0, 0] for i in range(int(labels.max()) + 1)])
    return palette[labels.reshape(32, 32)]


def overlay(rgb, score, scale):
    value = np.clip(np.maximum(score, 0) / scale, 0, 1)
    value = cv2.resize(value, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    heat = cv2.applyColorMap(np.round(value * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    alpha = (.7 * value)[..., None]
    return np.round(rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def wrap(draw, text, font, width):
    lines, current = [], ""
    for word in text.split():
        candidate = (current + " " + word).strip()
        if draw.textlength(candidate, font=font) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + [current]


def render(rgb, regions, pixel_score, group_score, scale, title, prompt):
    size, gap, width = 256, 12, 1084
    canvas = Image.new("RGB", (width, 422), "#f5f6f8")
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 13)
    draw.text((gap, 8), title, fill="black", font=font)
    for line, text in enumerate(wrap(draw, "Instruction: " + prompt, small, width - 2 * gap)):
        draw.text((gap, 34 + line * 18), text, fill="black", font=small)
    panels = [rgb, cv2.resize(region_image(regions), (256, 256), interpolation=cv2.INTER_NEAREST),
              overlay(rgb, pixel_score, scale), overlay(rgb, group_score, scale)]
    captions = ["Observed RGB", f"Automatic K groups: {len(np.unique(regions))}",
                "Whole instruction: pixel score", "Whole instruction: group score"]
    for col, (panel, caption) in enumerate(zip(panels, captions)):
        x = gap + col * (size + gap)
        draw.text((x, 91), caption, fill="black", font=small)
        canvas.paste(Image.fromarray(panel), (x, 117))
    draw.text((gap, 385), f"Task minus neutral-text score; positive values shown. Fixed scale: 0 to {scale:.3g}. No GT masks used.", fill="black", font=small)
    draw.text((gap, 403), "K groups are inferred regions, not GT objects; group colors are frame-local, not tracking identities.", fill="black", font=small)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/vace_prompt_k_guidance_20261006_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def log(**row):
        row["elapsed_seconds"] = round(time.monotonic() - start, 2)
        line = json.dumps(row)
        print(line, flush=True)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")

    torch.set_num_threads(4)
    previous = json.loads((args.cache / "report.json").read_text())
    if previous["start"] != 32 or previous["warmup_frames"] != 16 or previous["checkpoint_step"] != 1400:
        raise RuntimeError("Expected the existing step1400, start32, warmup16 K cache")
    cases = previous["cases"]
    video_root = ROOT / "runs/vace_tracking_all_suites_20261004/videos"
    tracks_root = ROOT / "runs/vace_tracking_all_suites_20261004/tracks"
    tasks = {case["episode"]: json.loads((video_root / f"episode_{case['episode']:05d}/metadata.json").read_text())["task"] for case in cases}
    log(stage="grouping_K", cases=len(cases), neighbors=args.neighbors)
    grouped = {}
    for case in cases:
        name = f"episode_{case['episode']:05d}_{case['camera']}"
        directory = args.output / name
        directory.mkdir()
        with np.load(args.cache / name / "model_fields.npz") as z:
            k, times = z["k"].astype(np.float32), z["times"]
        partitions, stats = k_regions(k, args.neighbors)
        masks = evaluation_masks(tracks_root / f"{name}_visible_surface.npz", times, 32, 256)
        group_metric = grouping_metrics(partitions, masks)
        grouped[name] = (partitions, times, stats, group_metric)
        np.savez_compressed(directory / "K_regions.npz", regions=partitions, times=times)
        log(stage="K_grouped", case=name, mean_groups=float(np.mean([s["groups"] for s in stats])), grouping=group_metric)
    device = torch.device("cuda:0")
    tokenizer, texts = encode_prompts(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", [NEUTRAL, SWAP, *tasks.values()], device)
    model_root = ROOT / "models/Wan2.1-VACE-1.3B"
    payload = torch.load(previous["checkpoint"], map_location="cpu", weights_only=False)
    tower = ControlTower.load(model_root / "diffusion_pytorch_model.safetensors", device)
    install_k_lora(tower, int(payload["layer"]), int(payload["head"]), int(payload["rank"]), float(payload["alpha"]))
    tower.load_state_dict(payload["lora"], strict=False)
    tower.eval().requires_grad_(False)
    vae = load_vae(model_root / "Wan2.1_VAE.pth", device)
    special_ids = set(tokenizer.all_special_ids)

    def text_inputs(prompt):
        embedding, ids = texts[prompt]
        positions = [i for i, token in enumerate(ids) if token not in special_ids]
        if not positions:
            raise RuntimeError("Instruction has no content tokens")
        return embedding.to(device), positions

    report = {"checkpoint": previous["checkpoint"], "checkpoint_step": 1400, "spatial_K": "independent empty-prompt pre-RoPE T13H11, full128 channels",
              "grouping": {"algorithm": "sparse feature-space nearest-neighbor graph + Louvain communities", "neighbors": args.neighbors,
                           "resolution": 1, "seed": 42, "fixed_object_count": False, "RGB_or_GT_used": False},
              "instruction_readout": "mean native attn2 logits over each K group; each complete-instruction token softmax over groups; mean all non-special tokens; multiply by number of groups",
              "contrast": "complete task instruction minus neutral sentence", "neutral_prompt": NEUTRAL,
              "no_noun_list_in_readout": True, "training": False,
              "head_selection": "episode39 agentview only, target objects versus unrelated annotated objects, fixed top3 thereafter",
              "GT_usage": "calibration head selection and evaluation only; never grouping or heatmap generation",
              "observation_only": True, "start": 32, "warmup": 16,
              "cases": []}
    heads, scale, zeros, overview = None, None, {}, []
    for case in cases:
        ep, camera = case["episode"], case["camera"]
        name = f"episode_{ep:05d}_{camera}"
        directory = args.output / name
        partitions, times, stats, group_metric = grouped[name]
        video = read_video(video_root / f"episode_{ep:05d}/{camera}.mp4")
        indices, _ = observation_window_indices(len(video), 32, int(times[-1] - times[0] + 1), 16)
        with torch.inference_mode():
            pixels = torch.from_numpy(video[indices].copy()).to(device).permute(0, 3, 1, 2).float()
            pixels = F.interpolate(pixels, size=(512, 512), mode="bilinear", align_corners=False).permute(1, 0, 2, 3) / 127.5 - 1
            control = observation_control(vae, pixels, 16, zeros)
            generator = torch.Generator(device=device).manual_seed(42 + ep * 37 + CAMERAS.index(camera))
            noise = torch.randn((1, 16, len(times), 64, 64), device=device, generator=generator).bfloat16()
            text, positions = text_inputs(NEUTRAL)
            log(stage="extracting_neutral", case=name)
            neutral_group, neutral_pixel = capture_instruction(tower, partitions, positions, noise, control, text)
            prompts = [tasks[ep]] + ([SWAP] if ep == 40 else [])
            selected_outputs = []
            for prompt in prompts:
                text, positions = text_inputs(prompt)
                log(stage="extracting_instruction", case=name, prompt=prompt, instruction_tokens=len(positions))
                native_group, native_pixel = capture_instruction(tower, partitions, positions, noise, control, text)
                delta_group, delta_pixel = native_group - neutral_group, native_pixel - neutral_pixel
                masks = evaluation_masks(tracks_root / f"{name}_visible_surface.npz", times, 32, 256)
                relevant = list(CASE_PHRASES[ep].values()) if prompt != SWAP else ["butter_1", "basket_1"]
                if case["calibration_only"]:
                    heads, ranking = select_heads(delta_group, masks, relevant)
                    report["selected_heads"] = heads
                    (args.output / "calibration_ranking.json").write_text(json.dumps(ranking, indent=2))
                    log(stage="calibrated", heads=heads, best_auc=ranking[0]["mean_auc"])
                group_score = np.stack([delta_group[l, h] for l, h in heads]).mean(0).reshape(len(times), 32, 32)
                pixel_score = np.stack([delta_pixel[l, h] for l, h in heads]).mean(0).reshape(len(times), 32, 32)
                if scale is None:
                    scale = max(float(np.percentile(np.r_[group_score.ravel(), pixel_score.ravel()], 98)), 1e-3)
                    report["display_scale"] = scale
                suffix = "task" if prompt == tasks[ep] else "swap"
                np.savez_compressed(directory / f"{suffix}_scores.npz", group=group_score, pixel=pixel_score,
                                    all_head_group_delta=delta_group.astype(np.float16), times=times)
                metrics = {"pixel": relevance_metrics(pixel_score, masks, relevant), "group": relevance_metrics(group_score, masks, relevant)}
                entry = {"episode": ep, "camera": camera, "suite": case["suite"], "prompt": prompt,
                         "swap": suffix == "swap", "calibration_only": case["calibration_only"],
                         "group_stats": stats, "grouping_metrics": group_metric, "metrics": metrics}
                frames = [render(video[frame], partitions[ti], pixel_score[ti], group_score[ti], scale,
                                 f"{name} | observed frame {frame}", prompt)
                          for ti, frame in enumerate(times)]
                chosen = [0, len(times) // 2, len(times) - 1]
                strip = Image.new("RGB", (frames[0].width, sum(frames[ti].height for ti in chosen)), "white")
                for row, ti in enumerate(chosen):
                    strip.paste(frames[ti], (0, row * frames[ti].height))
                strip.save(directory / f"{suffix}_comparison.jpg", quality=94)
                write_video(directory / f"{suffix}_comparison.mp4", np.stack([np.asarray(frame) for frame in frames]), 5)
                selected_outputs.append((suffix, frames, group_score, pixel_score))
                if camera == "agentview" and not case["calibration_only"] and suffix == "task":
                    overview.append(frames[len(times) // 2])
                report["cases"].append(entry)
                (args.output / "report.json").write_text(json.dumps(report, indent=2))
                log(stage="instruction_complete", case=name, prompt=suffix, metrics=metrics)
                del native_group, native_pixel, delta_group, delta_pixel
            if ep == 40:
                ti = int(np.argmin(np.abs(times - 44)))
                swap_image = Image.new("RGB", (1084, 844), "white")
                for row, (_, frames, _, _) in enumerate(selected_outputs):
                    swap_image.paste(frames[ti], (0, row * 422))
                swap_image.save(directory / "same_video_prompt_swap.jpg", quality=95)
                roi_comparison = {}
                for label in ("alphabet_soup_1", "butter_1", "basket_1"):
                    mask = masks[label]
                    roi_comparison[label] = {suffix: float(g[mask].mean()) if mask.any() else None
                                             for suffix, _, g, _ in selected_outputs}
                report.setdefault("prompt_swap", {})[camera] = roi_comparison
            del pixels, control, noise, neutral_group, neutral_pixel
            gc.collect()
            torch.cuda.empty_cache()
    for method in ("pixel", "group"):
        values = [row["metrics"][method]["mean_auc"] for row in report["cases"]
                  if not row["calibration_only"] and not row["swap"] and row["metrics"][method]["mean_auc"] is not None]
        report.setdefault("held_out_mean_auc", {})[method] = float(np.mean(values))
    if overview:
        image = Image.new("RGB", (overview[0].width, sum(frame.height for frame in overview)), "white")
        for row, frame in enumerate(overview):
            image.paste(frame, (0, row * frame.height))
        image.save(args.output / "held_out_overview.jpg", quality=94)
    report["elapsed_seconds"] = time.monotonic() - start
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    log(stage="complete", mean_auc=report["held_out_mean_auc"], prompt_swap=report.get("prompt_swap"))


if __name__ == "__main__":
    main()
