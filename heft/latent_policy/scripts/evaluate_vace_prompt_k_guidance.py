#!/usr/bin/env python3
"""Read tower text attention through an independent, empty-prompt K field."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse
from scipy.sparse.linalg import factorized
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from diagnose_vace_k_temporal_boundary import field
from scan_vace_prompt_attention import find_phrase_positions, phrase_token_ids
from search_vace_tracking_heads import CAMERAS, ControlTower, load_vae, read_video, write_video, xy_to_cell
from train_vace_k_vector_lora import capture_k, install_k_lora, split_episodes
from vace_vae_warmup import observation_control, observation_window_indices

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CASE_PHRASES = {
    0: {"black bowl": "akita_black_bowl_1", "ramekin": "glazed_rim_porcelain_ramekin_1", "plate": "plate_1"},
    39: {"black bowl": "akita_black_bowl_1", "plate": "plate_1", "wooden cabinet": "wooden_cabinet_1"},
    40: {"alphabet soup": "alphabet_soup_1", "basket": "basket_1"},
    85: {"bowl": "akita_black_bowl_1", "stove": "flat_stove_1"},
    123: {"alphabet soup": "alphabet_soup_1", "tomato sauce": "tomato_sauce_1", "basket": "basket_1"},
}
GRIPPER = {"gripper base", "left finger", "right finger"}


def encode_prompts(model_root, prompts, device):
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(
        model_root / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval().requires_grad_(False)
    outputs = {}
    for prompt in prompts:
        tokens = tokenizer([prompt], padding="max_length", max_length=512,
                           truncation=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            text = encoder(**tokens).last_hidden_state
            text[:, int(tokens.attention_mask.sum()):] = 0
        outputs[prompt] = (text.detach().cpu(), tokens.input_ids[0].cpu().tolist())
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return tokenizer, outputs


@torch.inference_mode()
def capture_semantics(tower, phrase_positions, noise, control, text, timestep):
    captures = {}

    def callback(layer):
        def store(query, key, attention_mask, is_causal, **_):
            if attention_mask is not None or is_causal:
                raise RuntimeError("This experiment expects native unmasked tower cross-attention")
            q, k = query[0].float(), key[0].float()
            values = []
            for start in range(0, q.shape[1], 1024):
                probability = (q[:, start:start + 1024] @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)
                values.append(torch.stack([probability[..., ids].sum(-1)
                                          for ids in phrase_positions.values()], dim=1).cpu())
            captures[layer] = torch.cat(values, dim=-1).numpy()
        return store

    for layer, block in enumerate(tower.vace_blocks):
        block.attn2._heft_capture = callback(layer)
    try:
        tower(noise, control, text, torch.tensor([timestep], device=noise.device), lambda *_: None)
    finally:
        for block in tower.vace_blocks:
            del block.attn2._heft_capture
    if set(captures) != set(range(len(tower.vace_blocks))):
        raise RuntimeError("Missing native attn2 captures")
    return np.stack([captures[i] for i in range(len(tower.vace_blocks))])


def evaluation_masks(path, times, grid, image_size):
    with np.load(path, allow_pickle=True) as z:
        xy, visible = z["xy"][times], z["visible"][times]
        labels = np.array(["robot gripper" if str(x) in GRIPPER else str(x) for x in z["point_label"]])
    cells = xy_to_cell(xy, grid, image_size)
    masks = {}
    for label in np.unique(labels):
        if label == "wrist":
            continue
        mask = np.zeros((len(times), grid * grid), bool)
        for ti in range(len(times)):
            ids = np.flatnonzero(visible[ti] & (labels == label) & np.isfinite(xy[ti]).all(-1))
            mask[ti, np.unique(cells[ti, ids])] = True
        masks[str(label)] = mask.reshape(len(times), grid, grid)
    # Surface-point occupancy is evaluation support, not a full segmentation mask.
    unmixed = np.stack(list(masks.values())).sum(0) == 1
    return {label: mask & unmixed for label, mask in masks.items()}


def score_heatmap(heatmap, masks, target):
    target_mask = masks[target]
    target_values = heatmap[target_mask]
    others = [(label, heatmap[mask]) for label, mask in masks.items()
              if label != target and mask.any()]
    if not len(target_values) or not others:
        return {"auc": None, "target_cells": int(len(target_values))}
    aucs = [roc_auc_score(np.r_[np.ones(len(target_values)), np.zeros(len(values))],
                          np.r_[target_values, values]) for _, values in others]
    target_mean = float(target_values.mean())
    other_mean = float(np.mean([values.mean() for _, values in others]))
    return {"auc": float(np.mean(aucs)), "target_cells": int(len(target_values)),
            "target_mean": target_mean, "other_object_mean": other_mean,
            "target_to_other": target_mean / max(other_mean, 1e-12),
            "distractor_auc": dict(zip([label for label, _ in others], map(float, aucs)))}


def calibrate(raw, phrases, masks, count):
    rows = []
    for layer in range(raw.shape[0]):
        for head in range(raw.shape[1]):
            scores = {phrase: score_heatmap(raw[layer, head, p], masks, label)
                      for p, (phrase, label) in enumerate(phrases.items())}
            valid = [value["auc"] for value in scores.values() if value["auc"] is not None]
            rows.append({"layer": layer, "head": head, "mean_auc": float(np.mean(valid)), "phrases": scores})
    rows.sort(key=lambda row: row["mean_auc"], reverse=True)
    return [(row["layer"], row["head"]) for row in rows[:count]], rows


def select_maps(raw, heads):
    # Equal spatial mass per head/phrase avoids one high-scale head dominating.
    selected = np.stack([raw[layer, head] for layer, head in heads])
    normalizer = np.maximum(selected.mean(axis=(-3, -2, -1), keepdims=True), 1e-12)
    return (selected / normalizer).mean(0).astype(np.float32)


def guide_heatmaps(heatmaps, k, strength, bandwidth_scale):
    phrases, frames, grid, _ = heatmaps.shape
    ids = np.arange(grid * grid).reshape(grid, grid)
    left = np.r_[ids[:, :-1].ravel(), ids[:-1, :].ravel()]
    right = np.r_[ids[:, 1:].ravel(), ids[1:, :].ravel()]
    identity = sparse.eye(grid * grid, format="csc")
    guided, isotropic, diagnostics = [], [], []
    for ti in range(frames):
        vectors = k[ti].astype(np.float64)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
        distance = np.maximum(1 - np.sum(vectors[left] * vectors[right], axis=-1), 0)
        bandwidth = max(float(np.median(distance)) * bandwidth_scale, 1e-8)
        weights = np.exp(-distance / bandwidth)

        def solve(w):
            graph = sparse.coo_matrix((np.r_[w, w], (np.r_[left, right], np.r_[right, left])),
                                      shape=(grid * grid, grid * grid)).tocsc()
            laplacian = sparse.diags(np.asarray(graph.sum(1)).ravel()) - graph
            solution = factorized(identity + strength * laplacian)(heatmaps[:, ti].reshape(phrases, -1).T)
            return solution.T.reshape(phrases, grid, grid).astype(np.float32)

        guided.append(solve(weights))
        # Same average conductance and strength: isolate the effect of K edges.
        isotropic.append(solve(np.full_like(weights, weights.mean())))
        diagnostics.append({"cosine_distance_median": float(np.median(distance)),
                            "bandwidth": bandwidth, "mean_edge_weight": float(weights.mean())})
    guided = np.stack(guided, axis=1)
    isotropic = np.stack(isotropic, axis=1)
    if not np.isfinite(guided).all() or not np.allclose(guided.sum((-2, -1)), heatmaps.sum((-2, -1)), rtol=1e-5, atol=1e-5):
        raise RuntimeError("Guidance must preserve heatmap mass and produce finite values")
    return guided, isotropic, diagnostics


def overlay(rgb, heatmap, limits):
    lo, hi = limits
    value = np.clip((heatmap - lo) / max(hi - lo, 1e-8), 0, 1)
    value = cv2.resize(value, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap(np.round(value * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    alpha = (.65 * value)[..., None]
    return np.round(rgb * (1 - alpha) + color * alpha).astype(np.uint8)


def figure(rgb, k_image, raw, guided, phrase, frame, limits, title, heads):
    size, gap = 256, 12
    canvas = Image.new("RGB", (4 * size + 5 * gap, size + 120), "#f5f6f8")
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 13)
    draw.text((gap, 8), title, fill="black", font=font)
    draw.text((gap, 32), f"frame {frame} | phrase: {phrase} | semantic {heads} | empty-prompt K: T13H11", fill="black", font=small)
    panels = [("Observed RGB", Image.fromarray(rgb)), ("Empty-prompt K (display only)", k_image),
              ("Task attn2: raw", Image.fromarray(overlay(rgb, raw, limits))),
              ("Task attn2 + K edges", Image.fromarray(overlay(rgb, guided, limits)))]
    for col, (caption, panel) in enumerate(panels):
        x = gap + col * (size + gap)
        draw.text((x, 60), caption, fill="black", font=small)
        canvas.paste(panel.resize((size, size), Image.Resampling.NEAREST), (x, 84))
    draw.text((gap, size + 94), f"Heat scale fixed across frames and raw/guided: {limits[0]:.3g} to {limits[1]:.3g}. No GT masks in guidance.", fill="black", font=small)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/vace_k_vector_lora_all_scenes_20261005_v4/k_vector_lora_step_1400.pt")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/tracks")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 40, 85, 123])
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=CAMERAS)
    parser.add_argument("--start", type=int, default=32)
    parser.add_argument("--frames", type=int, default=65)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--semantic-heads", type=int, default=3)
    parser.add_argument("--strength", type=float, default=8)
    parser.add_argument("--bandwidth-scale", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if 39 in args.episodes or any(ep not in CASE_PHRASES for ep in args.episodes):
        parser.error("Use the documented test episodes, excluding calibration episode39")
    if (args.frames - 1) % 4 or args.warmup % 4 or args.start < args.warmup:
        parser.error("Use 4n+1 retained frames and 4m actual warm-up frames")
    if args.strength <= 0 or args.bandwidth_scale <= 0 or args.semantic_heads < 1:
        parser.error("Guidance parameters and head count must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def log(**values):
        values["elapsed_seconds"] = round(time.monotonic() - started, 2)
        line = json.dumps(values)
        print(line, flush=True)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")

    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layer, head = int(payload["layer"]), int(payload["head"])
    if (layer, head) != (13, 11):
        raise RuntimeError("This comparison expects the existing T13H11 spatial K")
    _, _, held_out = split_episodes(args.video_root, args.seed)
    if not set(args.episodes) <= {row["episode"] for row in held_out}:
        raise RuntimeError("Evaluation episodes must be held out from spatial K training")
    cases = [(39, "agentview")] + [(ep, camera) for ep in args.episodes for camera in args.cameras]
    metadata = {ep: json.loads((args.video_root / f"episode_{ep:05d}/metadata.json").read_text())
                for ep in [39, *args.episodes]}
    prompts = {ep: row["task"] + ". The robot gripper is visible." for ep, row in metadata.items()}
    log(stage="encoding_text", cases=len(cases), checkpoint_step=int(payload["step"]))
    tokenizer, texts = encode_prompts(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", ["", *prompts.values()], device)
    display = dict(np.load(ROOT / "runs/vace_k_time_normalization_visual_20261006/fixed_display_projection.npz"))
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    install_k_lora(tower, layer, head, int(payload["rank"]), float(payload["alpha"]))
    if set(payload["lora"]) != {key for key in tower.state_dict() if "lora_" in key}:
        raise RuntimeError("Checkpoint LoRA keys do not match")
    tower.load_state_dict(payload["lora"], strict=False)
    tower.eval().requires_grad_(False)
    report = {"checkpoint": str(args.checkpoint), "checkpoint_step": int(payload["step"]),
              "spatial_k": "empty prompt; norm_k output before RoPE; T13H11; full128 channels",
              "semantic": "native tower attn2 softmax over all512 text keys; sum phrase token columns",
              "calibration": "episode39 agentview; fixed top heads by mean target-vs-distractor AUC across nouns",
              "guidance": "argmin ||u-r||^2 + strength * sum_local_edges w_ij (u_i-u_j)^2",
              "edge_weight": "exp(-(1-cos(K_i,K_j))/(median_local_distance*bandwidth_scale))",
              "strength": args.strength, "bandwidth_scale": args.bandwidth_scale,
              "observation_only": True, "start": args.start, "max_frames": args.frames,
              "warmup_frames": args.warmup, "shared_noise_between_branches": True,
              "inference_uses_gt": False, "training": False,
              "evaluation_limit": "visible surface-point cells only; mixed cells excluded; not full segmentation IoU",
              "display_scale": "per phrase whole-window 2%-98% of raw, held fixed across time and methods",
              "cases": []}
    zeros, semantic_heads, all_panels = {}, None, []
    for ep, camera in cases:
        name = f"episode_{ep:05d}_{camera}"
        directory = args.output / name
        directory.mkdir()
        video = read_video(args.video_root / f"episode_{ep:05d}/{camera}.mp4")
        length = min(args.frames, (len(video) - args.start - 1) // 4 * 4 + 1)
        indices, alignment = observation_window_indices(len(video), args.start, length, args.warmup)
        times = alignment.pop("frame_indices")
        phrases = {**CASE_PHRASES[ep], "robot gripper": "robot gripper"}
        prompt = prompts[ep]
        embedding, token_ids = texts[prompt]
        positions = {phrase: find_phrase_positions(token_ids, phrase_token_ids(tokenizer, phrase)) for phrase in phrases}
        log(stage="extracting", case=name, retained_frames=length, latent_frames=len(times), prompt=prompt)
        with torch.inference_mode():
            pixels = torch.from_numpy(video[indices].copy()).to(device).permute(0, 3, 1, 2).float()
            pixels = F.interpolate(pixels, size=(512, 512), mode="bilinear", align_corners=False).permute(1, 0, 2, 3) / 127.5 - 1
            control = observation_control(vae, pixels, args.warmup, zeros)
            generator = torch.Generator(device=device).manual_seed(args.seed + ep * 37 + CAMERAS.index(camera))
            noise = torch.randn((1, 16, len(times), 64, 64), generator=generator, device=device).bfloat16()
            k = capture_k(tower, layer, len(times), 32, noise, control, texts[""][0].to(device), 1000, device)[head].float().cpu().numpy()
            raw = capture_semantics(tower, positions, noise, control, embedding.to(device), 1000)
        raw = raw.reshape(15, 12, len(phrases), len(times), 32, 32)
        if not np.isfinite(raw).all() or not np.isfinite(k).all():
            raise RuntimeError("Non-finite model outputs")
        np.savez_compressed(directory / "model_fields.npz", k=k.astype(np.float16),
                            attn2=raw.astype(np.float16), times=times, phrases=np.array(list(phrases)))
        masks = evaluation_masks(args.tracks_root / f"episode_{ep:05d}_{camera}_visible_surface.npz", times, 32, video.shape[1])
        if ep == 39:
            semantic_heads, ranking = calibrate(raw, phrases, masks, args.semantic_heads)
            report["selected_heads"] = semantic_heads
            (args.output / "calibration_ranking.json").write_text(json.dumps(ranking, indent=2))
            log(stage="calibrated", selected_heads=semantic_heads,
                best_auc=ranking[0]["mean_auc"], per_phrase=ranking[0]["phrases"])
        selected = select_maps(raw, semantic_heads)
        guided, isotropic, edge_diagnostics = guide_heatmaps(selected, k, args.strength, args.bandwidth_scale)
        np.savez_compressed(directory / "heatmaps.npz", raw=selected, k_guided=guided,
                            isotropic=isotropic, times=times, phrases=np.array(list(phrases)))
        case = {"episode": ep, "camera": camera, "suite": metadata[ep]["suite"], "prompt": prompt,
                "calibration_only": ep == 39, "times": times.tolist(), "alignment": alignment,
                "phrase_tokens": positions, "edge_diagnostics": edge_diagnostics, "metrics": {}}
        head_names = "+".join(f"T{l}H{h}" for l, h in semantic_heads)
        for p, (phrase, label) in enumerate(phrases.items()):
            case["metrics"][phrase] = {method: score_heatmap(value[p], masks, label)
                                       for method, value in [("raw", selected), ("k_guided", guided), ("isotropic", isotropic)]}
            limits = tuple(float(v) for v in np.percentile(selected[p], [2, 98]))
            safe_phrase = phrase.replace(" ", "_")
            frames = []
            for ti, frame_index in enumerate(times):
                panel = figure(video[frame_index], field(k[ti], display), selected[p, ti], guided[p, ti],
                               phrase, int(frame_index), limits, f"{name} | {metadata[ep]['suite']}", head_names)
                frames.append(np.asarray(panel))
            chosen = [0, len(times) // 2, len(times) - 1]
            strip = Image.new("RGB", (frames[0].shape[1], len(chosen) * frames[0].shape[0]), "white")
            for row, ti in enumerate(chosen):
                strip.paste(Image.fromarray(frames[ti]), (0, row * frames[ti].shape[0]))
            strip.save(directory / f"{safe_phrase}_comparison.jpg", quality=94)
            write_video(directory / f"{safe_phrase}_comparison.mp4", np.stack(frames), 5)
            if ep != 39 and camera == "agentview" and p == 0:
                all_panels.append(Image.fromarray(frames[len(times) // 2]))
        (directory / "metrics.json").write_text(json.dumps(case, indent=2))
        report["cases"].append(case)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        log(stage="case_complete", case=name, metrics={p: {m: round(v["auc"], 4) if v["auc"] is not None else None
                                                         for m, v in ms.items()} for p, ms in case["metrics"].items()})
        del pixels, control, noise, k, raw, selected, guided, isotropic
        gc.collect()
        torch.cuda.empty_cache()
    if all_panels:
        overview = Image.new("RGB", (all_panels[0].width, sum(p.height for p in all_panels)), "white")
        y = 0
        for panel in all_panels:
            overview.paste(panel, (0, y))
            y += panel.height
        overview.save(args.output / "held_out_overview.jpg", quality=94)
    for method in ("raw", "k_guided", "isotropic"):
        values = [v[method]["auc"] for case in report["cases"] if not case["calibration_only"]
                  for v in case["metrics"].values() if v[method]["auc"] is not None]
        report.setdefault("held_out_mean_auc", {})[method] = float(np.mean(values))
    report["elapsed_seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    log(stage="complete", held_out_mean_auc=report["held_out_mean_auc"])


if __name__ == "__main__":
    main()
