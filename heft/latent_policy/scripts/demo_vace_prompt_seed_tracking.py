#!/usr/bin/env python3
"""Select prompt-responsive observation points and track them without GT anchors."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.special import logsumexp
from torch.nn import functional as F

from evaluate_vace_prompt_k_guidance import FONT, ROOT, encode_prompts
from heft_postprocess_vace_tracking import beam_track, load_lora
from search_vace_tracking_heads import CAMERAS, ControlTower, load_vae, read_video, write_video
from train_vace_k_vector_lora import install_k_lora
from vace_vae_warmup import observation_control, observation_window_indices

STOPWORDS = set("a an the of in on to into and or it its both between from up with at then".split())
COLORS = [(255, 80, 80), (55, 210, 255), (255, 215, 55),
          (190, 105, 255), (65, 245, 140), (255, 135, 45)]


def content_words(tokenizer, ids):
    """Read word spans from the instruction's SentencePiece tokens only."""
    words, positions = [], []
    specials = set(tokenizer.all_special_ids)
    for index, (token_id, piece) in enumerate(zip(ids, tokenizer.convert_ids_to_tokens(ids))):
        if token_id in specials:
            continue
        if piece.startswith(chr(0x2581)) or not words:
            words.append(piece.lstrip(chr(0x2581)))
            positions.append([index])
        else:
            words[-1] += piece
            positions[-1].append(index)
    kept = [(word, span) for word, span in zip(words, positions)
            if word.lower().strip(".,!?") not in STOPWORDS and any(c.isalpha() for c in word)]
    if not kept:
        raise ValueError("Instruction contains no content words")
    return [word for word, _ in kept], [span for _, span in kept]


@torch.inference_mode()
def capture_words(tower, positions, noise, control, text):
    captures = {}

    def callback(layer):
        def store(query, key, attention_mask, is_causal, **_):
            if attention_mask is not None or is_causal:
                raise RuntimeError("Expected native unmasked tower attn2")
            q, k = query[0].float(), key[0].float()
            chunks = []
            for start in range(0, q.shape[1], 1024):
                logits = q[:, start:start + 1024] @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
                logp = logits.log_softmax(-1)
                chunks.append(torch.stack([torch.logsumexp(logp[..., span], -1)
                                           for span in positions], 1).cpu())
            captures[layer] = torch.cat(chunks, -1).numpy()
        return store

    for layer, block in enumerate(tower.vace_blocks):
        block.attn2._heft_capture = callback(layer)
    try:
        tower(noise, control, text, torch.tensor([1000], device=noise.device), lambda *_: None)
    finally:
        for block in tower.vace_blocks:
            del block.attn2._heft_capture
    return np.stack([captures[layer] for layer in range(15)]).reshape(
        180, len(positions), noise.shape[2], 1024)


def select_seeds(logmaps, regions, words, maximum=6, spacing=3.0):
    """Rank region-local excess word probability; no semantic object identity."""
    values = logmaps[:, :, 0].astype(np.float64)
    baseline = logsumexp(values, -1) - math.log(values.shape[-1])
    proposals = []
    for region in np.unique(regions[0]):
        ids = np.flatnonzero(regions[0] == region)
        mean = logsumexp(values[:, :, ids], -1) - math.log(len(ids))
        positive = mean > baseline
        # Keep absolute native word mass: a tiny normalized peak must not win.
        score = np.full_like(mean, -np.inf)
        delta = np.minimum(baseline[positive] - mean[positive], -1e-12)
        score[positive] = mean[positive] + np.log(-np.expm1(delta))
        head, word = np.unravel_index(np.argmax(score), score.shape)
        if not np.isfinite(score[head, word]):
            continue
        cell = int(ids[np.argmax(values[head, word, ids])])
        proposals.append({"region": int(region), "cell": cell, "head_index": int(head),
                          "head": f"T{head // 12}H{head % 12}", "word_index": int(word),
                          "word": words[word], "log_excess_probability": float(score[head, word]),
                          "native_region_probability": float(np.exp(mean[head, word])),
                          "native_frame_probability": float(np.exp(baseline[head, word]))})
    proposals.sort(key=lambda row: row["log_excess_probability"], reverse=True)
    selected = []
    for candidate in proposals:
        xy = np.array([candidate["cell"] % 32, candidate["cell"] // 32])
        if any(np.linalg.norm(xy - np.array([row["cell"] % 32, row["cell"] // 32])) < spacing
               for row in selected):
            continue
        selected.append(candidate)
        if len(selected) == maximum:
            break
    if not selected:
        raise RuntimeError("No spatially nonuniform native word response")
    return selected


def heat_density(logmaps, regions, selected):
    maps = []
    for seed in selected:
        values = logmaps[seed["head_index"], seed["word_index"]].astype(np.float64)
        density = np.exp(values - logsumexp(values, -1, keepdims=True)) * 1024
        grouped = np.empty_like(density)
        for ti, labels in enumerate(regions):
            for label in np.unique(labels):
                ids = labels == label
                grouped[ti, ids] = density[ti, ids].mean()
        maps.append(grouped)
    return np.max(maps, axis=0).reshape(-1, 32, 32).astype(np.float32)


def overlay_heat(rgb, heat, scale):
    value = np.clip((heat - 1) / scale, 0, 1)
    value = cv2.resize(value, (256, 256), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap(np.round(value * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    alpha = (0.55 * value)[..., None]
    return np.round(rgb * (1 - alpha) + color * alpha).astype(np.uint8)


def draw_points(rgb, xy, paths=None, time_id=0):
    result = np.ascontiguousarray(rgb.copy())
    for pi, point in enumerate(xy):
        color = COLORS[pi % len(COLORS)]
        if paths is not None and time_id:
            line = np.round(paths[pi, :time_id + 1]).astype(np.int32)
            cv2.polylines(result, [line], False, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.polylines(result, [line], False, color, 1, cv2.LINE_AA)
        pos = tuple(np.round(point).astype(int))
        cv2.circle(result, pos, 5, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(result, pos, 3, color, -1, cv2.LINE_AA)
        cv2.putText(result, str(pi + 1), (pos[0] + 5, pos[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def render_case(case, output, video_root):
    directory = output / case["name"]
    with np.load(directory / "semantic_fields.npz") as z:
        times, heat = z["times"], z["heat"]
    with np.load(directory / "tracks.npz") as z:
        paths = z["xy"]
    seeds = json.loads((directory / "seeds.json").read_text())
    video = read_video(video_root / f"episode_{case['episode']:05d}/{case['camera']}.mp4")
    scale = max(float(np.quantile(np.maximum(heat - 1, 0), 0.99)), 1e-6)
    font, small = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 14)
    frames, panels = [], []
    for ti, frame in enumerate(times):
        canvas = Image.new("RGB", (1200, 524), "#f4f5f7")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"{case['suite']} | episode {case['episode']} | {case['camera']} | RGB frame {frame}", font=font, fill="black")
        prompt_lines = [case["prompt"][start:start + 120] for start in range(0, len(case["prompt"]), 120)]
        for index, line in enumerate(prompt_lines):
            draw.text((12, 32 + index * 18), line, font=small, fill="black")
        rgb = video[frame]
        seed_xy = np.array([[(seed["cell"] % 32 + 0.5) * 8 - 0.5,
                             (seed["cell"] // 32 + 0.5) * 8 - 0.5] for seed in seeds])
        heat_rgb = overlay_heat(rgb, heat[ti], scale)
        if ti == 0:
            heat_rgb = draw_points(heat_rgb, seed_xy)
        images = [rgb, heat_rgb, draw_points(rgb, paths[:, ti], paths, ti)]
        captions = ["Observed RGB", "Prompt response + K regions", "Prompt seeds -> empty-prompt T7H11 beam"]
        for col, (caption, image) in enumerate(zip(captions, images)):
            x = 12 + col * 396
            draw.text((x, 76), caption, font=small, fill="black")
            canvas.paste(Image.fromarray(image).resize((384, 384), Image.Resampling.NEAREST), (x, 100))
        legend = " | ".join(f"{i + 1}:{seed['word']} {seed['head']}" for i, seed in enumerate(seeds))
        draw.text((12, 490), legend, font=ImageFont.truetype(FONT, 12), fill="black")
        draw.text((12, 507), "No GT anchors or masks. Word response is not an object label. Heat scale fixed throughout clip.", font=ImageFont.truetype(FONT, 12), fill="black")
        frames.append(np.asarray(canvas))
        if ti in {0, len(times) // 2, len(times) - 1}:
            canvas.save(directory / f"frame_{frame:03d}.jpg", quality=95)
            panels.append(canvas)
    write_video(directory / "prompt_seed_tracks.mp4", np.stack(frames), fps=5)
    sheet = Image.new("RGB", (1200, 524 * len(panels)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 524))
    sheet.save(directory / "start_middle_end.jpg", quality=95)
    motion = np.linalg.norm(paths - paths[:, :1], axis=-1).max(1)
    return {"case": case["name"], "seeds": len(seeds),
            "max_displacement_px_per_seed": motion.tolist(), "heat_scale": scale,
            "words": [row["word"] for row in seeds], "heads": [row["head"] for row in seeds]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan", type=Path, default=ROOT / "runs/vace_semantics_all_suites_scan_20261006_v2")
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--tracking-checkpoint", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/training/tracking_lora_step_10000.pt")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    torch.set_num_threads(4)

    def log(**values):
        values["elapsed_seconds"] = round(time.monotonic() - start, 1)
        line = json.dumps(values)
        print(line, flush=True)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")

    tracking = torch.load(args.tracking_checkpoint, map_location="cpu", weights_only=False)
    training_config = json.loads((args.tracking_checkpoint.parent / "config.json").read_text())
    heldout = set(training_config["val_episodes"])
    examples = {}
    for row in json.loads((args.scan / "manifest.json").read_text()):
        if row["episode"] in heldout and row["suite"] not in examples:
            examples[row["suite"]] = {key: row[key] for key in ("episode", "suite", "task_id", "prompt")}
    if len(examples) != 4:
        raise RuntimeError("Expected one predetermined tracking-held-out episode in each suite")
    cases = [{**row, "camera": camera, "name": f"episode_{row['episode']:05d}_{camera}"}
             for row in examples.values() for camera in CAMERAS]
    (args.output / "protocol.json").write_text(json.dumps({
        "training": False, "GT_used_for_selection_or_tracking": False,
        "example_selection": "first manifest episode per suite in tracking validation split; no metric selection",
        "prompt_words": "instruction SentencePiece words, generic stopwords removed; no BDDL noun inventory",
        "head_selection": "all180 native heads; maximum absolute word probability excess over its frame mean in each K region",
        "seeds": "up to6 regions at RGB frame32; minimum3 grid-cell spacing; no object labels or visibility supervision",
        "K_regions": "existing empty-prompt pre-RoPE T13H11 K LoRA step1400; frozen frame-local groups",
        "tracking": "empty prompt, T7H11, step10000; beam top32 width8 motion0.45 bound4 cells",
        "observation": "RGB32..96 or shorter,16 actual past VAE warmup frames; no generated future",
        "scope": "visual seed-and-track demonstration, not ACT policy improvement or exact noun grounding",
        "cases": cases, "tracking_checkpoint": str(args.tracking_checkpoint.resolve())}, indent=2))
    log(stage="examples", cases=[row["name"] for row in cases])

    device = torch.device("cuda:0")
    model = ROOT / "models/Wan2.1-VACE-1.3B"
    tokenizer, texts = encode_prompts(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers",
                                      ["", *[row["prompt"] for row in examples.values()]], device)
    vae = load_vae(model / "Wan2.1_VAE.pth", device)
    zero_latents = {}
    for case in cases:
        directory = args.output / case["name"]
        directory.mkdir(exist_ok=True)
        control_path = directory / "control.pt"
        if not control_path.exists():
            with np.load(args.scan / case["name"] / "fields.npz") as z:
                times = z["times"]
            video = read_video(args.video_root / f"episode_{case['episode']:05d}/{case['camera']}.mp4")
            indices, alignment = observation_window_indices(len(video), int(times[0]), int(times[-1] - times[0] + 1), 16)
            if not np.array_equal(times, alignment["frame_indices"]):
                raise RuntimeError("Observation endpoints differ from frozen K region cache")
            with torch.inference_mode():
                pixels = torch.from_numpy(video[indices].copy()).to(device).permute(0, 3, 1, 2).float()
                pixels = F.interpolate(pixels, size=(512, 512), mode="bilinear", align_corners=False).permute(1, 0, 2, 3) / 127.5 - 1
                control = observation_control(vae, pixels, 16, zero_latents)
            torch.save({"control": control.cpu(), "times": times}, control_path)
            del video, pixels, control
            log(stage="observation_encoded", case=case["name"])
    del vae, zero_latents
    gc.collect()
    torch.cuda.empty_cache()

    def inputs(case):
        saved = torch.load(args.output / case["name"] / "control.pt", map_location="cpu", weights_only=False)
        generator = torch.Generator(device=device).manual_seed(42 + case["episode"] * 37 + CAMERAS.index(case["camera"]))
        noise = torch.randn((1, 16, len(saved["times"]), 64, 64), generator=generator, device=device).bfloat16()
        return noise, saved["control"].to(device), saved["times"]

    with np.load(args.scan / cases[0]["name"] / "fields.npz") as z:
        k_checkpoint = Path(str(z["checkpoint"]))
    kp = torch.load(k_checkpoint, map_location="cpu", weights_only=False)
    tower = ControlTower.load(model / "diffusion_pytorch_model.safetensors", device)
    install_k_lora(tower, int(kp["layer"]), int(kp["head"]), int(kp["rank"]), float(kp["alpha"]))
    tower.load_state_dict(kp["lora"], strict=False)
    tower.eval().requires_grad_(False)
    for case in cases:
        directory = args.output / case["name"]
        if (directory / "semantic_fields.npz").exists():
            continue
        words, positions = content_words(tokenizer, texts[case["prompt"]][1])
        noise, control, times = inputs(case)
        log(stage="prompt_forward", case=case["name"], words=words)
        logmaps = capture_words(tower, positions, noise, control, texts[case["prompt"]][0].to(device))
        with np.load(args.scan / case["name"] / "fields.npz") as z:
            if not np.array_equal(times, z["times"]) or str(k_checkpoint) != str(z["checkpoint"]):
                raise RuntimeError("K grouping cache differs")
            regions = z["regions"]
        selected = select_seeds(logmaps, regions, words)
        heat = heat_density(logmaps, regions, selected)
        np.savez_compressed(directory / "semantic_fields.npz", log_native_word_probability=logmaps.astype(np.float16),
                            heat=heat, times=times, words=np.array(words), regions=regions)
        (directory / "seeds.json").write_text(json.dumps(selected, indent=2))
        log(stage="seeds_selected", case=case["name"], points=[(s["word"], s["head"], s["cell"]) for s in selected])
        del noise, control, logmaps
    del tower, kp
    gc.collect()
    torch.cuda.empty_cache()

    tower = ControlTower.load(model / "diffusion_pytorch_model.safetensors", device)
    load_lora(tower, args.tracking_checkpoint, tracking["candidates"], tracking["rank"], tracking["alpha"])
    tower.eval().requires_grad_(False)
    summaries = []
    for case in cases:
        directory = args.output / case["name"]
        if not (directory / "tracks.npz").exists():
            selected = json.loads((directory / "seeds.json").read_text())
            noise, control, times = inputs(case)
            capture = {}

            def store(layer, q, k):
                if layer == 7:
                    capture["q"] = q[0, 11].detach().reshape(1, len(times), 1024, 128).contiguous()
                    capture["k"] = k[0, 11].detach().reshape(1, len(times), 1024, 128).contiguous()

            log(stage="tracking_forward", case=case["name"])
            with torch.inference_mode():
                tower(noise, control, texts[""][0].to(device), torch.tensor([1000], device=device), store)
                anchor_cells = np.array([seed["cell"] for seed in selected], dtype=np.int64)
                points = {"ids": np.arange(len(selected)), "anchor": np.zeros(len(selected), dtype=np.int64),
                          "cells": np.repeat(anchor_cells[:, None], len(times), axis=1)}
                xy = beam_track(capture["q"], capture["k"], points, 0, 32)
            np.savez_compressed(directory / "tracks.npz", xy=xy, times=times, anchor_cells=anchor_cells)
            del noise, control, capture
        summary = render_case(case, args.output, args.video_root)
        summaries.append(summary)
        log(stage="case_complete", **summary)
    del tower, texts
    gc.collect()
    torch.cuda.empty_cache()
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    for suite, row in examples.items():
        panels = [Image.open(args.output / f"episode_{row['episode']:05d}_{camera}" / "frame_096.jpg")
                  if (args.output / f"episode_{row['episode']:05d}_{camera}" / "frame_096.jpg").exists()
                  else Image.open(sorted((args.output / f"episode_{row['episode']:05d}_{camera}").glob("frame_*.jpg"))[-1])
                  for camera in CAMERAS]
        sheet = Image.new("RGB", (1200, 1048), "white")
        for index, panel in enumerate(panels):
            sheet.paste(panel, (0, index * 524))
        sheet.save(args.output / f"{suite}_both_views.jpg", quality=95)
    log(stage="complete", output=str(args.output.resolve()))


if __name__ == "__main__":
    main()
