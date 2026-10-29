#!/usr/bin/env python3
"""Measure natural object separation in main-branch K vectors without clustering."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(ROOT / "heft/latent_policy/scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from benchmark_vace_observed33 import observed_inputs  # noqa: E402
from evaluate_vace_pure_noise_future import load_transformer  # noqa: E402
from search_vace_tracking_heads import load_empty_text, load_vae, read_video  # noqa: E402


def read_window(path: Path, start: int, count: int):
    with av.open(str(path)) as container:
        frames = []
        for index, frame in enumerate(container.decode(video=0)):
            if index >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError(f"Expected {count} frames from {path}, got {len(frames)}")
    return np.stack(frames)


def tracked_grid(path: Path, start: int, frames: int, grid: int):
    times = np.arange(start, start + frames, 4)
    with np.load(path, allow_pickle=True) as source:
        xy = source["xy"][times]
        visible = source["visible"][times]
        labels = source["point_label"].astype(str)
    # -1 means unlabelled; labelled cells are used only for offline statistics.
    cell_labels = np.full((len(times), grid * grid), -1, dtype=np.int32)
    names = sorted(label for label in np.unique(labels) if label != "wrist")
    label_ids = {label: index for index, label in enumerate(names)}
    votes = [[defaultdict(int) for _ in range(grid * grid)] for _ in times]
    cells = np.floor((np.nan_to_num(xy) + 0.5) * grid / 256.0).astype(np.int64)
    cells = np.clip(cells, 0, grid - 1)
    for ti in range(len(times)):
        for point in range(len(labels)):
            if labels[point] == "wrist":
                continue
            if not visible[ti, point] or not np.isfinite(xy[ti, point]).all():
                continue
            cell = int(cells[ti, point, 1] * grid + cells[ti, point, 0])
            votes[ti][cell][label_ids[labels[point]]] += 1
        for cell, counts in enumerate(votes[ti]):
            if counts:
                cell_labels[ti, cell] = max(counts, key=counts.get)
    return times, cell_labels, names


def capture_k(model, layers, noise, control, text, timestep, scales, frames, grid):
    captured = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            values = output[0]
            expected = frames * grid * grid * model.config.num_attention_heads * model.config.attention_head_dim
            if values.numel() != expected:
                raise RuntimeError(f"Unexpected K shape at layer {layer}: {tuple(values.shape)}")
            values = values.reshape(frames, grid * grid,
                                    model.config.num_attention_heads,
                                    model.config.attention_head_dim)
            captured[layer] = values.permute(2, 0, 1, 3).float().cpu().numpy()
        return hook

    for layer in layers:
        handles.append(model.blocks[layer].attn1.norm_k.register_forward_hook(make_hook(layer)))
    try:
        with torch.inference_mode():
            model(hidden_states=noise, timestep=timestep, encoder_hidden_states=text,
                  control_hidden_states=control, control_hidden_states_scale=scales,
                  return_dict=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(layers) - set(captured))
    if missing:
        raise RuntimeError(f"Missing K captures: {missing}")
    return captured


def sample_pairs(indices_a, indices_b, rng, count):
    if len(indices_a) == 0 or len(indices_b) == 0:
        return np.empty((0, 2), dtype=np.int64)
    left = rng.choice(indices_a, size=count, replace=True)
    right = rng.choice(indices_b, size=count, replace=True)
    return np.stack((left, right), axis=1)


def separation_metrics(features, labels, rng, pair_count=2000):
    features = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
    known = labels >= 0
    unique = np.unique(labels[known])
    if len(unique) < 2:
        return {"known_cells": int(known.sum()), "objects": int(len(unique))}
    within, between = [], []
    nearest_same = []
    for label in unique:
        own = np.flatnonzero(labels == label)
        if len(own) >= 2:
            pairs = sample_pairs(own, own, rng, pair_count)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            within.extend(np.sum(features[pairs[:, 0]] * features[pairs[:, 1]], axis=-1).tolist())
            if len(own) > 1:
                others = np.flatnonzero(known & (labels != label))
                pairs = sample_pairs(own, others, rng, pair_count)
                between.extend(np.sum(features[pairs[:, 0]] * features[pairs[:, 1]], axis=-1).tolist())
        for cell in own:
            candidates = np.flatnonzero(known)
            candidates = candidates[candidates != cell]
            if len(candidates):
                similarity = features[candidates] @ features[cell]
                nearest_same.append(float(labels[candidates[int(np.argmax(similarity))]] == label))
    within = np.asarray(within, dtype=np.float32)
    between = np.asarray(between, dtype=np.float32)
    return {
        "known_cells": int(known.sum()),
        "objects": int(len(unique)),
        "within_cosine": float(within.mean()) if len(within) else None,
        "between_cosine": float(between.mean()) if len(between) else None,
        "separation_margin": float(within.mean() - between.mean()) if len(within) and len(between) else None,
        "nearest_same_label": float(np.mean(nearest_same)) if nearest_same else None,
    }


def save_pca_preview(video, keys, labels, output, grid):
    output.mkdir(parents=True, exist_ok=True)
    # Fit one PCA per layer/head over the complete observed window, then render
    # native temporal positions back on the corresponding RGB frames.
    for layer, values in keys.items():
        for head in range(values.shape[0]):
            flat = values[head].reshape(-1, values.shape[-1]).astype(np.float32)
            flat -= flat.mean(0, keepdims=True)
            components = PCA(n_components=3, random_state=0).fit_transform(flat)
            lo, hi = np.percentile(components, [2, 98], axis=0)
            colors = np.clip((components - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
            colors = np.uint8(colors.reshape(values.shape[1], grid, grid, 3) * 255)
            tiles = []
            for ti in (0, values.shape[1] // 2, values.shape[1] - 1):
                rgb = cv2.resize(video[min(ti * 4, len(video) - 1)], (512, 512), interpolation=cv2.INTER_AREA)
                color = cv2.resize(colors[ti], (512, 512), interpolation=cv2.INTER_NEAREST)
                tile = np.uint8(0.35 * rgb + 0.65 * color)
                cv2.putText(tile, f"main K L{layer}H{head} t={ti}", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            cv2.imwrite(str(output / f"pca_L{layer}H{head}.jpg"),
                        cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 39, 40, 85, 123])
    parser.add_argument("--camera", choices=("agentview", "robot0_eye_in_hand"), default="agentview")
    parser.add_argument("--start", type=int, default=48)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 7, 15, 22, 26])
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    args = parser.parse_args()
    if (args.frames - 1) % 4:
        parser.error("frames must be 4n+1 for the native VAE timeline")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    model = load_transformer(args.model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    text = load_empty_text(args.text_model, device)
    scales = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
    all_rows, manifests = [], []
    rng = np.random.default_rng(42)
    for episode in args.episodes:
        case_dir = args.data / "videos" / f"episode_{episode:05d}"
        video_path = case_dir / f"{args.camera}.mp4"
        video = read_window(video_path, args.start, args.frames)
        with torch.inference_mode():
            control, encoded = observed_inputs(vae, video, argparse.Namespace(size=args.size), device)
            noise = torch.randn((1, 16, encoded.shape[1], args.size // 8, args.size // 8),
                                device=device, dtype=torch.bfloat16,
                                generator=torch.Generator(device=device).manual_seed(42 + episode))
        grid = args.size // 16
        times, labels, label_names = tracked_grid(
            args.data / "tracks" / f"episode_{episode:05d}_{args.camera}_visible_surface.npz",
            args.start, args.frames, grid)
        keys = capture_k(model, args.layers, noise, control, text,
                         torch.tensor([1000], device=device), scales, len(times), grid)
        case_rows = []
        for layer, values in keys.items():
            for head in range(values.shape[0]):
                for ti, frame in enumerate(times):
                    metric = separation_metrics(values[head, ti], labels[ti], rng)
                    row = {"episode": episode, "camera": args.camera, "frame": int(frame),
                           "layer": layer, "head": head, **metric}
                    case_rows.append(row)
                    all_rows.append(row)
        manifests.append({"episode": episode, "camera": args.camera, "start": args.start,
                          "frames": args.frames, "times": times.tolist(), "labels": label_names})
        np.savez_compressed(args.output / f"episode_{episode:05d}_{args.camera}_k.npz",
                            **{f"L{layer}": values.astype(np.float16) for layer, values in keys.items()},
                            labels=labels, times=times, label_names=np.asarray(label_names), rgb=video)
        save_pca_preview(video, keys, labels, args.output / f"episode_{episode:05d}_{args.camera}_previews", grid)
        del control, encoded, noise, keys
        gc.collect()
        torch.cuda.empty_cache()
    (args.output / "manifest.json").write_text(json.dumps({"protocol": {
        "main_input": "pure Gaussian noise",
        "control_input": "observed RGB frames only",
        "prompt": "empty",
        "clustering": "none; pairwise K separation only",
        "labels_used": "offline statistics only",
        "layers": args.layers,
    }, "cases": manifests}, indent=2))
    (args.output / "rows.json").write_text(json.dumps(all_rows, indent=2, allow_nan=False))
    aggregate = defaultdict(list)
    for row in all_rows:
        if row.get("separation_margin") is not None:
            aggregate[(row["layer"], row["head"])].append(row)
    summary = []
    for (layer, head), rows in aggregate.items():
        summary.append({"layer": layer, "head": head, "frames": len(rows),
                        "mean_margin": float(np.mean([r["separation_margin"] for r in rows])),
                        "mean_nearest_same": float(np.mean([r["nearest_same_label"] for r in rows]))})
    summary.sort(key=lambda row: (-row["mean_margin"], -row["mean_nearest_same"]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"output": str(args.output), "cases": len(manifests), "top": summary[:12]}, indent=2))


if __name__ == "__main__":
    main()
