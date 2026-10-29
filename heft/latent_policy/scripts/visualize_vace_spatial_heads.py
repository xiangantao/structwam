#!/usr/bin/env python3
"""Visualize full-image groups decoded from cached, untrained VACE head Q/K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import SpectralClustering


PALETTE = np.asarray([
    (239, 92, 93), (62, 157, 222), (75, 187, 123), (244, 176, 64),
    (158, 109, 207), (44, 185, 183), (232, 126, 182), (136, 167, 56),
    (222, 118, 62), (98, 120, 208), (151, 132, 111), (102, 191, 210),
    (178, 85, 135), (186, 189, 106), (85, 155, 149), (162, 163, 175),
], dtype=np.uint8)


def decode_groups(q: np.ndarray, k: np.ndarray, groups: int, seed: int):
    query = torch.from_numpy(q.astype(np.float32))
    key = torch.from_numpy(k.astype(np.float32))
    scores = query @ key.T / q.shape[-1] ** 0.5
    scores.fill_diagonal_(-torch.inf)
    probabilities = scores.softmax(-1).numpy()
    affinity = (probabilities + probabilities.T) * 0.5
    labels = SpectralClustering(
        n_clusters=groups, affinity="precomputed", assign_labels="cluster_qr",
        eigen_solver="arpack", random_state=seed,
    ).fit_predict(affinity)
    grid = int(round(len(labels) ** 0.5))
    if grid * grid != len(labels):
        raise ValueError("Cached tokens must form a square spatial grid")
    labels = labels.reshape(grid, grid)
    # Spatial ordering changes display colors only, not the inferred partition.
    order = sorted(np.unique(labels), key=lambda value: tuple(
        np.argwhere(labels == value).mean(0)))
    lookup = np.zeros(groups, dtype=np.int32)
    lookup[order] = np.arange(groups)
    labels = lookup[labels]
    stats = {
        "group_areas_tokens": np.bincount(labels.ravel(), minlength=groups).tolist(),
        "mean_peak_attention": float(probabilities.max(-1).mean()),
    }
    return labels, stats


def render(frame: np.ndarray, labels: np.ndarray, size: int, alpha: float):
    rgb = cv2.resize(frame, (size, size), interpolation=cv2.INTER_CUBIC)
    expanded = cv2.resize(labels, (size, size), interpolation=cv2.INTER_NEAREST)
    colors = PALETTE[expanded % len(PALETTE)]
    overlay = ((1 - alpha) * rgb.astype(np.float32) + alpha * colors).astype(np.uint8)
    edges = np.zeros_like(expanded, dtype=np.uint8)
    edges[:, 1:] |= expanded[:, 1:] != expanded[:, :-1]
    edges[1:, :] |= expanded[1:, :] != expanded[:-1, :]
    halo = cv2.dilate(edges, np.ones((3, 3), np.uint8)).astype(bool)
    overlay[halo] = (overlay[halo].astype(np.float32) * 0.6).astype(np.uint8)
    overlay[edges.astype(bool)] = 245
    return rgb, overlay, colors


def panel(title: str, pixels: np.ndarray, size: int):
    result = Image.new("RGB", (size, size + 42), "white")
    result.paste(Image.fromarray(pixels), (0, 42))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    ImageDraw.Draw(result).text((12, 10), title, fill=(25, 25, 25), font=font)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", default="0:1,10:2,0:11,10:7")
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlay-alpha", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.groups < 2 or args.groups > len(PALETTE) or not 0 <= args.overlay_alpha <= 1:
        parser.error("groups must be between 2 and 16; overlay alpha must be in [0, 1]")
    heads = [tuple(map(int, value.split(":"))) for value in args.heads.split(",")]
    if len(heads) != 4:
        parser.error("The comparison sheet expects exactly four heads")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    records, saved = [], {}
    with np.load(args.cache) as source:
        frames = source["frame_indices"].astype(int)
        rgb = source["rgb"]
        if len(rgb) != len(frames):
            raise ValueError("RGB frames and cached Q/K frame indices must agree")
        cached = {layer: (source[f"layer_{layer}_q"], source[f"layer_{layer}_k"])
                  for layer, _ in heads}
    for ti, frame_index in enumerate(frames):
        visuals = {}
        for layer, head in heads:
            name = f"T{layer}H{head}"
            q, k = cached[layer]
            labels, stats = decode_groups(q[head, ti], k[head, ti], args.groups, args.seed)
            raw, overlay, colors = render(rgb[ti], labels, args.size, args.overlay_alpha)
            saved[f"frame_{frame_index:04d}_{name}"] = labels
            Image.fromarray(overlay).save(args.output / f"frame_{frame_index:04d}_{name}_overlay.png")
            Image.fromarray(colors).save(args.output / f"frame_{frame_index:04d}_{name}_labels.png")
            visuals[name] = overlay
            if (layer, head) == heads[0]:
                first_colors = colors
            records.append({"frame": int(frame_index), "head": name, **stats})
            print(json.dumps(records[-1]), flush=True)
        panels = [panel(f"RGB | frame {frame_index}", raw, args.size)]
        panels += [panel(f"Untrained T{layer}H{head} | {args.groups} groups",
                         visuals[f"T{layer}H{head}"], args.size) for layer, head in heads[:2]]
        panels.append(panel(f"T{heads[0][0]}H{heads[0][1]} | group map", first_colors, args.size))
        panels += [panel(f"Untrained T{layer}H{head} | {args.groups} groups",
                         visuals[f"T{layer}H{head}"], args.size) for layer, head in heads[2:]]
        gap = 12
        sheet = Image.new("RGB", (3 * args.size + 4 * gap,
                                  2 * (args.size + 42) + 3 * gap), (235, 237, 240))
        for index, item in enumerate(panels):
            sheet.paste(item, (gap + (index % 3) * (args.size + gap),
                              gap + (index // 3) * (args.size + 42 + gap)))
        sheet.save(args.output / f"comparison_frame_{frame_index:04d}.jpg", quality=95)
    np.savez_compressed(args.output / "groups.npz", frame_indices=frames, **saved)
    report = {
        "protocol": "Untrained empty-prompt attn1; full same-frame QK; exclude diagonal; native softmax; symmetrized graph; spectral clustering",
        "ground_truth_used": False, "rgb_used_for_clustering": False,
        "background_removed": False, "colors_are_instance_names": False,
        "temporal_identity_matching": False,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()}, "frames": records,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.output), "done": True}), flush=True)


if __name__ == "__main__":
    main()
