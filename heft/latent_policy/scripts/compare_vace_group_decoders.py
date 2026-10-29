#!/usr/bin/env python3
"""Compare head grouping decoders on identical frames without labels or RGB cues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans

from visualize_vace_spatial_heads import decode_groups, panel, render


def rotary_frequencies(time: int, grid: int, dim: int):
    spatial_dim = 2 * (dim // 6)
    dimensions = (dim - 2 * spatial_dim, spatial_dim, spatial_dim)
    y, x = np.indices((grid, grid))
    positions = (np.full(grid * grid, time), y.ravel(), x.ravel())
    phases = [position[:, None] / (10000.0 ** (np.arange(0, width, 2) / width))
              for position, width in zip(positions, dimensions, strict=True)]
    return np.exp(1j * np.concatenate(phases, axis=-1))


def undo_rope(values: np.ndarray, frequencies: np.ndarray):
    paired = values.astype(np.float64).reshape(len(values), -1, 2)
    complex_values = paired[..., 0] + 1j * paired[..., 1]
    content = complex_values * frequencies.conj()
    recovered = np.stack((content.real, content.imag), axis=-1).reshape(values.shape)
    reconstructed = content * frequencies
    roundtrip = np.stack((reconstructed.real, reconstructed.imag), axis=-1).reshape(values.shape)
    error = float(np.max(np.abs(roundtrip - values)))
    if error > 1e-8:
        raise RuntimeError(f"Inverse rotary round-trip failed: {error}")
    return recovered.astype(np.float32), error


def normalize(values):
    values = values.astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-6)


def cluster_content(features: np.ndarray, groups: int, seed: int):
    # Centering and L2 normalization are identical for every frame and head.
    features = normalize(features - features.mean(0, keepdims=True))
    model = KMeans(n_clusters=groups, n_init=10, random_state=seed)
    labels = model.fit_predict(features)
    grid = int(round(len(labels) ** .5))
    labels = labels.reshape(grid, grid)
    order = sorted(np.unique(labels), key=lambda value: tuple(np.argwhere(labels == value).mean(0)))
    lookup = np.zeros(groups, dtype=np.int32)
    lookup[order] = np.arange(groups)
    return lookup[labels], {"inertia": float(model.inertia_)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--layer-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", default="0:1,10:2")
    parser.add_argument("--groups", type=int, default=12)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-layer", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    heads = [tuple(map(int, item.split(":"))) for item in args.heads.split(",")]
    with np.load(args.cache) as z:
        frames, rgb = z["frame_indices"].astype(int), z["rgb"]
        cached = {layer: (z[f"layer_{layer}_q"], z[f"layer_{layer}_k"])
                  for layer in {layer for layer, _ in heads}}
    with np.load(args.layer_cache) as z:
        all_frames = z["frame_indices"].astype(int)
        hidden = z[f"layer_{args.reference_layer}"]
    grid = int(round(cached[heads[0][0]][0].shape[-2] ** .5))
    hidden = hidden.reshape(len(all_frames), grid * grid, -1)
    saved, records = {}, []
    titles = ("RGB", "Post-RoPE QK | spectral", "Pre-RoPE QK | spectral",
              "Pre-RoPE K | content clusters", "Pre-RoPE Q/K | content clusters",
              f"L{args.reference_layer} input | content clusters")
    for fi, frame in enumerate(frames):
        matches = np.flatnonzero(all_frames == frame)
        if len(matches) != 1:
            raise ValueError("Cached frame does not occur exactly once in layer cache")
        time = int(matches[0])
        reference, reference_stats = cluster_content(hidden[time], args.groups, args.seed)
        for layer, head in heads:
            q, k = (values[head, fi] for values in cached[layer])
            frequencies = rotary_frequencies(time, grid, q.shape[-1])
            content_q, q_error = undo_rope(q, frequencies)
            content_k, k_error = undo_rope(k, frequencies)
            post, post_stats = decode_groups(q, k, args.groups, args.seed)
            pre, pre_stats = decode_groups(content_q, content_k, args.groups, args.seed)
            keys, keys_stats = cluster_content(content_k, args.groups, args.seed)
            combined = np.concatenate((normalize(content_q), normalize(content_k)), axis=-1)
            qk, qk_stats = cluster_content(combined, args.groups, args.seed)
            labels = (post, pre, keys, qk, reference)
            name = f"T{layer}H{head}"
            panels, stats = [], (post_stats, pre_stats, keys_stats, qk_stats, reference_stats)
            for index, (partition, values) in enumerate(zip(labels, stats, strict=True)):
                raw, overlay, colors = render(rgb[fi], partition, args.size, .28)
                if index == 0:
                    panels.append(panel(f"RGB | frame {frame} | {name}", raw, args.size))
                panels.append(panel(titles[index + 1], overlay, args.size))
                stem = f"frame_{frame:04d}_{name}_method{index + 1}"
                Image.fromarray(overlay).save(args.output / f"{stem}_overlay.png")
                saved[stem] = partition
                records.append({"frame": int(frame), "head": name, "method": titles[index + 1],
                                "groups": int(len(np.unique(partition))),
                                "inverse_rope_roundtrip_max_error": max(q_error, k_error), **values})
            gap = 12
            sheet = Image.new("RGB", (3 * args.size + 4 * gap,
                                      2 * (args.size + 42) + 3 * gap), (235, 237, 240))
            for index, item in enumerate(panels):
                sheet.paste(item, (gap + (index % 3) * (args.size + gap),
                                  gap + (index // 3) * (args.size + 42 + gap)))
            sheet.save(args.output / f"decoder_comparison_frame_{frame:04d}_{name}.jpg", quality=95)
            print(json.dumps({"frame": int(frame), "head": name, "done": True}), flush=True)
    np.savez_compressed(args.output / "groups.npz", frame_indices=frames, **saved)
    report = {"ground_truth_used": False, "rgb_used_for_clustering": False,
              "background_removed": False, "temporal_identity_matching": False,
              "content_affinity_is_native_attention": False,
              "inverse_rope_source": "WanRotaryPosEmbed and WanAttnProcessor2_0; cached BF16/FP16 precision retained",
              "config": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}, "records": records}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.output), "done": True}), flush=True)


if __name__ == "__main__":
    main()
