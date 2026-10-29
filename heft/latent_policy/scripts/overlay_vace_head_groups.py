#!/usr/bin/env python3
"""Reproduce the original eight-group overlay for single or concatenated heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from compare_vace_group_decoders import rotary_frequencies, undo_rope


COLORS = np.asarray([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--layer-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    schemes = {
        "T0H1_QK": [(0, 1)],
        "T10H2_QK": [(10, 2)],
        "selected4_QK": [(0, 1), (0, 11), (10, 2), (10, 7)],
        "T5_all12_QK": [(5, head) for head in range(12)],
    }
    with np.load(args.cache) as source:
        frames = source["frame_indices"].astype(int)
        rgb = source["rgb"]
        cached = {layer: (source[f"layer_{layer}_q"], source[f"layer_{layer}_k"])
                  for layer in {layer for pairs in schemes.values() for layer, _ in pairs}}
    with np.load(args.layer_cache) as source:
        all_frames = source["frame_indices"].astype(int)
        hidden = source["layer_5"].reshape(len(all_frames), 1024, 1536)
    strips = {name: [] for name in ["RGB", "L5_input", *schemes]}
    labels_saved, records = {}, []
    for frame_pos, frame in enumerate(frames):
        matches = np.flatnonzero(all_frames == frame)
        if len(matches) != 1:
            raise ValueError("Head and whole-layer caches must refer to the same observation frames")
        time = int(matches[0])
        frequencies = rotary_frequencies(time, 32, 128)
        content = {}
        for layer, head in {pair for pairs in schemes.values() for pair in pairs}:
            content[(layer, head)] = tuple(
                undo_rope(values[head, frame_pos], frequencies)[0]
                for values in cached[layer]
            )
        features = {"L5_input": hidden[time].astype(np.float32)}
        features.update({name: np.concatenate([values for pair in pairs
                                              for values in content[pair]], axis=-1)
                         for name, pairs in schemes.items()})
        original = rgb[frame_pos].copy()
        cv2.putText(original, f"RGB t={frame}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    .36, (255, 255, 255), 1, cv2.LINE_AA)
        strips["RGB"].append(original)
        for name, values in features.items():
            values = values.copy()
            values /= np.linalg.norm(values, axis=1, keepdims=True) + 1e-6
            model = MiniBatchKMeans(n_clusters=8, batch_size=1024, n_init=5, random_state=0)
            labels = model.fit_predict(values).reshape(32, 32)
            color_map = cv2.resize(COLORS[labels], (256, 256), interpolation=cv2.INTER_NEAREST)
            overlay = (rgb[frame_pos] * .45 + color_map * .55).astype(np.uint8)
            cv2.putText(overlay, f"{name} {values.shape[-1]}D t={frame}", (3, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1, cv2.LINE_AA)
            strips[name].append(overlay)
            cv2.imwrite(str(args.output / f"{name}_frame_{frame:04d}.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            labels_saved[f"{name}_frame_{frame:04d}"] = labels
            records.append({"frame": int(frame), "representation": name,
                            "channels": int(values.shape[-1]), "inertia": float(model.inertia_)})
            print(json.dumps(records[-1]), flush=True)
    for name, panels in strips.items():
        cv2.imwrite(str(args.output / f"{name}_first_mid_last.jpg"),
                    cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    np.savez_compressed(args.output / "groups.npz", frame_indices=frames, **labels_saved)
    report = {
        "protocol": "Original per-frame L2 normalization and eight-group MiniBatchKMeans; no centering or PCA; original 45% RGB / 55% color overlay",
        "heads": schemes, "ground_truth_used": False, "rgb_used_for_clustering": False,
        "explicit_rope_removed_from_qk": True, "native_attention_matrix_used": False,
        "temporal_identity_matching": False, "records": records,
        "config": {name: str(value) for name, value in vars(args).items()},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"output": str(args.output), "done": True}), flush=True)


if __name__ == "__main__":
    main()
