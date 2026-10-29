#!/usr/bin/env python3
"""Train-free fusion of already decoded VACE trajectories.

The method deliberately operates after fixed-anchor Q->K decoding. It does
not modify Q/K coordinates or use GT: robust coordinate fusion across heads,
followed by an optional causal/short-window temporal filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_vace_tracking_heads import read_video, render_preview


HEADS = ("T6H1", "T10H6", "T7H11", "T14H8")


def median_filter_tracks(tracks: np.ndarray, visible: np.ndarray, radius: int = 1) -> np.ndarray:
    """Apply a short robust temporal median without crossing invisible gaps."""
    output = tracks.copy()
    for point_id in range(tracks.shape[0]):
        for time_id in range(tracks.shape[1]):
            if not visible[point_id, time_id]:
                continue
            start = max(0, time_id - radius)
            stop = min(tracks.shape[1], time_id + radius + 1)
            indices = np.flatnonzero(visible[point_id, start:stop]) + start
            if len(indices):
                output[point_id, time_id] = np.median(tracks[point_id, indices], axis=0)
    return output


def metrics(prediction, visible, data, group, displaced_only=False):
    mask = data["valid"] & visible & (data["groups"][:, None] == group)
    if displaced_only:
        mask &= data["displaced"]
    if not mask.any():
        return {"epe_px": None, "within8": None, "trajectory80_at8": None, "count": 0}
    error = np.linalg.norm(prediction - data["gt_xy"], axis=-1)
    point_ids = np.flatnonzero(mask.any(1))
    return {
        "epe_px": float(error[mask].mean()),
        "within4": float((error[mask] <= 4).mean()),
        "within8": float((error[mask] <= 8).mean()),
        "trajectory80_at8": float(np.mean([(error[i, mask[i]] <= 8).mean() >= 0.8 for i in point_ids])),
        "count": int(mask.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--radius", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    all_cases = []
    for case_dir in sorted(args.input.glob(f"episode_{args.episode:05d}_*")):
        camera = case_dir.name.split("_", 2)[2]
        source = np.load(case_dir / "trajectories.npz", allow_pickle=True)
        data = {key: source[key] for key in ("gt_xy", "valid", "displaced", "groups", "labels", "anchor", "frame_indices")}
        data["xy"] = data["gt_xy"]
        visible = source["T7H11__raw_anchor__visible"]
        head_tracks = np.stack([source[f"{head}__raw_anchor"] for head in HEADS])
        fused = np.median(head_tracks, axis=0)
        fused_filtered = median_filter_tracks(fused, visible, args.radius)
        output_case = args.output / case_dir.name
        output_case.mkdir()
        np.savez_compressed(output_case / "trajectories.npz", fused=fused,
                            fused_median=fused_filtered, **data)
        rows = []
        for name, prediction in (("head_median", fused), ("head_median_temporal_median", fused_filtered)):
            row = {"variant": name}
            for group in ("moving_object", "gripper"):
                row[group] = metrics(prediction, visible, data, group, displaced_only=group == "moving_object")
            rows.append(row)
        (output_case / "metrics.json").write_text(json.dumps(rows, indent=2))
        video = read_video(args.video_root / f"episode_{args.episode:05d}" / f"{camera}.mp4")
        for name, prediction in (("head_median", fused), ("head_median_temporal_median", fused_filtered)):
            for group in ("moving_object", "gripper"):
                render_preview(video, data["frame_indices"], prediction, data, group,
                               name, output_case / f"{name}_{group}.mp4")
        all_cases.append({"name": case_dir.name, "metrics": rows})
    (args.output / "results.json").write_text(json.dumps(all_cases, indent=2))
    print(json.dumps({"cases": len(all_cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
