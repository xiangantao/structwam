#!/usr/bin/env python3
"""Render coarse region trajectories from saved point tracks.

This view intentionally ignores fine object boundaries. Each semantic label is
summarized by a robust center, a coarse spread, and a short fading trail.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2
import numpy as np


COLORS = ((240, 60, 210), (40, 220, 90), (60, 180, 250), (250, 180, 50))


def read_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def write_video(path: Path, frames: list[np.ndarray], fps: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def robust_region(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray | None, float]:
    values = points[valid]
    if len(values) == 0:
        return None, 0.0
    center = np.median(values, axis=0)
    radius = float(np.percentile(np.linalg.norm(values - center, axis=1), 80))
    return center, max(5.0, min(42.0, radius + 3.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--prediction-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--trail", type=int, default=8)
    args = parser.parse_args()

    video = read_video(args.video)
    data = np.load(args.tracks, allow_pickle=True)
    prediction = data[args.prediction_key].astype(np.float32)
    visibility_key = f"{args.prediction_key}__visible"
    visibility = data[visibility_key].astype(bool) if visibility_key in data.files else np.ones(prediction.shape[:2], bool)
    gt = data["gt_xy"].astype(np.float32)
    valid = data["valid"].astype(bool)
    labels = data["labels"].astype(str)
    times = data["frame_indices"].astype(int)
    if len(times) != prediction.shape[1]:
        raise ValueError("frame_indices and prediction time dimension differ")
    if times.max(initial=0) >= len(video):
        raise ValueError("track frame index exceeds video length")

    selected = [label for label in args.labels if np.any(labels == label)]
    if not selected:
        raise ValueError(f"none of the requested labels are present: {args.labels}")
    label_ids = {label: np.flatnonzero(labels == label) for label in selected}
    centers_pred: dict[str, list[np.ndarray | None]] = {label: [] for label in selected}
    centers_gt: dict[str, list[np.ndarray | None]] = {label: [] for label in selected}
    radii: dict[str, list[float]] = {label: [] for label in selected}
    for ti in range(len(times)):
        for label, ids in label_ids.items():
            pred_center, radius = robust_region(prediction[ids, ti], visibility[ids, ti])
            gt_center, _ = robust_region(gt[ids, ti], valid[ids, ti])
            centers_pred[label].append(pred_center)
            centers_gt[label].append(gt_center)
            radii[label].append(radius)

    frames: list[np.ndarray] = []
    for ti, frame_id in enumerate(times):
        image = video[frame_id].copy()
        overlay = image.copy()
        for color_index, label in enumerate(selected):
            color = COLORS[color_index % len(COLORS)]
            pred_center = centers_pred[label][ti]
            gt_center = centers_gt[label][ti]
            radius = radii[label][ti]
            if pred_center is not None:
                for past in range(max(0, ti - args.trail), ti):
                    point = centers_pred[label][past]
                    if point is None:
                        continue
                    alpha = (past - max(0, ti - args.trail) + 1) / max(1, args.trail)
                    cv2.circle(overlay, tuple(np.round(point).astype(int)), 2, color, -1)
                center_xy = tuple(np.round(pred_center).astype(int))
                cv2.circle(overlay, center_xy, max(3, int(radius)), color, 1)
                cv2.circle(overlay, center_xy, 3, color, -1)
                ids = label_ids[label]
                point_mask = visibility[ids, ti]
                for point in prediction[ids[point_mask], ti]:
                    cv2.circle(overlay, tuple(np.round(point).astype(int)), 1, color, -1)
            if gt_center is not None:
                gt_xy = tuple(np.round(gt_center).astype(int))
                cv2.circle(overlay, gt_xy, 4, (40, 240, 70), 1)
                cv2.drawMarker(overlay, gt_xy, (40, 240, 70), cv2.MARKER_CROSS, 9, 1)
        image = cv2.addWeighted(overlay, 0.82, image, 0.18, 0)
        cv2.rectangle(image, (0, 0), (256, 38), (20, 20, 20), -1)
        cv2.putText(image, f"{args.prediction_key}: coarse region", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1)
        cv2.putText(image, "colored: predicted region / green cross: GT center", (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
        cv2.putText(image, f"frame {int(frame_id)}", (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (210, 210, 210), 1)
        frames.append(image)
    write_video(args.output, frames)
    sample = np.linspace(0, len(frames) - 1, min(6, len(frames))).astype(int)
    sheet = np.concatenate([frames[i] for i in sample], axis=1)
    cv2.imwrite(str(args.output.with_suffix(".jpg")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print({"output": str(args.output), "labels": selected, "frames": len(frames)})


if __name__ == "__main__":
    main()
