#!/usr/bin/env python3
"""Render dense ground-truth object surface trajectories on one RGB frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2
import numpy as np


PALETTE = ((255, 90, 70), (50, 205, 255), (255, 210, 45), (190, 100, 255), (70, 225, 130), (255, 140, 200))


def read_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs="*", help="Object labels; defaults to all non-robot labels")
    parser.add_argument("--background-frame", type=int, default=-1)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--line-thickness", type=int, default=1)
    args = parser.parse_args()

    video = read_video(args.video)
    data = np.load(args.tracks, allow_pickle=True)
    xy = data["xy"].astype(np.float32)
    visible = data["visible"].astype(bool)
    labels = data["point_label"].astype(str)
    frame = video[args.background_frame].copy()
    object_labels = [x for x in np.unique(labels) if x not in {"wrist", "gripper base", "left finger", "right finger"}]
    if args.labels:
        object_labels = [x for x in args.labels if x in set(labels)]
    if not object_labels:
        raise ValueError("no object labels selected")

    canvas = frame.astype(np.float32)
    for color_index, label in enumerate(object_labels):
        color = np.asarray(PALETTE[color_index % len(PALETTE)], dtype=np.float32)
        ids = np.flatnonzero(labels == label)[::max(1, args.point_stride)]
        # Draw every visible consecutive segment, preserving gaps caused by occlusion.
        for pid in ids:
            for t in range(1, xy.shape[0]):
                if not (visible[t - 1, pid] and visible[t, pid]):
                    continue
                a = np.round(xy[t - 1, pid]).astype(int)
                b = np.round(xy[t, pid]).astype(int)
                layer = canvas.astype(np.uint8)
                cv2.line(layer, tuple(a), tuple(b), tuple(int(v) for v in color), args.line_thickness, cv2.LINE_AA)
                canvas = canvas * 0.88 + layer.astype(np.float32) * 0.12
        # Mark the final visible location of each dense track.
        layer = canvas.astype(np.uint8)
        for pid in ids:
            valid = np.flatnonzero(visible[:, pid])
            if len(valid):
                point = tuple(np.round(xy[valid[-1], pid]).astype(int))
                cv2.circle(layer, point, 1, tuple(int(v) for v in color), -1, cv2.LINE_AA)
        canvas = canvas * 0.75 + layer.astype(np.float32) * 0.25

    output = np.clip(canvas, 0, 255).astype(np.uint8)
    # Add a compact legend without covering most of the image.
    height = output.shape[0]
    cv2.rectangle(output, (0, 0), (256, min(18 + 14 * len(object_labels), height)), (15, 15, 15), -1)
    cv2.putText(output, "dense GT object trajectories", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (255, 255, 255), 1)
    for i, label in enumerate(object_labels):
        y = 25 + i * 13
        color = tuple(int(v) for v in PALETTE[i % len(PALETTE)])
        cv2.circle(output, (7, y - 3), 2, color, -1)
        cv2.putText(output, label[:31], (13, y), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (235, 235, 235), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print({"output": str(args.output), "labels": object_labels, "points": int(sum((labels == x).sum() for x in object_labels)), "frames": int(xy.shape[0])})


if __name__ == "__main__":
    main()
