#!/usr/bin/env python3
"""Add fading trails to saved LIBERO surface-point annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.utils.io_utils import write_video

from render_libero_surface_tracks import CAMERAS, contact_sheet, overlay, paired_views, read_frames, save_legend


def representative_points(world_xyz, point_labels, count):
    selected = []
    for label in np.unique(point_labels):
        candidates = np.flatnonzero(point_labels == label)
        points = world_xyz[0, candidates].astype(np.float64)
        center = points.mean(axis=0)
        choice = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
        distances = np.full(len(points), np.inf)
        used = np.zeros(len(points), dtype=bool)
        for _ in range(min(count, len(candidates))):
            selected.append(int(candidates[choice]))
            used[choice] = True
            distances = np.minimum(distances, np.sum((points - points[choice]) ** 2, axis=1))
            distances[used] = -1
            choice = int(np.argmax(distances))
    return np.asarray(selected, dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--tracks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=12)
    parser.add_argument("--trails-per-subject", type=int, default=24)
    parser.add_argument("--max-episodes", type=int, default=3)
    args = parser.parse_args()
    if min(args.tail, args.trails_per_subject, args.max_episodes) <= 0:
        parser.error("tail, trails-per-subject and max-episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.tracks_root / "manifest.json").read_text())[: args.max_episodes]
    source_report = json.loads((args.tracks_root / "report.json").read_text())
    episode_reports = {item["episode"]: item for item in source_report["episodes"]}
    report = {
        "video_root": str(args.video_root.resolve()), "tracks_root": str(args.tracks_root.resolve()),
        "tail_frames": args.tail, "trails_per_subject": args.trails_per_subject, "episodes": [],
    }
    for item in manifest:
        ep = item["episode"]
        episode = args.video_root / f"episode_{ep:05d}"
        metadata = json.loads((episode / "metadata.json").read_text())
        subjects = episode_reports[ep]["subjects"]
        rendered = {}
        trails = None
        reference_ids = None
        for camera in CAMERAS:
            path = args.tracks_root / f"episode_{ep:05d}_{camera}_visible_surface.npz"
            with np.load(path) as data:
                if trails is None:
                    trails = representative_points(data["world_xyz"], data["point_label"], args.trails_per_subject)
                    reference_ids = data["body_id"].copy(), data["geom_id"].copy(), data["local_xyz"].copy()
                else:
                    for key, expected in zip(("body_id", "geom_id", "local_xyz"), reference_ids, strict=True):
                        if not np.array_equal(data[key], expected):
                            raise ValueError(f"Point identities differ between cameras: {key}")
                frames = read_frames(episode / f"{camera}.mp4")
                if len(frames) != len(data["xy"]):
                    raise ValueError("RGB and annotation frame counts differ")
                colors = {int(bid): tuple(subjects[label]) for bid, label in zip(data["body_id"], data["point_label"], strict=True)}
                labels = {int(bid): str(label) for bid, label in zip(data["body_id"], data["point_label"], strict=True)}
                rendered[camera] = overlay(frames, data["xy"], data["visible"], data["body_id"], colors, args.tail, trails)
            basename = f"episode_{ep:05d}_{camera}_surface_trails"
            write_video(args.output / f"{basename}.mp4", rendered[camera], int(metadata["fps"]))
            contact_sheet(rendered[camera], args.output / f"{basename}.jpg")
        save_legend(colors, labels, args.output / f"episode_{ep:05d}_legend.png")
        paired = paired_views(rendered)
        basename = f"episode_{ep:05d}_two_views_surface_trails"
        write_video(args.output / f"{basename}.mp4", paired, int(metadata["fps"]))
        contact_sheet(paired, args.output / f"{basename}.jpg")
        entry = {"episode": ep, "frames": len(paired), "fps": metadata["fps"], "trail_point_ids": trails.tolist()}
        report["episodes"].append(entry)
        print(json.dumps({"episode": ep, "frames": len(paired), "trail_points": len(trails)}), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
