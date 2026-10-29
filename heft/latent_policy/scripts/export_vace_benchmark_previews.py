#!/usr/bin/env python3
"""Export another saved benchmark stage without running the video model again."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmark_vace_motion_heads_all_suites import ROOT, overlay, read_video, write_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    parser.add_argument("--state", type=int, default=22)
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    index = [f"# All-suite previews after{args.state} updates", "",
             "Fixed main L15H4 on the left. Frozen control11H1 for objects or7H11 for gripper on the right. GT green, QK magenta. Original RGB background; only native temporal samples are shown.", ""]
    images = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        index += ["## " + suite, "", "| Task | Camera | Moving object | Gripper | Image |", "| --- | --- | --- | --- | --- |"]
        for case in manifest:
            if case["suite"] != suite:
                continue
            directory = args.output / "cases" / case["name"]
            if not (directory / "completed.json").exists():
                raise ValueError(f"Case is not complete: {case['name']}")
            with np.load(directory / "points.npz") as archive:
                points = {key: archive[key] for key in archive.files if not key.startswith("mask_")}
                points["phase_masks"] = {key.removeprefix("mask_"): archive[key] for key in archive.files if key.startswith("mask_")}
            with np.load(directory / "fixed_trajectories.npz") as archive:
                trajectories = {key: archive[key] for key in archive.files if key.startswith(f"gt_assisted_after{args.state:02d}_")}
            video = read_video(args.data / "videos" / f"episode_{case['episode']:05d}" / f"{case['camera']}.mp4")
            paths, tiles = [], []
            for phase in ("moving_object_displaced", "gripper_visible"):
                path = directory / f"{phase}_after{args.state:02d}.mp4"
                if not points["phase_masks"][phase].any():
                    paths.append(None)
                    continue
                layer, head = (11, 1) if phase.startswith("moving") else (7, 11)
                main = trajectories[f"gt_assisted_after{args.state:02d}_main_L15H4_anchor_qk"]
                control = trajectories[f"gt_assisted_after{args.state:02d}_control_L{layer}H{head}_anchor_qk"]
                frames = []
                for ti in range(len(points["times"])):
                    left = overlay(video, points, main, phase, f"{case['name']} after{args.state} main L15H4", ti)
                    right = overlay(video, points, control, phase, f"after{args.state} control L{layer}H{head}", ti)
                    frames.append(np.concatenate((left, right), 1))
                write_video(path, np.stack(frames * 3), 5)
                paths.append(path)
                tiles.append(frames[-1])
                if phase.startswith("moving") and case["camera"] == "agentview" and suite not in images and not case["calibration_episode"]:
                    images[suite] = frames[-1]
            image_path = directory / f"endpoint_comparison_after{args.state:02d}.jpg"
            if tiles:
                cv2.imwrite(str(image_path), cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))
            links = [f"[video]({path.resolve()})" if path is not None else "no displaced object points" for path in paths]
            links.append(f"[image]({image_path.resolve()})" if tiles else "n/a")
            index.append(f"| {case['task_id']}: {case['task']} | {case['camera']} | " + " | ".join(links) + " |")
        index.append("")
    (args.output / f"videos_after{args.state:02d}.md").write_text("\n".join(index))
    if images:
        cv2.imwrite(str(args.output / f"suite_overview_after{args.state:02d}.jpg"),
                    cv2.cvtColor(np.concatenate(list(images.values()), 0), cv2.COLOR_RGB2BGR))
    print(json.dumps({"state": args.state, "cases": len(manifest), "suite_examples": list(images)}), flush=True)


if __name__ == "__main__":
    main()
