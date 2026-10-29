#!/usr/bin/env python3
"""Render LIBERO tracks with wrist-safe camera projection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import av
import cv2
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.utils.io_utils import write_video
from robosuite.utils.camera_utils import get_camera_transform_matrix

CAMERAS = ("agentview", "robot0_eye_in_hand")


def read_frames(path):
    container = av.open(str(path))
    stream = container.streams.video[0]
    return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]


def project(sim, camera, points, height=256, width=256):
    matrix = get_camera_transform_matrix(sim, camera, height, width)
    hp = (matrix @ np.c_[points, np.ones(len(points))].T).T
    valid = hp[:, 2] > 1e-6
    xy = np.full((len(points), 2), np.nan, dtype=np.float32)
    xy[valid] = hp[valid, :2] / hp[valid, 2:3]
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    xy[~valid] = np.nan
    return xy


def draw(frames, tracks, colors, tail):
    output = []
    for t, frame in enumerate(frames):
        image = frame.copy()
        for name, points in tracks.items():
            if not np.isfinite(points[t]).all():
                continue
            color = colors[name]
            for j in range(max(1, t - tail + 1), t + 1):
                if np.isfinite(points[j - 1]).all() and np.isfinite(points[j]).all():
                    cv2.line(image, tuple(points[j - 1].astype(int)), tuple(points[j].astype(int)), color, 2, cv2.LINE_AA)
            p = tuple(points[t].astype(int))
            cv2.circle(image, p, 3, color, -1, cv2.LINE_AA)
            cv2.putText(image, name, (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        output.append(image)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--tail", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.input / "manifest.json").read_text())[: args.max_episodes]
    palette = [(255, 80, 0), (255, 0, 255), (60, 220, 80), (60, 160, 255), (220, 80, 220), (255, 200, 40)]
    for item in manifest:
        episode = args.input / f"episode_{item['episode']:05d}"
        metadata = json.loads((episode / "metadata.json").read_text())
        state = np.load(episode / "sim_state.npz")
        names = metadata["body_names"]
        ids = {name: i for i, name in enumerate(names)}
        selected = [name for name in ("gripper0_eef", "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip") if name in ids]
        selected += [name for name in names if name.endswith("_main")][:4]
        colors = {name: palette[i % len(palette)] for i, name in enumerate(selected)}
        suite = benchmark.get_benchmark_dict()[metadata["suite"]]()
        task = suite.get_task(int(metadata["task_id"]))
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=list(CAMERAS), camera_heights=256, camera_widths=256, camera_depths=False, control_freq=20, horizon=1000, ignore_done=True)
        env.reset()
        try:
            tracks = {camera: {name: [] for name in selected} for camera in CAMERAS}
            for t in range(len(state["qpos"])):
                sim = env.env.sim
                sim.data.qpos[:] = state["qpos"][t]
                sim.data.qvel[:] = state["qvel"][t]
                sim.forward()
                points = np.asarray([sim.data.body_xpos[ids[name]] for name in selected])
                for camera in CAMERAS:
                    px = project(sim, camera, points)
                    for i, name in enumerate(selected):
                        tracks[camera][name].append(px[i])
        finally:
            env.close()
        for camera in CAMERAS:
            frames = read_frames(episode / f"{camera}.mp4")
            write_video(args.output / f"episode_{item['episode']:05d}_{camera}_tracks.mp4", draw(frames, tracks[camera], colors, args.tail), int(metadata["fps"]))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(args.output), "episodes": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
