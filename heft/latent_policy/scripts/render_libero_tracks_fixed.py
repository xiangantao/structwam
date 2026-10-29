#!/usr/bin/env python3
"""Render LIBERO body-center tracks while discarding points behind a camera."""

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


def make_env(metadata, height=256, width=256):
    suite = benchmark.get_benchmark_dict()[metadata["suite"]]()
    task = suite.get_task(int(metadata["task_id"]))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_names=list(CAMERAS),
        camera_heights=height,
        camera_widths=width,
        camera_depths=False,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
    )
    env.reset()
    return env


def project(sim, camera, points, height=256, width=256):
    matrix = get_camera_transform_matrix(sim, camera, height, width)
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)
    projected = (matrix @ homogeneous.T).T
    valid = projected[:, 2] > 1e-6
    xy = np.full((len(points), 2), np.nan, dtype=np.float32)
    xy[valid] = projected[valid, :2] / projected[valid, 2:3]
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    # Return x,y for OpenCV; the recorded RGB has already been vertically flipped.
    return xy


def read_frames(path):
    container = av.open(str(path))
    stream = container.streams.video[0]
    return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]


def draw(frames, tracks, colors, tail):
    output = []
    for index, frame in enumerate(frames):
        image = frame.copy()
        for name, points in tracks.items():
            p = points[index]
            if not np.isfinite(p).all():
                continue
            color = colors[name]
            start = max(0, index - tail)
            for j in range(start + 1, index + 1):
                if np.isfinite(points[j - 1]).all() and np.isfinite(points[j]).all():
                    cv2.line(image, tuple(points[j - 1].astype(int)), tuple(points[j].astype(int)), color, 2, cv2.LINE_AA)
            cv2.circle(image, tuple(p.astype(int)), 3, color, -1, cv2.LINE_AA)
            cv2.putText(image, name, (int(p[0]) + 4, int(p[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        output.append(image)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        name_to_index = {name: i for i, name in enumerate(names)}
        selected = [name for name in ("gripper0_eef", "robot0_right_hand") if name in name_to_index]
        selected += [name for name in names if name.endswith("_main") and name not in selected][:4]
        colors = {name: palette[i % len(palette)] for i, name in enumerate(selected)}
        env = make_env(metadata)
        try:
            tracks = {camera: {name: [] for name in selected} for camera in CAMERAS}
            for frame in range(len(state["qpos"])):
                sim = env.env.sim
                sim.data.qpos[:] = state["qpos"][frame]
                sim.data.qvel[:] = state["qvel"][frame]
                sim.forward()
                points = np.asarray([sim.data.body_xpos[name_to_index[name]] for name in selected])
                for camera in CAMERAS:
                    pixels = project(sim, camera, points)
                    for i, name in enumerate(selected):
                        tracks[camera][name].append(pixels[i])
        finally:
            env.close()
        for camera in CAMERAS:
            frames = read_frames(episode / f"{camera}.mp4")
            write_video(args.output / f"episode_{item['episode']:05d}_{camera}_tracks.mp4", draw(frames, tracks[camera], colors, args.tail), int(metadata["fps"]))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(args.output), "episodes": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
