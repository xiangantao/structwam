#!/usr/bin/env python3
"""Overlay simulator body-center trajectories on recorded LIBERO RGB videos."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import av
import numpy as np
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
from lerobot.utils.io_utils import write_video


CAMERAS = {"agentview": "agentview", "robot0_eye_in_hand": "robot0_eye_in_hand"}


def read_video(path: Path) -> list[np.ndarray]:
    container = av.open(str(path))
    stream = container.streams.video[0]
    return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]


def make_env(suite_name: str, task_id: int, height: int, width: int):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_names=list(CAMERAS.values()),
        camera_heights=height,
        camera_widths=width,
        camera_depths=False,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
    )
    env.reset()
    return env


def project(sim, camera: str, points: np.ndarray, height: int, width: int) -> np.ndarray:
    matrix = get_camera_transform_matrix(sim, camera, height, width)
    return project_points_from_world_to_camera(points, matrix, height, width)


def draw_tracks(frames, tracks, colors, radius=3, tail=24):
    import cv2

    output = []
    for frame_index, frame in enumerate(frames):
        image = frame.copy()
        for body, points in tracks.items():
            color = colors[body]
            start = max(0, frame_index - tail)
            for j in range(start + 1, frame_index + 1):
                p0 = tuple(int(x) for x in points[j - 1][::-1])
                p1 = tuple(int(x) for x in points[j][::-1])
                cv2.line(image, p0, p1, color, 2, cv2.LINE_AA)
            p = tuple(int(x) for x in points[frame_index][::-1])
            cv2.circle(image, p, radius, color, -1, cv2.LINE_AA)
            cv2.putText(image, body, (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
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
    colors = {
        "gripper0_eef": (255, 80, 0),
        "robot0_right_hand": (255, 0, 255),
    }
    for item in manifest:
        episode_dir = args.input / f"episode_{item['episode']:05d}"
        state = np.load(episode_dir / "sim_state.npz")
        metadata = json.loads((episode_dir / "metadata.json").read_text())
        names = metadata["body_names"]
        name_to_index = {name: i for i, name in enumerate(names)}
        object_names = [name for name in names if name.endswith("_main") and not name.startswith("robot")]
        selected = [name for name in ("gripper0_eef", "robot0_right_hand") if name in name_to_index]
        selected += [name for name in object_names[:4] if name not in selected]
        for offset, name in enumerate(selected):
            colors.setdefault(name, ((80 + 55 * offset) % 256, (210 - 35 * offset) % 256, (50 + 75 * offset) % 256))
        env = make_env(metadata["suite"], int(metadata["task_id"]), 256, 256)
        try:
            trajectories = {camera: {name: [] for name in selected} for camera in CAMERAS}
            qpos = state["qpos"]
            qvel = state["qvel"]
            for frame_index in range(len(qpos)):
                sim = env.env.sim
                sim.data.qpos[:] = qpos[frame_index]
                sim.data.qvel[:] = qvel[frame_index]
                sim.forward()
                points = np.asarray([sim.data.body_xpos[name_to_index[name]] for name in selected])
                for camera in CAMERAS:
                    pixels = project(sim, camera, points, 256, 256)
                    for body_index, name in enumerate(selected):
                        trajectories[camera][name].append(pixels[body_index])
        finally:
            env.close()
        for camera in CAMERAS:
            frames = read_video(episode_dir / f"{camera}.mp4")
            overlaid = draw_tracks(frames, trajectories[camera], colors, tail=args.tail)
            write_video(args.output / f"episode_{item['episode']:05d}_{camera}_tracks.mp4", overlaid, int(metadata["fps"]))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(args.output), "episodes": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
