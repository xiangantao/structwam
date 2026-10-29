#!/usr/bin/env python3
"""Record long, simulator-backed LIBERO RGB/depth videos for representation probes.

The recorder deliberately does not import or run a visual encoder.  It replays the
local LeRobot actions from selected episodes in a LIBERO simulator task and stores
the rendered observations together with simulator state.  The replay is a fresh
simulation trajectory; it is not claimed to be pixel-identical to the source demo
because the local dataset does not contain its original simulator initial state.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as arrow_dataset
import torch

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from lerobot.utils.io_utils import write_video


CAMERA_NAMES = ("agentview", "robot0_eye_in_hand")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def canonical(text: str) -> str:
    return " ".join(str(text).lower().strip().rstrip(".").split())


def save_manifest(output: Path, manifest: list) -> None:
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output / "manifest.json")


def load_task_names(root: Path) -> dict[int, str]:
    table = arrow_dataset.dataset(str(root / "meta" / "tasks.parquet"), format="parquet").to_table()
    rows = table.to_pylist()
    return {int(row["task_index"]): canonical(row["__index_level_0__"]) for row in rows}


def load_episode_actions(root: Path) -> dict[int, tuple[int, np.ndarray]]:
    columns = ("episode_index", "task_index", "action")
    table = arrow_dataset.dataset(str(root / "data"), format="parquet").to_table(columns=list(columns))
    grouped: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    for episode, task, action in zip(
        table.column("episode_index").to_pylist(),
        table.column("task_index").to_pylist(),
        table.column("action").to_pylist(),
        strict=True,
    ):
        grouped[int(episode)].append((int(task), action))
    result: dict[int, tuple[int, np.ndarray]] = {}
    for episode, rows in grouped.items():
        tasks = {task for task, _ in rows}
        if len(tasks) != 1:
            raise ValueError(f"Episode {episode} has multiple task indices: {tasks}")
        result[episode] = (next(iter(tasks)), np.asarray([action for _, action in rows], dtype=np.float32))
    return result


def make_env(suite_name: str, task_id: int, height: int, width: int, control_freq: int, horizon: int):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_names=list(CAMERA_NAMES),
        camera_heights=height,
        camera_widths=width,
        camera_depths=[True, True],
        control_freq=control_freq,
        horizon=horizon,
        ignore_done=True,
    )
    init_path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    init_states = torch.load(init_path, weights_only=False)
    return suite, task, env, np.asarray(init_states)


def raw_observation(env):
    # OffScreenRenderEnv wraps the task environment in `.env`.
    return env.env._get_observations()


def simulator_state(env):
    sim = env.env.sim
    state = {
        "qpos": np.asarray(sim.data.qpos, dtype=np.float32).copy(),
        "qvel": np.asarray(sim.data.qvel, dtype=np.float32).copy(),
        "body_xpos": np.asarray(sim.data.body_xpos, dtype=np.float32).copy(),
        "body_xmat": np.asarray(sim.data.body_xmat, dtype=np.float32).copy(),
    }
    names = getattr(sim.model, "body_names", None)
    if names is not None:
        state["body_names"] = np.asarray([str(name) for name in names])
    return state


def save_episode(
    out_dir: Path,
    episode_id: int,
    suite_name: str,
    task_id: int,
    task_name: str,
    init_state_id: int,
    frames: dict[str, list[np.ndarray]],
    depths: dict[str, list[np.ndarray]],
    states: list[dict[str, np.ndarray]],
    actions: np.ndarray,
    fps: int,
    source_episode: int,
):
    episode_dir = out_dir / f"episode_{episode_id:05d}"
    episode_dir.mkdir(parents=True, exist_ok=False)
    for camera, camera_frames in frames.items():
        write_video(episode_dir / f"{camera}.mp4", camera_frames, fps)
        np.savez_compressed(
            episode_dir / f"{camera}_depth.npz",
            depth=np.asarray(depths[camera], dtype=np.float16),
        )
    np.savez_compressed(
        episode_dir / "sim_state.npz",
        actions=actions.astype(np.float32),
        qpos=np.stack([state["qpos"] for state in states]),
        qvel=np.stack([state["qvel"] for state in states]),
        body_xpos=np.stack([state["body_xpos"] for state in states]),
        body_xmat=np.stack([state["body_xmat"] for state in states]),
    )
    body_names = states[0].get("body_names", np.asarray([], dtype=str)).tolist()
    (episode_dir / "metadata.json").write_text(
        json.dumps(
            {
                "suite": suite_name,
                "task_id": task_id,
                "task": task_name,
                "source_episode": source_episode,
                "init_state_id": init_state_id,
                "fps": fps,
                "num_frames": len(actions) + 1,
                "cameras": list(CAMERA_NAMES),
                "body_names": body_names,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, action="append", help="Suite(s); default is all four")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/libero"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--control-fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Reuse completed recordings in the output folder")
    args = parser.parse_args()
    if args.episodes_per_task < 1 or args.max_steps < 1:
        parser.error("episodes-per-task and max-steps must be positive")
    suites = args.suite or list(SUITES)
    root = args.dataset_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    task_names = load_task_names(root)
    episodes = load_episode_actions(root)
    by_task: dict[str, list[int]] = defaultdict(list)
    for episode, (task_index, _) in episodes.items():
        by_task[task_names[task_index]].append(episode)
    rng = np.random.default_rng(args.seed)
    manifest = []
    next_episode = 0

    for suite_name in suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        dataset_task_by_name = {name: task_index for task_index, name in task_names.items()}
        for task_id in range(suite.get_num_tasks()):
            task = suite.get_task(task_id)
            task_name = canonical(task.language)
            if task_name not in by_task:
                raise ValueError(f"No local episodes for {suite_name} task {task_id}: {task.language}")
            candidates = sorted(by_task[task_name])
            if len(candidates) < args.episodes_per_task:
                raise ValueError(f"Not enough distinct demonstrations for {suite_name} task {task_id}")
            chosen = sorted(rng.choice(candidates, args.episodes_per_task, replace=False).tolist())
            for source_episode in chosen:
                dataset_task = dataset_task_by_name[task_name]
                action_sequence = episodes[source_episode][1][: args.max_steps]
                row = {
                    "episode": next_episode, "suite": suite_name, "task_id": task_id,
                    "task": task.language, "source_episode": source_episode,
                    "dataset_task_index": dataset_task, "steps": len(action_sequence),
                }
                episode_dir = output / f"episode_{next_episode:05d}"
                if args.resume and (episode_dir / "metadata.json").exists():
                    metadata = json.loads((episode_dir / "metadata.json").read_text())
                    if any(metadata[key] != value for key, value in {
                        "suite": suite_name, "task_id": task_id, "source_episode": source_episode,
                        "fps": args.control_fps, "num_frames": len(action_sequence) + 1,
                    }.items()):
                        raise ValueError(f"Resume configuration differs for {episode_dir}")
                    manifest.append(row)
                    next_episode += 1
                    continue
                _, _, env, init_states = make_env(
                    suite_name, task_id, args.height, args.width, args.control_fps, args.max_steps
                )
                init_state_id = source_episode % len(init_states)
                try:
                    env.reset()
                    env.set_init_state(init_states[init_state_id])
                    frames = {camera: [] for camera in CAMERA_NAMES}
                    depths = {camera: [] for camera in CAMERA_NAMES}
                    states = []
                    for step in range(len(action_sequence) + 1):
                        raw = raw_observation(env)
                        for camera in CAMERA_NAMES:
                            frames[camera].append(np.asarray(raw[f"{camera}_image"])[::-1].copy())
                            depths[camera].append(np.asarray(raw[f"{camera}_depth"])[::-1, :, 0].copy())
                        states.append(simulator_state(env))
                        if step == len(action_sequence):
                            break
                        env.step(np.clip(action_sequence[step], -1.0, 1.0))
                    save_episode(
                        output,
                        next_episode,
                        suite_name,
                        task_id,
                        task.language,
                        init_state_id,
                        frames,
                        depths,
                        states,
                        action_sequence,
                        args.control_fps,
                        source_episode,
                    )
                    manifest.append(row)
                    next_episode += 1
                    save_manifest(output, manifest)
                    print(json.dumps({"stage": "recorded", **row}), flush=True)
                finally:
                    env.close()
    save_manifest(output, manifest)
    print(json.dumps({"output": str(output), "episodes": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
