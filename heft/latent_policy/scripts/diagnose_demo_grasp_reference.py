#!/usr/bin/env python3
"""Reference grasp geometry from the re-simulated held-out demonstrations (simulator ground truth).

For each demo, the recorded state of every frame is restored and the eef position, finger qpos,
commanded gripper (the demo's action) and the target object's position are logged; the same
``grasp_metrics`` as the instrumented closed-loop runs are then computed, so the demos say what a
successful grasp looks like at the first close (offset, descent speed, finger gap).
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_libero_closed_loop import grasp_metrics, make_env  # noqa: E402
from train_latent_dp_online import load_frames  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    frames, _ = load_frames(args.dataset_root)
    results = []
    for directory in sorted(args.episodes_root.glob("episode_*")):
        meta = json.loads((directory / "metadata.json").read_text())
        if meta["suite"] != args.suite:
            continue
        state = np.load(directory / "sim_state.npz")
        rows = np.flatnonzero(frames["episode"] == meta["source_episode"])
        actions = frames["action"][rows]
        env, language = make_env(suite, meta["task_id"])
        env.reset()
        name = env.env.obj_of_interest[0]
        target = env.env.obj_body_id[name]
        log = {"eef": [], "fingers": [], "command": [], "object": []}
        for i in range(len(rows)):
            raw = env.set_init_state(np.concatenate(([0.0], state["qpos"][i + 1], state["qvel"][i + 1])))
            log["eef"].append(np.array(raw["robot0_eef_pos"]))
            log["fingers"].append(np.array(raw["robot0_gripper_qpos"]))
            log["command"].append(float(actions[i, -1]))              # action i leads to state i + 1
            log["object"].append(np.array(env.env.sim.data.body_xpos[target]))
        env.close()
        result = {"episode": directory.name, "task_id": meta["task_id"], "object": name,
                  "grasp": grasp_metrics(log)}
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
