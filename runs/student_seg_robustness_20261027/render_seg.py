"""Render LIBERO-plus initial scenes with simulator instance segmentation (evaluation only).

Per perturbation category, ``--per-category`` tasks are sampled evenly over the four suites
(seed fixed).  Each scene: reset, the task's first init state, 30 zero-action settle steps as in
the policy eval (LIBERO-plus applies its sensor noise to the agentview image inside ``step``),
then the agentview and wrist RGB plus their instance segmentation are saved, rotated 180 degrees
like the policy input.  The segmentation is ground truth for scoring only.

    python render_seg.py --output DIR --per-category 24 --shard I --shards N
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

CAMERAS = (("agentview", "head"), ("robot0_eye_in_hand", "wrist"))
ROBOT = ("Panda0", "PandaGripper0", "RethinkMount0")


def sample_tasks(libero_root: Path, per_category: int, seed: int):
    cls = json.loads((libero_root / "libero/libero/benchmark/task_classification.json").read_text())
    rng = np.random.default_rng(seed)
    by_cat = {}
    for suite, items in cls.items():
        for item in items:
            by_cat.setdefault(item["category"], {}).setdefault(suite, []).append(item)
    jobs = []
    for category in sorted(by_cat):
        suites = sorted(by_cat[category])
        for k in range(per_category):
            suite = suites[k % len(suites)]
            pool = by_cat[category][suite]
            item = pool[int(rng.integers(len(pool)))]
            jobs.append({"suite": suite, "task_id": int(item["id"]) - 1, "category": category,
                         "difficulty": item.get("difficulty_level"), "name": item["name"]})
    return jobs


def render(job, out: Path, size: int):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[job["suite"]]()
    task = suite.get_task(job["task_id"])
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=size, camera_widths=size,
                             camera_segmentations="instance")
    try:
        env.reset()
        init = suite.get_task_init_states(job["task_id"])
        obs = env.set_init_state(init[0]) if len(init) else None
        for _ in range(30):
            obs, _, _, _ = env.step(np.zeros(7, np.float32))
        names = list(env.env.model.instances_to_ids.keys())
        record = {}
        for key, camera in CAMERAS:
            rgb = np.asarray(obs[f"{key}_image"])[::-1, ::-1]
            seg = np.asarray(obs[f"{key}_segmentation_instance"])[::-1, ::-1]
            record[f"{camera}_rgb"] = np.ascontiguousarray(rgb).astype(np.uint8)
            record[f"{camera}_seg"] = np.ascontiguousarray(seg.squeeze(-1) if seg.ndim == 3 else seg).astype(np.int32)
        stem = f"{job['suite']}_{job['task_id']:05d}"
        np.savez_compressed(out / f"{stem}.npz", **record)
        meta = {**job, "language": task.language, "instances": names, "robot": [n for n in names if n in ROBOT]}
        (out / f"{stem}.json").write_text(json.dumps(meta))
        return stem
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-category", type=int, default=24)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    libero_root = Path(os.environ["LIBERO_PLUS_PATH"])
    jobs = sample_tasks(libero_root, args.per_category, args.seed)
    if args.shard == 0:
        (out / "jobs.json").write_text(json.dumps(jobs, indent=1))
    for i, job in enumerate(jobs):
        if i % args.shards != args.shard:
            continue
        stem = f"{job['suite']}_{job['task_id']:05d}"
        if (out / f"{stem}.npz").exists():
            continue
        try:
            print("done", render(job, out, args.size), job["category"], flush=True)
        except Exception as exc:  # keep going; failures are listed
            print("FAIL", stem, job["category"], repr(exc)[:300], flush=True)


if __name__ == "__main__":
    main()
