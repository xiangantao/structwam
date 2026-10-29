#!/usr/bin/env python3
"""Create a per-task episode holdout for four-suite surface tracking."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=4)
    args = parser.parse_args()
    if args.episodes_per_task < 2:
        parser.error("each task needs training and held-out episodes")
    rows = json.loads((args.videos / "manifest.json").read_text())
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["task_id"])].append(row)
        episode_dir = args.videos / f"episode_{row['episode']:05d}"
        for camera in ("agentview", "robot0_eye_in_hand"):
            for path in (episode_dir / f"{camera}.mp4",
                         args.tracks / f"episode_{row['episode']:05d}_{camera}_visible_surface.npz"):
                if not path.is_file():
                    raise FileNotFoundError(path)
    expected = {(suite, task) for suite in SUITES for task in range(10)}
    if set(grouped) != expected:
        raise ValueError(f"Task coverage differs: missing={sorted(expected - set(grouped))}")
    train, val = [], []
    for key, episodes in sorted(grouped.items()):
        episodes = sorted(episodes, key=lambda row: row["source_episode"])
        if len(episodes) != args.episodes_per_task or len({row["source_episode"] for row in episodes}) != len(episodes):
            raise ValueError(f"Task {key} needs {args.episodes_per_task} distinct source episodes")
        for row in episodes[:-1]:
            row["split"] = "train"
            train.append(row)
        episodes[-1]["split"] = "val"
        val.append(episodes[-1])
    if {row["source_episode"] for row in train} & {row["source_episode"] for row in val}:
        raise ValueError("Source demonstrations overlap across train and validation")
    result = {
        "protocol": "observed-video tracking; per-task source-episode holdout; both cameras share the split",
        "provenance": "Fresh simulator replays; original demonstration initial states were unavailable",
        "train_episodes": sorted(row["episode"] for row in train),
        "val_episodes": sorted(row["episode"] for row in val),
        "episodes": sorted(train + val, key=lambda row: row["episode"]),
        "counts": {
            "tasks": len(grouped), "train_episodes": len(train), "val_episodes": len(val),
            "train_by_suite": dict(Counter(row["suite"] for row in train)),
            "val_by_suite": dict(Counter(row["suite"] for row in val)),
            "rgb_frames": sum(row["steps"] + 1 for row in rows),
        },
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"stage": "split_ready", **result["counts"]}), flush=True)


if __name__ == "__main__":
    main()
