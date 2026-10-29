#!/usr/bin/env python3
"""Fixed small window sets for fast cached-teacher experiments.

``train``: per task, ``--train-per-task`` LeRobot episodes (at most all of them), both
cameras, ``--starts-per-episode`` distinct starts per episode shared by the two cameras
(the first start is the same as with one start).  ``heldout``: the tracking dataset's validation
episodes (one per task, with simulator surface tracks and depth), mapped to their
LeRobot source episodes and excluded from ``train``.  Held-out starts are multiples
of 4 so every native readout time lands on an annotated GT frame.  GT is only used
for evaluation.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vace_latent_distill.data import stable_seed


def window(row, start, split, seed, **extra):
    absolute = row["episode_start_frame"] + start
    sample_id = f"{row['id']}-s{start:04d}-{split}"
    return {**row, **extra, "split": split, "start": start, "sample_id": sample_id,
            "frame_indices": list(range(absolute, absolute + 49)),
            "teacher_seed": stable_seed(seed, row["id"], start, "local-cache-20261015")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-task", type=int, default=3)
    parser.add_argument("--starts-per-episode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    by_episode = defaultdict(dict)
    for row in rows:
        by_episode[row["episode"]][row["camera"]] = row
    tracking = json.loads((args.tracking_root / "videos/manifest.json").read_text())
    validation = set(json.loads((args.tracking_root / "split.json").read_text())["val_episodes"])
    heldout, held_sources = [], set()
    for item in sorted(tracking, key=lambda value: value["episode"]):
        if item["episode"] not in validation:
            continue
        cameras = by_episode[item["source_episode"]]
        length = min(row["length"] for row in cameras.values())
        start = min(4 * round((length - 49) / 8), 4 * ((length - 49) // 4))
        held_sources.add(item["source_episode"])
        for camera in sorted(cameras):
            heldout.append(window(cameras[camera], start, "heldout", args.seed,
                                  tracking_episode=item["episode"], suite=item["suite"]))
    by_task = defaultdict(list)
    for episode, cameras in by_episode.items():
        if episode not in held_sources and len(cameras) == 2:
            by_task[next(iter(cameras.values()))["task"]].append(episode)
    train = []
    for task in sorted(by_task):
        rng = np.random.default_rng(stable_seed(args.seed, task, "cache-train"))
        count = min(args.train_per_task, len(by_task[task]))
        for episode in sorted(rng.choice(sorted(by_task[task]), count, replace=False)):
            cameras = by_episode[int(episode)]
            length = min(row["length"] for row in cameras.values())
            draws = np.random.default_rng(stable_seed(args.seed, int(episode)))
            starts = []
            while len(starts) < min(args.starts_per_episode, length - 48):
                start = int(draws.integers(0, length - 48))
                if start not in starts:
                    starts.append(start)
            for start in starts:
                for camera in sorted(cameras):
                    train.append(window(cameras[camera], start, "train", args.seed))
    args.output.mkdir(parents=True, exist_ok=True)
    windows = train + heldout
    if len({w["sample_id"] for w in windows}) != len(windows):
        raise RuntimeError("Duplicate window ids")
    (args.output / "windows.jsonl").write_text("".join(json.dumps(w) + "\n" for w in windows))
    summary = {"train_windows": len(train), "heldout_windows": len(heldout),
               "tasks": len(by_task), "train_per_task": args.train_per_task,
               "starts_per_episode": args.starts_per_episode,
               "heldout_tracking_episodes": sorted(validation),
               "heldout_source_episodes": sorted(held_sources), "seed": args.seed}
    (args.output / "windows_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("heldout_")}))


if __name__ == "__main__":
    main()
