#!/usr/bin/env python3
"""Success rates (overall and per task) of ``launch_libero_eval.sh`` runs, from the shard jsonl files."""

import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ma-user/work/2337888765/workspace/runs/libero_eval_20261018/full")
expected = int(sys.argv[2]) if len(sys.argv) > 2 else 500
for model in sorted(p for p in root.iterdir() if p.is_dir()):
    episodes = {}
    for path in model.glob("episodes_shard*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                episodes[(row["task_id"], row["episode"])] = row
    if not episodes:
        print(f"{model.name}: no episodes yet")
        continue
    tasks = defaultdict(list)
    for (task, _), row in episodes.items():
        tasks[task].append(row)
    rate = sum(r["success"] for r in episodes.values()) / len(episodes)
    seconds = sum(r["seconds"] for r in episodes.values()) / len(episodes)
    print(f"{model.name}: {len(episodes)}/{expected} episodes, success {rate:.3f}, {seconds:.0f} s/episode")
    for task in sorted(tasks):
        rows = tasks[task]
        print(f"  task {task}: {sum(r['success'] for r in rows)}/{len(rows)}  {rows[0]['task']}")
