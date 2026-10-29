#!/usr/bin/env python3
"""Print the main numbers of a ``check_policy_maps.py`` output directory."""

import json
import sys
from pathlib import Path

table = json.loads((Path(sys.argv[1]) / "summary.json").read_text())
HEAT = ["windows", "target_1_hit", "target_1_chance", "target_1_right_instance", "words_best_target_hit",
        "words_right_instance", "destination_1_hit_+16", "destination_1_chance_+16", "destination_1_hit",
        "words_any_destination_hit", "target_segment_hit", "destination_segment_hit", "target_1_on_things"]
FILTER = ["raw_on_things", "kept_on_things", "hand_on_gripper", "gripper_found", "moving_recall_raw", "moving_recall_kept"]
for group in ("all", "third", "wrist"):
    row = table.get(group, {})
    print(f"[{group}] heat   ", "  ".join(f"{k}={row.get(k)}" for k in HEAT))
    print(f"[{group}] filter ", "  ".join(f"{k}={row.get(k)}" for k in FILTER))
    for role in ("hand", "target", "destination"):
        keys = sorted(k for k in row if k.startswith(role + "_") and not k.endswith("_n")
                      and any(t in k for t in ("epe", "hold", "valid_steps", "on_things")) and "_all_" in k or k in
                      (f"{role}_valid_steps", f"{role}_on_things"))
        print(f"[{group}] C {role:11s}", "  ".join(f"{k[len(role) + 1:]}={row.get(k)}" for k in keys))
for group in sorted(g for g in table if "/" in g):
    row = table[group]
    print(group, "  ".join(f"{k}={row.get(k)}" for k in ("windows", "target_1_hit", "target_1_chance",
                                                          "destination_1_hit_+16", "hand_on_gripper", "gripper_found",
                                                          "kept_on_things")))
