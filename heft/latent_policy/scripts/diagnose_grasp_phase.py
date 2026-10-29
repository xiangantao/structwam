#!/usr/bin/env python3
"""Offline action error by phase around the grasp, on held-out libero_spatial demonstrations.

Both policies (DP + maps / tracks / trunk, and vanilla DP) predict chunks from the dataset's own
frames at every ``--stride``-th step of each held-out episode; errors (min-max units, executed 8
actions) are binned by the decision time relative to the demonstration's first gripper close.
Also reports when each policy first predicts closing relative to the demonstration.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import os
if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_libero_closed_loop import Policy  # noqa: E402
from latent_dp.data import SUITES, minmax  # noqa: E402
from train_latent_dp_online import load_frames  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-checkpoint", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--window", type=int, default=24, help="decisions within +-window steps of the grasp")
    args = parser.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    policies = {"maps": Policy(args.maps_checkpoint, "maps", device, args.student, args.roles, 10, 2),
                "rgb": Policy(args.rgb_checkpoint, "rgb", device)}
    frames, names = load_frames(args.dataset_root)
    dataset = LeRobotDataset("local/libero", root=str(args.dataset_root), video_backend="pyav")
    heldout = set(json.loads(args.heldout.read_text())["heldout_source_episodes"])
    suite_tasks = list(SUITES["libero_spatial"])
    episodes = sorted(e for e in heldout if frames["task"][np.flatnonzero(frames["episode"] == e)[0]] in suite_tasks)
    records = []
    for episode in episodes:
        rows = np.flatnonzero(frames["episode"] == episode)
        actions, states = frames["action"][rows], list(frames["state"][rows])
        grasp = int(np.flatnonzero(actions[:, -1] > 0)[0])                    # first close command
        task = names[int(frames["task"][rows[0]])]
        images = []
        for g in rows:
            item = dataset[int(g)]
            images.append(np.stack([(item[k].numpy() * 255).round().astype(np.uint8)
                                    for k in ("observation.images.image", "observation.images.image2")]))
        for t in range(max(grasp - args.window, 0), min(grasp + args.window, len(rows) - 8) + 1, args.stride):
            target = actions[t:t + 8]
            rec = {"episode": int(episode), "t": t, "rel": t - grasp}
            for model, policy in policies.items():
                low, high = policy.stats["action_min"], policy.stats["action_max"]
                pred = policy.act(images[:t + 1], states[:t + 1], task, seed=2000 + t)
                err = np.abs(minmax(pred, low, high) - minmax(target, low, high))
                rec[f"{model}_xyz"] = float(err[:, :3].mean())
                rec[f"{model}_rot"] = float(err[:, 3:6].mean())
                rec[f"{model}_grip"] = float(err[:, 6].mean())
                close = np.flatnonzero(pred[:, -1] > 0)
                rec[f"{model}_first_close"] = int(close[0]) if len(close) else None   # within the chunk
            demo_close = np.flatnonzero(target[:, -1] > 0)
            rec["demo_first_close"] = int(demo_close[0]) if len(demo_close) else None
            records.append(rec)
            print(json.dumps(rec), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records) + "\n")
    bins = [(-24, -9), (-8, -1), (0, 7), (8, 24)]
    for lo, hi in bins:
        sel = [r for r in records if lo <= r["rel"] <= hi]
        if not sel:
            continue
        line = f"rel {lo:+d}..{hi:+d} (n={len(sel)}):"
        for model in policies:
            line += "  %s xyz %.3f rot %.3f grip %.3f |" % (model, *[np.mean([r[f"{model}_{k}"] for r in sel]) for k in ("xyz", "rot", "grip")])
        print(line)
    policies["maps"].online.close()


if __name__ == "__main__":
    main()
