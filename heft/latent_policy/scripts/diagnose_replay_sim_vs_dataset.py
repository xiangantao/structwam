#!/usr/bin/env python3
"""Open-loop check: the same held-out demonstrations as training-video frames vs simulator renders.

For each re-simulated held-out demonstration (``vace_tracking_all_suites_20261004``: qpos / qvel
per frame), every frame is rendered from its recorded state (rotated 180 deg like the eval) and
paired with the dataset's own frame.  At every ``--stride``-th step both policies predict the
chunk from (a) dataset frames + states and (b) simulator renders + states, with the same noise;
the executed 8 actions are scored against the demonstration (min-max units).  For the maps
policy the student features of the two inputs are compared directly.  No closed loop involved:
if a policy is fine on dataset frames but not on renders, its inputs do not transfer.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_libero_closed_loop import Policy, make_env, observe  # noqa: E402
from latent_dp.data import minmax  # noqa: E402
from latent_dp.policy_maps import MAP_CHANNELS  # noqa: E402
from train_latent_dp_online import load_frames  # noqa: E402


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-9 else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", type=Path, required=True, help="vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--maps-checkpoint", type=Path, required=True)
    parser.add_argument("--rgb-checkpoint", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    from libero.libero import benchmark
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    suite = benchmark.get_benchmark_dict()[args.suite]()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    policies = {"maps": Policy(args.maps_checkpoint, "maps", device, args.student, args.roles, 10, 2),
                "rgb": Policy(args.rgb_checkpoint, "rgb", device)}
    frames, _ = load_frames(args.dataset_root)
    dataset = LeRobotDataset("local/libero", root=str(args.dataset_root), video_backend="pyav")
    episodes = [d for d in sorted(args.episodes_root.glob("episode_*"))
                if json.loads((d / "metadata.json").read_text())["suite"] == args.suite][args.shard::args.num_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    for directory in episodes:
        if (args.output / f"{directory.name}.json").exists():               # resume after a restart
            continue
        meta = json.loads((directory / "metadata.json").read_text())
        state = np.load(directory / "sim_state.npz")
        rows = np.flatnonzero(frames["episode"] == meta["source_episode"])
        env, language = make_env(suite, meta["task_id"])
        env.reset()
        sim_images, sim_states, data_images = [], [], []
        for i, g in enumerate(rows):
            raw = env.set_init_state(np.concatenate(([0.0], state["qpos"][i], state["qvel"][i])))
            images, st = observe(raw)
            sim_images.append(images)
            sim_states.append(st)
            item = dataset[int(g)]
            data_images.append(np.stack([(item[k].numpy() * 255).round().astype(np.uint8)
                                         for k in ("observation.images.image", "observation.images.image2")]))
        env.close()
        data_states = list(frames["state"][rows])
        actions = frames["action"][rows]
        sources = {"data": (data_images, data_states), "sim": (sim_images, sim_states)}
        result = {"episode": directory.name, "task_id": meta["task_id"], "task": language, "length": len(rows),
                  "pixel_abs_diff": float(np.mean([np.abs(a.astype(int) - b).mean() for a, b in zip(sim_images, data_images)])),
                  "state_abs_diff": float(np.abs(np.stack(sim_states) - np.stack(data_states)).mean()), "steps": []}
        for t in range(0, len(rows) - 8, args.stride):
            target = actions[t:t + 8]
            step = {"t": t}
            for model, policy in policies.items():
                low, high = policy.stats["action_min"], policy.stats["action_max"]
                for name, (images, states) in sources.items():
                    predicted = policy.act(images[:t + 1], states[:t + 1], language, seed=1000 + t)
                    step[f"{model}_{name}_l1"] = float(np.abs(minmax(predicted, low, high) - minmax(target, low, high)).mean())
            # Student features of the two inputs.
            online = policies["maps"].online
            feats = {}
            for name, (images, _) in sources.items():
                window = [images[max(i, 0)] for i in range(t - 32, t + 1)]
                rgb = torch.from_numpy(np.stack(window).transpose(1, 0, 2, 3, 4)).to(device)
                maps, tokens, used, trunk, _ = online(rgb, [language, language], [False, True])
                feats[name] = (maps, tokens, used, trunk.float().cpu().numpy())
            (md, td, ud, jd), (ms, ts, us, js) = feats["data"], feats["sim"]
            step["maps_channel_corr"] = {MAP_CHANNELS[c]: round(np.nanmean([corr(md[k, c], ms[k, c]) for k in range(2)]), 3)
                                         for c in range(len(MAP_CHANNELS))}
            step["kept_cells"] = [int(md[k, MAP_CHANNELS.index("kept")].sum()) for k in range(2)] + \
                                 [int(ms[k, MAP_CHANNELS.index("kept")].sum()) for k in range(2)]
            step["keypoints_used"] = [int(ud[k].sum()) for k in range(2)] + [int(us[k].sum()) for k in range(2)]
            both = ud & us
            step["track_token_abs_diff"] = float(np.abs(td[both] - ts[both]).mean()) if both.any() else None
            jd, js = jd.reshape(-1, jd.shape[-1]), js.reshape(-1, js.shape[-1])
            step["trunk_cosine"] = float(((jd * js).sum(-1) / np.maximum(
                np.linalg.norm(jd, axis=-1) * np.linalg.norm(js, axis=-1), 1e-9)).mean())
            result["steps"].append(step)
            print(json.dumps({k: v for k, v in step.items() if k != "maps_channel_corr"}), flush=True)
        (args.output / f"{directory.name}.json").write_text(json.dumps(result) + "\n")
    if "maps" in policies:
        policies["maps"].online.close()


if __name__ == "__main__":
    main()
