#!/usr/bin/env python3
"""Vanilla Diffusion Policy plus the VACE student's policy maps (A), keypoint tracks (C) and
V-JEPA2 trunk, computed online, for one LIBERO suite.

Launch with ``torchrun --nproc_per_node=N``.  Everything vanilla matches ``train_dp_rgb.py``
(ResNet-18 GroupNorm per camera, crop 224, 2 obs steps, UNet / DDPM 100 / epsilon, AdamW
1e-4 betas 0.95/0.999 wd 1e-6, cosine with 500 warmup, grad clip 10, EMA, global batch 64,
same frames, held-out episodes and validation).  On top, every step the frozen student runs
on the 33-frame window of both cameras for every sample (no feature cache):

* A: ``policy_maps.MAP_CHANNELS`` per camera, 32x32 (role / word heat now and +16, filtered
  displacements at -32/-16/-8/+8/+16 frames, wrist camera motion, hand / target /
  destination / kept masks, K PCA, wrist depth);
* C: up to 32 filtered keypoint tracks per camera (hand / target / destination), 13 steps;
* J: the student's V-JEPA2 trunk, all 17 time steps.

Validation also reports the EMA policy with each branch zeroed (``val_l1_without_*``).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latent_dp.data import SUITES, minmax  # noqa: E402
from latent_dp.model import EMA  # noqa: E402
from latent_dp.online import OnlineMaps  # noqa: E402
from latent_dp.policy_maps import MAP_CHANNELS  # noqa: E402
from latent_dp.rgb import MapsDiffusionPolicy  # noqa: E402
from train_latent_dp_online import Batcher, learning_rate, load_frames, log, shared_frames  # noqa: E402


def features(online, raw, device):
    rgb = torch.from_numpy(raw["rgb"]).to(device)                                   # [B, 2, 33, 3, H, W]
    size = len(rgb)
    maps, tokens, used, trunk, k_sample = online(rgb.flatten(0, 1), [n for n in raw["names"] for _ in range(2)],
                                                [False, True] * size)
    return {"maps": maps.reshape(size, 2, *maps.shape[1:]), "tracks": tokens.reshape(size, 2, *tokens.shape[1:]),
            "track_used": used.reshape(size, 2, -1), "trunk": trunk.reshape(size, 2, *trunk.shape[1:]),
            "k_sample": k_sample}


def to_batch(feats, raw, stats, device):
    batch = {"maps": torch.from_numpy(feats["maps"]).to(device),
             "tracks": torch.from_numpy(feats["tracks"]).to(device),
             "track_used": torch.from_numpy(feats["track_used"]).to(device),
             "trunk": feats["trunk"],
             "images": torch.from_numpy(raw["images"]).to(device).float().div_(255),
             "state": torch.from_numpy(minmax(raw["state"], stats["state_min"], stats["state_max"]).astype(np.float32)).to(device),
             "action": torch.from_numpy(minmax(raw["action"], stats["action_min"], stats["action_max"]).astype(np.float32)).to(device),
             "action_is_pad": torch.from_numpy(raw["pad"]).to(device)}
    return batch


@torch.no_grad()
def validate(online, policy, batcher, indices, stats, device, steps, ablations=True):
    policy.eval()
    variants = {"": ()}
    if ablations:
        variants.update({f"_without_{name}": (name,) for name in policy.BRANCHES})
    totals = {v: {"abs": 0.0, "sq": 0.0, "grip": 0.0, "dims": 0.0} for v in variants}
    count = 0
    for begin in range(0, len(indices), 16):
        raw = batcher(indices[begin:begin + 16])
        batch = to_batch(features(online, raw, device), raw, stats, device)
        target = policy.executed(batch["action"])
        valid = ~policy.executed(batch["action_is_pad"].unsqueeze(-1)).squeeze(-1)
        for name, drop in variants.items():
            generator = torch.Generator(device=device).manual_seed(begin)
            chunk = policy.sample({**batch, "drop": drop}, scheduler="DDIM", steps=steps, generator=generator)
            error = (policy.executed(chunk) - target)[valid]
            totals[name]["abs"] += float(error.abs().sum())
            totals[name]["sq"] += float(error.square().sum())
            totals[name]["dims"] = totals[name]["dims"] + error.abs().sum(0).cpu().numpy()
            totals[name]["grip"] += float(((policy.executed(chunk)[..., -1] > 0) == (target[..., -1] > 0))[valid].sum())
        count += int(valid.sum())
    policy.train()
    dims = policy.action_dim
    out = {"val_actions": count}
    for name, t in totals.items():
        out[f"val_l1{name}"] = t["abs"] / max(count * dims, 1)
        if not name:
            out["val_mse"] = t["sq"] / max(count * dims, 1)
            out["val_gripper_acc"] = t["grip"] / max(count, 1)
            out["val_l1_dims"] = [round(float(v), 4) for v in np.asarray(t["dims"]) / max(count, 1)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="frozen VACE student")
    parser.add_argument("--roles", type=Path, required=True, help="select_policy_roles.py output")
    parser.add_argument("--manifest", type=Path, required=True, help="episodes.jsonl (per episode and camera)")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000, help="length of the LR schedule")
    parser.add_argument("--stop-step", type=int, default=0, help="end early, same LR schedule")
    parser.add_argument("--batch-size", type=int, default=64, help="global")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--branch-dropout", type=float, default=0.1)
    parser.add_argument("--appearance-samples", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-stride", type=int, default=8)
    parser.add_argument("--val-limit", type=int, default=0, help="first N validation frames only (smoke tests)")
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--map-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchors", action="store_true",
                        help="SCDP-style image features sampled at the student's keypoints (multi-scale ResNet)")
    parser.add_argument("--phase-strength", type=float, default=0.0,
                        help="GAP-style: scale the student branches' gradients by 1 - strength * rho, rho = "
                             "proximity to the demo's gripper open/close events (0 = off)")
    parser.add_argument("--phase-sigma", type=float, default=8.0, help="rho = exp(-d^2 / 2 sigma^2), d in steps")
    parser.add_argument("--rgb-weights", type=Path, default=None,
                        help="torchvision resnet18-f37072fd.pth: ImageNet-pretrained backbone with BatchNorm "
                             "(LeRobot 0.6 default); omitted = from scratch with GroupNorm")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist.init_process_group("hccl" if os.environ.get("VACE_NPU") == "1" and world_size > 1 else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.batch_size % world:
        raise ValueError("Global batch must divide by the number of processes")
    per_rank = args.batch_size // world
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = args.output / "metrics.jsonl"
    torch.manual_seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)

    frames, task_names = load_frames(args.dataset_root)
    suite_tasks = list(SUITES[args.suite])
    heldout = set(json.loads(args.heldout.read_text())["heldout_source_episodes"])
    in_suite = np.isin(frames["task"], suite_tasks)
    episodes = set(np.unique(frames["episode"][in_suite]).tolist())
    change = np.flatnonzero(np.diff(frames["episode"])) + 1
    first, last = np.concatenate(([0], change)), np.concatenate((change, [len(frames["episode"])]))
    starts, ends = np.repeat(first, last - first), np.repeat(last, last - first)
    # Grasp / release transitions of each demonstration: steps where the commanded gripper flips.
    rho = np.zeros(len(frames["action"]), np.float32)
    for a, b in zip(first, last):
        grip = frames["action"][a:b, -1] > 0
        events = a + np.flatnonzero(np.diff(grip.astype(np.int8)) != 0) + 1
        if len(events):
            distance = np.abs(np.arange(a, b)[:, None] - events[None]).min(1)
            rho[a:b] = np.exp(-distance.astype(np.float32) ** 2 / (2 * args.phase_sigma ** 2))
    suite_rows = np.flatnonzero(in_suite)
    local_of = {(int(frames["episode"][g]), int(frames["frame"][g])): i for i, g in enumerate(suite_rows)}
    near_end = (ends - np.arange(len(ends))) <= 7                                   # DP drops the last 7 frames
    train_idx = np.flatnonzero(in_suite & ~np.isin(frames["episode"], list(heldout)) & ~near_end)
    val_idx = np.flatnonzero(in_suite & np.isin(frames["episode"], list(heldout)) & ~near_end)[:: args.val_stride]
    if args.val_limit:
        val_idx = val_idx[: args.val_limit]
    fit_idx = np.sort(np.random.default_rng(1).choice(train_idx, len(val_idx), replace=False))  # same metric on training frames
    store = shared_frames(args, rank, episodes, local_of, args.manifest)

    online = OnlineMaps(args.checkpoint, args.roles, device, args.map_workers)
    prompts = np.zeros((len(suite_tasks), 3, 66), np.float32)                      # unused by this policy
    batcher = Batcher(frames, store, local_of, starts, ends, prompts, task_names, args.suite)

    # Rank 0: appearance PCA of the present K from random training windows; action/state ranges.
    stats_box = [None]
    if rank == 0:
        pick = np.random.default_rng(args.seed).choice(train_idx, args.appearance_samples, replace=False)
        samples = [features(online, batcher(pick[b:b + 16]), device)["k_sample"] for b in range(0, len(pick), 16)]
        online.fit_appearance(samples)
        rows = np.flatnonzero(in_suite & ~np.isin(frames["episode"], list(heldout)))
        stats_box = [{"state_min": frames["state"][rows].min(0), "state_max": frames["state"][rows].max(0),
                      "action_min": frames["action"][rows].min(0), "action_max": frames["action"][rows].max(0),
                      "k_mean": online.k_basis[0], "k_basis": online.k_basis[1]}]
    dist.broadcast_object_list(stats_box, src=0)
    stats = stats_box[0]
    online.k_basis = (stats["k_mean"], stats["k_basis"])

    probe = features(online, batcher(train_idx[:1]), device)
    policy = MapsDiffusionPolicy(len(MAP_CHANNELS), probe["tracks"].shape[-1], branch_dropout=args.branch_dropout,
                                 anchors=args.anchors,
                                 rgb_imagenet=args.rgb_weights is not None, rgb_weights=args.rgb_weights).to(device)
    model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank]) if world > 1 else policy
    ema = EMA(policy) if rank == 0 else None
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    stop = args.stop_step or args.steps
    if rank == 0:
        np.savez(args.output / "stats.npz", **stats)
        branches = {"maps": sum(p.numel() for p in policy.maps.parameters()),
                    "tracks": sum(p.numel() for p in policy.tracks.parameters()),
                    "trunk": sum(p.numel() for p in policy.trunk.parameters())}
        config = {"suite": args.suite, "rgb_imagenet": args.rgb_weights is not None,
                  "rgb_weights": str(args.rgb_weights) if args.rgb_weights else None,
                  "observation": "rgb resnet18 " + ("imagenet batchnorm" if args.rgb_weights else "groupnorm")
                                 + " crop 224 + policy maps A + tracks C + trunk J",
                  "dp": policy.dp, "lr": args.lr, "global_batch": args.batch_size, "processes": world,
                  "schedule_steps": args.steps, "stop_step": stop, "warmup": args.warmup,
                  "branch_dropout": args.branch_dropout, "map_channels": list(MAP_CHANNELS),
                  "anchors": bool(args.anchors), "phase_strength": args.phase_strength, "phase_sigma": args.phase_sigma,
                  "phase_rho_mean_train": float(rho[train_idx].mean()),
                  "track_token_dim": int(probe["tracks"].shape[-1]), "train_frames": int(len(train_idx)),
                  "val_frames": int(len(val_idx)), "student": str(args.checkpoint), "online_features": True,
                  "branch_parameters_m": {k: round(v / 1e6, 2) for k, v in branches.items()},
                  "parameters_m": round(sum(p.numel() for p in policy.parameters()) / 1e6, 2)}
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        log(metrics, stage="start", **{k: config[k] for k in ("train_frames", "val_frames", "processes", "parameters_m",
                                                               "branch_parameters_m")})

    timing = {"data": 0.0, "student": 0.0, "dp": 0.0}
    running, tick = [], time.time()
    for step in range(1, stop + 1):
        t0 = time.time()
        picked = rng.choice(train_idx, per_rank, replace=False)
        raw = batcher(picked)
        t1 = time.time()
        batch = to_batch(features(online, raw, device), raw, stats, device)
        if args.phase_strength > 0:
            batch["grad_scale"] = torch.from_numpy(np.clip(1 - args.phase_strength * rho[picked], 0, 1)).to(device)
        t2 = time.time()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step - 1, args.lr, args.warmup, args.steps)
        trajectory = batch["action"]
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(0, policy.dp["num_train_timesteps"], (len(trajectory),), device=device).long()
        noisy = policy.scheduler.add_noise(trajectory, noise, timesteps)
        loss = torch.nn.functional.mse_loss(model(noisy, timesteps, batch), noise)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
        if ema is not None:
            ema.update(policy)
        torch.cuda.synchronize(device)
        t3 = time.time()
        timing["data"] += t1 - t0
        timing["student"] += t2 - t1
        timing["dp"] += t3 - t2
        running.append(float(loss))
        if step % args.log_every == 0 and rank == 0:
            log(metrics, step=step, loss=float(np.mean(running)), lr=optimizer.param_groups[0]["lr"], grad_norm=float(grad),
                s_per_step=round((time.time() - tick) / args.log_every, 2),
                **{f"t_{k}": round(v / args.log_every, 2) for k, v in timing.items()})
            running, tick = [], time.time()
            timing = dict.fromkeys(timing, 0.0)
        if step % args.val_every == 0 or step == stop:
            if rank == 0:
                log(metrics, step=step, **validate(online, ema.model, batcher, val_idx, stats, device, args.val_steps))
                fit = validate(online, ema.model, batcher, fit_idx, stats, device, args.val_steps, ablations=False)
                log(metrics, step=step, **{k.replace("val_", "fit_"): v for k, v in fit.items()})
            dist.barrier()
        if (step % args.save_every == 0 or step == stop) and rank == 0:
            torch.save({"step": step, "model": policy.state_dict(), "ema": ema.model.state_dict(), "stats": stats,
                        "config": config}, args.output / f"step_{step:07d}.pt")
    if rank == 0:
        log(metrics, stage="done", step=stop)
    online.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
