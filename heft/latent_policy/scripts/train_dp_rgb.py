#!/usr/bin/env python3
"""Vanilla Diffusion Policy (ResNet-18 on the two cameras) -- control for the latent policy.

Everything except the observation matches ``train_latent_dp_online.py``: the suite's
frames, held-out episodes, DP's dropped last 7 frames, 2 proprio steps, action chunk
t-1 .. t+14 copy-padded, min-max normalization, UNet / DDPM / epsilon, AdamW 1e-4 with
cosine over ``--steps`` and 500 warmup, grad clip 10, EMA, global batch 64, and the
validation (EMA, DDIM 10 steps, executed 8 actions, every ``--val-every`` steps).
Observation: images at t-1 and t from both cameras (uint8 frames in shared memory).
``--stop-step`` ends the run early with the same learning-rate schedule.
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
from latent_dp.rgb import RgbDiffusionPolicy  # noqa: E402
from train_latent_dp_online import learning_rate, load_frames, log, shared_frames  # noqa: E402


class ImageBatcher:
    def __init__(self, frames, store, local_of, starts, ends):
        self.f, self.store, self.local_of, self.starts, self.ends = frames, store, local_of, starts, ends

    def __call__(self, indices):
        images, state, action, pad = [], [], [], []
        for g in indices:
            start, end = self.starts[g], self.ends[g]
            obs = np.clip(np.arange(g - 1, g + 1), start, end - 1)
            local = [self.local_of[(int(self.f["episode"][o]), int(self.f["frame"][o]))] for o in obs]
            images.append(np.asarray(self.store[local]))                             # [2 steps, 2 cams, 3, H, W]
            span = np.arange(g - 1, g + 15)
            state.append(self.f["state"][obs])
            action.append(self.f["action"][np.clip(span, start, end - 1)])
            pad.append((span < start) | (span >= end))
        return {"images": np.stack(images), "state": np.stack(state), "action": np.stack(action),
                "pad": np.stack(pad)}


def to_batch(raw, stats, device):
    return {"images": torch.from_numpy(raw["images"]).to(device).float().div_(255),
            "state": torch.from_numpy(minmax(raw["state"], stats["state_min"], stats["state_max"]).astype(np.float32)).to(device),
            "action": torch.from_numpy(minmax(raw["action"], stats["action_min"], stats["action_max"]).astype(np.float32)).to(device),
            "action_is_pad": torch.from_numpy(raw["pad"]).to(device)}


@torch.no_grad()
def validate(policy, batcher, indices, stats, device, steps):
    policy.eval()
    squared = absolute = gripper = 0.0
    per_dim = 0.0
    count = 0
    generator = torch.Generator(device=device).manual_seed(0)
    for begin in range(0, len(indices), 32):
        batch = to_batch(batcher(indices[begin:begin + 32]), stats, device)
        chunk = policy.sample(batch, scheduler="DDIM", steps=steps, generator=generator)
        predicted, target = policy.executed(chunk), policy.executed(batch["action"])
        valid = ~policy.executed(batch["action_is_pad"].unsqueeze(-1)).squeeze(-1)
        error = (predicted - target)[valid]
        squared += float(error.square().sum())
        absolute += float(error.abs().sum())
        per_dim = per_dim + error.abs().sum(0).cpu().numpy()
        gripper += float(((predicted[..., -1] > 0) == (target[..., -1] > 0))[valid].sum())
        count += int(valid.sum())
    policy.train()
    dims = policy.action_dim
    return {"val_mse": squared / max(count * dims, 1), "val_l1": absolute / max(count * dims, 1),
            "val_gripper_acc": gripper / max(count, 1), "val_actions": count,
            "val_l1_dims": [round(float(v), 4) for v in np.asarray(per_dim) / max(count, 1)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000, help="length of the LR schedule")
    parser.add_argument("--stop-step", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64, help="global")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-stride", type=int, default=8)
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rgb-weights", type=Path, default=None,
                        help="torchvision resnet18-f37072fd.pth: ImageNet-pretrained backbone with BatchNorm "
                             "(LeRobot 0.6 default); omitted = from scratch with GroupNorm")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("hccl" if os.environ.get("VACE_NPU") == "1" else "nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    per_rank = args.batch_size // world
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = args.output / "metrics.jsonl"
    torch.manual_seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)

    frames, _ = load_frames(args.dataset_root)
    heldout = set(json.loads(args.heldout.read_text())["heldout_source_episodes"])
    in_suite = np.isin(frames["task"], list(SUITES[args.suite]))
    episodes = set(np.unique(frames["episode"][in_suite]).tolist())
    change = np.flatnonzero(np.diff(frames["episode"])) + 1
    first, last = np.concatenate(([0], change)), np.concatenate((change, [len(frames["episode"])]))
    starts, ends = np.repeat(first, last - first), np.repeat(last, last - first)
    suite_rows = np.flatnonzero(in_suite)
    local_of = {(int(frames["episode"][g]), int(frames["frame"][g])): i for i, g in enumerate(suite_rows)}
    near_end = (ends - np.arange(len(ends))) <= 7
    train_idx = np.flatnonzero(in_suite & ~np.isin(frames["episode"], list(heldout)) & ~near_end)
    val_idx = np.flatnonzero(in_suite & np.isin(frames["episode"], list(heldout)) & ~near_end)[:: args.val_stride]
    fit_idx = np.sort(np.random.default_rng(1).choice(train_idx, len(val_idx), replace=False))  # same metric on training frames
    store = shared_frames(args, rank, episodes, local_of, args.manifest)
    batcher = ImageBatcher(frames, store, local_of, starts, ends)
    rows = np.flatnonzero(in_suite & ~np.isin(frames["episode"], list(heldout)))
    stats = {"state_min": frames["state"][rows].min(0), "state_max": frames["state"][rows].max(0),
             "action_min": frames["action"][rows].min(0), "action_max": frames["action"][rows].max(0)}

    policy = RgbDiffusionPolicy(rgb_imagenet=args.rgb_weights is not None, rgb_weights=args.rgb_weights).to(device)
    model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank])
    ema = EMA(policy) if rank == 0 else None
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    stop = args.stop_step or args.steps
    if rank == 0:
        np.savez(args.output / "stats.npz", **stats)
        config = {"suite": args.suite, "dp": policy.dp, "rgb_imagenet": args.rgb_weights is not None,
                  "rgb_weights": str(args.rgb_weights) if args.rgb_weights else None,
                  "observation": "rgb resnet18 " + ("imagenet batchnorm" if args.rgb_weights else "groupnorm") + ", crop 224",
                  "lr": args.lr, "global_batch": args.batch_size, "processes": world, "schedule_steps": args.steps,
                  "stop_step": stop, "train_frames": int(len(train_idx)), "val_frames": int(len(val_idx)),
                  "parameters_m": round(sum(p.numel() for p in policy.parameters()) / 1e6, 2)}
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        log(metrics, stage="start", **{k: config[k] for k in ("train_frames", "val_frames", "processes", "parameters_m")})

    running, tick = [], time.time()
    for step in range(1, stop + 1):
        batch = to_batch(batcher(rng.choice(train_idx, per_rank, replace=False)), stats, device)
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
        running.append(float(loss))
        if step % args.log_every == 0 and rank == 0:
            log(metrics, step=step, loss=float(np.mean(running)), lr=optimizer.param_groups[0]["lr"],
                grad_norm=float(grad), s_per_step=round((time.time() - tick) / args.log_every, 2))
            running, tick = [], time.time()
        if step % args.val_every == 0 or step == stop:
            if rank == 0:
                log(metrics, step=step, **validate(ema.model, batcher, val_idx, stats, device, args.val_steps))
                fit = validate(ema.model, batcher, fit_idx, stats, device, args.val_steps)
                log(metrics, step=step, **{k.replace("val_", "fit_"): v for k, v in fit.items()})
            dist.barrier()
        if (step % args.save_every == 0 or step == stop) and rank == 0:
            torch.save({"step": step, "model": policy.state_dict(), "ema": ema.model.state_dict(), "stats": stats,
                        "config": config}, args.output / f"step_{step:07d}.pt")
    if rank == 0:
        log(metrics, stage="done", step=stop)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
