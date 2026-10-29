#!/usr/bin/env python3
"""Train a Diffusion Policy for one LIBERO suite on post-processed VACE student features.

Settings follow Chi et al. 2023 / LeRobot's DiffusionConfig: 2 observation steps, horizon 16,
8 executed actions, UNet (512, 1024, 2048) with FiLM, DDPM with 100 squared-cosine steps and
epsilon prediction, AdamW (lr 1e-4, betas 0.95/0.999, weight decay 1e-6), cosine schedule
with 500 warmup steps, batch 64, EMA of weights.  The tracking-validation source episodes
(one per task, never seen by the student) are held out; every ``--val-every`` steps the EMA
policy predicts their action chunks (DDIM, ``--val-steps`` denoising steps) and the error of
the executed actions is logged.  Closed-loop LIBERO evaluation is a separate step.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_dp.data import SUITES, Features, LatentDPDataset, compute_stats, suite_frames  # noqa: E402
from latent_dp.model import EMA, LatentDiffusionPolicy  # noqa: E402


def log(path, **values):
    line = json.dumps({"time": round(time.time(), 1), **values})
    print(line, flush=True)
    with path.open("a") as stream:
        stream.write(line + "\n")


def learning_rate(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(policy, loader, device, scheduler, steps):
    policy.eval()
    squared, absolute, gripper, count = 0.0, 0.0, 0.0, 0
    generator = torch.Generator(device=device).manual_seed(0)
    for batch in loader:
        batch = to_device(batch, device)
        chunk = policy.sample(batch, scheduler=scheduler, steps=steps, generator=generator)
        predicted, target = policy.executed(chunk), policy.executed(batch["action"])
        valid = ~policy.executed(batch["action_is_pad"][..., None].float()).bool()[..., 0]
        error = (predicted - target)[valid]
        squared += float(error.square().sum())
        absolute += float(error.abs().sum())
        gripper += float(((predicted[..., -1] > 0) == (target[..., -1] > 0))[valid].sum())
        count += int(valid.sum())
    policy.train()
    if not count:
        return {"val_actions": 0}
    dims = predicted.shape[-1]
    return {"val_mse": squared / (count * dims), "val_l1": absolute / (count * dims),
            "val_gripper_acc": gripper / count, "val_actions": count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--heldout", type=Path, required=True, help="windows_summary.json with heldout_source_episodes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=10000)
    parser.add_argument("--val-stride", type=int, default=8)
    parser.add_argument("--val-scheduler", default="DDIM")
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    metrics = args.output / "metrics.jsonl"

    features = Features(args.features)
    heldout = json.loads(args.heldout.read_text())["heldout_source_episodes"]
    train_idx = suite_frames(features, args.suite, exclude=heldout)
    val_idx = suite_frames(features, args.suite, episodes=heldout)[:: args.val_stride]
    stats = compute_stats(features, train_idx)
    np.savez(args.output / "stats.npz", **stats)
    train = LatentDPDataset(features, train_idx, args.suite, stats)
    val = LatentDPDataset(features, val_idx, args.suite, stats)
    loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                         num_workers=args.workers, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=256, shuffle=False, num_workers=2)
    policy = LatentDiffusionPolicy(tasks=len(SUITES[args.suite])).to(device)
    ema = EMA(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    config = {"suite": args.suite, "dp": policy.dp, "lr": args.lr, "batch_size": args.batch_size, "steps": args.steps,
              "warmup": args.warmup, "train_frames": len(train_idx), "val_frames": len(val_idx),
              "train_episodes": int(len(np.unique(features.frames["episode"][train_idx]))),
              "heldout_episodes": sorted(set(heldout) & set(features.frames["episode"][val_idx].tolist())),
              "parameters_m": round(sum(p.numel() for p in policy.parameters()) / 1e6, 2),
              "features": str(args.features), "feature_meta": {k: features.meta[k] for k in ("columns", "cameras")}}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log(metrics, stage="start", **{k: config[k] for k in ("train_frames", "val_frames", "train_episodes", "parameters_m")})

    step, tick, running = 0, time.time(), []
    while step < args.steps:
        for batch in loader:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, args.lr, args.warmup, args.steps)
            loss = policy.loss(to_device(batch, device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            ema.update(policy)
            running.append(float(loss))
            step += 1
            if step % args.log_every == 0:
                log(metrics, step=step, loss=float(np.mean(running)), lr=optimizer.param_groups[0]["lr"],
                    grad_norm=float(grad), steps_per_s=round(args.log_every / (time.time() - tick), 2))
                running, tick = [], time.time()
            if step % args.val_every == 0 or step == args.steps:
                log(metrics, step=step, **validate(ema.model, val_loader, device, args.val_scheduler, args.val_steps))
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"step": step, "model": policy.state_dict(), "ema": ema.model.state_dict(),
                            "config": config, "stats": stats}, args.output / f"step_{step:07d}.pt")
            if step >= args.steps:
                break
    log(metrics, stage="done", step=step)


if __name__ == "__main__":
    main()
