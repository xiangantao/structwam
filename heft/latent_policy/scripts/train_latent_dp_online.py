#!/usr/bin/env python3
"""Diffusion Policy for one LIBERO suite with VACE student features computed online.

Launch with ``torchrun --nproc_per_node=N``.  Every step each process takes its share of
the global batch, runs the frozen student on the 33-frame window of both cameras for every
sample (no feature cache), post-processes the fields into K blocks / point tracks / word
relevance, and trains the DP (LeRobot UNet, DDPM 100, epsilon, AdamW 1e-4, betas
0.95/0.999, wd 1e-6, cosine with 500 warmup, grad clip 10, EMA).  DP sees 2 proprio
steps; the student keeps its own 33-frame history.  The suite's video frames are decoded
once into shared memory (the dataset, not features).  Feature normalization and the
appearance PCA come from a warm-up pass over random training windows.  The tracking-
validation episodes (one per task, unseen by the student) are held out and scored every
``--val-every`` steps with the EMA policy (DDIM, ``--val-steps`` steps).
"""

import argparse
import glob
import json
import math
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

from latent_dp.data import SUITES, minmax  # noqa: E402
from latent_dp.model import EMA, LatentDiffusionPolicy  # noqa: E402
from latent_dp.rgb import RgbLatentDiffusionPolicy  # noqa: E402
from latent_dp.online import OnlineFeatures, prompt_table  # noqa: E402

CAMERAS = ("observation.images.image", "observation.images.image2")
WINDOW = 33


def log(path, **values):
    line = json.dumps({"time": round(time.time(), 1), **values})
    print(line, flush=True)
    with path.open("a") as stream:
        stream.write(line + "\n")


def load_frames(root):
    import pyarrow.parquet as pq

    columns = ["episode_index", "frame_index", "index", "task_index", "observation.state", "action"]
    tables = [pq.read_table(p, columns=columns).to_pydict() for p in sorted(glob.glob(str(root / "data/chunk-*/*.parquet")))]
    data = {c: np.concatenate([np.asarray(t[c]) for t in tables]) for c in columns}
    order = np.argsort(data["index"])
    data = {c: v[order] for c, v in data.items()}
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    names = {int(row["task_index"]): name for name, row in tasks.iterrows()}
    return {"episode": data["episode_index"], "frame": data["frame_index"], "task": data["task_index"],
            "state": np.stack(data["observation.state"]).astype(np.float32),
            "action": np.stack(data["action"]).astype(np.float32)}, names


def decode_job(job):
    from vace_latent_distill.data import decode_indices, resize_rgb

    path, row, positions, camera = job
    frames = decode_indices(row["video"], range(row["episode_start_frame"], row["episode_start_frame"] + row["length"]),
                            row["fps"])
    frames = resize_rgb(torch.from_numpy(frames), 256).numpy()
    store = np.load(path, mmap_mode="r+")
    store[positions, camera] = frames
    store.flush()
    return len(frames)


def shared_frames(args, rank, episodes, local_of, manifest):
    """uint8 [suite frames, 2, 3, 256, 256] in /dev/shm, decoded once by rank 0."""
    path = Path(f"/dev/shm/latent_dp_{args.suite}_frames.npy")
    done = path.with_suffix(".done")
    if rank == 0 and not done.exists():
        from multiprocessing import get_context

        store = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(local_of), 2, 3, 256, 256))
        del store
        rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        jobs = []
        for row in rows:
            if row["episode"] in episodes:
                positions = np.asarray([local_of[(row["episode"], t)] for t in range(row["length"])])
                jobs.append((path, row, positions, CAMERAS.index(row["camera"])))
        with get_context("spawn").Pool(32) as pool:
            total = sum(pool.map(decode_job, jobs))
        done.write_text(str(total))
    dist.barrier()
    return np.load(path, mmap_mode="r")


def learning_rate(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


class Batcher:
    """Raw samples: windows of both cameras at frame t, states t-1/t, action chunk t-1..t+14."""

    def __init__(self, frames, store, local_of, starts, ends, prompts, task_names, suite):
        self.f, self.store, self.local_of = frames, store, local_of
        self.starts, self.ends, self.prompts, self.task_names = starts, ends, prompts, task_names
        self.offset = SUITES[suite].start

    def __call__(self, indices):
        rgb, images, state, action, pad, task, names, prompt = [], [], [], [], [], [], [], []
        for g in indices:
            start, end = self.starts[g], self.ends[g]
            window = np.clip(np.arange(g - WINDOW + 1, g + 1), start, g)
            local = [self.local_of[(int(self.f["episode"][w]), int(self.f["frame"][w]))] for w in window]
            rgb.append(np.asarray(self.store[local]).transpose(1, 0, 2, 3, 4))      # [2, 33, 3, H, W]
            obs = np.clip(np.arange(g - 1, g + 1), start, end - 1)
            images.append(np.asarray(self.store[[self.local_of[(int(self.f["episode"][o]), int(self.f["frame"][o]))]
                                                 for o in obs]]))                       # [2 steps, 2 cams, 3, H, W]
            span = np.arange(g - 1, g + 15)
            state.append(self.f["state"][obs])
            action.append(self.f["action"][np.clip(span, start, end - 1)])
            pad.append((span < start) | (span >= end))
            t = int(self.f["task"][g])
            task.append(t - self.offset)
            names.append(self.task_names[t])
            prompt.append(self.prompts[t - self.offset])
        return {"rgb": np.stack(rgb), "images": np.stack(images), "state": np.stack(state), "action": np.stack(action), "pad": np.stack(pad),
                "task": np.asarray(task), "names": names, "prompt": np.stack(prompt)}


def features_for(online, raw, device):
    """Student + post-processing for every window of a raw batch -> (objects, tracks, owner) numpy."""
    rgb = torch.from_numpy(raw["rgb"]).to(device)                                   # [B, 2, 33, 3, H, W]
    return online.raw(rgb.flatten(0, 1), [n for n in raw["names"] for _ in range(2)], [False, True] * len(rgb))


def to_batch(online, results, raw, stats, device):
    objects, tracks, owner = online.assemble(results)
    size = len(raw["task"])
    objects, tracks, owner = (objects.reshape(size, 2, *objects.shape[1:]), tracks.reshape(size, 2, *tracks.shape[1:]),
                              owner.reshape(size, 2, -1))
    valid = objects[..., 0] > 0.5
    objects = np.where(valid[..., None], (objects - stats["object_mean"][:, None]) / stats["object_std"][:, None], 0)
    tracks = np.where((owner >= 0)[..., None, None],
                      (tracks - stats["track_mean"][:, None, None]) / stats["track_std"][:, None, None], 0)
    batch = {"objects": objects, "object_valid": valid, "tracks": tracks, "owner": owner, "prompt": raw["prompt"],
             "word_valid": raw["prompt"][..., -1] > 0.5, "task": raw["task"],
             "state": minmax(raw["state"], stats["state_min"], stats["state_max"]),
             "action": minmax(raw["action"], stats["action_min"], stats["action_max"]), "action_is_pad": raw["pad"]}
    out = {}
    for key, value in batch.items():
        value = torch.as_tensor(np.asarray(value))
        out[key] = value.to(device, torch.float32 if value.is_floating_point() else value.dtype)
    out["images"] = torch.from_numpy(raw["images"]).to(device).float().div_(255)    # used by --with-images
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="frozen VACE student")
    parser.add_argument("--words", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="episodes.jsonl (per episode and camera)")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64, help="global")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--stats-samples", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-stride", type=int, default=8)
    parser.add_argument("--val-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--block-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-images", action="store_true",
                        help="vanilla DP image encoder plus the latents as an additional condition")
    parser.add_argument("--stop-step", type=int, default=0, help="end early, same LR schedule")
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("hccl" if os.environ.get("VACE_NPU") == "1" else "nccl")
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
    suite_rows = np.flatnonzero(in_suite)
    local_of = {(int(frames["episode"][g]), int(frames["frame"][g])): i for i, g in enumerate(suite_rows)}
    near_end = (ends - np.arange(len(ends))) <= 7                                   # DP drops the last 7 frames
    train_idx = np.flatnonzero(in_suite & ~np.isin(frames["episode"], list(heldout)) & ~near_end)
    val_idx = np.flatnonzero(in_suite & np.isin(frames["episode"], list(heldout)) & ~near_end)[:: args.val_stride]
    fit_idx = np.sort(np.random.default_rng(1).choice(train_idx, len(val_idx), replace=False))  # same metric on training frames
    store = shared_frames(args, rank, episodes, local_of, args.manifest)

    online = OnlineFeatures(args.checkpoint, args.words, device, args.block_workers)
    words = json.loads(args.words.read_text())
    prompts = prompt_table(words, online.text_cache, [task_names[t] for t in suite_tasks])
    batcher = Batcher(frames, store, local_of, starts, ends, prompts, task_names, args.suite)

    # Warm-up (rank 0): appearance PCA and feature statistics from random training windows.
    stats_box = [None]
    if rank == 0:
        pick = np.random.default_rng(args.seed).choice(train_idx, args.stats_samples, replace=False)
        results = []
        for begin in range(0, len(pick), 16):
            results += features_for(online, batcher(pick[begin:begin + 16]), device)
        online.fit_appearance(results)
        objects, tracks, owner = online.assemble(results)
        objects = objects.reshape(-1, 2, *objects.shape[1:])
        tracks, owner = tracks.reshape(-1, 2, *tracks.shape[1:]), owner.reshape(-1, 2, owner.shape[-1])
        stats = {"object_mean": np.zeros((2, 42), np.float32), "object_std": np.ones((2, 42), np.float32),
                 "track_mean": np.zeros((2, 3), np.float32), "track_std": np.ones((2, 3), np.float32)}
        for camera in range(2):
            rows = objects[:, camera][objects[:, camera, :, 0] > 0.5]
            stats["object_mean"][camera], stats["object_std"][camera] = rows.mean(0), rows.std(0) + 1e-6
            points = tracks[:, camera][owner[:, camera] >= 0].reshape(-1, 3)
            stats["track_mean"][camera], stats["track_std"][camera] = points.mean(0), points.std(0) + 1e-6
        stats["object_mean"][:, 0], stats["object_std"][:, 0] = 0.0, 1.0
        rows = np.flatnonzero(in_suite & ~np.isin(frames["episode"], list(heldout)))
        stats.update(state_min=frames["state"][rows].min(0), state_max=frames["state"][rows].max(0),
                     action_min=frames["action"][rows].min(0), action_max=frames["action"][rows].max(0),
                     k_mean=online.k_mean, k_basis=online.k_basis)
        stats_box = [stats]
    dist.broadcast_object_list(stats_box, src=0)
    stats = stats_box[0]
    online.k_mean, online.k_basis = stats["k_mean"], stats["k_basis"]
    if rank == 0:
        np.savez(args.output / "stats.npz", **stats)

    policy = (RgbLatentDiffusionPolicy(tasks=len(suite_tasks)) if args.with_images
              else LatentDiffusionPolicy(tasks=len(suite_tasks))).to(device)
    model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[local_rank])
    ema = EMA(policy) if rank == 0 else None
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    if rank == 0:
        config = {"suite": args.suite, "dp": policy.dp, "lr": args.lr, "global_batch": args.batch_size,
                  "processes": world, "steps": args.steps, "warmup": args.warmup, "train_frames": int(len(train_idx)),
                  "val_frames": int(len(val_idx)), "student": str(args.checkpoint), "online_features": True,
                  "observation": "rgb resnet18 + latents" if args.with_images else "latents only",
                  "parameters_m": round(sum(p.numel() for p in policy.parameters()) / 1e6, 2)}
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        log(metrics, stage="start", **{k: config[k] for k in ("train_frames", "val_frames", "processes", "parameters_m")})

    timing = {"data": 0.0, "student": 0.0, "dp": 0.0}
    running = []
    tick = time.time()
    stop = args.stop_step or args.steps
    for step in range(1, stop + 1):
        t0 = time.time()
        raw = batcher(rng.choice(train_idx, per_rank, replace=False))
        t1 = time.time()
        batch = to_batch(online, features_for(online, raw, device), raw, stats, device)
        t2 = time.time()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step - 1, args.lr, args.warmup, args.steps)
        trajectory = batch["action"]
        noise = torch.randn_like(trajectory)
        steps = torch.randint(0, policy.dp["num_train_timesteps"], (len(trajectory),), device=device).long()
        noisy = policy.scheduler.add_noise(trajectory, noise, steps)
        prediction = model(noisy, steps, batch)
        loss = torch.nn.functional.mse_loss(prediction, noise)
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
                fit = validate(online, ema.model, batcher, fit_idx, stats, device, args.val_steps)
                log(metrics, step=step, **{k.replace("val_", "fit_"): v for k, v in fit.items()})
            dist.barrier()
        if (step % args.save_every == 0 or step == stop) and rank == 0:
            torch.save({"step": step, "model": policy.state_dict(), "ema": ema.model.state_dict(), "stats": stats,
                        "config": config}, args.output / f"step_{step:07d}.pt")
    if rank == 0:
        log(metrics, stage="done", step=stop)
    online.close()
    dist.destroy_process_group()


@torch.no_grad()
def validate(online, policy, batcher, indices, stats, device, steps):
    policy.eval()
    squared = absolute = gripper = 0.0
    per_dim = 0.0
    count = 0
    generator = torch.Generator(device=device).manual_seed(0)
    for begin in range(0, len(indices), 16):
        raw = batcher(indices[begin:begin + 16])
        batch = to_batch(online, features_for(online, raw, device), raw, stats, device)
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


if __name__ == "__main__":
    main()
