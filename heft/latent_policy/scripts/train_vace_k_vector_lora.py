#!/usr/bin/env python3
"""Train a small K-vector LoRA for texture-invariant LIBERO instances.

The loss is applied to the spatial K vectors directly, before RoPE. It does
not build QK matrices, select query centers, run KMeans, or assume an object
count. Simulator surface labels are used only as training/validation pairs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from search_vace_tracking_heads import CAMERAS, ControlTower, load_empty_text, load_vae, read_video
from train_vace_tracking_lora import HeadLoRALinear, make_case


def log(**values):
    print(json.dumps(values), flush=True)


def install_k_lora(tower, layer, head, rank, alpha):
    block = tower.vace_blocks[layer]
    block.attn1.to_k = HeadLoRALinear(block.attn1.to_k, head, 128, rank, alpha)
    tower.to(next(tower.parameters()).device)
    params = [p for p in tower.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("K LoRA has no trainable parameters")
    return params


def photometric_view(video, indices, seed, device, size):
    rng = np.random.default_rng(seed)
    pixels = torch.from_numpy(video[indices].copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
    pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=(size, size), mode="bilinear",
                           align_corners=False).permute(1, 0, 2, 3)
    # Geometry is unchanged; this only perturbs color/texture statistics.
    brightness = float(rng.uniform(.78, 1.22))
    contrast = float(rng.uniform(.78, 1.22))
    channel = torch.as_tensor(rng.uniform(.86, 1.14, 3), device=device, dtype=pixels.dtype)[:, None, None, None]
    pixels = ((pixels - pixels.mean(dim=(1, 2, 3), keepdim=True)) * contrast +
              pixels.mean(dim=(1, 2, 3), keepdim=True)) * brightness
    pixels = pixels * channel
    if rng.random() < .7:
        noise = torch.randn_like(pixels) * float(rng.uniform(.005, .025))
        pixels = pixels + noise
    return pixels.clamp(-1, 1)


@torch.no_grad()
def make_controls(case, video_root, model_root, vae, zero_latents, device, size, seed):
    n = int(case["times"][-1] - case["start_frame"] + 1)
    indices = np.arange(case["start_frame"], case["start_frame"] + n)
    video = read_video(video_root / f"episode_{case['episode']:05d}" / f"{case['camera']}.mp4")
    base_pixels = photometric_view(video, indices, seed + 991, device, size)
    aug_pixels = photometric_view(video, indices, seed + 1991, device, size)
    observed = vae.encode([base_pixels])[0]
    augmented = vae.encode([aug_pixels])[0]
    if n not in zero_latents:
        zero_latents[n] = vae.encode([torch.zeros_like(base_pixels)])[0].cpu()
    reactive = zero_latents[n].to(device)
    def build(observed_latent):
        return torch.cat((observed_latent, reactive, torch.zeros((64, *observed_latent.shape[1:]), device=device)), dim=0).unsqueeze(0).bfloat16().cpu()
    generator = torch.Generator(device=device).manual_seed(seed + case["episode"] * 10000 + case["start_frame"] * 10 + CAMERAS.index(case["camera"]))
    noise = torch.randn((1, *observed.shape), generator=generator, device=device, dtype=torch.float32).bfloat16().cpu()
    aug_control = build(augmented)
    base_control = build(observed)
    del video, base_pixels, aug_pixels, observed, augmented, reactive
    torch.cuda.empty_cache()
    return {"control": base_control, "aug_control": aug_control, "noise": noise}


def capture_k(tower, layer, latent_frames, grid, noise, control, text, timestep, device):
    captured = {}
    def hook(_module, _inputs, output):
        # This VACE processor applies norm_k before head reshape:
        # [B,text+video,H*D]. Reshape only the video suffix; no QK matrix.
        values = output
        video_tokens = latent_frames * grid * grid
        if values.ndim != 3 or values.shape[-2] < video_tokens:
            raise RuntimeError(f"Unexpected norm_k shape: {tuple(values.shape)}")
        hidden = values.shape[-1]
        if hidden % 12:
            raise RuntimeError(f"K hidden dimension is not divisible by 12: {hidden}")
        head_dim = hidden // 12
        video = values[:, -video_tokens:].reshape(1, latent_frames, grid * grid, 12, head_dim)
        captured["k"] = video.permute(0, 3, 1, 2, 4).contiguous()
    handle = tower.vace_blocks[layer].attn1.norm_k.register_forward_hook(hook)
    try:
        tower(noise.to(device), control.to(device), text, torch.tensor([timestep], device=device), lambda *_args: None)
    finally:
        handle.remove()
    if "k" not in captured:
        raise RuntimeError("norm_k hook did not capture K")
    return captured["k"][0]


def point_vectors(k, points, normalize=True):
    # k: [T,S,D], points cells: [P,T].
    cells = torch.as_tensor(points["cells"], device=k.device, dtype=torch.long)
    time_index = torch.arange(k.shape[0], device=k.device)[:, None].expand_as(cells.T)
    vectors = k[time_index, cells.T].permute(1, 0, 2)
    # Equivalent indexing above is [P,T,D].
    return F.normalize(vectors.float(), dim=-1) if normalize else vectors.float()


def vector_object_loss(k, points, margin_cos=0.2, temporal_weight=0.15):
    z = point_vectors(k, points)
    visible = torch.as_tensor(points["visible"], device=z.device, dtype=torch.bool)
    labels = np.asarray(["gripper" if str(x) in {"gripper base", "left finger", "right finger"} else str(x)
                         for x in points["labels"]])
    pulls, pushes, temporal = [], [], []
    for t in range(z.shape[1]):
        ids = np.flatnonzero(visible[:, t].cpu().numpy())
        if len(ids) < 2:
            continue
        sim = z[:, t] @ z[:, t].T
        for label in np.unique(labels[ids]):
            group = ids[labels[ids] == label]
            if len(group) >= 2:
                gi = torch.as_tensor(group, device=z.device)
                off = ~torch.eye(len(group), device=z.device, dtype=torch.bool)
                pulls.append((1 - sim[gi[:, None], gi[None, :]][off]).mean())
        pair_ids = np.asarray([(a, b) for ia, a in enumerate(ids) for b in ids[ia + 1:] if labels[a] != labels[b]], dtype=np.int64)
        if len(pair_ids):
            pair = torch.as_tensor(pair_ids, device=z.device)
            pushes.append(F.relu(sim[pair[:, 0], pair[:, 1]] - margin_cos).pow(2).mean())
    for p in range(z.shape[0]):
        for t in range(z.shape[1] - 1):
            if visible[p, t] and visible[p, t + 1] and (points["groups"][p] == "gripper" or points["displaced"][p, t + 1]):
                temporal.append(1 - (z[p, t] * z[p, t + 1]).sum())
    zero = k.sum() * 0
    pull = torch.stack(pulls).mean() if pulls else zero
    push = torch.stack(pushes).mean() if pushes else zero
    temporal_value = torch.stack(temporal).mean() if temporal else zero
    # Variance floor prevents the selected visible tokens from collapsing.
    std = z[visible].float().std(0) if visible.any() else z.sum(0).flatten() * 0
    variance = F.relu(.05 - std).mean()
    return pull, push, temporal_value, variance


def all_temporal_consistency(k, points):
    """Temporal invariance for every visible surface point, including static objects."""
    z = point_vectors(k, points)
    visible = torch.as_tensor(points["visible"], device=z.device, dtype=torch.bool)
    pairs = visible[:, :-1] & visible[:, 1:]
    if not pairs.any():
        return k.sum() * 0
    return (1 - (z[:, :-1] * z[:, 1:]).sum(-1))[pairs].mean()


def consistency_loss(base_k, aug_k, points):
    base = point_vectors(base_k, points)
    aug = point_vectors(aug_k, points)
    visible = torch.as_tensor(points["visible"], device=base.device, dtype=torch.bool)
    if not visible.any():
        return base.sum() * 0
    return (1 - (base[visible] * aug[visible]).sum(-1)).mean()


@torch.no_grad()
def geometry_metrics(k, points):
    z = point_vectors(k, points).float()
    visible = np.asarray(points["visible"], dtype=bool)
    labels = np.asarray(["gripper" if str(x) in {"gripper base", "left finger", "right finger"} else str(x)
                         for x in points["labels"]])
    ratios = []
    for t in range(z.shape[1]):
        ids = np.flatnonzero(visible[:, t])
        for label in np.unique(labels[ids]):
            same = ids[labels[ids] == label]
            others = ids[labels[ids] != label]
            if len(same) < 2 or not len(others):
                continue
            center = z[same, t].mean(0)
            within = torch.cdist(z[same, t], center[None]).mean().item()
            cross = torch.cdist(z[same, t].mean(0, keepdim=True), z[others, t]).min().item()
            ratios.append(cross / max(within, 1e-5))
    return {"cross_to_within_ratio": float(np.mean(ratios)) if ratios else None, "count": len(ratios)}


def split_episodes(video_root, seed):
    rows = []
    for directory in sorted(video_root.glob("episode_*")):
        metadata_path = directory / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            rows.append({"episode": int(directory.name.split("_")[-1]), **metadata})
    if len(rows) < 100:
        raise RuntimeError(f"Expected the full LIBERO set, found only {len(rows)} episodes")
    rng = np.random.default_rng(seed)
    by_task = {}
    for row in rows:
        by_task.setdefault((row["suite"], row["task_id"]), []).append(row)
    train, val = [], []
    for key, task_rows in sorted(by_task.items()):
        task_rows = list(task_rows)
        rng.shuffle(task_rows)
        n_val = max(1, int(round(len(task_rows) * .2)))
        val.extend(task_rows[:n_val])
        train.extend(task_rows[n_val:])
    return rows, train, val


def make_case_index(video_root, tracks_root, rows, cameras, args, device, seed):
    cases = []
    for row in rows:
        episode = row["episode"]
        for camera in cameras:
            if not (tracks_root / f"episode_{episode:05d}_{camera}_visible_surface.npz").exists():
                continue
            last_start = max(args.skip_frames, ((int(row["num_frames"]) - args.max_frames) // 4) * 4)
            starts = np.unique(np.linspace(args.skip_frames, last_start, args.windows_per_episode).astype(int) // 4 * 4)
            for start in starts:
                try:
                    case = make_case(video_root, tracks_root, args.model, episode, camera, args.size,
                                     int(start), args.max_frames, args.points_per_subject, args.min_motion,
                                     device, seed, video=[None] * int(row["num_frames"]),
                                     points_only=True, include_static=True)
                except ValueError:
                    continue
                case.update(suite=row["suite"], task_id=row["task_id"])
                cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/tracks")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--cameras", nargs="+", default=["agentview", "robot0_eye_in_hand"], choices=list(CAMERAS))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--skip-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=129)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--points-per-subject", type=int, default=24)
    parser.add_argument("--min-motion", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--head", type=int, default=11)
    parser.add_argument("--margin-cos", type=float, default=.2)
    parser.add_argument("--all-temporal-weight", type=float, default=0.0,
                        help="extra temporal consistency for every visible point; 0 preserves the baseline")
    parser.add_argument("--resume-lora", type=Path,
                        help="initialize LoRA from a prior K-vector checkpoint; optimizer starts fresh")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.size % 16 or args.steps < 1:
        parser.error("size must be divisible by 16 and steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    all_rows, train_rows, val_rows = split_episodes(args.video_root, args.seed)
    log(stage="split", episodes=len(all_rows), train_episodes=len(train_rows), val_episodes=len(val_rows),
        suites=sorted({row["suite"] for row in all_rows}), tasks=len({(row["suite"], row["task_id"]) for row in all_rows}))
    text = load_empty_text(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", device)
    log(stage="index_train")
    train_cases = make_case_index(args.video_root, args.tracks_root, train_rows, args.cameras, args, device, args.seed)
    log(stage="index_val")
    val_cases = make_case_index(args.video_root, args.tracks_root, val_rows, args.cameras, args, device, args.seed)
    if not train_cases or not val_cases:
        raise RuntimeError("No usable all-scene LIBERO cases")
    manifest = {"all_episodes": all_rows, "train_episodes": train_rows, "val_episodes": val_rows,
                "train_cases": [{"name": c["name"], "suite": c["suite"], "task_id": c["task_id"], "points": len(c["points"]["ids"])} for c in train_cases],
                "val_cases": [{"name": c["name"], "suite": c["suite"], "task_id": c["task_id"], "points": len(c["points"]["ids"])} for c in val_cases]}
    (args.output / "split.json").write_text(json.dumps(manifest, indent=2))
    log(stage="indexed", train_cases=len(train_cases), val_cases=len(val_cases))
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    zero_latents = {}
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    params = install_k_lora(tower, args.layer, args.head, args.rank, args.alpha)
    if args.resume_lora:
        checkpoint = torch.load(args.resume_lora, map_location="cpu", weights_only=False)
        if checkpoint.get("layer") != args.layer or checkpoint.get("head") != args.head:
            raise ValueError("Resume checkpoint layer/head does not match")
        tower.load_state_dict(checkpoint["lora"], strict=False)
        log(stage="initialized_lora", checkpoint=str(args.resume_lora), source_step=checkpoint.get("step"),
            all_temporal_weight=args.all_temporal_weight)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    buckets = {}
    for index, case in enumerate(train_cases):
        buckets.setdefault((case["suite"], case["task_id"]), []).append(index)
    rng = np.random.default_rng(args.seed)
    task_keys = sorted(buckets)
    def ensure(case):
        if "control" not in case:
            case.update(make_controls(case, args.video_root, args.model, vae, zero_latents, device, args.size,
                                      args.seed + case["episode"] * 37 + case["start_frame"]))
    def evaluate(cases, limit=24):
        tower.eval()
        selected = cases if len(cases) <= limit else [cases[i] for i in np.linspace(0, len(cases) - 1, limit).astype(int)]
        rows = []
        with torch.inference_mode():
            for case in selected:
                ensure(case)
                k = capture_k(tower, args.layer, len(case["times"]), args.size // 16,
                              case["noise"], case["control"], text, 1000, device)[args.head]
                rows.append(geometry_metrics(k, case["points"]))
        valid = [x["cross_to_within_ratio"] for x in rows if x["cross_to_within_ratio"] is not None]
        tower.train()
        return {"cross_to_within_ratio": float(np.mean(valid)) if valid else None,
                "cases": len(rows), "points": int(sum(x["count"] for x in rows))}
    log(stage="before", validation=evaluate(val_cases))
    losses = []
    start_time = time.monotonic()
    for step in range(1, args.steps + 1):
        key = task_keys[int(rng.integers(len(task_keys)))]
        case = train_cases[int(rng.choice(buckets[key]))]
        ensure(case)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base = capture_k(tower, args.layer, len(case["times"]), args.size // 16,
                             case["noise"], case["control"], text, 1000, device)[args.head]
            aug = capture_k(tower, args.layer, len(case["times"]), args.size // 16,
                            case["noise"], case["aug_control"], text, 1000, device)[args.head]
            pull, push, temporal, variance = vector_object_loss(base, case["points"], args.margin_cos)
            all_temporal = all_temporal_consistency(base, case["points"])
            aug_loss = consistency_loss(base, aug, case["points"])
            reg = torch.stack([p.float().pow(2).mean() for p in params]).mean()
            loss = (pull + .8 * push + .15 * temporal + aug_loss + .1 * variance +
                    args.all_temporal_weight * all_temporal + .02 * reg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {case['name']}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % 20 == 0 or step == args.steps:
            log(stage="train", step=step, loss=losses[-1], mean_last20=float(np.mean(losses[-20:])),
                pull=float(pull.detach()), push=float(push.detach()), temporal=float(temporal.detach()),
                all_temporal=float(all_temporal.detach()), texture=float(aug_loss.detach()),
                variance=float(variance.detach()), grad_norm=float(grad_norm),
                case=case["name"], suite=case["suite"], task_id=case["task_id"],
                seconds_per_step=(time.monotonic() - start_time) / step,
                peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 2))
        del base, aug, pull, push, temporal, all_temporal, variance, aug_loss, reg, loss
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"step": step, "layer": args.layer, "head": args.head, "rank": args.rank,
                        "alpha": args.alpha, "all_temporal_weight": args.all_temporal_weight,
                        "lora": {k: v.detach().cpu() for k, v in tower.state_dict().items() if "lora_" in k},
                        "optimizer": optimizer.state_dict()}, args.output / f"k_vector_lora_step_{step:04d}.pt")
            (args.output / "loss.json").write_text(json.dumps(losses))
            log(stage="checkpoint", step=step, validation=evaluate(val_cases))
    torch.save({"step": args.steps, "layer": args.layer, "head": args.head, "rank": args.rank,
                "alpha": args.alpha, "all_temporal_weight": args.all_temporal_weight,
                "lora": {k: v.detach().cpu() for k, v in tower.state_dict().items() if "lora_" in k}},
               args.output / "k_vector_lora.pt")
    (args.output / "loss.json").write_text(json.dumps(losses))
    log(stage="done", output=str(args.output), final_validation=evaluate(val_cases))


if __name__ == "__main__":
    main()
