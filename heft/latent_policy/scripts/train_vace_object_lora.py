#!/usr/bin/env python3
"""Batched object-only LoRA with phase-balanced, multi-hop observed tracking."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from search_vace_tracking_heads import (
    CAMERAS, GRIPPER, ROOT, ControlTower, cell_to_xy, load_empty_text,
    load_vae, read_video, select_points, xy_to_cell,
)
from train_vace_tracking_lora import install_lora
from scan_vace_object_tracking_heads import decode_native, measure
from summarize_vace_object_head_scan import full_track_quality
from vace_vae_warmup import observation_control, observation_window_indices


PHASES = ("moving", "near_stationary")


def log(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


class Observations:
    """Keep raw observations/annotations on CPU; no teacher feature cache."""

    def __init__(self, args):
        self.args = args
        self.cache = OrderedDict()

    def get(self, episode, camera):
        key = episode, camera
        if key not in self.cache:
            directory = self.args.video_root / f"episode_{episode:05d}"
            video = read_video(directory / f"{camera}.mp4")
            meta = json.loads((directory / "metadata.json").read_text())
            with np.load(directory / "sim_state.npz") as z:
                hand = z["body_xpos"][:, meta["body_names"].index("gripper0_eef")].copy()
            with np.load(self.args.tracks_root / f"episode_{episode:05d}_{camera}_visible_surface.npz") as z:
                data = {k: z[k] for k in ("xy", "visible", "world_xyz", "point_label",
                                          "local_xyz", "annotated_frame_indices")}
            self.cache[key] = (video, data, hand)
            if len(self.cache) > self.args.raw_cache_cases:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def points(self, case):
        _, data, hand = self.get(case["episode"], case["camera"])
        times = np.arange(case["start_frame"], case["start_frame"] + case["rgb_frames"], 4)
        if not np.isin(times, data["annotated_frame_indices"]).all():
            raise ValueError("Unannotated retained frame")
        points = select_points(data, times, self.args.size // 16, self.args.points_per_subject,
                               0.02, surface_motion=True, include_static=True)
        keep = ~np.isin(points["labels"], list(GRIPPER))
        for key in ("ids", "anchor", "groups", "xy", "valid", "displaced", "cells", "labels"):
            points[key] = points[key][keep]
        if not len(points["ids"]):
            raise ValueError("No visible object points")
        points["visible"] = data["visible"][times][:, points["ids"]].T
        labels = np.asarray(["gripper" if x in GRIPPER else str(x) for x in data["point_label"]])
        identities = sorted(set(labels))
        owners = np.asarray([identities.index(x) for x in labels])
        point_owners = np.asarray([identities.index(str(x)) for x in points["labels"]])
        grid = self.args.size // 16
        occupancy = np.full((len(times), grid * grid), -1, np.int32)
        all_cells = xy_to_cell(data["xy"][times], grid, 256)
        for ti, frame in enumerate(times):
            cells, current_owners = all_cells[ti, data["visible"][frame]], owners[data["visible"][frame]]
            low, high = np.full(grid * grid, len(identities), int), np.full(grid * grid, -1, int)
            np.minimum.at(low, cells, current_owners)
            np.maximum.at(high, cells, current_owners)
            occupancy[ti, low == high] = high[low == high]
            occupancy[ti, (high >= 0) & (low != high)] = -2
        world = data["world_xyz"][times][:, points["ids"]].transpose(1, 0, 2)
        anchors = points["anchor"]
        initial = world[np.arange(len(anchors)), anchors]
        world_delta = np.linalg.norm(world - initial[:, None], axis=-1)
        hand = hand[times]
        hand_delta = np.linalg.norm(hand[None] - hand[anchors, None], axis=-1)
        near = np.linalg.norm(world - hand[None], axis=-1) <= 0.15
        moving_labels = [row["label"] for row in points["motion"] if row["surface_excursion_m"] >= 0.02]
        phases = {
            "moving": points["displaced"] & np.isin(points["labels"], moving_labels)[:, None],
            "near_stationary": points["valid"] & (world_delta <= 0.01) & near & (hand_delta > 0.02),
        }
        active_labels = set(points["labels"][(phases["moving"] | phases["near_stationary"]).any(1)])
        phases["all_visible"] = points["valid"] & np.isin(points["labels"], list(active_labels))[:, None]
        points.update(phases=phases, occupancy=occupancy, point_owners=point_owners,
                      gripper_owner=identities.index("gripper") if "gripper" in identities else -3)
        return points

    @torch.no_grad()
    def control(self, case, vae, device, zero_latents):
        video, _, _ = self.get(case["episode"], case["camera"])
        indices, _ = observation_window_indices(len(video), case["start_frame"], case["rgb_frames"],
                                                self.args.warmup_frames)
        pixels = torch.from_numpy(video[indices].copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
        pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=(self.args.size, self.args.size),
                               mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        return observation_control(vae, pixels, self.args.warmup_frames, zero_latents)


def index_windows(args, split):
    """Index possible starts, not fixed features. GT only guides training sampling."""
    pools = {length: {phase: [] for phase in PHASES} for length in args.lengths}
    val = []
    allowed_train, allowed_val = set(split["train_episodes"]), set(split["val_episodes"])
    for record in split["episodes"]:
        ep = record["episode"]
        directory = args.video_root / f"episode_{ep:05d}"
        meta = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "sim_state.npz") as z:
            positions = z["body_xpos"].copy()
        names = meta["body_names"]
        hand = names.index("gripper0_eef")
        roots = [i for i, name in enumerate(names) if name.endswith("_main")]
        bodies = [i for i, name in enumerate(names)
                  if any(name.startswith(names[root].removesuffix("_main")) for root in roots)]
        for length in args.lengths:
            for start in range(args.warmup_frames, len(positions) - length + 1, 4):
                times = np.arange(start, start + length, 4)
                world = positions[times]
                excursions = np.linalg.norm(world[:, bodies] - world[0, bodies], axis=-1).max(0)
                hand_motion = np.linalg.norm(world[:, hand] - world[0, hand], axis=-1).max()
                hand_near = np.linalg.norm(world[:, roots] - world[:, hand, None], axis=-1).min() < 0.25
                phases = []
                if excursions.max() > 0.02:
                    phases.append("moving")
                if hand_motion > 0.02 and hand_near:
                    phases.append("near_stationary")
                case = {k: record[k] for k in ("episode", "suite", "task_id", "task", "source_episode")}
                case.update(start_frame=start, rgb_frames=length)
                if ep in allowed_train:
                    for phase in phases:
                        for camera in CAMERAS:
                            pools[length][phase].append(case | {"camera": camera})
        if ep in allowed_val:
            length = max(x for x in args.lengths if x + args.warmup_frames <= len(positions))
            options = []
            for start in range(args.warmup_frames, len(positions) - length + 1, 4):
                world = positions[np.arange(start, start + length, 4)]
                score = np.linalg.norm(world[:, bodies] - world[0, bodies], axis=-1).max()
                options.append((float(score), start))
            start = max(options)[1]
            for camera in CAMERAS:
                val.append({k: record[k] for k in ("episode", "suite", "task_id", "task", "source_episode")}
                           | {"start_frame": start, "rgb_frames": length, "camera": camera})
    summary = {}
    for length, phases in pools.items():
        summary[str(length)] = {phase: {"windows": len(rows),
                                      "tasks": len({(r['suite'], r['task_id']) for r in rows}),
                                      "episodes": len({r['episode'] for r in rows})}
                               for phase, rows in phases.items()}
    return pools, val, summary


class WindowSampler:
    def __init__(self, pools, observations, seed):
        self.pools, self.observations = pools, observations
        self.rng = np.random.default_rng(seed)
        self.groups = {}
        for length, phases in pools.items():
            self.groups[length] = {}
            for phase, cases in phases.items():
                by_task = {}
                for case in cases:
                    by_task.setdefault((case["suite"], case["task_id"], case["camera"]), []).append(case)
                self.groups[length][phase] = list(by_task.values())

    def sample(self, length, batch_size):
        result = []
        for index in range(batch_size):
            desired = "moving" if index < round(0.75 * batch_size) else "near_stationary"
            for attempt in range(100):
                groups = self.groups[length][desired]
                if not groups:
                    raise RuntimeError(f"No {desired} training windows for length {length}")
                pool = groups[int(self.rng.integers(len(groups)))]
                case = pool[int(self.rng.integers(len(pool)))].copy()
                try:
                    points = self.observations.points(case)
                except ValueError:
                    continue
                if points["phases"][desired].sum() < 3:
                    continue
                result.append((case, points))
                break
            else:
                raise RuntimeError(f"No annotated {desired} samples after 100 attempts")
        return result


def selected_qk(tower, noise, control, text, timestep, layer, head, verify=False):
    # Frozen prefix needs no activation graph; supervised Q/K need no later blocks.
    with torch.no_grad():
        rotary = tower.rope(noise)
        initial = tower.patch_embedding(noise).flatten(2).transpose(1, 2)
        c = tower.vace_patch_embedding(control).flatten(2).transpose(1, 2)
        _, temb, encoded_text, _ = tower.condition_embedder(timestep, text)
        temb = temb.unflatten(1, (6, -1))
        for block in tower.vace_blocks[:layer]:
            _, c = block(initial, encoded_text, c, temb, rotary)
        block = tower.vace_blocks[layer]
        before = c
        if block.proj_in is not None:
            c = block.proj_in(c) + initial
        shift, scale, *_ = (block.scale_shift_table + temb.float()).chunk(6, dim=1)
        normalized = (block.norm1(c.float()) * (1 + scale) + shift).type_as(c)
    attn = block.attn1
    query, key = attn.norm_q(attn.to_q(normalized)), attn.norm_k(attn.to_k(normalized))

    def rotate(value):
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)[:, head:head + 1]
        complex_value = torch.view_as_complex(value.to(torch.float64).unflatten(3, (-1, 2)))
        return torch.view_as_real(complex_value * rotary).flatten(3, 4).type_as(value)[:, 0]

    query, key = rotate(query), rotate(key)
    if verify:
        captured = {}
        attn._heft_capture = lambda **kw: captured.update(q=kw["query"][:, head], k=kw["key"][:, head])
        try:
            with torch.no_grad():
                block(initial[:1], encoded_text[:1], before[:1], temb[:1], rotary)
        finally:
            del attn._heft_capture
        for name, value in (("q", query[:1]), ("k", key[:1])):
            torch.testing.assert_close(value.detach(), captured[name], rtol=0.02, atol=0.04)
        log(stage="native_qk_verified", q_max_error=float((query[:1] - captured["q"]).abs().max()),
            k_max_error=float((key[:1] - captured["k"]).abs().max()))
    shape = (noise.shape[0], noise.shape[2], control.shape[3] * control.shape[4] // 4, 128)
    return query.reshape(shape), key.reshape(shape)


def object_mean(values, mask, labels):
    terms = []
    for label in np.unique(labels):
        selected = mask & torch.as_tensor(labels == label, device=mask.device)[:, None]
        if selected.any():
            terms.append(values[selected].mean())
    return torch.stack(terms).mean() if terms else None


def window_loss(q, k, points, args, rng):
    q, k = q.float(), k.float()
    device = q.device
    cells = torch.as_tensor(points["cells"], device=device)
    anchor = torch.as_tensor(points["anchor"], device=device)
    visible = torch.as_tensor(points["visible"], device=device)
    valid = torch.as_tensor(points["valid"], device=device)
    masks = {phase: torch.as_tensor(points["phases"][phase], device=device) for phase in PHASES}
    count, frames = cells.shape
    pids = torch.arange(count, device=device)
    query = q[anchor, cells[pids, anchor]]
    logits = torch.einsum("pd,tsd->pts", query, k) / math.sqrt(128)
    ce = F.cross_entropy(logits.flatten(0, 1), cells.flatten(), reduction="none").reshape(count, frames)
    previous = q[torch.arange(frames - 1, device=device)[None], cells[:, :-1]]
    step_logits = torch.einsum("ptd,tsd->pts", previous, k[1:]) / math.sqrt(128)
    step_ce = F.cross_entropy(step_logits.flatten(0, 1), cells[:, 1:].flatten(), reduction="none").reshape(count, frames - 1)
    terms = {phase: {} for phase in PHASES}
    for phase in PHASES:
        terms[phase]["anchor"] = object_mean(ce, masks[phase], points["labels"])
        terms[phase]["adjacent"] = object_mean(step_ce, masks[phase][:, 1:] & visible[:, :-1], points["labels"])

    occupancy = torch.as_tensor(points["occupancy"], device=device)
    owners = torch.as_tensor(points["point_owners"], device=device)
    negative = (occupancy[None] >= 0) & (occupancy[None] != owners[:, None, None])
    clean = occupancy[torch.arange(frames)[None], cells] == owners[:, None]
    positive = logits.gather(-1, cells[..., None]).squeeze(-1)
    strongest = logits.masked_fill(~negative, -torch.inf).amax(-1)
    has_negative = negative.any(-1) & clean
    separation = F.relu(args.margin + strongest.masked_fill(~has_negative, 0) - positive)
    for phase in PHASES:
        terms[phase]["separation"] = object_mean(separation, masks[phase] & has_negative, points["labels"])

    # Soft sparse propagation uses its own support, with no GT resets after the start.
    rollout_values = {phase: [] for phase in PHASES}
    starts = rng.choice(frames - 2, size=min(args.rollout_starts, frames - 2), replace=False)
    for start in starts:
        start = int(start)
        active = visible[:, start] & (anchor <= start) & (masks["moving"] | masks["near_stationary"])[:, start + 1:].any(1)
        ids = torch.nonzero(active, as_tuple=False).flatten()
        if not len(ids):
            continue
        nodes = cells[ids, start:start + 1]
        log_mass = torch.zeros((len(ids), 1), device=device)
        for hop in range(1, min(max(args.rollout_horizons), frames - start - 1) + 1):
            t = start + hop
            source = q[t - 1, nodes]
            transition = torch.einsum("pkd,sd->pks", source, k[t]) / math.sqrt(128)
            log_distribution = torch.logsumexp(log_mass[:, :, None] + F.log_softmax(transition, -1), dim=1)
            if hop in args.rollout_horizons:
                nll = -log_distribution.gather(1, cells[ids, t:t + 1]).squeeze(1)
                for phase in PHASES:
                    mask = masks[phase][ids, t:t + 1]
                    value = object_mean(nll[:, None], mask, points["labels"][ids.cpu().numpy()])
                    if value is not None:
                        rollout_values[phase].append(value)
            log_mass, nodes = log_distribution.topk(args.rollout_topk, dim=-1)
            log_mass = log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True)
    for phase in PHASES:
        terms[phase]["multi_hop"] = torch.stack(rollout_values[phase]).mean() if rollout_values[phase] else None
    return terms


def batch_loss(q, k, samples, args, rng):
    collected = {phase: {key: [] for key in ("anchor", "adjacent", "multi_hop", "separation")} for phase in PHASES}
    for index, (_, points) in enumerate(samples):
        terms = window_loss(q[index], k[index], points, args, rng)
        for phase in PHASES:
            for key, value in terms[phase].items():
                if value is not None:
                    collected[phase][key].append(value)
    weights = {"anchor": 1.0, "adjacent": 1.0, "multi_hop": args.rollout_weight,
               "separation": args.separation_weight}
    total, metrics = q.sum() * 0, {}
    for phase, phase_weight in zip(PHASES, (0.7, 0.3)):
        if not collected[phase]["anchor"]:
            raise RuntimeError(f"Batch lost {phase} supervision")
        for key, values in collected[phase].items():
            if values:
                value = torch.stack(values).mean()
                total = total + phase_weight * weights[key] * value
                metrics[f"{phase}_{key}"] = float(value.detach())
    return total, metrics


def save_checkpoint(tower, optimizer, scheduler, sampler, rng, args, step):
    path = args.output / f"object_tracking_lora_step_{step:05d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save({"step": step, "candidates": [(args.layer, args.head)], "rank": args.rank,
                "alpha": args.alpha, "lora": {k: v.detach().cpu() for k, v in tower.state_dict().items() if "lora_" in k},
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "sampler_rng": sampler.rng.bit_generator.state, "loss_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "batch_size": args.batch_size, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, temporary)
    temporary.replace(path)
    dump(args.output / "latest_checkpoint.json", {"step": step, "checkpoint": str(path)})
    log(stage="checkpoint", step=step, path=str(path))


@torch.no_grad()
def evaluate(tower, vae, text, observations, cases, args, step, zero_latents):
    device = text.device
    selected = []
    for suite in sorted({c["suite"] for c in cases}):
        pool = [c for c in cases if c["suite"] == suite]
        # Fixed source-episode holdout, rotated task coverage at later snapshots.
        tasks = sorted({c["task_id"] for c in pool})
        task_ids = {tasks[(step // args.eval_every + offset) % len(tasks)] for offset in range(args.eval_tasks_per_suite)}
        selected.extend(c for c in pool if c["task_id"] in task_ids)
    rows = []
    for case in selected:
        points = observations.points(case)
        control = observations.control(case, vae, device, zero_latents)
        generator = torch.Generator(device=device).manual_seed(args.seed + case["episode"] * 11 + CAMERAS.index(case["camera"]))
        noise = torch.randn((1, 16, control.shape[2], control.shape[3], control.shape[4]), device=device,
                            dtype=torch.bfloat16, generator=generator)
        q, k = selected_qk(tower, noise, control, text, torch.tensor([args.timestep], device=device), args.layer, args.head)
        fixed, rolling = decode_native(q, k, points, args.size // 16)
        baseline = cell_to_xy(points["cells"][np.arange(len(points["anchor"])), points["anchor"]], args.size // 16, 256)
        baseline = np.repeat(baseline[:, None], len(points["cells"][0]), axis=1)
        quality = full_track_quality(np.stack((fixed[0], rolling[0], baseline)), points)
        for name, prediction, full in zip(("anchor", "rolling", "fixed_position"), (fixed[0], rolling[0], baseline), quality):
            rows.append(case | {"decoder": name, "metrics": measure(prediction, points, args.size // 16), **full})
        log(stage="eval_case", step=step, episode=case["episode"], camera=case["camera"])
        del control, noise, q, k
    summary = {}
    for name in ("anchor", "rolling", "fixed_position"):
        result = {}
        for phase in PHASES:
            for metric in ("epe_px", "within8", "known_wrong_identity", "gripper_swap"):
                strata = []
                for suite in sorted({r["suite"] for r in rows}):
                    for camera in CAMERAS:
                        values = [r["metrics"][phase][metric] for r in rows
                                  if r["suite"] == suite and r["camera"] == camera and r["decoder"] == name
                                  and r["metrics"][phase][metric] is not None]
                        if values:
                            strata.append(float(np.mean(values)))
                result[f"{phase}_{metric}"] = float(np.mean(strata)) if strata else None
        for metric in ("full_track80", "joint_track80"):
            values = [r[metric] for r in rows if r["decoder"] == name and r[metric] is not None]
            result[metric] = float(np.mean(values)) if values else None
        summary[name] = result
    dump(args.output / f"eval_step_{step:05d}.json", {"step": step, "summary": summary, "per_case": rows})
    log(stage="eval", step=step, **summary)
    return summary


def parse_args():
    base = ROOT / "runs/vace_tracking_all_suites_20261004"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video-root", type=Path, default=base / "videos")
    p.add_argument("--tracks-root", type=Path, default=base / "tracks")
    p.add_argument("--split", type=Path, default=base / "split.json")
    p.add_argument("--model-root", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    p.add_argument("--text-root", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--head", type=int, default=1)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--warmup-frames", type=int, default=16)
    p.add_argument("--lengths", type=lambda s: [int(x) for x in s.split(",")], default=[49, 65, 97, 129])
    p.add_argument("--length-weights", type=lambda s: [float(x) for x in s.split(",")], default=[5, 3, 1, 1])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--points-per-subject", type=int, default=12)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--timestep", type=float, default=1000)
    p.add_argument("--rollout-horizons", type=lambda s: [int(x) for x in s.split(",")], default=[2, 4, 8])
    p.add_argument("--rollout-topk", type=int, default=16)
    p.add_argument("--rollout-starts", type=int, default=2)
    p.add_argument("--rollout-weight", type=float, default=1)
    p.add_argument("--separation-weight", type=float, default=0.25)
    p.add_argument("--margin", type=float, default=1)
    p.add_argument("--raw-cache-cases", type=int, default=8)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-tasks-per-suite", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if args.batch_size < 4 or args.size % 16 or args.warmup_frames % 4:
        p.error("Use a real batch >=4, size divisible by16, warmup divisible by4")
    if len(args.lengths) != len(args.length_weights) or any((x - 1) % 4 for x in args.lengths):
        p.error("Each length must be4n+1, with a corresponding sampling weight")
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    dump(args.output / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} |
         {"protocol": "observed full window; empty prompt; inactive observed video; no future prediction",
          "phase_weights": {"moving": 0.7, "near_stationary": 0.3},
          "trainable": "selected control attn1 Q/K head LoRA only; separate object adapter",
          "feature_cache": False, "frozen_prefix_no_grad": True, "real_batched_windows": True})
    split = json.loads(args.split.read_text())
    pools, val, summary = index_windows(args, split)
    dump(args.output / "data_summary.json", {"buckets": summary, "train_episodes": split["train_episodes"],
                                           "val_episodes": split["val_episodes"], "val_cases": val})
    log(stage="data_indexed", buckets=summary, val_views=len(val))
    observations = Observations(args)
    sampler = WindowSampler(pools, observations, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    text = load_empty_text(args.text_root, device)
    tower = ControlTower.load(args.model_root / "diffusion_pytorch_model.safetensors", device)
    params = install_lora(tower, [(args.layer, args.head)], args.rank, args.alpha)
    vae = load_vae(args.model_root / "Wan2.1_VAE.pth", device)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    def schedule(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        ratio = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(ratio, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = tower.load_state_dict(state["lora"], strict=False)
        if any("lora_" in x for x in missing + unexpected):
            raise RuntimeError("Adapter checkpoint mismatch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.rng.bit_generator.state = state["sampler_rng"]
        rng.bit_generator.state = state["loss_rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start = state["step"]
    zero_latents = {}
    torch.cuda.reset_peak_memory_stats()
    log(stage="ready", trainable_parameters=sum(p.numel() for p in params), batch_size=args.batch_size,
        head=f"T{args.layer}H{args.head}", gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    started = time.monotonic()
    weights = np.asarray(args.length_weights) / sum(args.length_weights)
    for step in range(start + 1, args.steps + 1):
        tick = time.monotonic()
        length = args.lengths[0] if step == 1 else int(rng.choice(args.lengths, p=weights))
        samples = sampler.sample(length, args.batch_size)
        control = torch.cat([observations.control(case, vae, device, zero_latents) for case, _ in samples])
        shape = (args.batch_size, 16, control.shape[2], control.shape[3], control.shape[4])
        noise = torch.randn(shape, device=device, dtype=torch.bfloat16)
        optimizer.zero_grad(set_to_none=True)
        q, k = selected_qk(tower, noise, control, text.expand(args.batch_size, -1, -1),
                           torch.full((args.batch_size,), args.timestep, device=device), args.layer, args.head,
                           verify=step == 1)
        loss, metrics = batch_loss(q, k, samples, args, rng)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step{step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        if step == 1 and float(grad_norm) == 0:
            raise RuntimeError("No gradient reached the object LoRA")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        info = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm), "lr": lr,
                "batch_size": args.batch_size, "rgb_frames": length, "latent_frames": shape[2],
                "seconds": time.monotonic() - tick, "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, **metrics}
        log(stage="train", **info)
        dump(args.output / "status.json", info | {"pid": os.getpid(), "state": "training"})
        del q, k, loss, control, noise
        if step == 1 or step % args.save_every == 0 or step == args.steps:
            save_checkpoint(tower, optimizer, scheduler, sampler, rng, args, step)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(tower, vae, text, observations, val, args, step, zero_latents)
            gc.collect()
    dump(args.output / "completed.json", {"step": args.steps, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
