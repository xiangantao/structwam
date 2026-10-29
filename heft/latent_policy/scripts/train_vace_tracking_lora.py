#!/usr/bin/env python3
"""LoRA-align selected VACE self-attention heads to LIBERO surface trajectories."""

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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from search_vace_tracking_heads import (
    CAMERAS,
    ControlTower,
    cell_to_xy,
    decode_matches,
    load_empty_text,
    load_vae,
    read_video,
    select_points,
    track_metrics,
)


def log(**values):
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(values), flush=True)


class HeadLoRALinear(nn.Module):
    """LoRA on one attention head's output rows, leaving other heads untouched."""

    def __init__(self, base: nn.Linear, head: int, head_dim: int, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.head = int(head)
        self.start = self.head * head_dim
        self.stop = self.start + head_dim
        self.scaling = float(alpha) / float(rank)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(head_dim, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        # Nested adapters allow several selected heads in the same projection.
        for name, parameter in self.base.named_parameters():
            if 'lora_' not in name:
                parameter.requires_grad_(False)

    def forward(self, x):
        base = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b)
        out = torch.zeros_like(base)
        out[..., self.start:self.stop] = (delta * self.scaling).to(base.dtype)
        return base + out


def parse_candidates(value):
    result = []
    for item in value.split(","):
        layer, head = item.split(":")
        result.append((int(layer), int(head)))
    if not result:
        raise ValueError("at least one candidate is required")
    if len(set(result)) != len(result):
        raise ValueError("duplicate candidate")
    return result


def install_lora(tower, candidates, rank, alpha):
    for layer, head in candidates:
        block = tower.vace_blocks[layer]
        block.attn1.to_q = HeadLoRALinear(block.attn1.to_q, head, 128, rank, alpha)
        block.attn1.to_k = HeadLoRALinear(block.attn1.to_k, head, 128, rank, alpha)
    tower.to(next(tower.parameters()).device)
    params = [p for p in tower.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("LoRA installation produced no trainable parameters")
    return params


@torch.no_grad()
def make_case(video_root, tracks_root, model_root, episode, camera, size, skip_frames, max_frames,
              points_per_subject, min_motion, device, seed, vae=None, zero_latents=None, video=None,
              points_only=False, include_static=False):
    name = f"episode_{episode:05d}_{camera}_start{skip_frames:04d}"
    if video is None:
        video = read_video(video_root / f"episode_{episode:05d}" / f"{camera}.mp4")
    n = min(len(video) - skip_frames, max_frames)
    if n < 25:
        raise ValueError("Tracking windows need at least 25 RGB frames")
    n = 1 + 4 * ((n - 1) // 4)
    indices = np.arange(skip_frames, skip_frames + n)
    times = indices[::4]
    with np.load(tracks_root / f"episode_{episode:05d}_{camera}_visible_surface.npz") as source:
        data = {key: source[key] for key in ("xy", "visible", "world_xyz", "point_label", "local_xyz")}
        if "annotated_frame_indices" in source.files:
            data["annotated_frame_indices"] = source["annotated_frame_indices"]
    if "annotated_frame_indices" in data and not np.isin(times, data["annotated_frame_indices"]).all():
        raise ValueError("Training window requests frames without visibility annotations")
    points = select_points(data, times, size // 16, points_per_subject, min_motion,
                           surface_motion=True, include_static=include_static)
    points["visible"] = data["visible"][times][:, points["ids"]].T
    case = {"name": name, "episode": episode, "camera": camera, "start_frame": skip_frames,
            "points": points, "times": times}
    if points_only:
        log(stage="indexed", case=name, rgb_frames=n, latent_frames=len(times), points=len(points["ids"]))
        return case
    pixels = torch.from_numpy(video[indices].copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
    pixels = F.interpolate(
        pixels.permute(1, 0, 2, 3), size=(size, size), mode="bilinear", align_corners=False
    ).permute(1, 0, 2, 3)
    if vae is None:
        vae = load_vae(model_root / "Wan2.1_VAE.pth", device)
    observed = vae.encode([pixels])[0]
    if zero_latents is None:
        zero_latents = {}
    if n not in zero_latents:
        zero_latents[n] = vae.encode([torch.zeros_like(pixels)])[0].cpu()
    reactive = zero_latents[n].to(device)
    control = torch.cat(
        (observed, reactive, torch.zeros((64, *observed.shape[1:]), device=device)), dim=0
    ).unsqueeze(0).half().cpu()
    generator = torch.Generator(device=device).manual_seed(seed + episode * 10000 + skip_frames * 10 + CAMERAS.index(camera))
    noise = torch.randn((1, *observed.shape), generator=generator, device=device, dtype=torch.float32).bfloat16().cpu()
    del pixels, observed, reactive
    torch.cuda.empty_cache()
    log(stage="prepared", case=name, rgb_frames=n, latent_frames=len(times), points=len(points["ids"]))
    return case | {"control": control, "noise": noise}


def capture_selected(tower, candidates):
    captures = {}
    for layer, _ in candidates:
        tower.vace_blocks[layer].attn1._heft_capture = (
            lambda layer=layer, **kw: captures.__setitem__(layer, (kw["query"], kw["key"]))
        )
    return captures


def remove_captures(tower, candidates):
    for layer, _ in candidates:
        if hasattr(tower.vace_blocks[layer].attn1, "_heft_capture"):
            del tower.vace_blocks[layer].attn1._heft_capture


def trajectory_loss(q, k, points, head, grid, rollout_weight=1.0, separation_weight=0.25, margin=1.0,
                    balance_groups=False):
    # q/k are [1, heads, T*S, 128], already RMS-normalized and RoPE-rotated.
    q = q[0, head].reshape(-1, grid * grid, 128)
    k = k[0, head].reshape(-1, grid * grid, 128)
    anchor = torch.as_tensor(points["anchor"], device=q.device, dtype=torch.long)
    cells = torch.as_tensor(points["cells"], device=q.device, dtype=torch.long)
    target = cells
    query = q[anchor, cells[torch.arange(len(anchor), device=q.device), anchor]]
    logits = torch.einsum("pd,tsd->pts", query.float(), k.float()) / (128**0.5)
    valid = torch.as_tensor(points["valid"], device=q.device, dtype=torch.bool)
    # Emphasize true movement while retaining visible gripper/object observations.
    displaced = torch.as_tensor(points["displaced"], device=q.device, dtype=torch.bool)
    gripper = torch.from_numpy(points["groups"] == "gripper").to(q.device)
    valid = valid & (displaced | gripper[:, None])
    if not valid.any():
        return logits.sum() * 0

    def cross_entropy(scores, targets, mask):
        if not balance_groups:
            return F.cross_entropy(scores[mask], targets[mask])
        terms = []
        for group in (gripper, ~gripper):
            selected = mask & group[:, None]
            if selected.any():
                terms.append(F.cross_entropy(scores[selected], targets[selected]))
        return torch.stack(terms).mean()

    anchor_loss = cross_entropy(logits, target, valid)

    # Teacher-forced one-step rollout: train q at the previous true location
    # to select the next true location. At inference this query is replaced by
    # the previous predicted location, so the objective matches rolling decode.
    visible = torch.as_tensor(
        points.get("visible", points["valid"] | (np.arange(q.shape[0])[None] == points["anchor"][:, None])),
        device=q.device, dtype=torch.bool,
    )
    roll_valid = valid[:, 1:] & visible[:, :-1]
    if q.shape[0] > 1 and roll_valid.any():
        roll_queries = torch.stack(
            [q[t - 1, cells[:, t - 1]] for t in range(1, q.shape[0])], dim=1
        )
        roll_logits = torch.einsum("ptd,tsd->pts", roll_queries.float(), k[1:].float()) / (128**0.5)
        rollout_loss = cross_entropy(roll_logits, target[:, 1:], roll_valid)
    else:
        rollout_loss = anchor_loss * 0

    # Explicit object separation. All gripper surface labels share one object
    # ID, while each independently moving scene object has its own ID.
    labels = np.asarray(points["labels"])
    object_ids = np.asarray([
        "gripper" if label in {"gripper base", "left finger", "right finger"} else str(label)
        for label in labels
    ])
    separation_terms = []
    for t in range(q.shape[0]):
        visible_now = visible[:, t]
        if not valid[:, t].any():
            continue
        # Convert other-object point identities into a mask over the full key
        # grid. Multiple surface points can quantize to one cell.
        negative = torch.zeros((len(anchor), grid * grid), dtype=torch.bool, device=q.device)
        visible_ids = torch.nonzero(visible_now, as_tuple=False).flatten()
        for point_id in range(len(anchor)):
            other_ids = visible_ids[torch.from_numpy(object_ids[visible_ids.cpu().numpy()] != object_ids[point_id]).to(q.device)]
            if len(other_ids):
                negative[point_id, cells[other_ids, t]] = True
            # Different surfaces can occupy the same coarse token cell.
            negative[point_id, cells[point_id, t]] = False
        positive = logits[:, t].gather(1, target[:, t:t + 1]).squeeze(1)
        strongest_negative = logits[:, t].masked_fill(~negative, -torch.inf).amax(1)
        has_negative = torch.isfinite(strongest_negative) & valid[:, t]
        if has_negative.any():
            separation_terms.append(
                F.softplus(strongest_negative[has_negative] - positive[has_negative] + margin).mean()
            )
    separation_loss = torch.stack(separation_terms).mean() if separation_terms else anchor_loss * 0
    return anchor_loss + rollout_weight * rollout_loss + separation_weight * separation_loss



def spatial_group_loss(q, k, points, head, grid, margin=0.5):
    """Train same-object off-diagonal affinity on known visible surfaces.

    Sparse annotations cannot identify background. Unknown and mixed-object
    cells are ignored, and every same-object positive must receive probability.
    """
    q = q[0, head].reshape(-1, grid * grid, 128)
    k = k[0, head].reshape(-1, grid * grid, 128)
    cells = torch.as_tensor(points["cells"], device=q.device, dtype=torch.long)
    visible_np = np.asarray(points.get("visible"), dtype=bool)
    if visible_np.shape != (len(points["ids"]), q.shape[0]):
        visible_np = np.asarray(points["valid"], dtype=bool)
    labels = np.asarray(points["labels"]).astype(str)
    object_ids = np.asarray([
        "gripper" if label in {"gripper base", "left finger", "right finger"} else label
        for label in labels
    ])
    object_terms = {}
    for t in range(q.shape[0]):
        visible_ids = np.flatnonzero(visible_np[:, t])
        if len(visible_ids) < 2:
            continue
        frame_cells = np.asarray(points['cells'])[visible_ids, t]
        clean_cells, clean_objects = [], []
        for cell in np.unique(frame_cells):
            occupants = np.unique(object_ids[visible_ids[frame_cells == cell]])
            if len(occupants) == 1:
                clean_cells.append(int(cell))
                clean_objects.append(occupants[0])
        if len(clean_cells) < 3:
            continue
        clean_objects = np.asarray(clean_objects)
        query_cells = torch.as_tensor(clean_cells, device=q.device)
        scores = q[t, query_cells].float() @ k[t, query_cells].float().T / (128 ** 0.5)
        self_mask = torch.eye(len(clean_cells), dtype=torch.bool, device=q.device)
        log_probs = F.log_softmax(scores.masked_fill(self_mask, -torch.inf), dim=-1)
        for object_id in np.unique(clean_objects):
            indices = np.flatnonzero(clean_objects == object_id)
            if len(indices) < 2 or len(indices) == len(clean_cells):
                continue
            rows = torch.as_tensor(indices, device=q.device)
            positive = log_probs[rows[:, None], rows[None, :]]
            off_diagonal = ~torch.eye(len(indices), dtype=torch.bool, device=q.device)
            object_terms.setdefault(object_id, []).append(-positive[off_diagonal].mean())
    if not object_terms:
        return (q.sum() + k.sum()) * 0
    # Equalize objects, so a cabinet with many visible surface points cannot
    # dominate smaller movable objects.
    terms = [torch.stack(values).mean() for values in object_terms.values()]
    return torch.stack(terms).mean()


@torch.no_grad()
def spatial_group_metrics(q, k, points, head, grid):
    q = q[0, head].reshape(-1, grid * grid, 128)
    k = k[0, head].reshape(-1, grid * grid, 128)
    labels = np.asarray(['gripper' if label in {'gripper base', 'left finger', 'right finger'}
                         else str(label) for label in points['labels']])
    rows_by_object = {}
    for t in range(q.shape[0]):
        ids = np.flatnonzero(points['visible'][:, t])
        cells = np.asarray(points['cells'])[ids, t]
        clean_cells, objects = [], []
        for cell in np.unique(cells):
            occupants = np.unique(labels[ids[cells == cell]])
            if len(occupants) == 1:
                clean_cells.append(int(cell))
                objects.append(occupants[0])
        if len(clean_cells) < 3:
            continue
        objects = np.asarray(objects)
        indices = torch.as_tensor(clean_cells, device=q.device)
        scores = q[t, indices].float() @ k[t, indices].float().T / (128 ** .5)
        diagonal = torch.eye(len(indices), device=q.device, dtype=torch.bool)
        probs = scores.masked_fill(diagonal, -torch.inf).softmax(-1)
        same = torch.as_tensor(objects[:, None] == objects[None, :], device=q.device) & ~diagonal
        mass = (probs * same).sum(-1).cpu().numpy()
        correct = same.gather(1, probs.argmax(-1)[:, None]).squeeze(1).cpu().numpy()
        chance = same.float().sum(-1).cpu().numpy() / (len(indices) - 1)
        for obj in np.unique(objects):
            selected = objects == obj
            if selected.sum() > 1 and not selected.all():
                rows_by_object.setdefault(obj, []).extend(zip(mass[selected], correct[selected], chance[selected]))
    result = {'count': sum(len(rows) for rows in rows_by_object.values())}
    for index, name in enumerate(('same_object_mass', 'same_object_top1', 'chance_mass')):
        result[name] = (float(np.mean([np.mean(np.asarray(rows)[:, index])
                                      for rows in rows_by_object.values()])) if rows_by_object else None)
    return result


@torch.no_grad()
def evaluate_case(tower, text, case, candidates, timestep, device, size, spatial=False):
    tower.eval()
    captures = {}
    layers = {layer for layer, _ in candidates}
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tower(
                case["noise"].to(device).bfloat16(),
                case["control"].to(device).bfloat16(),
                text,
                torch.tensor([timestep], device=device),
                lambda layer, q, k: captures.__setitem__(layer, (q, k)) if layer in layers else None,
            )
    finally:
        remove_captures(tower, candidates)
    rows = {}
    points, grid = case["points"], size // 16
    for layer, head in candidates:
        q, k = captures[layer]
        decoded, _, _ = decode_matches(
            q[0].reshape(12, -1, grid * grid, 128),
            k[0].reshape(12, -1, grid * grid, 128),
            points["anchor"], points["cells"], grid,
        )
        prediction = cell_to_xy(decoded[head], grid, 256)
        rows[f"T{layer}H{head}"] = {
            "moving_object": track_metrics(prediction, points, "moving_object", True),
            "gripper": track_metrics(prediction, points, "gripper", True),
        }
        if spatial:
            rows[f'T{layer}H{head}']['spatial'] = spatial_group_metrics(q, k, points, head, grid)
    return rows


def aggregate(rows):
    if not rows:
        return {}
    result = {}
    for key in rows[0]:
        result[key] = {}
        for group in ("moving_object", "gripper"):
            result[key][group] = {}
            for metric in ("epe_px", "within8", "trajectory80_at8", "count"):
                values = [r[key][group][metric] for r in rows if r[key][group][metric] is not None]
                result[key][group][metric] = (int(sum(values)) if metric == "count" else float(np.mean(values))) if values else None
        if 'spatial' in rows[0][key]:
            result[key]['spatial'] = {}
            for metric in ('same_object_mass', 'same_object_top1', 'chance_mass', 'count'):
                values = [row[key]['spatial'][metric] for row in rows
                          if row[key]['spatial'][metric] is not None]
                result[key]['spatial'][metric] = (int(sum(values)) if metric == 'count' else float(np.mean(values))) if values else None
    return result


def validation_report(tower, text, cases, candidates, args, device, ensure_encoded=None):
    rows = []
    for case in cases:
        if ensure_encoded is not None:
            ensure_encoded(case)
        rows.append(evaluate_case(tower, text, case, candidates, args.timestep, device, args.size,
                                  spatial=args.spatial_weight > 0))
    suites = sorted({case.get("suite", "unspecified") for case in cases})
    return {
        "per_case": [{"name": case["name"], "suite": case.get("suite"),
                      "task_id": case.get("task_id"), "metrics": row} for case, row in zip(cases, rows)],
        "mean": aggregate(rows),
        "by_suite": {suite: aggregate([row for case, row in zip(cases, rows)
                                       if case.get("suite", "unspecified") == suite]) for suite in suites},
    }


def case_summary(case):
    return {key: case[key] for key in ("name", "episode", "camera", "start_frame", "suite", "task_id")} | {
        "rgb_frames": int(case["times"][-1] - case["times"][0] + 1),
        "points": int(len(case["points"]["ids"])),
        "moving_object_point_times": int((case["points"]["displaced"] &
                                          (case["points"]["groups"][:, None] == "moving_object")).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/libero_video_spatial_long_20261003")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/libero_video_spatial_objects_gripper_trails_20261003/annotations")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--train-episodes", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--val-episodes", type=int, nargs="+", default=[2])
    parser.add_argument("--split-manifest", type=Path, help="Episode split covering all four suites")
    parser.add_argument("--windows-per-episode", type=int, default=1)
    parser.add_argument("--resume-lora", type=Path, help="Initialize LoRA weights; optimizer starts fresh")
    parser.add_argument("--fresh-noise", action="store_true")
    parser.add_argument("--balance-groups", action="store_true")
    parser.add_argument("--lazy-controls", action="store_true", help="Encode each window on its first sampled use")
    parser.add_argument("--skip-initial-eval", action="store_true", help="Run full held-out evaluation at checkpoints")
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS), choices=CAMERAS)
    parser.add_argument("--candidates", default="6:1,10:6,7:11,14:8")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--skip-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=129)
    parser.add_argument("--points-per-subject", type=int, default=24)
    parser.add_argument("--min-motion", type=float, default=0.02)
    parser.add_argument("--timestep", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--separation-weight", type=float, default=0.25)
    parser.add_argument("--spatial-weight", type=float, default=0.0,
                        help="weight for same-frame object grouping loss")
    parser.add_argument("--temporal-weight", type=float, default=1.0,
                        help="weight for the existing trajectory loss")
    parser.add_argument("--include-static", action="store_true",
                        help="include static scene objects in spatial grouping supervision")
    parser.add_argument('--truncate-after-candidates', action='store_true',
                        help='skip later frozen blocks after the last supervised candidate')
    parser.add_argument("--separation-margin", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--ddp", action="store_true", help="use torchrun multi-GPU training")
    parser.add_argument("--dry-run", action="store_true", help="prepare one case and check forward/loss/backward without updating weights")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.size % 16 or args.steps <= 0 or args.windows_per_episode < 1 or args.save_every < 1 or args.max_frames < 25:
        parser.error("size must be divisible by 16; steps, windows and save-every must be positive")
    episode_metadata = {}
    if args.split_manifest:
        split = json.loads(args.split_manifest.read_text())
        args.train_episodes = split["train_episodes"]
        args.val_episodes = split["val_episodes"]
        episode_metadata = {row["episode"]: row for row in split["episodes"]}
    if set(args.train_episodes) & set(args.val_episodes):
        parser.error("training and validation episodes overlap")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    distributed = bool(args.ddp)
    if distributed:
        if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
            parser.error("--ddp must be launched with torchrun")
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if distributed:
        dist.barrier()
    candidates = parse_candidates(args.candidates)
    device = torch.device("cuda", local_rank)
    if rank == 0:
        (args.output / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    log(stage="text")
    text = load_empty_text(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", device)
    log(stage="prepare_data")
    train_cases, val_cases = [], []
    skipped = []
    vae = None if args.lazy_controls else load_vae(args.model / "Wan2.1_VAE.pth", device)
    zero_latents = {}
    for split_name, episode_ids, cases in (("train", args.train_episodes, train_cases),
                                           ("val", args.val_episodes, val_cases)):
        for episode in episode_ids:
            metadata = episode_metadata.get(episode) or json.loads(
                (args.video_root / f"episode_{episode:05d}/metadata.json").read_text())
            for camera in args.cameras:
                video = ([None] * int(metadata.get("num_frames", metadata.get("steps", 0) + 1))
                         if args.lazy_controls else read_video(args.video_root / f"episode_{episode:05d}/{camera}.mp4"))
                last_start = max(args.skip_frames, ((len(video) - args.max_frames) // 4) * 4)
                window_count = args.windows_per_episode
                if window_count > 1:
                    # Add a middle window when the two endpoints would leave a gap.
                    window_count = max(window_count, 1 + int(np.ceil(
                        (last_start - args.skip_frames) / (4 * ((args.max_frames - 1) // 4)))))
                starts = np.unique(np.linspace(args.skip_frames, last_start, window_count).astype(int) // 4 * 4)
                for start in starts:
                    try:
                        case = make_case(args.video_root, args.tracks_root, args.model, episode, camera,
                                         args.size, int(start), args.max_frames, args.points_per_subject,
                                         args.min_motion, device, args.seed, vae, zero_latents, video,
                                         points_only=args.lazy_controls, include_static=args.include_static)
                    except ValueError as exc:
                        if str(exc) not in {"No visible moving subjects after warmup", "Tracking windows need at least 25 RGB frames"}:
                            raise
                        skipped.append({"split": split_name, "episode": episode, "camera": camera,
                                        "start_frame": int(start), "reason": str(exc)})
                        log(stage="skipped_window", **skipped[-1])
                        continue
                    case.update(suite=metadata["suite"], task_id=metadata["task_id"])
                    cases.append(case)
                del video
    if not args.lazy_controls:
        vae = None
        zero_latents.clear()
    gc.collect()
    torch.cuda.empty_cache()
    if not train_cases or not val_cases:
        raise RuntimeError("Both training and validation need usable windows")
    if args.split_manifest:
        required_tasks = {(row["suite"], row["task_id"]) for row in episode_metadata.values()}
        for split_name, cases in (("train", train_cases), ("val", val_cases)):
            missing = required_tasks - {(case["suite"], case["task_id"]) for case in cases}
            if missing:
                raise RuntimeError(f"No usable {split_name} tracking window for tasks: {sorted(missing)}")
    if rank == 0:
        (args.output / "data_summary.json").write_text(json.dumps({
            "train": [case_summary(case) for case in train_cases],
            "val": [case_summary(case) for case in val_cases], "skipped": skipped,
        }, indent=2))
    log(stage="data_ready", train_cases=len(train_cases), val_cases=len(val_cases), skipped=len(skipped))

    def ensure_encoded(case):
        nonlocal vae
        if "control" in case:
            return
        if vae is None:
            vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
        case.update(make_case(args.video_root, args.tracks_root, args.model, case["episode"], case["camera"],
                              args.size, case["start_frame"], args.max_frames, args.points_per_subject,
                              args.min_motion, device, args.seed, vae, zero_latents,
                              include_static=args.include_static))

    log(stage="tower")
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    params = install_lora(tower, candidates, args.rank, args.alpha)
    if args.truncate_after_candidates:
        tower.vace_blocks = nn.ModuleList(list(tower.vace_blocks)[:max(layer for layer, _ in candidates) + 1])
    if args.resume_lora:
        checkpoint = torch.load(args.resume_lora, map_location="cpu", weights_only=False)
        if [tuple(item) for item in checkpoint["candidates"]] != candidates or checkpoint["rank"] != args.rank or checkpoint["alpha"] != args.alpha:
            raise ValueError("Checkpoint LoRA configuration does not match requested heads/rank/alpha")
        expected = {key for key in tower.state_dict() if "lora_" in key}
        if set(checkpoint["lora"]) != expected:
            raise ValueError("Checkpoint LoRA parameter names do not match")
        tower.load_state_dict(checkpoint["lora"], strict=False)
        log(stage="initialized_lora", checkpoint=str(args.resume_lora), trainable_parameters=sum(p.numel() for p in params))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    if args.dry_run:
        if distributed:
            raise RuntimeError("--dry-run is single-GPU only; use test_vace_tracking_ddp.py for DDP smoke tests")
        torch.cuda.reset_peak_memory_stats(device)
        case = train_cases[0]
        ensure_encoded(case)
        captures = {}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tower(
                case["noise"].to(device).bfloat16(), case["control"].to(device).bfloat16(), text,
                torch.tensor([args.timestep], device=device),
                lambda layer, q, k: captures.__setitem__(layer, (q, k)),
            )
            spatial_for_heads, temporal_for_heads, losses_for_heads = [], [], []
            for layer, head in candidates:
                spatial_value = spatial_group_loss(
                    captures[layer][0], captures[layer][1], case["points"], head, args.size // 16,
                    args.separation_margin,
                )
                temporal_value = trajectory_loss(
                    captures[layer][0], captures[layer][1], case["points"], head, args.size // 16,
                    args.rollout_weight, args.separation_weight, args.separation_margin, args.balance_groups,
                )
                spatial_for_heads.append(spatial_value)
                temporal_for_heads.append(temporal_value)
                losses_for_heads.append(args.spatial_weight * spatial_value + args.temporal_weight * temporal_value)
            loss = torch.stack(losses_for_heads).mean()
        loss.backward()
        grad_norm = float(torch.sqrt(sum((p.grad.float().pow(2).sum() for p in params if p.grad is not None))).detach().cpu())
        log(stage="dry_run", loss=float(loss.detach().cpu()),
            spatial_loss=float(torch.stack(spatial_for_heads).mean().detach().cpu()),
            temporal_loss=float(torch.stack(temporal_for_heads).mean().detach().cpu()),
            trainable_parameters=sum(p.numel() for p in params), grad_norm=grad_norm,
            captured_layers=sorted(captures),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved(device) / 2**30, 2))
        return
    if rank == 0 and not args.skip_initial_eval:
        before = validation_report(tower, text, val_cases, candidates, args, device, ensure_encoded)
        (args.output / "before.json").write_text(json.dumps(before, indent=2))
        log(stage="before", validation=before["mean"], by_suite=before["by_suite"])
    elif rank == 0:
        (args.output / "before.json").write_text(json.dumps({"skipped": True,
            "reason": "Full validation will run at each checkpoint; prior LoRA weights are retained."}, indent=2))
        log(stage="before_skipped")
    if distributed:
        dist.barrier()
        train_model = DDP(tower, device_ids=[local_rank], output_device=local_rank,
                          broadcast_buffers=False, find_unused_parameters=False)
    else:
        train_model = tower
    train_model.train()
    rng = np.random.default_rng(args.seed)
    buckets = {}
    for index, case in enumerate(train_cases):
        buckets.setdefault((case["suite"], case["task_id"]), []).append(index)
    task_keys = sorted(buckets)
    sample_order = []
    while len(sample_order) < args.steps * world_size:
        for task_index in rng.permutation(len(task_keys)):
            sample_order.append(int(rng.choice(buckets[task_keys[int(task_index)]])))
    losses = []
    train_started = time.monotonic()
    layers = {layer for layer, _ in candidates}
    for step in range(1, args.steps + 1):
        case = train_cases[sample_order[(step - 1) * world_size + rank]]
        ensure_encoded(case)
        captures = {}
        optimizer.zero_grad(set_to_none=True)
        noise = torch.randn_like(case["noise"], device=device) if args.fresh_noise else case["noise"].to(device).bfloat16()
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                train_model(noise, case["control"].to(device).bfloat16(), text,
                      torch.tensor([args.timestep], device=device),
                      lambda layer, q, k: captures.__setitem__(layer, (q, k)) if layer in layers else None)
                spatial_for_heads, temporal_for_heads, losses_for_heads = [], [], []
                for layer, head in candidates:
                    q, k = captures[layer]
                    spatial_value = spatial_group_loss(
                        q, k, case["points"], head, args.size // 16, args.separation_margin,
                    )
                    temporal_value = trajectory_loss(
                        q, k, case["points"], head, args.size // 16,
                        args.rollout_weight, args.separation_weight, args.separation_margin, args.balance_groups,
                    )
                    spatial_for_heads.append(spatial_value)
                    temporal_for_heads.append(temporal_value)
                    losses_for_heads.append(args.spatial_weight * spatial_value + args.temporal_weight * temporal_value)
                loss = torch.stack(losses_for_heads).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}, {case['name']}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
        finally:
            remove_captures(tower, candidates)
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % 20 == 0 or step == args.steps:
            log(stage="train", step=step, loss=losses[-1], mean_last20=float(np.mean(losses[-20:])),
                spatial_loss=float(torch.stack(spatial_for_heads).mean().detach()),
                temporal_loss=float(torch.stack(temporal_for_heads).mean().detach()),
                grad_norm=float(grad_norm), case=case["name"], suite=case["suite"], task_id=case["task_id"],
                per_head={f"T{layer}H{head}": float(value.detach())
                          for (layer, head), value in zip(candidates, losses_for_heads)},
                seconds_per_step=(time.monotonic() - train_started) / step,
                peak_allocated_gib=round(torch.cuda.max_memory_allocated(device) / 2**30, 2))
        # Release the just-completed graph before a large validation pass.
        del captures, spatial_for_heads, temporal_for_heads, losses_for_heads, loss, noise
        if step % args.save_every == 0 or step == args.steps:
            if rank == 0:
                torch.save({"step": step, "candidates": candidates, "rank": args.rank, "alpha": args.alpha,
                            "lora": {k: v.detach().cpu() for k, v in tower.state_dict().items() if "lora_" in k},
                            "optimizer": optimizer.state_dict()}, args.output / f"tracking_lora_step_{step:04d}.pt")
                (args.output / "loss.json").write_text(json.dumps(losses))
                log(stage="checkpoint", step=step)
                current = validation_report(tower, text, val_cases, candidates, args, device, ensure_encoded)
                (args.output / f"val_{step:04d}.json").write_text(json.dumps(current, indent=2))
                log(stage="validation", step=step, validation=current["mean"], by_suite=current["by_suite"])
            if distributed:
                dist.barrier()
            train_model.train()
    if rank == 0:
        torch.save({"step": args.steps, "candidates": candidates, "rank": args.rank, "alpha": args.alpha,
                    "lora": {k: v.detach().cpu() for k, v in tower.state_dict().items() if "lora_" in k}}, args.output / "tracking_lora.pt")
        (args.output / "loss.json").write_text(json.dumps(losses))
        log(stage="done", output=str(args.output), world_size=world_size)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
