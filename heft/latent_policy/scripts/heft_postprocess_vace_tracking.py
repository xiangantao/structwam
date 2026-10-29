#!/usr/bin/env python3
"""Train-free HEFT-style postprocessing for VACE Q/K trajectories.

This script reruns only the frozen VACE tower (optionally with an existing
tracking LoRA), then decodes trajectories with local soft-argmax, descriptor
EMA, optional RoPE-frequency filtering, and one-step forward/backward checks.
No weights are updated by this script.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from itertools import product
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(ROOT / "heft/src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch.nn import functional as F

from heft.tracking.config import TrackingConfig
from heft.tracking.features import RopePairing, RopeSpec
from heft.tracking.ops import TrackingOps

from search_vace_tracking_heads import (
    CAMERAS,
    ControlTower,
    cell_to_xy,
    decode_matches,
    load_empty_text,
    load_vae,
    read_video,
    render_preview,
    select_points,
    track_metrics,
)
from train_vace_tracking_lora import install_lora, parse_candidates


WAN_ROPE = RopeSpec(
    pairing=RopePairing.ADJACENT,
    temporal_pairs=22,
    height_pairs=21,
    width_pairs=21,
)


def log(**values):
    print(json.dumps(values), flush=True)


def load_lora(tower, checkpoint: Path, candidates, rank: int, alpha: float):
    install_lora(tower, candidates, rank=rank, alpha=alpha)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = payload.get("lora", payload)
    missing, unexpected = tower.load_state_dict(saved, strict=False)
    missing_lora = [x for x in missing if "lora_" in x]
    unexpected = [x for x in unexpected if "lora_" in x]
    if missing_lora or unexpected:
        raise RuntimeError(f"LoRA checkpoint mismatch: missing_lora={missing_lora}, unexpected={unexpected}")
    return payload


def _grid_position(cell: int, grid: int) -> torch.Tensor:
    return torch.tensor(
        [float(cell % grid), float(cell // grid)], dtype=torch.float32
    )


def _grid_to_image(points: torch.Tensor, grid: int) -> np.ndarray:
    values = points.detach().cpu().numpy()
    return (values + 0.5) * (256.0 / grid) - 0.5


def _select_channels(x: torch.Tensor, frequency_range: tuple[float, float]) -> torch.Tensor:
    indices = WAN_ROPE.channel_indices(frequency_range)
    return x[..., list(indices)]


@torch.inference_mode()
def heft_track(
    q: torch.Tensor,
    k: torch.Tensor,
    points: dict,
    head: int,
    grid: int,
    *,
    query_kind: str,
    frequency_range: tuple[float, float],
    search_radius: float,
    argmax_radius: float,
    ema_alpha: float,
    fb_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Track all independently anchored points at one head on the token grid."""

    # q/k are [heads, time, spatial, dim]. Keeping the grid resolution here
    # avoids an artificial 256x256 interpolation before the correspondence
    # is stabilized; positions are converted to image coordinates afterward.
    qh = _select_channels(q[head].float(), frequency_range).reshape(q.shape[1], grid, grid, -1)
    kh = _select_channels(k[head].float(), frequency_range).reshape(k.shape[1], grid, grid, -1)
    qh = qh.permute(0, 3, 1, 2).contiguous()
    kh = kh.permute(0, 3, 1, 2).contiguous()
    frames = qh.shape[0]
    config = TrackingConfig(
        query_feature="key",
        target_feature="key",
        update_feature="key",
        upsample_features=False,
        align_chunk_features=True,
        argmax_radius=argmax_radius,
        search_radius=search_radius,
        visibility_threshold=fb_threshold,
        feature_ema_alpha=ema_alpha,
        feature_update_sampling_radius=1,
        point_batch_size=128,
    )
    ops = TrackingOps(frame_size=(grid, grid), config=config, device=q.device)
    tracks = np.full((len(points["ids"]), frames, 2), np.nan, dtype=np.float32)
    visible = np.zeros((len(points["ids"]), frames), dtype=bool)
    cells = np.asarray(points["cells"])
    anchors = np.asarray(points["anchor"])

    for point_id, anchor in enumerate(anchors):
        if anchor >= frames:
            continue
        source = qh[anchor] if query_kind == "q_to_k" else kh[anchor]
        position = _grid_position(int(cells[point_id, anchor]), grid).to(q.device)
        descriptors = ops.make_descriptors(source, position[None])
        tracks[point_id, anchor] = _grid_to_image(position[None], grid)[0]
        visible[point_id, anchor] = True
        previous_frame = kh[anchor]
        previous_position = position

        for frame_index in range(anchor + 1, frames):
            target = kh[frame_index]
            predicted = ops.predict(
                descriptors,
                target,
                previous_positions=previous_position[None],
                previous_visibility=torch.ones(1, dtype=torch.bool, device=q.device),
                apply_search_mask=True,
                softmax_before_resize=False,
            )[0]

            # A one-step reverse check removes locally plausible but drifting
            # matches without using GT. The descriptor is taken at the current
            # predicted point and queried back into the previous feature map.
            backward_descriptor = ops.make_descriptors(target, predicted[None])
            backward = ops.predict(
                backward_descriptor,
                previous_frame,
                previous_positions=predicted[None],
                previous_visibility=torch.ones(1, dtype=torch.bool, device=q.device),
                apply_search_mask=True,
                softmax_before_resize=False,
            )[0]
            is_visible = bool(torch.linalg.vector_norm(backward - previous_position) <= fb_threshold)
            if is_visible:
                visible[point_id, frame_index] = True
                tracks[point_id, frame_index] = _grid_to_image(predicted[None], grid)[0]
                ops.update_descriptor(
                    descriptors,
                    target,
                    predicted[None],
                    torch.ones(1, dtype=torch.bool, device=q.device),
                )
                previous_position = predicted
                previous_frame = target
            else:
                # Keep the last valid location for a stable visualization, but
                # mark it invisible so metrics report the actual coverage.
                tracks[point_id, frame_index] = tracks[point_id, frame_index - 1]

    return tracks, visible


def metrics_with_visibility(prediction, visibility, points, group, displaced_only=False):
    mask = points["valid"] & visibility & (points["groups"][:, None] == group)
    if displaced_only:
        mask &= points["displaced"]
    if not mask.any():
        return {"epe_px": None, "within4": None, "within8": None,
                "trajectory80_at8": None, "coverage": 0.0, "count": 0}
    error = np.linalg.norm(prediction - points["xy"], axis=-1)
    point_ids = np.flatnonzero(mask.any(1))
    consistency = [float((error[p, mask[p]] <= 8).mean() >= 0.8) for p in point_ids]
    return {
        "epe_px": float(error[mask].mean()),
        "within4": float((error[mask] <= 4).mean()),
        "within8": float((error[mask] <= 8).mean()),
        "trajectory80_at8": float(np.mean(consistency)),
        "coverage": float(mask.mean()),
        "count": int(mask.sum()),
    }



def metrics_for_labels(prediction, visibility, points, labels):
    """Score one semantic surface label without merging base and fingers."""
    mask = points["valid"] & visibility & np.isin(points["labels"], labels)[:, None]
    if not mask.any():
        return {"epe_px": None, "within8": None, "trajectory80_at8": None, "count": 0}
    error = np.linalg.norm(prediction - points["xy"], axis=-1)
    point_ids = np.flatnonzero(mask.any(1))
    return {
        "epe_px": float(error[mask].mean()),
        "within4": float((error[mask] <= 4).mean()),
        "within8": float((error[mask] <= 8).mean()),
        "trajectory80_at8": float(np.mean([(error[i, mask[i]] <= 8).mean() >= 0.8 for i in point_ids])),
        "count": int(mask.sum()),
    }


def render_tracking_previews(video, times, prediction, points, title, directory, prefix, labels):
    preview_points = dict(points)
    preview_groups = points["groups"].copy()
    # Show the base and both fingers instead of taking only the first label.
    preview_groups[preview_groups == "gripper"] = "hidden"
    for label in ("gripper base", "left finger", "right finger"):
        ids = np.flatnonzero(points["labels"] == label)
        if len(ids):
            chosen = ids[np.linspace(0, len(ids) - 1, min(4, len(ids))).astype(int)]
            preview_groups[chosen] = "gripper"
    preview_points["groups"] = preview_groups
    for group in ("moving_object", "gripper"):
        render_preview(video, times, prediction, preview_points, group, title,
                       directory / f"{prefix}_{group}.mp4")
    for label in labels or ():
        if not np.any(points["labels"] == label):
            continue
        subject_points = dict(points)
        subject_points["groups"] = np.where(points["labels"] == label, "moving_object", "hidden")
        render_preview(video, times, prediction, subject_points, "moving_object", f"{title} {label}",
                       directory / f"{prefix}_{label}.mp4")


@torch.inference_mode()
def beam_track(
    q,
    k,
    points,
    head,
    grid,
    topk=32,
    beam_size=8,
    motion_weight=0.45,
    max_step_cells=4.0,
    local_weight=0.0,
    descriptor_ema=0.0,
):
    """Causal top-k Q->K decoding with a bounded constant-velocity prior.

    A high-scoring Q/K match can still jump to a visually similar object. The
    hard step bound makes that failure recoverable: when no candidate lies near
    the predicted position, the tracker keeps the previous location instead
    of teleporting to a global match.
    """
    query = q[head].float()
    target = k[head].float()
    frames, spatial, dim = target.shape
    topk = min(topk, spatial)
    point_count = len(points["ids"])
    cells = np.asarray(points["cells"])
    anchors = np.asarray(points["anchor"])
    anchor_cells = cells[np.arange(point_count), anchors]
    anchor_queries = query[
        torch.as_tensor(anchors, device=q.device),
        torch.as_tensor(anchor_cells, device=q.device),
    ]
    output = np.full((point_count, frames, 2), np.nan, dtype=np.float32)
    anchor_xy = np.stack((anchor_cells % grid, anchor_cells // grid), axis=1).astype(np.float32)
    output[np.arange(point_count), anchors] = _grid_to_image(torch.from_numpy(anchor_xy), grid)
    beams = []
    for point_id in range(point_count):
        # Keep a per-beam local descriptor so a visually similar object cannot
        # win repeatedly after a small drift away from the anchor.
        beams.append([(
            0.0, anchor_xy[point_id].copy(), np.zeros(2, np.float32),
            [anchor_xy[point_id].copy()], anchor_queries[point_id].clone(),
        )])
    for time_id in range(frames):
        if time_id == 0:
            continue
        logits = anchor_queries @ target[time_id].transpose(0, 1) / math.sqrt(dim)
        values, indices = torch.topk(logits, topk, dim=1)
        values = (values - torch.logsumexp(logits, dim=1, keepdim=True)).cpu().numpy()
        indices = indices.cpu().numpy()
        for point_id in range(point_count):
            if time_id <= anchors[point_id]:
                continue
            coords = np.stack((indices[point_id] % grid, indices[point_id] // grid), axis=1).astype(np.float32)
            candidates = []
            for score, previous, velocity, path, descriptor in beams[point_id]:
                forecast = previous + velocity
                distance = ((coords - forecast[None]) ** 2).sum(axis=1)
                if local_weight:
                    candidate_index_tensor = torch.as_tensor(
                        indices[point_id], device=target.device, dtype=torch.long
                    )
                    candidate_features = target[time_id][candidate_index_tensor]
                    local_logits = (candidate_features @ descriptor) / math.sqrt(dim)
                    local_logits = local_logits - torch.logsumexp(local_logits, dim=0)
                    local_values = local_logits.cpu().numpy()
                else:
                    local_values = np.zeros(topk, dtype=np.float32)
                allowed = distance <= max_step_cells**2
                candidate_ids = np.flatnonzero(allowed)
                if len(candidate_ids) == 0:
                    candidate_ids = np.asarray([int(np.argmin(distance))])
                for candidate_id in candidate_ids:
                    candidate = coords[candidate_id]
                    candidate_velocity = candidate - previous
                    next_descriptor = descriptor
                    if local_weight and descriptor_ema:
                        next_descriptor = (
                            (1.0 - descriptor_ema) * descriptor
                            + descriptor_ema * candidate_features[candidate_id]
                        ).detach()
                    candidates.append((
                        score
                        + float(values[point_id, candidate_id])
                        + local_weight * float(local_values[candidate_id])
                        - motion_weight * float(distance[candidate_id]),
                        candidate.copy(), candidate_velocity.copy(),
                        path + [candidate.copy()], next_descriptor,
                    ))
            candidates.sort(key=lambda item: item[0], reverse=True)
            beams[point_id] = candidates[:beam_size]
    for point_id in range(point_count):
        path = beams[point_id][0][3]
        for time_id, cell_xy in enumerate(path, start=int(anchors[point_id])):
            output[point_id, time_id] = (cell_xy + 0.5) * (256.0 / grid) - 0.5
        for time_id in range(int(anchors[point_id]) + 1, frames):
            delta = output[point_id, time_id] - output[point_id, time_id - 1]
            if np.linalg.norm(delta) > max_step_cells * (256.0 / grid):
                output[point_id, time_id] = output[point_id, time_id - 1]
    return output



@torch.inference_mode()
def joint_beam_track(
    qk_pairs, points, grid, topk=32, beam_size=8,
    motion_weight=0.45, agreement_weight=0.2,
    max_step_cells=4.0, max_agreement_cells=3.0,
):
    """Jointly search candidate paths from multiple attention heads."""
    head_count = len(qk_pairs)
    if head_count < 2:
        raise ValueError("joint_beam_track needs at least two heads")
    frames, spatial, dim = qk_pairs[0][1].shape
    if any(k.shape != (frames, spatial, dim) for _, k in qk_pairs):
        raise ValueError("all joint heads must have the same shape")
    topk = min(topk, spatial)
    cells = np.asarray(points["cells"])
    anchors = np.asarray(points["anchor"])
    point_count = len(points["ids"])
    anchor_cells = cells[np.arange(point_count), anchors]
    anchor_xy = np.stack((anchor_cells % grid, anchor_cells // grid), axis=1).astype(np.float32)
    anchor_queries = []
    for query, _ in qk_pairs:
        anchor_queries.append(query[
            torch.as_tensor(anchors, device=query.device),
            torch.as_tensor(anchor_cells, device=query.device),
        ])
    beams = []
    for point_id in range(point_count):
        beams.append([(
            0.0,
            tuple(anchor_xy[point_id].copy() for _ in range(head_count)),
            tuple(np.zeros(2, np.float32) for _ in range(head_count)),
            tuple([[anchor_xy[point_id].copy()] for _ in range(head_count)]),
            np.zeros(head_count, np.float32),
        )])
    for time_id in range(1, frames):
        candidates_by_head = []
        for (query, target), anchor_query in zip(qk_pairs, anchor_queries):
            logits = anchor_query.float() @ target[time_id].float().transpose(0, 1) / math.sqrt(dim)
            values, indices = torch.topk(logits, topk, dim=1)
            values = (values - torch.logsumexp(logits, dim=1, keepdim=True)).cpu().numpy()
            candidates_by_head.append((values, indices.cpu().numpy()))
        for point_id in range(point_count):
            if time_id <= anchors[point_id]:
                continue
            point_candidates = []
            for values, indices in candidates_by_head:
                coords = np.stack(
                    (indices[point_id] % grid, indices[point_id] // grid), axis=1
                ).astype(np.float32)
                point_candidates.append((coords, values[point_id]))
            next_beams = []
            for score, positions, velocities, paths, head_scores in beams[point_id]:
                choices_by_head = []
                for head_id, (coords, values) in enumerate(point_candidates):
                    forecast = positions[head_id] + velocities[head_id]
                    distance = ((coords - forecast[None]) ** 2).sum(axis=1)
                    ids = np.flatnonzero(distance <= max_step_cells**2)
                    if len(ids) == 0:
                        ids = np.asarray([int(np.argmin(distance))])
                    if len(ids) > 12:
                        ids = ids[np.argsort(values[ids])[::-1][:12]]
                    choices_by_head.append((coords[ids], values[ids]))
                for choice in product(*[range(len(item[0])) for item in choices_by_head]):
                    new_positions = tuple(
                        choices_by_head[h][0][choice[h]].copy() for h in range(head_count)
                    )
                    new_velocities = tuple(
                        new_positions[h] - positions[h] for h in range(head_count)
                    )
                    motion = sum(float(np.sum(
                        (new_positions[h] - (positions[h] + velocities[h])) ** 2
                    )) for h in range(head_count))
                    pairwise = [float(np.sum((new_positions[i] - new_positions[j]) ** 2))
                                for i in range(head_count) for j in range(i + 1, head_count)]
                    agreement = float(np.mean(pairwise)) if pairwise else 0.0
                    local_values = np.asarray([
                        choices_by_head[h][1][choice[h]] for h in range(head_count)
                    ], dtype=np.float32)
                    new_paths = tuple(paths[h] + [new_positions[h]] for h in range(head_count))
                    next_beams.append((
                        score + float(local_values.sum())
                        - motion_weight * motion - agreement_weight * agreement,
                        new_positions, new_velocities, new_paths,
                        head_scores + local_values,
                    ))
            next_beams.sort(key=lambda state: state[0], reverse=True)
            beams[point_id] = next_beams[:beam_size]
    output = np.full((point_count, frames, 2), np.nan, dtype=np.float32)
    for point_id, states in enumerate(beams):
        _, _, _, paths, head_scores = states[0]
        primary = int(np.argmax(head_scores))
        for path_index in range(frames - int(anchors[point_id])):
            time_id = int(anchors[point_id]) + path_index
            positions = np.stack([paths[h][path_index] for h in range(head_count)])
            if np.max(np.linalg.norm(positions - positions[primary], axis=1)) <= max_agreement_cells:
                cell_xy = positions.mean(axis=0)
            else:
                cell_xy = positions[primary]
            output[point_id, time_id] = (cell_xy + 0.5) * (256.0 / grid) - 0.5
    return output

def consensus_trajectory(predictions, max_disagreement_px=24.0):
    """Combine head tracks while rejecting frames with no head agreement."""
    stack = np.asarray(predictions, dtype=np.float32)
    consensus = np.nanmedian(stack, axis=0)
    visible = np.isfinite(consensus).all(axis=-1)
    if stack.shape[0] > 1:
        disagreement = np.nanmedian(
            np.linalg.norm(stack - consensus[None], axis=-1), axis=0
        )
        visible &= disagreement <= max_disagreement_px
    for point_id in range(consensus.shape[0]):
        for time_id in range(1, consensus.shape[1]):
            if not visible[point_id, time_id]:
                consensus[point_id, time_id] = consensus[point_id, time_id - 1]
                visible[point_id, time_id] = visible[point_id, time_id - 1]
    return consensus, visible


def prepare_case(video_root, tracks_root, model_root, episode, camera, size, skip_frames,
                 max_frames, points_per_subject, min_motion, device, seed):
    name = f"episode_{episode:05d}_{camera}"
    video = read_video(video_root / f"episode_{episode:05d}" / f"{camera}.mp4")
    n = min(len(video) - skip_frames, max_frames)
    n = 1 + 4 * ((n - 1) // 4)
    indices = np.arange(skip_frames, skip_frames + n)
    times = indices[::4]
    with np.load(tracks_root / f"{name}_visible_surface.npz") as source:
        data = {key: source[key] for key in source.files}
    # Match training: articulated or rotating objects can have substantial
    # surface motion even when their centroid moves less than the threshold.
    points = select_points(data, times, size // 16, points_per_subject, min_motion,
                           surface_motion=True)
    pixels = torch.from_numpy(video[indices].copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
    pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=(size, size), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    vae = load_vae(model_root / "Wan2.1_VAE.pth", device)
    observed, reactive = vae.encode([pixels, torch.zeros_like(pixels)])
    control = torch.cat((observed, reactive, torch.zeros((64, *observed.shape[1:]), device=device)), 0).unsqueeze(0).bfloat16()
    generator = torch.Generator(device=device).manual_seed(seed + episode * 100 + CAMERAS.index(camera))
    noise = torch.randn((1, *observed.shape), generator=generator, device=device, dtype=torch.float32).bfloat16()
    del vae, pixels, observed, reactive
    torch.cuda.empty_cache()
    return name, video, times, points, control, noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/libero_video_spatial_long_20261003")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/libero_video_spatial_objects_gripper_trails_20261003/annotations")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--episodes", type=int, nargs="+", default=[2])
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS), choices=CAMERAS)
    parser.add_argument("--candidates", default="6:1,10:6,7:11,14:8")
    parser.add_argument("--eval-candidates", help="Subset to decode; all candidates still load their LoRA")
    parser.add_argument("--variants", nargs="+", choices=(
        "raw_anchor", "raw_rolling", "heft_full", "heft_rope_high", "heft_q_to_k_high", "beam_q_to_k"
    ), help="Decode only selected variants")
    parser.add_argument("--preview-labels", nargs="+", help="Also render individual target objects")
    parser.add_argument("--preview-heads", nargs="+", default=("T7H11", "T6H1"),
                        help="Heads to render; metrics still include every eval candidate")
    parser.add_argument("--preview-variants", nargs="+",
                        default=("beam_q_to_k", "raw_anchor"),
                        choices=("raw_anchor", "raw_rolling", "heft_full", "heft_rope_high",
                                 "heft_q_to_k_high", "beam_q_to_k"),
                        help="Variants to render for preview heads")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--skip-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=129)
    parser.add_argument("--points-per-subject", type=int, default=24)
    parser.add_argument("--min-motion", type=float, default=0.02)
    parser.add_argument("--timestep", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-radius", type=float, default=6.0)
    parser.add_argument("--argmax-radius", type=float, default=2.0)
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--fb-threshold", type=float, default=4.5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--beam-topk", type=int, default=32)
    parser.add_argument("--beam-size", type=int, default=8)
    parser.add_argument("--beam-motion-weight", type=float, default=0.45)
    parser.add_argument("--beam-max-step-cells", type=float, default=4.0)
    parser.add_argument("--beam-local-weight", type=float, default=0.0)
    parser.add_argument("--beam-descriptor-ema", type=float, default=0.0)
    parser.add_argument("--joint-beam-size", type=int, default=8)
    parser.add_argument("--joint-beam-motion-weight", type=float, default=0.45)
    parser.add_argument("--joint-beam-agreement-weight", type=float, default=0.2)
    parser.add_argument("--joint-beam-max-step-cells", type=float, default=4.0)
    parser.add_argument("--joint-beam-max-agreement-cells", type=float, default=3.0)
    parser.add_argument("--consensus-max-disagreement-px", type=float, default=24.0)
    parser.add_argument("--enable-consensus", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    candidates = parse_candidates(args.candidates)
    eval_candidates = parse_candidates(args.eval_candidates) if args.eval_candidates else candidates
    if not set(eval_candidates) <= set(candidates):
        parser.error("eval-candidates must be a subset of candidates")
    (args.output / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    text = load_empty_text(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", device)
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    if args.checkpoint is not None:
        payload = load_lora(tower, args.checkpoint, candidates, args.lora_rank, args.lora_alpha)
        log(stage="lora_loaded", checkpoint=str(args.checkpoint), step=payload.get("step"))
    tower.eval()
    results = []
    for episode in args.episodes:
        for camera in args.cameras:
            started = time.monotonic()
            log(stage="preparing", episode=episode, camera=camera)
            name, video, times, points, control, noise = prepare_case(
                args.video_root, args.tracks_root, args.model, episode, camera,
                args.size, args.skip_frames, args.max_frames, args.points_per_subject,
                args.min_motion, device, args.seed,
            )
            case_dir = args.output / name
            case_dir.mkdir()
            captures = {}
            wanted_layers = sorted({layer for layer, _ in eval_candidates})
            log(stage="tower_forward", case=name, frames=len(times), points=len(points["ids"]))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                tower(noise, control, text, torch.tensor([args.timestep], device=device),
                      lambda layer, q, k: captures.__setitem__(layer, (q.detach(), k.detach())) if layer in wanted_layers else None)
            torch.cuda.synchronize(device)
            log(stage="decoding", case=name, elapsed_seconds=round(time.monotonic() - started, 1))
            grid = args.size // 16
            case_rows = []
            predictions = {}
            beam_predictions = []
            joint_inputs = {}
            for layer, head in eval_candidates:
                q, k = captures[layer]
                q = q[0].reshape(12, len(times), grid * grid, 128)
                k = k[0].reshape(12, len(times), grid * grid, 128)
                joint_inputs[f"T{layer}H{head}"] = (q[head], k[head])
                for variant, query_kind, freq in (
                    ("raw_anchor", None, (0.0, 1.0)),
                    ("raw_rolling", None, (0.0, 1.0)),
                    ("heft_full", "key_to_key", (0.0, 1.0)),
                    ("heft_rope_high", "key_to_key", (0.15, 1.0)),
                    ("heft_q_to_k_high", "q_to_k", (0.15, 1.0)),
                    ("beam_q_to_k", None, (0.0, 1.0)),
                ):
                    if args.variants is not None and variant not in args.variants:
                        continue
                    if variant == "beam_q_to_k":
                        pred = beam_track(
                            q, k, points, head, grid, args.beam_topk, args.beam_size,
                            args.beam_motion_weight, args.beam_max_step_cells,
                            args.beam_local_weight, args.beam_descriptor_ema,
                        )
                        vis = points["valid"].copy()
                        vis[np.arange(len(vis)), points["anchor"]] = True
                        beam_predictions.append(pred)
                    elif variant.startswith("raw_"):
                        decoded, _, rolling = decode_matches(q, k, points["anchor"], points["cells"], grid)
                        pred = cell_to_xy(decoded[head] if variant == "raw_anchor" else rolling[head], grid, 256)
                        vis = points["valid"].copy()
                        vis[np.arange(len(vis)), points["anchor"]] = True
                    else:
                        pred, vis = heft_track(
                            q, k, points, head, grid, query_kind=query_kind,
                            frequency_range=freq, search_radius=args.search_radius,
                            argmax_radius=args.argmax_radius, ema_alpha=args.ema_alpha,
                            fb_threshold=args.fb_threshold,
                        )
                    key = f"T{layer}H{head}/{variant}"
                    predictions[key] = (pred, vis)
                    row = {"head": f"T{layer}H{head}", "variant": variant}
                    for group in ("moving_object", "gripper"):
                        row[group] = metrics_with_visibility(pred, vis, points, group,
                                                             displaced_only=group == "moving_object")
                    for label in ("gripper base", "left finger", "right finger"):
                        row[label] = metrics_for_labels(pred, vis, points, (label,))
                    case_rows.append(row)
            joint_key = "joint/T6H1_T7H11"
            if all(name in joint_inputs for name in ("T6H1", "T7H11")):
                log(stage="joint_beam", case=name)
                joint_pred = joint_beam_track(
                    [joint_inputs["T6H1"], joint_inputs["T7H11"]],
                    points, grid, topk=args.beam_topk, beam_size=args.joint_beam_size,
                    motion_weight=args.joint_beam_motion_weight,
                    agreement_weight=args.joint_beam_agreement_weight,
                    max_step_cells=args.joint_beam_max_step_cells,
                    max_agreement_cells=args.joint_beam_max_agreement_cells,
                )
                joint_vis = points["valid"].copy()
                joint_vis[np.arange(len(joint_vis)), points["anchor"]] = True
                predictions[joint_key] = (joint_pred, joint_vis)
                row = {"head": "T6H1+T7H11", "variant": "joint_beam"}
                for group in ("moving_object", "gripper"):
                    row[group] = metrics_with_visibility(
                        joint_pred, joint_vis, points, group,
                        displaced_only=group == "moving_object"
                    )
                for label in ("gripper base", "left finger", "right finger"):
                    row[label] = metrics_for_labels(joint_pred, joint_vis, points, (label,))
                case_rows.append(row)
            if args.enable_consensus and beam_predictions:
                consensus_pred, consensus_vis = consensus_trajectory(
                    beam_predictions, args.consensus_max_disagreement_px
                )
                predictions["multihead/beam_consensus"] = (consensus_pred, consensus_vis)
                row = {"head": "multihead", "variant": "beam_consensus"}
                for group in ("moving_object", "gripper"):
                    row[group] = metrics_with_visibility(
                        consensus_pred, consensus_vis, points, group,
                        displaced_only=group == "moving_object"
                    )
                for label in ("gripper base", "left finger", "right finger"):
                    row[label] = metrics_for_labels(consensus_pred, consensus_vis, points, (label,))
                case_rows.append(row)
            (case_dir / "metrics.json").write_text(json.dumps(case_rows, indent=2))
            np.savez_compressed(case_dir / "trajectories.npz",
                                **{key.replace("/", "__"): value[0] for key, value in predictions.items()},
                                **{f"{key.replace('/', '__')}__visible": value[1] for key, value in predictions.items()},
                                gt_xy=points["xy"], valid=points["valid"], displaced=points["displaced"],
                                anchor=points["anchor"], groups=points["groups"], labels=points["labels"],
                                frame_indices=times)
            # Visualize the trained head and the first two train-free variants.
            for head_name in args.preview_heads:
                for variant in args.preview_variants:
                    key = f"{head_name}/{variant}"
                    if key not in predictions:
                        continue
                    pred, vis = predictions[key]
                    render_tracking_previews(
                        video, times, pred, points, f"{head_name} {variant}", case_dir,
                        f"{head_name}_{variant}", args.preview_labels,
                    )
            if joint_key in predictions:
                render_tracking_previews(
                    video, times, predictions[joint_key][0], points, "T6H1+T7H11 joint_beam", case_dir,
                    "T6H1_T7H11_joint_beam", args.preview_labels,
                )
            if "multihead/beam_consensus" in predictions:
                for group in ("moving_object", "gripper"):
                    render_preview(
                        video, times, consensus_pred, points, group,
                        "multihead beam_consensus",
                        case_dir / f"multihead_beam_consensus_{group}.mp4",
                    )
            results.append({"name": name, "metrics": case_rows})
            log(stage="case_done", case=name, rows=len(case_rows), elapsed_seconds=round(time.monotonic() - started, 1))
            del captures, control, noise
            gc.collect()
            torch.cuda.empty_cache()
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    log(stage="done", cases=len(results), output=str(args.output))


if __name__ == "__main__":
    main()
