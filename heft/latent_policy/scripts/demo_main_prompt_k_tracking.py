#!/usr/bin/env python3
"""Demo: main prompt heatmap -> K region -> dense Q/K tracking."""

from __future__ import annotations

import argparse
import math
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(ROOT / "heft/latent_policy/scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from benchmark_vace_observed33 import observed_inputs  # noqa: E402
from evaluate_vace_pure_noise_future import load_transformer  # noqa: E402
from scan_vace_main_prompt_attention import (  # noqa: E402
    find_phrase_positions,
    load_text,
    phrase_token_ids,
    read_window,
)
from search_vace_tracking_heads import load_empty_text, load_vae, write_video  # noqa: E402


def capture(model, prompt_positions, noise, control, text, timestep, scales,
            frames, grid, semantic_layer, semantic_head, k_heads, track_layer, track_head):
    semantic = {}
    qk = {}
    k_values = {}

    def semantic_callback(query, key, **_):
        q, k = query[0].float(), key[0].float()
        values = []
        for start in range(0, q.shape[1], 2048):
            logits = torch.einsum("hnd,htd->hnt", q[:, start:start + 2048], k) / math.sqrt(q.shape[-1])
            probability = torch.softmax(logits, dim=-1)
            values.append(probability[..., prompt_positions].sum(-1).cpu())
        semantic[semantic_layer] = torch.cat(values, dim=-1)

    def qk_callback(query, key, **_):
        qk["q"] = query[0, track_head].float().cpu()
        qk["k"] = key[0, track_head].float().cpu()

    def make_k_hook(layer):
        def hook(_module, _inputs, output):
            values = output[0].reshape(frames, grid * grid, model.config.num_attention_heads,
                                       model.config.attention_head_dim)
            k_values[layer] = values[:, :, k_heads[layer], :].float().cpu()
        return hook

    main_block = model.blocks[semantic_layer]
    main_block.attn2._heft_capture = semantic_callback
    track_block = model.blocks[track_layer]
    track_block.attn1._heft_capture = qk_callback
    handles = [model.blocks[layer].attn1.norm_k.register_forward_hook(make_k_hook(layer))
               for layer in k_heads.keys()]
    try:
        with torch.inference_mode():
            model(hidden_states=noise, timestep=timestep, encoder_hidden_states=text,
                  control_hidden_states=control, control_hidden_states_scale=scales,
                  return_dict=False)
    finally:
        if hasattr(main_block.attn2, "_heft_capture"):
            del main_block.attn2._heft_capture
        if hasattr(track_block.attn1, "_heft_capture"):
            del track_block.attn1._heft_capture
        for handle in handles:
            handle.remove()
    if semantic_layer not in semantic or "q" not in qk or set(k_values) != set(k_heads):
        raise RuntimeError("Missing one or more main attention captures")
    return semantic[semantic_layer][semantic_head], qk["q"], qk["k"], k_values


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def region_from_prompt_and_k(heatmap, k_values, anchor, grid, seed_fraction=0.05, keep_fraction=0.15):
    # Prompt creates seeds; K only expands/filters their spatially similar region.
    heat = heatmap[anchor].reshape(grid * grid).astype(np.float32)
    heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-8)
    seed_count = max(4, int(grid * grid * seed_fraction))
    seeds = np.argpartition(heat, -seed_count)[-seed_count:]
    seed_features = []
    for values in k_values.values():
        field = normalize(values[anchor])
        seed_features.append(field[seeds].mean(0))
    similarities = []
    for values, seed in zip(k_values.values(), seed_features, strict=True):
        field = normalize(values[anchor])
        similarities.append(field @ (seed / max(np.linalg.norm(seed), 1e-8)))
    affinity = np.mean(similarities, axis=0)
    affinity = (affinity - affinity.min()) / max(float(affinity.max() - affinity.min()), 1e-8)
    score = heat * affinity
    keep = max(seed_count, int(grid * grid * keep_fraction))
    selected = np.argpartition(score, -keep)[-keep:]
    return score.reshape(grid, grid), np.asarray(selected, dtype=np.int64)


def k_region_over_time(k_values, seeds, anchor_time, grid):
    prototypes = []
    for values in k_values.values():
        field = normalize(values[anchor_time])
        prototype = field[seeds].mean(0)
        prototypes.append(prototype / max(np.linalg.norm(prototype), 1e-8))
    scores = []
    for time in range(next(iter(k_values.values())).shape[0]):
        affinities = []
        for values, prototype in zip(k_values.values(), prototypes, strict=True):
            affinities.append(normalize(values[time]) @ prototype)
        affinity = np.mean(affinities, axis=0)
        affinity = (affinity - affinity.min()) / max(float(affinity.max() - affinity.min()), 1e-8)
        scores.append(affinity)
    return np.stack(scores)


def track_dense(q, k, anchors, anchor_time, grid, region_scores, topk=32,
                beam_size=4, max_step=3.5, motion_weight=0.45,
                local_weight=0.35, region_weight=0.25, acceleration_weight=0.15,
                descriptor_ema=0.1, cycle_weight=0.0, direction="q_to_k"):
    """Beam decode with motion, local descriptor, region, and cycle constraints."""
    q = normalize(q.reshape(len(q) // (grid * grid), grid * grid, -1))
    k = normalize(k.reshape(len(k) // (grid * grid), grid * grid, -1))
    paths = np.zeros((len(anchors), q.shape[0] - anchor_time, 2), dtype=np.float32)
    topk = min(topk, grid * grid)
    for point_id, anchor in enumerate(anchors):
        anchor_xy = np.asarray([anchor % grid, anchor // grid], np.float32)
        descriptor = k[anchor_time, anchor].copy()
        beams = [(0.0, int(anchor), np.zeros(2, np.float32), [anchor_xy.copy()], descriptor)]
        for time in range(anchor_time + 1, q.shape[0]):
            candidates_by_beam = []
            for total, previous_cell, velocity, path, previous_descriptor in beams:
                if direction == "q_to_k":
                    logits = q[time - 1, previous_cell] @ k[time].T
                elif direction == "k_to_q":
                    logits = k[time - 1, previous_cell] @ q[time].T
                else:
                    raise ValueError(f"Unknown tracking direction: {direction}")
                candidate_cells = np.argpartition(logits, -topk)[-topk:]
                previous_xy = np.asarray([previous_cell % grid, previous_cell // grid], np.float32)
                forecast = previous_xy + velocity
                candidate_xy = np.stack((candidate_cells % grid, candidate_cells // grid), axis=-1).astype(np.float32)
                distances = np.sum((candidate_xy - forecast[None]) ** 2, axis=-1)
                allowed = distances <= max_step ** 2
                if not allowed.any():
                    allowed[np.argmin(distances)] = True
                for index, cell in enumerate(candidate_cells):
                    if not allowed[index]:
                        continue
                    step = candidate_xy[index] - previous_xy
                    acceleration = np.linalg.norm(step - velocity)
                    local = float(k[time, cell] @ previous_descriptor)
                    # Reject one-way matches: a valid correspondence should
                    # also read the previous cell when queried backwards.
                    reverse_logits = q[time, cell] @ k[time - 1].T
                    cycle_margin = float(reverse_logits[previous_cell] - reverse_logits.max())
                    score = (total + float(logits[cell]) + local_weight * local
                             + region_weight * float(region_scores[time, cell])
                             + cycle_weight * cycle_margin
                             - motion_weight * float(distances[index])
                             - acceleration_weight * float(acceleration ** 2))
                    next_descriptor = ((1.0 - descriptor_ema) * previous_descriptor
                                       + descriptor_ema * k[time, cell])
                    candidates_by_beam.append((score, int(cell), step, path + [candidate_xy[index].copy()], next_descriptor))
            candidates_by_beam.sort(key=lambda item: item[0], reverse=True)
            beams = candidates_by_beam[:beam_size]
        paths[point_id] = np.asarray(beams[0][3], dtype=np.float32)
    return paths


def regularize_paths(paths, max_step=3.0, max_acceleration=1.5, passes=2):
    """Remove isolated readout jumps while preserving sustained motion."""
    filtered = np.asarray(paths, dtype=np.float32).copy()
    for _ in range(max(1, passes)):
        for point_id in range(filtered.shape[0]):
            previous_velocity = np.zeros(2, dtype=np.float32)
            for time in range(1, filtered.shape[1]):
                velocity = filtered[point_id, time] - filtered[point_id, time - 1]
                speed = float(np.linalg.norm(velocity))
                if speed > max_step:
                    velocity *= max_step / max(speed, 1e-8)
                if time > 1:
                    acceleration = velocity - previous_velocity
                    magnitude = float(np.linalg.norm(acceleration))
                    if magnitude > max_acceleration:
                        velocity = previous_velocity + acceleration * (max_acceleration / magnitude)
                filtered[point_id, time] = filtered[point_id, time - 1] + velocity
                previous_velocity = velocity
    return filtered


def render(video, heat, anchor_region, paths, anchor_time, grid, output, fps=5):
    frames = []
    for ti in range(heat.shape[0]):
        rgb_index = min(ti * 4, len(video) - 1)
        image = cv2.resize(video[rgb_index], (512, 512), interpolation=cv2.INTER_AREA)
        values = anchor_region if ti == anchor_time else heat[ti]
        values = (values - values.min()) / max(float(values.max() - values.min()), 1e-8)
        values = cv2.resize(values.astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)
        colored = cv2.applyColorMap(np.uint8(values * 255), cv2.COLORMAP_JET)[:, :, ::-1]
        image = np.uint8(0.7 * image + 0.3 * colored)
        for point in range(paths.shape[0]):
            for prev in range(1, ti - anchor_time + 1):
                if ti < anchor_time + prev:
                    break
                p0 = paths[point, prev - 1] * (512 / grid) + 512 / (2 * grid)
                p1 = paths[point, prev] * (512 / grid) + 512 / (2 * grid)
                cv2.line(image, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), (60, 255, 60), 1)
            if ti >= anchor_time:
                p = paths[point, ti - anchor_time] * (512 / grid) + 512 / (2 * grid)
                cv2.circle(image, tuple(np.round(p).astype(int)), 2, (255, 255, 255), -1)
        cv2.putText(image, f"prompt -> K region -> L15H4 track | native t={ti}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(image)
    write_video(output, np.stack(frames * 3), fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postprocess-max-step", type=float, default=3.0)
    parser.add_argument("--postprocess-max-acceleration", type=float, default=1.5)
    parser.add_argument("--postprocess-passes", type=int, default=2)
    parser.add_argument("--episode", type=int, default=39)
    parser.add_argument("--camera", choices=("agentview", "robot0_eye_in_hand"), default="agentview")
    parser.add_argument("--start", type=int, default=48)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--prompt", default="pick up the black bowl on the wooden cabinet and place it on the plate")
    parser.add_argument("--phrase", default="black bowl")
    parser.add_argument("--semantic-layer", type=int, default=22)
    parser.add_argument("--semantic-head", type=int, default=4)
    parser.add_argument("--track-layer", type=int, default=15)
    parser.add_argument("--track-head", type=int, default=4)
    parser.add_argument("--k-layers", type=int, nargs="+", default=[22, 15])
    parser.add_argument("--k-heads", type=int, nargs="+", default=[9, 7])
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--beam-topk", type=int, default=32)
    parser.add_argument("--max-step", type=float, default=4.0)
    parser.add_argument("--motion-weight", type=float, default=0.15)
    parser.add_argument("--local-weight", type=float, default=0.25)
    parser.add_argument("--region-weight", type=float, default=0.08)
    parser.add_argument("--acceleration-weight", type=float, default=0.05)
    parser.add_argument("--descriptor-ema", type=float, default=0.1)
    parser.add_argument("--cycle-weight", type=float, default=0.0)
    parser.add_argument("--track-direction", choices=("q_to_k", "k_to_q"), default="q_to_k")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    args = parser.parse_args()
    if len(args.k_layers) != len(args.k_heads):
        parser.error("--k-layers and --k-heads must have equal length")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    video_path = args.data / "videos" / f"episode_{args.episode:05d}" / f"{args.camera}.mp4"
    video = read_window(video_path, args.start, args.frames)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    control, encoded = observed_inputs(vae, video, argparse.Namespace(size=args.size), device)
    noise = torch.randn((1, 16, encoded.shape[1], args.size // 8, args.size // 8),
                        device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(42))
    del vae, encoded
    torch.cuda.empty_cache()
    text, input_ids, _ = load_text(args.text_model, args.prompt, device)
    phrase_ids = phrase_token_ids(args.text_model, args.phrase)
    positions = find_phrase_positions(input_ids, phrase_ids)
    model = load_transformer(args.model, device)
    scales = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
    grid = args.size // 16
    k_heads = {layer: head for layer, head in zip(args.k_layers, args.k_heads, strict=True)}
    heat, q, k, k_values = capture(
        model, positions, noise, control, text, torch.tensor([1000], device=device), scales,
        (args.frames - 1) // 4 + 1, grid, args.semantic_layer, args.semantic_head, k_heads,
        args.track_layer, args.track_head)
    heat = heat.numpy().reshape((args.frames - 1) // 4 + 1, grid, grid)
    k_values = {layer: values.numpy() for layer, values in k_values.items()}
    anchor_time = 4
    combined_heat, selected = region_from_prompt_and_k(heat, k_values, anchor_time, grid)
    # Dense seeds are selected from the prompt/K region, with no object center or label.
    seeds = selected[np.argsort(combined_heat.reshape(-1)[selected])[-min(64, len(selected)):]]
    region_scores = k_region_over_time(k_values, seeds, anchor_time, grid)
    paths = track_dense(q.numpy(), k.numpy(), seeds, anchor_time, grid, region_scores,
                         topk=args.beam_topk, beam_size=args.beam_size, max_step=args.max_step,
                         motion_weight=args.motion_weight, local_weight=args.local_weight,
                         region_weight=args.region_weight, acceleration_weight=args.acceleration_weight,
                         descriptor_ema=args.descriptor_ema, cycle_weight=args.cycle_weight,
                         direction=args.track_direction)
    raw_paths = paths.copy()
    if args.postprocess:
        paths = regularize_paths(paths, max_step=args.postprocess_max_step,
                                 max_acceleration=args.postprocess_max_acceleration,
                                 passes=args.postprocess_passes)
    render(video, heat, combined_heat, paths,
           anchor_time, grid, args.output / "prompt_k_dense_tracking.mp4")
    np.savez_compressed(args.output / "readout.npz", prompt_heatmap=heat, combined_region=combined_heat, region_scores=region_scores,
                        selected_cells=selected, seeds=seeds, raw_paths=raw_paths, paths=paths)
    (args.output / "protocol.json").write_text(json.dumps({
        "input": "observed 33-frame video through control tower; main pure noise; full task prompt",
        "prompt": args.prompt, "phrase": args.phrase, "semantic_head": f"L{args.semantic_layer}H{args.semantic_head}",
        "k_heads": [f"L{layer}H{head}" for layer, head in k_heads.items()],
        "track_head": f"L{args.track_layer}H{args.track_head}",
        "gt_used_for_readout": False, "cluster_centers": False,
        "decoder": f"{args.track_direction} beam with velocity, acceleration, local K descriptor, and time-varying K-region gate",
        "postprocess": {"enabled": bool(args.postprocess), "max_step_cells": args.postprocess_max_step,
                        "max_acceleration_cells": args.postprocess_max_acceleration, "passes": args.postprocess_passes},
    }, indent=2))
    print(json.dumps({"output": str(args.output), "video": str(args.output / "prompt_k_dense_tracking.mp4"),
                      "native_frames": int(heat.shape[0]), "selected_region_cells": int(len(selected)),
                      "dense_seeds": int(len(seeds))}, indent=2))


if __name__ == "__main__":
    main()
