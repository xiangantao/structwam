#!/usr/bin/env python3
"""Read native QK future trajectories from saved GT-assisted denoising states."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_vace_gt_denoising import prepare_latents
from evaluate_vace_pure_noise_future import (
    ROOT, dump, load_transformer, load_vae, log, read_prefix, read_video, text_conditions, write_video,
)
from search_vace_tracking_heads import GRIPPER, cell_to_xy, xy_to_cell


def points_for_case(args, protocol):
    anchor_frame = protocol["start"] + protocol["observed"] - 1
    latent_times = np.arange(protocol["start"], protocol["start"] + protocol["frames"], 4)
    anchor_time = (protocol["observed"] - 1) // 4
    times = latent_times[anchor_time:]
    path = Path(protocol["data"]) / "tracks" / f"episode_{protocol['episode']:05d}_{protocol['camera']}_visible_surface.npz"
    with np.load(path) as z:
        data = {key: z[key] for key in ("xy", "visible", "world_xyz", "point_label", "annotated_frame_indices")}
    if not np.isin(times, data["annotated_frame_indices"]).all():
        raise ValueError("Missing native-grid annotation times")
    target = protocol["diagnostic_target_body"].removesuffix("_main")
    selected = []
    for label in [target, *sorted(GRIPPER)]:
        eligible = np.flatnonzero((data["point_label"] == label) & data["visible"][anchor_frame])
        if not len(eligible):
            continue
        xy = data["xy"][anchor_frame, eligible]
        cells = xy_to_cell(xy, args.grid, 256)
        remaining = set(range(len(eligible)))
        used_cells, chosen = set(), []
        distances = np.linalg.norm(xy - xy.mean(0), axis=-1)
        choice = int(np.argmin(distances))
        while remaining and len(chosen) < args.points_per_label:
            if cells[choice] not in used_cells:
                selected.append(int(eligible[choice]))
                chosen.append(choice)
                used_cells.add(int(cells[choice]))
            remaining.remove(choice)
            if not remaining:
                break
            if chosen:
                distances = np.min(np.linalg.norm(xy[:, None] - xy[chosen][None], axis=-1), axis=1)
            choice = max(remaining, key=lambda i: distances[i])
    ids = np.asarray(selected)
    labels = data["point_label"][ids]
    groups = np.asarray(["gripper" if label in GRIPPER else "moving_object" for label in labels])
    xy = data["xy"][times][:, ids].transpose(1, 0, 2)
    visible = data["visible"][times][:, ids].T
    valid = visible.copy()
    valid[:, 0] = False
    world = data["world_xyz"][times][:, ids].transpose(1, 0, 2)
    displaced = valid & (np.linalg.norm(xy - xy[:, :1], axis=-1) > 8)
    displaced &= np.linalg.norm(world - world[:, :1], axis=-1) > 0.01
    recent = np.asarray([anchor_frame - 8, anchor_frame - 4, anchor_frame])
    recent_visible = data["visible"][recent][:, ids].all(0)
    past = data["xy"][recent][:, ids].transpose(1, 0, 2)
    velocity = (past[:, 2] - past[:, 0]) / 8
    velocity[~recent_visible] = 0
    static = np.repeat(xy[:, :1], len(times), axis=1)
    extrapolated = xy[:, :1] + velocity[:, None] * (times - anchor_frame)[None, :, None]
    return {"ids": ids, "labels": labels, "groups": groups, "xy": xy, "valid": valid,
            "displaced": displaced, "cells": xy_to_cell(xy, args.grid, 256),
            "anchor_time": anchor_time, "times": times, "latent_times": latent_times,
            "velocity_available": recent_visible, "static": static, "velocity": extrapolated}


def metrics(prediction, points, group, displaced_only=False):
    mask = points["displaced"] if displaced_only else points["valid"]
    mask = mask & (points["groups"][:, None] == group)
    if not mask.any():
        return {"epe_px": None, "within8": None, "count": 0}
    error = np.linalg.norm(prediction - points["xy"], axis=-1)
    return {"epe_px": float(error[mask].mean()), "within8": float((error[mask] <= 8).mean()),
            "count": int(mask.sum()), "endpoint_epe_px": float(error[:, -1][mask[:, -1]].mean())
            if mask[:, -1].any() else None}


def describe(prediction, points):
    return {group + ("_displaced" if displaced else ""): metrics(prediction, points, group, displaced)
            for group in ("moving_object", "gripper") for displaced in (False, True)}


class Capture:
    def __init__(self, points, grid):
        self.points, self.grid = points, grid
        self.state, self.predictions, self.rows = None, {}, []

    def __call__(self, branch, layer, query, key):
        p, g = self.points, self.grid
        heads, temporal, spatial, dim = query.shape[1], len(p["latent_times"]), g * g, query.shape[-1]
        if query.shape != key.shape or query.shape[2] != temporal * spatial:
            raise ValueError("Native attention grid disagrees with trajectory annotation")
        q = query[0].reshape(heads, temporal, spatial, dim)
        k = key[0].reshape(heads, temporal, spatial, dim)
        anchor = p["anchor_time"]
        initial_cells = torch.as_tensor(p["cells"][:, 0], device=q.device)
        seed = q[:, anchor, initial_cells].float()
        logits = torch.einsum("hpd,htsd->hpts", seed, k[:, anchor + 1:].float()) / math.sqrt(dim)
        fixed = torch.cat((initial_cells[None, :, None].expand(heads, -1, -1), logits.argmax(-1)), dim=-1)
        current = initial_cells[None].expand(heads, -1).clone()
        paths = [current]
        head_ids = torch.arange(heads, device=q.device)[:, None]
        for ti in range(anchor + 1, temporal):
            previous = q[:, ti - 1][head_ids, current].float()
            scores = torch.einsum("hpd,hsd->hps", previous, k[:, ti].float()) / math.sqrt(dim)
            current = scores.argmax(-1)
            paths.append(current)
        rolling = torch.stack(paths, dim=-1)
        for decoder, cells in (("anchor_qk", fixed), ("rolling_qk", rolling)):
            predictions = cell_to_xy(cells.cpu().numpy(), g, 256)
            predictions[:, :, 0] = p["xy"][:, 0]
            self.predictions[(self.state, branch, layer, decoder)] = predictions
            for head in range(heads):
                self.rows.append({"state_after_steps": self.state, "branch": branch, "layer": layer,
                                  "head": head, "decoder": decoder, **describe(predictions[head], p)})


def render(video, points, prediction, group, title, path):
    ids = np.flatnonzero(points["groups"] == group)[:6]
    frames = []
    for ti, frame_index in enumerate(points["times"]):
        frame = cv2.resize(video[frame_index], (512, 512))
        cv2.rectangle(frame, (0, 0), (512, 42), (0, 0, 0), -1)
        cv2.putText(frame, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "GT green / native QK magenta", (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for pid in ids:
            if ti and not points["valid"][pid, ti]:
                continue
            gt = np.round(points["xy"][pid] * 2).astype(int)
            pred = np.round(prediction[pid] * 2).astype(int)
            for prev in range(ti):
                if (prev == 0 or points["valid"][pid, prev]) and points["valid"][pid, prev + 1]:
                    cv2.line(frame, tuple(gt[prev]), tuple(gt[prev + 1]), (40, 240, 60), 1)
                    cv2.line(frame, tuple(pred[prev]), tuple(pred[prev + 1]), (240, 40, 210), 1)
            cv2.circle(frame, tuple(gt[ti]), 3, (40, 240, 60), -1)
            cv2.circle(frame, tuple(pred[ti]), 3, (240, 40, 210), -1)
        frames.append(frame)
    write_video(path, np.stack(frames * 3), 5)
    return frames[-1]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/vace_future16_sigma080_steps_20261006")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, nargs="+", default=[0, 10, 13, 18, 22])
    parser.add_argument("--points-per-label", type=int, default=12)
    options = parser.parse_args()
    protocol = json.loads((options.source / "protocol.json").read_text())
    args = argparse.Namespace(**protocol)
    args.data, args.model, args.text_model = Path(args.data), Path(args.model), Path(args.text_model)
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    options.grid = args.size // 16
    points = points_for_case(options, protocol)
    source = args.data / "videos" / f"episode_{args.episode:05d}" / f"{args.camera}.mp4"
    video = read_video(source)
    truth = video[args.start:args.start + args.frames]
    prefix = read_prefix(source, args.start, args.observed)
    study_protocol = protocol | {
        "study": "Exploratory native attention future-trajectory readout on one privileged-GT clip",
        "states_after_updates": options.states,
        "state_matching": "After n updates, read the saved input to forward n+1 at its recorded timestep",
        "annotation_coordinates": "256-pixel original simulation camera; model input upsampled to512",
        "head_indices": "zero-based; main30 layers x12 heads, control15 layers x12 heads",
        "query_seed": "Surface points visible in the last observed frame; no future GT query refresh",
        "targets": points["times"].tolist(), "grid": options.grid,
        "decoder": "Native post-RoPE QK dot product, argmax restricted to each future time's complete spatial grid",
        "selection": "Head ranking uses GT of this diagnostic clip; not a generalization benchmark",
    }
    dump(options.output / "protocol.json", study_protocol)
    np.savez_compressed(options.output / "points.npz", **points)
    baselines = {name: describe(prediction, points) for name, prediction in
                 (("hold_position", points["static"]), ("recent_velocity", points["velocity"]),
                  ("grid_quantization_oracle", cell_to_xy(points["cells"], options.grid, 256)))}
    dump(options.output / "baselines.json", baselines)
    log(stage="points", points=len(points["ids"]), labels={label: int((points["labels"] == label).sum())
                                                          for label in np.unique(points["labels"])},
        baselines=baselines)
    positive, negative, _ = text_conditions(args.text_model, args.task, device)
    model = load_transformer(args.model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    control, _, _ = prepare_latents(vae, prefix, truth, args, device)
    del vae
    torch.cuda.empty_cache()
    capture = Capture(points, options.grid)
    for branch, blocks in (("main", model.blocks), ("control", model.vace_blocks)):
        for layer, block in enumerate(blocks):
            block.attn1._heft_capture = lambda branch=branch, layer=layer, **kw: capture(branch, layer, kw["query"], kw["key"])
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for state in options.states:
        snapshot = torch.load(options.source / f"step_{state + 1:03d}" / "latents.pt",
                              map_location="cpu", weights_only=True)
        capture.state = state
        logits = model(hidden_states=snapshot["model_input"].to(device),
                       timestep=torch.tensor([snapshot["timestep"]], device=device),
                       encoder_hidden_states=positive, control_hidden_states=control,
                       control_hidden_states_scale=torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16),
                       return_dict=False)[0]
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite model prediction")
        dump(options.output / "rows.json", capture.rows)
        log(stage="state_scanned", state_after_steps=state, sigma=snapshot["input_sigma"],
            rows=len(capture.rows), elapsed_seconds=time.monotonic() - started)
        del logits, snapshot
    for blocks in (model.blocks, model.vace_blocks):
        for block in blocks:
            del block.attn1._heft_capture
    summaries, preview_tiles = [], []
    for state in options.states:
        for branch in ("main", "control"):
            rows = [row for row in capture.rows if row["state_after_steps"] == state and row["branch"] == branch]
            for group in ("moving_object", "gripper"):
                key = group + "_displaced"
                eligible = [row for row in rows if row[key]["epe_px"] is not None]
                ranked = sorted(eligible, key=lambda row: row[key]["epe_px"])
                best = ranked[0]
                prediction = capture.predictions[(state, branch, best["layer"], best["decoder"])][best["head"]]
                preview_path = options.output / f"after{state:02d}_{branch}_{group}_best.mp4"
                tile = render(video, points, prediction, group,
                              f"after{state} {branch} L{best['layer']}H{best['head']} {best['decoder']}", preview_path)
                if branch == "main":
                    preview_tiles.append(tile)
                summaries.append({"state_after_steps": state, "branch": branch, "group": group,
                                  "best": best, "median_epe_px": float(np.median([row[key]["epe_px"] for row in ranked])),
                                  "preview": str(preview_path)})
    fixed_curves = []
    for branch in ("main", "control"):
        for group in ("moving_object", "gripper"):
            reference_rows = [row for row in capture.rows if row["branch"] == branch and row["state_after_steps"] == 13]
            key = group + "_displaced"
            reference = min(reference_rows, key=lambda row: row[key]["epe_px"])
            curve = [row for row in capture.rows if row["branch"] == branch and row["layer"] == reference["layer"]
                     and row["head"] == reference["head"] and row["decoder"] == reference["decoder"]]
            fixed_curves.append({"branch": branch, "group": group, "selected_at_state": 13,
                                 "layer": reference["layer"], "head": reference["head"],
                                 "decoder": reference["decoder"], "curve": curve})
    keys = list(capture.predictions)
    np.savez_compressed(options.output / "predictions.npz",
                        predictions=np.stack([capture.predictions[key] for key in keys]),
                        keys=np.asarray([json.dumps(key) for key in keys]))
    cv2.imwrite(str(options.output / "main_best_endpoint_overview.jpg"),
                cv2.cvtColor(np.concatenate([np.concatenate(preview_tiles[i:i + 2], axis=1)
                                             for i in range(0, len(preview_tiles), 2)]), cv2.COLOR_RGB2BGR))
    completed = {"elapsed_seconds": time.monotonic() - started, "states": options.states,
                 "rows": len(capture.rows), "summaries": summaries, "fixed_head_curves": fixed_curves,
                 "baselines": baselines, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    dump(options.output / "completed.json", completed)
    log(stage="completed", seconds=completed["elapsed_seconds"], rows=completed["rows"])


if __name__ == "__main__":
    main()
