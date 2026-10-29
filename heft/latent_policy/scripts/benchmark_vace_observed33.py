#!/usr/bin/env python3
"""Matched 33-frame observed-only QK comparison for main and control branches."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from benchmark_vace_motion_heads_all_suites import (
    FIXED, Capture, dump, encode_texts, load_transformer, load_vae, read_video, seed_ids,
)
from evaluate_vace_pure_noise_future import ROOT
from search_vace_tracking_heads import GRIPPER, cell_to_xy, xy_to_cell


def points_for_case(args, case):
    anchor_rgb = case["start"] + 16
    times = np.arange(anchor_rgb, case["start"] + 33, 4)
    path = args.data / "tracks" / f"episode_{case['episode']:05d}_{case['camera']}_visible_surface.npz"
    with np.load(path) as archive:
        data = {key: archive[key] for key in ("xy", "visible", "world_xyz", "point_label", "body_id", "annotated_frame_indices")}
    if not np.isin(times, data["annotated_frame_indices"]).all():
        raise ValueError(f"Missing annotations for {case['name']}")
    ids = seed_ids(data, anchor_rgb, args.size // 16, args.points_per_body)
    xy = data["xy"][times][:, ids].transpose(1, 0, 2)
    valid = data["visible"][times][:, ids].T.copy()
    valid[:, 0] = False
    world = data["world_xyz"][times][:, ids].transpose(1, 0, 2)
    world_delta = np.linalg.norm(world - world[:, :1], axis=-1)
    displaced = valid & (world_delta > .01) & (np.linalg.norm(xy - xy[:, :1], axis=-1) > 8)
    labels = data["point_label"][ids]
    gripper = np.isin(labels, list(GRIPPER))
    moving_labels = []
    for label in np.unique(data["point_label"]):
        body = np.flatnonzero(data["point_label"] == label)
        excursion = np.linalg.norm(data["world_xyz"][times][:, body] - data["world_xyz"][times[0], body], axis=-1).max()
        if excursion >= .02:
            moving_labels.append(str(label))
    moving = np.isin(labels, moving_labels) & ~gripper
    phase_masks = {
        "moving_object_displaced": displaced & moving[:, None],
        "moving_object_visible": valid & moving[:, None],
        "gripper_displaced": displaced & gripper[:, None],
        "gripper_visible": valid & gripper[:, None],
    }
    return {
        "ids": ids, "labels": labels, "xy": xy, "valid": valid, "displaced": displaced,
        "cells": xy_to_cell(xy, args.size // 16, 256), "times": times,
        "anchor_time": 4, "latent_times": np.arange(case["start"], case["start"] + 33, 4),
        "phase_masks": phase_masks, "score_labels": np.where(gripper, "gripper", labels),
    }


@torch.no_grad()
def observed_inputs(vae, video, args, device):
    pixels = torch.from_numpy(video.copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
    pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=(args.size, args.size), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    encoded = vae.encode([pixels])[0]
    reactive = vae.encode([torch.zeros_like(pixels)])[0]
    mask = torch.zeros((64, *encoded.shape[1:]), device=device, dtype=encoded.dtype)
    control = torch.cat((encoded, reactive, mask), 0)[None].bfloat16()
    return control, encoded


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--episodes", type=int, nargs="+", default=[3, 7, 39, 47, 83, 103, 139, 159])
    parser.add_argument("--camera", choices=("agentview", "robot0_eye_in_hand"), default="agentview")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--points-per-body", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split = json.loads((args.data / "split.json").read_text())
    records = {row["episode"]: row for row in split["episodes"]}
    meta = {}
    cases = []
    for episode in args.episodes:
        directory = args.data / "videos" / f"episode_{episode:05d}"
        metadata = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "sim_state.npz") as archive:
            positions, rotations = archive["body_xpos"], archive["body_xmat"]
        names = metadata["body_names"]
        roots = [i for i, name in enumerate(names) if name.endswith("_main")]
        hand = names.index("gripper0_eef")
        best = None
        for start in range(16, len(positions) - 33 + 1, 4):
            times = np.arange(start + 32, start + 33, 4)
            object_motion = float(np.linalg.norm(positions[times][:, roots] - positions[start + 16, roots], axis=-1).max())
            hand_motion = float(np.linalg.norm(positions[times, hand] - positions[start + 16, hand], axis=-1).max())
            candidate = (object_motion + .01 * hand_motion, start)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise ValueError(f"Episode {episode} too short")
        cases.append(records[episode] | {"episode": episode, "camera": args.camera, "start": best[1],
                                        "name": f"episode_{episode:05d}_{args.camera}", "selection_score": best[0]})
    dump(args.output / "protocol.json", {
        "input": "33 recorded RGB frames, all supplied to VACE control condition; no future frames",
        "query": "17th RGB frame (native temporal position4), evaluated on native positions5-8 through the final 33rd frame",
        "main": "pure Gaussian noise at timestep1000, empty prompt",
        "control": "encoded 33-frame video, zero mask, zero reactive branch",
        "same_forward": "Main and control Q/K captured in the same model forward; no GT latent, denoising or LoRA",
        "metrics": "Original256 coordinates; displaced object requires >1cm world and >8px image motion",
        "cases": cases, "fixed_heads": FIXED, "points_per_body": args.points_per_body,
    })
    device = torch.device("cuda:0")
    text = encode_texts(args.text_model, [""], device)[""].to(device)
    model, vae = load_transformer(args.model, device), load_vae(args.model / "Wan2.1_VAE.pth", device)
    scale = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
    active = [None]
    for branch, blocks in (("main", model.blocks), ("control", model.vace_blocks)):
        for layer, block in enumerate(blocks):
            block.attn1._heft_capture = lambda branch=branch, layer=layer, **kw: active[0](branch, layer, kw["query"], kw["key"])
    rows = []
    for index, case in enumerate(cases):
        video_path = args.data / "videos" / f"episode_{case['episode']:05d}" / f"{args.camera}.mp4"
        video = read_video(video_path)[case["start"]:case["start"] + 33]
        points = points_for_case(args, case)
        control, encoded = observed_inputs(vae, video, args, device)
        noise = torch.randn((1, 16, encoded.shape[1], args.size // 8, args.size // 8), device=device,
                            generator=torch.Generator(device=device).manual_seed(args.seed)).bfloat16()
        capture = Capture(points, args.size // 16)
        capture.experiment, capture.state, capture.enabled = "observed33", 0, True
        active[0] = capture
        output = model(hidden_states=noise, timestep=torch.tensor([1000], device=device), encoder_hidden_states=text,
                       control_hidden_states=control, control_hidden_states_scale=scale, return_dict=False)[0]
        if not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite output")
        rows.extend([row | {"case": case["name"], "suite": case["suite"]} for row in capture.rows])
        log = {"case": case["name"], "finished": index + 1, "rows": len(capture.rows)}
        print(json.dumps(log), flush=True)
        del output, capture, control, encoded, noise, points
        active[0] = None
    dump(args.output / "rows.json", rows)
    dump(args.output / "summary.json", {"cases": len(cases), "rows": len(rows), "input_protocol": "33 observed frames only"})
    for blocks in (model.blocks, model.vace_blocks):
        for block in blocks:
            del block.attn1._heft_capture


if __name__ == "__main__":
    main()
