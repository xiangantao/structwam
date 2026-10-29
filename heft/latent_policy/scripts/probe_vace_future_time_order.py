#!/usr/bin/env python3
"""Test whether a fixed native QK head follows reordered privileged future content."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_vace_future_motion_heads import Capture, describe
from evaluate_vace_gt_denoising import prepare_latents
from evaluate_vace_pure_noise_future import (
    dump, load_transformer, load_vae, log, read_prefix, read_video, text_conditions,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--states", type=int, nargs="+", default=[0, 13])
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--head", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads((args.study / "protocol.json").read_text())
    source = args.source
    original = json.loads((source / "protocol.json").read_text())
    model_args = argparse.Namespace(**original)
    for name in ("data", "model", "text_model"):
        setattr(model_args, name, Path(getattr(model_args, name)))
    with np.load(args.study / "points.npz") as archive:
        points = {name: archive[name] for name in archive.files}
    rows = json.loads((args.study / "rows.json").read_text())
    order = np.r_[0, np.arange(len(points["times"]) - 1, 0, -1)]
    reordered = dict(points)
    for name in ("xy", "valid", "displaced", "cells"):
        reordered[name] = points[name][:, order]
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    video_path = model_args.data / "videos" / f"episode_{model_args.episode:05d}" / f"{model_args.camera}.mp4"
    video = read_video(video_path)
    truth = video[model_args.start:model_args.start + model_args.frames]
    prefix = read_prefix(video_path, model_args.start, model_args.observed)
    positive, _, _ = text_conditions(model_args.text_model, model_args.task, device)
    model = load_transformer(model_args.model, device)
    vae = load_vae(model_args.model / "Wan2.1_VAE.pth", device)
    control, _, _ = prepare_latents(vae, prefix, truth, model_args, device)
    del vae
    torch.cuda.empty_cache()
    capture = Capture(points, protocol["grid"])
    model.blocks[args.layer].attn1._heft_capture = lambda **kw: capture(
        "main", args.layer, kw["query"], kw["key"])
    results = []
    for state in args.states:
        snapshot = torch.load(source / f"step_{state + 1:03d}" / "latents.pt", map_location="cpu", weights_only=True)
        inputs = snapshot["model_input"].to(device).clone()
        anchor = int(points["anchor_time"])
        inputs[:, :, anchor + 1:] = inputs[:, :, anchor + 1:].flip(2)
        if not torch.equal(inputs[:, :, :anchor + 1].cpu(), snapshot["model_input"][:, :, :anchor + 1]):
            raise RuntimeError("Observed native positions changed during the future-only intervention")
        capture.state = state
        model(hidden_states=inputs, timestep=torch.tensor([snapshot["timestep"]], device=device),
              encoder_hidden_states=positive, control_hidden_states=control,
              control_hidden_states_scale=torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16),
              return_dict=False)
        for decoder in ("anchor_qk", "rolling_qk"):
            prediction = capture.predictions[(state, "main", args.layer, decoder)][args.head]
            reference = next(row for row in rows if row["state_after_steps"] == state and row["branch"] == "main"
                             and row["layer"] == args.layer and row["head"] == args.head and row["decoder"] == decoder)
            result = {"state_after_steps": state, "sigma": snapshot["input_sigma"],
                      "layer": args.layer, "head": args.head, "decoder": decoder,
                      "unaltered_input_original_targets": {key: value for key, value in reference.items()
                                                           if key.startswith(("moving_object", "gripper"))},
                      "reordered_input_original_targets": describe(prediction, points),
                      "reordered_input_reordered_targets": describe(prediction, reordered)}
            results.append(result)
            log(stage="future_order_control", **result)
    dump(args.study / "future_order_control.json", {
        "intervention": "Reverse only the four future native temporal blocks, including their existing noise; observed inputs, control and prompt remain identical",
        "original_target_frames": points["times"].tolist(),
        "reordered_target_frames": points["times"][order].tolist(),
        "head_selection": "The fixed main L15H4 was selected on the original clip at state13 before this intervention",
        "interpretation": "Single-forward content sensitivity diagnostic, not a generated counterfactual or autonomous prediction test",
        "results": results,
    })
    del model.blocks[args.layer].attn1._heft_capture


if __name__ == "__main__":
    main()
