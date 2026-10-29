#!/usr/bin/env python3
"""Read local prompt hotspots from frozen native attention and track their seeds."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.special import logsumexp

from demo_vace_prompt_seed_tracking import heat_density, render_case, select_seeds
from evaluate_vace_prompt_k_guidance import ROOT, encode_prompts
from heft_postprocess_vace_tracking import beam_track, load_lora
from search_vace_tracking_heads import CAMERAS, ControlTower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    start = time.monotonic()
    protocol = json.loads((args.source / "protocol.json").read_text())
    cases = protocol["cases"]
    protocol["head_selection"] = "all180 heads; frame-wise spatial normalization of native word probability, then region-local excess; no GT selection"
    protocol["source"] = str(args.source.resolve())
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    def log(**values):
        values["elapsed_seconds"] = round(time.monotonic() - start, 1)
        line = json.dumps(values)
        print(line, flush=True)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")

    for case in cases:
        directory = args.output / case["name"]
        directory.mkdir(exist_ok=True)
        with np.load(args.source / case["name"] / "semantic_fields.npz") as z:
            native = z["log_native_word_probability"].astype(np.float64)
            regions, words, times = z["regions"], z["words"].tolist(), z["times"]
        normalized = native - logsumexp(native, -1, keepdims=True)
        seeds = select_seeds(normalized, regions, words)
        for seed in seeds:
            seed["ranking"] = "spatially normalized hotspot, not absolute native attention mass"
            seed["spatial_region_probability"] = seed.pop("native_region_probability")
            seed["spatial_frame_probability"] = seed.pop("native_frame_probability")
            seed["log_excess_spatial_probability"] = seed.pop("log_excess_probability")
        heat = heat_density(native, regions, seeds)
        np.savez_compressed(directory / "semantic_fields.npz", heat=heat, times=times)
        (directory / "seeds.json").write_text(json.dumps(seeds, indent=2))
        log(stage="seeds_selected", case=case["name"], points=[(s["word"], s["head"], s["cell"]) for s in seeds])

    device = torch.device("cuda:0")
    _, texts = encode_prompts(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", [""], device)
    checkpoint = Path(protocol["tracking_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tower = ControlTower.load(ROOT / "models/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors", device)
    load_lora(tower, checkpoint, payload["candidates"], payload["rank"], payload["alpha"])
    tower.eval().requires_grad_(False)
    summaries = []
    for case in cases:
        directory = args.output / case["name"]
        seeds = json.loads((directory / "seeds.json").read_text())
        saved = torch.load(args.source / case["name"] / "control.pt", map_location="cpu", weights_only=False)
        times = saved["times"]
        generator = torch.Generator(device=device).manual_seed(42 + case["episode"] * 37 + CAMERAS.index(case["camera"]))
        noise = torch.randn((1, 16, len(times), 64, 64), generator=generator, device=device).bfloat16()
        control = saved["control"].to(device)
        capture = {}

        def store(layer, q, k):
            if layer == 7:
                capture["q"] = q[0, 11].detach().reshape(1, len(times), 1024, 128).contiguous()
                capture["k"] = k[0, 11].detach().reshape(1, len(times), 1024, 128).contiguous()

        with torch.inference_mode():
            tower(noise, control, texts[""][0].to(device), torch.tensor([1000], device=device), store)
            cells = np.array([seed["cell"] for seed in seeds], dtype=np.int64)
            points = {"ids": np.arange(len(seeds)), "anchor": np.zeros(len(seeds), dtype=np.int64),
                      "cells": np.repeat(cells[:, None], len(times), axis=1)}
            xy = beam_track(capture["q"], capture["k"], points, 0, 32)
        np.savez_compressed(directory / "tracks.npz", xy=xy, times=times, anchor_cells=cells)
        del noise, control, saved, capture
        summary = render_case(case, args.output, ROOT / "runs/vace_tracking_all_suites_20261004/videos")
        summaries.append(summary)
        log(stage="case_complete", **summary)
    del tower, texts
    gc.collect()
    torch.cuda.empty_cache()
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    for suite in sorted({case["suite"] for case in cases}):
        selected = [case for case in cases if case["suite"] == suite]
        sheet = Image.new("RGB", (1200, 1048), "white")
        for index, case in enumerate(selected):
            path = sorted((args.output / case["name"]).glob("frame_*.jpg"))[-1]
            with Image.open(path) as panel:
                sheet.paste(panel, (0, index * 524))
        sheet.save(args.output / f"{suite}_both_views.jpg", quality=95)
    log(stage="complete", output=str(args.output.resolve()))


if __name__ == "__main__":
    main()
