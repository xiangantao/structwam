#!/usr/bin/env python3
"""Time resident online teacher extraction with fixed sparse-denoise endpoints."""

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_vace_pure_noise_future import read_video
from vace_latent_distill.config import NEGATIVE_PROMPT, PipelineConfig
from vace_latent_distill.teacher import VaceTeacher, content_token_masks


def progress(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vace_structured.json")
    parser.add_argument("--source", type=Path,
                        default=WORKSPACE / "runs/vace_gt_denoising_future16_sigma080_20261006/noise_0p80")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.repeats < 1:
        parser.error("Batch size and repeats must be positive")
    if args.output.exists():
        parser.error("Use a new output directory")
    if os.environ.get("VACE_CFG_PARALLEL", "1") != "1":
        parser.error("This benchmark requires parallel CFG")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    config = PipelineConfig.load(args.config)
    reference_config = dataclasses.replace(config.teacher)
    if reference_config.denoise_reference_updates != 13:
        raise ValueError("The reference endpoint is the existing 13-update schedule")
    source = json.loads((args.source / "protocol.json").read_text())
    rgb = []
    for camera in ("agentview", "robot0_eye_in_hand"):
        path = Path(source["data"]) / "videos" / f"episode_{source['episode']:05d}" / f"{camera}.mp4"
        frames = read_video(path)[source["start"]:source["start"] + 49]
        if len(frames) != 49:
            raise ValueError("Incomplete benchmark window")
        rgb.append(torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).contiguous())
    rgbs = torch.stack([rgb[index % 2] for index in range(args.batch_size)])
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    text = [text_cache[source["task"]]] * args.batch_size
    masks = [text[0].abs().sum(-1) != 0] * args.batch_size
    content = content_token_masks(config.teacher.text_root, [source["task"]])
    contents = [content[source["task"]]] * args.batch_size
    seeds = [42 + index for index in range(args.batch_size)]
    device = torch.device("cuda:0")
    progress(stage="load_teacher", batch_size=args.batch_size, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    teacher = VaceTeacher(config.teacher, "shared", device, text_cache[NEGATIVE_PROMPT])
    original_forward, original_encode, original_readout = teacher._forward, teacher.vae.encode, teacher._readout
    record = {}

    def timed_forward(latent, control, embedding, timestep, capture=None):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        if capture is None and "denoise_started" not in record:
            record["denoise_started"] = started
        if capture is not None and latent.shape[2] == 13:
            record["denoise_seconds"] = started - record.pop("denoise_started")
        result = original_forward(latent, control, embedding, timestep, capture)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        phase = "denoise_forwards" if capture is None else (
            "observed_capture" if latent.shape[2] == 12 else "future_capture"
        )
        record.setdefault(phase, []).append(elapsed)
        if capture is not None or len(record[phase]) % 4 == 0:
            progress(stage=phase, updates=record["updates"], warmup=record["warmup"],
                     completed=len(record[phase]), seconds=elapsed)
        return result

    def timed_encode(*values, **kwargs):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = original_encode(*values, **kwargs)
        torch.cuda.synchronize(device)
        record.setdefault("vae_encodes", []).append(time.perf_counter() - started)
        return result

    def timed_readout(*values, **kwargs):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = original_readout(*values, **kwargs)
        torch.cuda.synchronize(device)
        record.setdefault("cpu_readouts", []).append(time.perf_counter() - started)
        return result

    teacher._forward, teacher.vae.encode, teacher._readout = timed_forward, timed_encode, timed_readout
    rows = []
    try:
        # Warm all VAE, capture, and CPU decoder paths before recording timings.
        runs = [(1, True)] + [(updates, False) for _ in range(args.repeats) for updates in (1, 13)]
        for updates, warmup in runs:
            teacher.config = dataclasses.replace(reference_config, denoise_updates=updates)
            record = {"updates": updates, "warmup": warmup}
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            target = teacher.extract_batch(rgbs, text, masks, seeds, contents)
            torch.cuda.synchronize(device)
            total = time.perf_counter() - started
            nodes = teacher.protocol["denoise_schedule"]
            if abs(nodes[-1]["sigma"] - 0.4996665418148041) > 1e-7:
                raise RuntimeError("Benchmark changed the readout endpoint")
            if len(target) != args.batch_size:
                raise RuntimeError("Teacher returned an incomplete target batch")
            row = {"updates": updates, "warmup": warmup, "batch_size": args.batch_size,
                   "initial_sigma": nodes[0]["sigma"], "endpoint_sigma": nodes[-1]["sigma"],
                   "denoise_seconds": record["denoise_seconds"],
                   "denoise_forward_seconds": sum(record["denoise_forwards"]),
                   "future_capture_seconds": sum(record["future_capture"]),
                   "observed_capture_seconds": sum(record["observed_capture"]),
                   "vae_seconds": sum(record["vae_encodes"]),
                   "cpu_readout_seconds": sum(record["cpu_readouts"]),
                   "full_teacher_seconds": total,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}
            row["future_denoise_and_capture_seconds"] = row["denoise_seconds"] + row["future_capture_seconds"]
            rows.append(row)
            (args.output / "timings.json").write_text(json.dumps(rows, indent=2) + "\n")
            progress(stage="run_completed", **row)
            del target
    finally:
        teacher.close()
    metrics = ("denoise_seconds", "future_denoise_and_capture_seconds", "observed_capture_seconds",
               "future_capture_seconds", "vae_seconds", "cpu_readout_seconds", "full_teacher_seconds")
    summary = {str(updates): {metric: statistics.median([row[metric] for row in rows
                                                       if row["updates"] == updates and not row["warmup"]])
                              for metric in metrics} for updates in (1, 13)}
    speedups = {metric: summary["13"][metric] / summary["1"][metric]
                for metric in ("denoise_seconds", "future_denoise_and_capture_seconds", "full_teacher_seconds")}
    result = {"gpu": torch.cuda.get_device_name(device), "batch_size": args.batch_size,
              "cfg_batch_size": args.batch_size * 2, "task": "shared", "repeats": args.repeats,
              "warmup_excluded": True, "same_endpoint": True,
              "input": "33 observed +16 future, 512px teacher, all five heads, production extract_batch and CPU beam decoder",
              "scope": "Resident teacher only; excludes weights/text/video loading, preview rendering, student and distributed training",
              "timing": "CUDA synchronized at phase boundaries on shared development GPU",
              "summary_seconds": summary, "speedups_13_to_1": speedups,
              "production_configuration_modified": False}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    progress(stage="completed", **result)


if __name__ == "__main__":
    main()
