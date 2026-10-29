#!/usr/bin/env python3
"""Compare sparse UniPC updates with the existing 13-update teacher readout.

Every schedule starts from the same noisy GT and ends at exactly the original
step-13 sigma/timestep. Metrics measure readout agreement, not GT accuracy.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_vace_denoise_readouts import describe, dump, fixed_paths, progress, summarize
from vace_latent_distill.config import FUTURE_ANCHORS, NEGATIVE_PROMPT, PipelineConfig, TeacherConfig
from vace_latent_distill.data import decode_indices
from vace_latent_distill.schedules import sparse_schedule
from vace_latent_distill.teacher import Capture, VaceTeacher, content_token_masks, multi_anchor_paths


@torch.no_grad()
def prepare(teacher, items):
    rgb = torch.stack([item["teacher_rgb"] for item in items])
    pixels = teacher._pixels_batch(rgb)
    clean = torch.stack(teacher.vae.encode(list(pixels)), 0).float()
    control, _ = teacher._inputs_batch(pixels, 33)
    del pixels
    generators = [torch.Generator(device=teacher.device).manual_seed(item["metadata"]["teacher_seed"])
                  for item in items]
    for generator in generators:
        torch.randn((clean.shape[1], 12, *clean.shape[-2:]), device=teacher.device, generator=generator)
    noise = torch.stack([torch.randn(clean.shape[1:], device=teacher.device, generator=generator)
                         for generator in generators])
    text = torch.stack([item["text"] for item in items]).to(teacher.device, dtype=torch.bfloat16)
    masks = torch.stack([item["text_mask"] for item in items]).to(teacher.device)
    return clean, noise, control, text, masks


@torch.no_grad()
def run_schedule(teacher, prepared, updates, label):
    clean, noise, control, text, masks = prepared
    scheduler, begin, schedule = sparse_schedule(teacher.scheduler, teacher.config, updates, teacher.device)
    latent = (1 - schedule[0]["sigma"]) * clean + schedule[0]["sigma"] * noise
    batch = len(clean)
    pair_control = torch.cat((control, control), 0)
    pair_text = torch.cat((text, teacher.negative.expand(batch, -1, -1)), 0)
    torch.cuda.synchronize(teacher.device)
    started = time.monotonic()
    for index in range(updates):
        timestep = scheduler.timesteps[begin + index]
        velocity = teacher._forward(torch.cat((latent, latent), 0), pair_control, pair_text, timestep)
        conditional, unconditional = velocity.chunk(2)
        guided = unconditional.float() + teacher.config.guidance * (conditional.float() - unconditional.float())
        latent = scheduler.step(guided, timestep, latent, return_dict=False)[0]
        del velocity, conditional, unconditional, guided
        progress(stage="denoise_update", mode=label, update=index + 1, updates=updates,
                 sigma=float(scheduler.sigmas[begin + index + 1]))
    torch.cuda.synchronize(teacher.device)
    denoise_seconds = time.monotonic() - started
    capture = Capture(teacher.model, "shared", clean.shape[2], teacher.config.image_size // 16, masks)
    teacher._forward(latent, control, text, scheduler.timesteps[begin + updates], capture)
    torch.cuda.synchronize(teacher.device)
    total_seconds = time.monotonic() - started
    if abs(float(scheduler.sigmas[scheduler.step_index]) - schedule[-1]["sigma"]) > 1e-7:
        raise RuntimeError("Sparse denoising missed the requested endpoint")
    progress(stage="endpoint_captured", mode=label, updates=updates,
             denoise_seconds=denoise_seconds, denoise_and_capture_seconds=total_seconds,
             sigma=schedule[-1]["sigma"], timestep=schedule[-1]["timestep"])
    return capture.values, schedule, {"denoise_seconds": denoise_seconds,
                                      "denoise_and_capture_seconds": total_seconds}


@torch.no_grad()
def run_direct(teacher, prepared, endpoint):
    clean, noise, control, text, masks = prepared
    capture = Capture(teacher.model, "shared", clean.shape[2], teacher.config.image_size // 16, masks)
    sigma = endpoint["sigma"]
    torch.cuda.synchronize(teacher.device)
    started = time.monotonic()
    teacher._forward((1 - sigma) * clean + sigma * noise, control, text,
                     torch.tensor(endpoint["timestep"], device=teacher.device), capture)
    torch.cuda.synchronize(teacher.device)
    return capture.values, {"denoise_seconds": 0.0,
                            "denoise_and_capture_seconds": time.monotonic() - started}


def plot_summary(summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = sorted(summary, key=lambda row: row["step"])
    measures = [("k22/spatial_centered_cosine", "Partition L22H9: centered cosine"),
                ("k15/spatial_centered_cosine", "Partition L15H7: centered cosine"),
                ("semantic/content_heat_correlation", "Prompt heatmap correlation"),
                ("depth/spearman", "Relative depth: rank correlation"),
                ("qk_relation/changed_pixels/argmax_delta_px", "QK match disagreement: changed pixels (px)"),
                ("beam_paths/delta_reference_moving_px", "Beam disagreement: reference-moving points (px)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    labels = ["Direct" if row["step"] == 0 else str(row["step"]) for row in series]
    for ax, (key, title) in zip(axes.flat, measures):
        ax.plot(labels, [row["metrics"][key] for row in series], marker="o")
        ax.set(title=title, xlabel="Denoising updates; identical endpoint sigma")
        ax.grid(alpha=0.25)
    fig.suptitle("Agreement with the 13-update teacher; these are not GT errors")
    fig.savefig(output / "sparse_denoise_readouts.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vace_structured.json")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    config = PipelineConfig.load(args.config)
    config.teacher = dataclasses.replace(
        config.teacher, denoise_updates=config.teacher.denoise_reference_updates
    )
    source_protocol = json.loads((args.reference_run / "protocol.json").read_text())
    reference_teacher = TeacherConfig(**source_protocol["config"]["teacher"])
    if dataclasses.asdict(config.teacher) != dataclasses.asdict(reference_teacher):
        raise ValueError("Teacher settings must match the previous comparison")
    if config.teacher.denoise_updates != 13 or args.batch_size < 1:
        raise ValueError("This comparison requires the existing 13-update reference and positive batch size")
    counts = list(dict.fromkeys(args.updates))
    if any(not 1 <= count < 13 for count in counts):
        raise ValueError("Sparse schedules must use 1..12 updates")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty output directory to preserve earlier results")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    items = []
    for metadata in source_protocol["samples"]:
        start = metadata["episode_start_frame"] + metadata["start"]
        rgb = torch.from_numpy(decode_indices(metadata["video"], np.arange(start, start + 49), metadata["fps"]))
        text = text_cache[metadata["task"]].float()
        items.append({"teacher_rgb": rgb, "text": text, "text_mask": text.abs().sum(-1) != 0,
                      "metadata": metadata})
    content = content_token_masks(config.teacher.text_root, [item["metadata"]["task"] for item in items])
    protocol = {"config": config.to_dict(), "samples": source_protocol["samples"],
                "sparse_updates": counts, "reference_updates": 13,
                "reference_run": str(args.reference_run.resolve()), "cfg_parallel": True,
                "schedule": "Subsample original UniPC nodes; preserve exact endpoints, reset history per run, retain suffix.",
                "noise_rng": source_protocol["noise_rng"], "scope": source_protocol["scope"],
                "reference": source_protocol["reference"], "changed_pixels": source_protocol["changed_pixels"],
                "fixed_seeds": source_protocol["fixed_seeds"],
                "timing": "GPU synchronized; excludes shared VAE/control preparation and CPU beam decoding."}
    dump(args.output / "protocol.json", protocol)
    progress(stage="load_teacher", samples=len(items), updates=counts, physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
    teacher = VaceTeacher(config.teacher, "shared", device, text_cache[NEGATIVE_PROMPT])
    started = time.monotonic()
    records, timings = [], []
    try:
        for begin in range(0, len(items), args.batch_size):
            batch = items[begin:begin + args.batch_size]
            prepared = prepare(teacher, batch)
            reference_values, reference_schedule, timing = run_schedule(teacher, prepared, 13, "reference")
            timings.append({"batch": begin, "updates": 13, "samples": len(batch), **timing})
            candidates = [("sparse_vs_step13", 13, reference_values)]
            schedules = {"13": reference_schedule}
            for count in counts:
                values, schedule, timing = run_schedule(teacher, prepared, count, f"sparse_{count}")
                if schedule[0] != reference_schedule[0] or any(
                    schedule[-1][key] != reference_schedule[-1][key] for key in ("sigma", "timestep", "reference_step")
                ):
                    raise RuntimeError("The compared schedules have different endpoints")
                candidates.append(("sparse_vs_step13", count, values))
                schedules[str(count)] = schedule
                timings.append({"batch": begin, "updates": count, "samples": len(batch), **timing})
            direct_values, timing = run_direct(teacher, prepared, reference_schedule[-1])
            candidates.append(("direct_vs_step13", 0, direct_values))
            timings.append({"batch": begin, "updates": 0, "samples": len(batch), **timing})
            dump(args.output / f"schedule_batch{begin:02d}.json", schedules)
            del prepared
            for index, item in enumerate(batch):
                raw_rgb = item["teacher_rgb"]
                grid = config.teacher.image_size // 16
                reduced = F.adaptive_avg_pool2d(raw_rgb.float(), (grid, grid))
                energy = torch.stack([(reduced[t * 4] - reduced[(t - 1) * 4]).abs().mean(0).flatten()
                                      for t in range(9, 13)])
                reference = {name: value[index].float() for name, value in reference_values.items()}
                content_mask = content[item["metadata"]["task"]]
                seeds = multi_anchor_paths(reference, FUTURE_ANCHORS, config.teacher, content_mask)["seeds"].numpy()
                reference_paths = fixed_paths(reference, seeds, config.teacher)
                cell_px = raw_rgb.shape[-1] / grid
                for mode, count, values in candidates:
                    current = {name: value[index].float() for name, value in values.items()}
                    metrics = describe(current, reference, energy, content_mask, seeds, reference_paths,
                                       config.teacher, config.train.relation_temperature, cell_px)
                    records.append({"sample": item["metadata"]["id"], "camera": item["metadata"]["camera"],
                                    "mode": mode, "step": count, "sigma": reference_schedule[-1]["sigma"],
                                    "metrics": metrics})
                dump(args.output / "rows.json", records)
                progress(stage="sample_compared", sample=item["metadata"]["id"], completed=begin + index + 1, total=len(items))
            dump(args.output / "timings.json", timings)
            del candidates, reference_values, values, direct_values
    finally:
        teacher.close()
    summary = summarize(records)
    dump(args.output / "summary.json", summary)
    dump(args.output / "summary_by_camera.json", {
        camera: summarize([row for row in records if row["camera"] == camera])
        for camera in sorted({row["camera"] for row in records})
    })
    plot_summary(summary, args.output)
    completed = {"samples": len(items), "elapsed_seconds": time.monotonic() - started,
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    dump(args.output / "completed.json", completed)
    progress(stage="completed", output=str(args.output), **completed)


if __name__ == "__main__":
    main()
