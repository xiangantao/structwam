#!/usr/bin/env python3
"""Export every clean estimate and noisy state of an existing VACE diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_vace_gt_denoising import prepare_latents
from evaluate_vace_pure_noise_future import (
    ROOT, UniPCMultistepScheduler, dump, labelled, load_transformer, load_vae,
    log, read_prefix, read_video, text_conditions, write_video,
)


@torch.no_grad()
def decode_future(vae, latent, args):
    decoded = vae.decode([latent[0]])[0].permute(1, 2, 3, 0).float().cpu().numpy()
    if len(decoded) != args.frames or not np.isfinite(decoded).all():
        raise RuntimeError("Invalid intermediate VAE decode")
    return np.round(np.clip((decoded[args.observed:] + 1) / 2, 0, 1) * 255).astype(np.uint8)


def video_grid(videos, labels, playback_fps, path, repeats=1):
    if not 1 <= len(videos) <= 6 or len(labels) != len(videos):
        raise ValueError("A grid needs one to six labelled clips")
    count = len(videos[0])
    blank = np.zeros_like(videos[0])
    videos = [*videos, *([blank] * (6 - len(videos)))]
    labels = [*labels, *([""] * (6 - len(labels)))]
    frames = []
    for t in range(count):
        tiles = [labelled(video[t], label, width=384) for video, label in zip(videos, labels, strict=True)]
        frames.append(np.concatenate((np.concatenate(tiles[:3], axis=1),
                                      np.concatenate(tiles[3:], axis=1)), axis=0))
    frames = np.concatenate([np.stack(frames)] * repeats, axis=0)
    write_video(path, frames, playback_fps)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/vace_gt_denoising_future16_sigma080_20261006")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise", type=float, default=0.8)
    options = parser.parse_args()
    source_protocol = json.loads((options.run / "protocol.json").read_text())
    args = argparse.Namespace(**source_protocol)
    args.output = options.output
    args.data, args.model, args.text_model = Path(args.data), Path(args.model), Path(args.text_model)
    reference_output = options.run / f"noise_{options.noise:.2f}".replace(".", "p")
    source_sampling = json.loads((reference_output / "sampling.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.data / "videos" / f"episode_{args.episode:05d}" / f"{args.camera}.mp4"
    prefix = read_prefix(source, args.start, args.observed)
    truth = read_video(source)[args.start:args.start + args.frames]
    if len(truth) != args.frames:
        raise ValueError("Incomplete source timeline")
    truth_future = truth[args.observed:]
    write_video(args.output / "recorded_future.mp4", truth_future, args.fps)
    protocol = source_protocol | {
        "output": str(args.output), "source_run": str(options.run),
        "export": "Every actual denoising step, both model clean estimate and noisy state after update",
        "clean_estimate": "Scheduler x0 estimate at the step input sigma; no additional denoising",
        "noisy_state": "Sampler state after the step, at the next sigma; direct VAE decode",
        "grid_playback": "10 FPS and repeated3 times; each source clip remains16 frames at20 FPS",
        "brightness": "Fixed [-1,1] to [0,255] with clipping, no per-frame normalization",
    }
    dump(args.output / "protocol.json", protocol)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    positive, negative, _ = text_conditions(args.text_model, args.task, device)
    model = load_transformer(args.model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    control, clean, _ = prepare_latents(vae, prefix, truth, args, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(args.noise_source_shape, dtype=torch.float32, device=device, generator=generator)
    noise = noise[:, :, :clean.shape[2]].contiguous()
    scheduler = UniPCMultistepScheduler.from_pretrained(
        args.text_model / "scheduler", local_files_only=True, flow_shift=args.flow_shift)
    if not scheduler.config.use_flow_sigmas or scheduler.config.prediction_type != "flow_prediction" or not scheduler.predict_x0:
        raise ValueError("Clean estimates require the original flow-prediction, predict-x0 scheduler")
    scheduler.set_timesteps(args.steps, device=device)
    begin = source_sampling["begin_index"]
    scheduler.set_begin_index(begin)
    timesteps = scheduler.timesteps[begin:]
    latents = scheduler.add_noise(clean, noise, timesteps[:1])
    reference_initial = torch.load(reference_output / "noisy_input_latents.pt", map_location="cpu",
                                   weights_only=True)["latents"]
    if not torch.equal(latents.bfloat16().cpu(), reference_initial):
        raise RuntimeError("Initial model input does not reproduce the existing short-horizon run")
    scales = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
    estimates, states, records = [], [], []
    for index, timestep in enumerate(timesteps):
        step_number = index + 1
        input_latents = latents.bfloat16()
        inputs = {"hidden_states": input_latents, "timestep": timestep.reshape(1),
                  "control_hidden_states": control, "control_hidden_states_scale": scales,
                  "return_dict": False}
        conditional = model(**inputs, encoder_hidden_states=positive)[0]
        unconditional = model(**inputs, encoder_hidden_states=negative)[0]
        velocity = unconditional.float() + args.guidance * (conditional.float() - unconditional.float())
        input_sigma = float(scheduler.sigmas[begin + index])
        latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0]
        estimate = scheduler.model_outputs[-1]
        if not torch.isfinite(latents).all() or not torch.isfinite(estimate).all():
            raise RuntimeError("Nonfinite denoising state")
        remaining_sigma = float(scheduler.sigmas[begin + index + 1])
        estimate_video = decode_future(vae, estimate, args)
        state_video = decode_future(vae, latents, args)
        directory = args.output / f"step_{step_number:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        estimate_path = directory / "clean_estimate_future.mp4"
        state_path = directory / "noisy_state_future.mp4"
        write_video(estimate_path, estimate_video, args.fps)
        write_video(state_path, state_video, args.fps)
        comparison = np.stack([np.concatenate((labelled(truth_future[t], "Recorded GT"),
                                               labelled(estimate_video[t], f"Step{step_number}: x0 estimate"),
                                               labelled(state_video[t], f"After step{step_number}: sigma={remaining_sigma:.3f}")), axis=1)
                               for t in range(len(truth_future))])
        write_video(directory / "comparison.mp4", comparison, args.fps)
        record = {"step": step_number, "schedule_position": begin + index + 1,
                  "timestep": int(timestep), "input_sigma": input_sigma,
                  "remaining_sigma_after_step": remaining_sigma,
                  "clean_estimate_video": str(estimate_path), "noisy_state_video": str(state_path)}
        dump(directory / "metadata.json", record)
        torch.save({"model_input": input_latents.cpu(), "x0_estimate": estimate.cpu(), **record},
                   directory / "latents.pt")
        estimates.append(estimate_video)
        states.append(state_video)
        records.append(record)
        status = record | {"steps": len(timesteps), "pid": os.getpid(), "state": "exporting",
                           "elapsed_seconds": time.monotonic() - started,
                           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        dump(args.output / "status.json", status)
        log(stage="step_exported", **status)
        del conditional, unconditional, velocity, input_latents, inputs, comparison
    final_video = decode_future(vae, latents, args)
    write_video(args.output / "final_future.mp4", final_video, args.fps)
    reference_final = torch.load(reference_output / "generated_latents.pt", map_location="cpu",
                                 weights_only=True)["latents"]
    delta = latents.cpu() - reference_final
    matching = {"final_latent_exact_match": bool(torch.equal(latents.cpu(), reference_final)),
                "final_latent_max_difference": float(delta.abs().max()),
                "final_latent_rms_difference": float(delta.square().mean().sqrt())}
    pages = []
    for start in range(0, len(records), 6):
        end = min(start + 6, len(records))
        clean_page = args.output / f"clean_steps_{start + 1:02d}_{end:02d}.mp4"
        noisy_page = args.output / f"noisy_steps_{start + 1:02d}_{end:02d}.mp4"
        clean_labels = [f"Step{row['step']}: x0 @sigma={row['input_sigma']:.3f}" for row in records[start:end]]
        noisy_labels = [f"After{row['step']}: sigma={row['remaining_sigma_after_step']:.3f}" for row in records[start:end]]
        video_grid(estimates[start:end], clean_labels, max(1, int(args.fps / 2)), clean_page, repeats=3)
        video_grid(states[start:end], noisy_labels, max(1, int(args.fps / 2)), noisy_page, repeats=3)
        pages.append({"steps": list(range(start + 1, end + 1)), "clean_estimates": str(clean_page),
                      "noisy_states": str(noisy_page)})
    endpoint_tiles = [labelled(truth_future[-1], "Recorded GT: future16", width=256)]
    endpoint_tiles.extend(labelled(video[-1], f"Step{row['step']}: sigma={row['input_sigma']:.3f}", width=256)
                          for row, video in zip(records, estimates, strict=True))
    while len(endpoint_tiles) % 6:
        endpoint_tiles.append(np.zeros_like(endpoint_tiles[0]))
    overview = np.concatenate([np.concatenate(endpoint_tiles[i:i + 6], axis=1)
                               for i in range(0, len(endpoint_tiles), 6)])
    cv2.imwrite(str(args.output / "all_steps_future_endpoint.jpg"), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
    completed = {"steps": len(records), "frames_per_video": len(truth_future), "fps": args.fps,
                 "elapsed_seconds": time.monotonic() - started, "step_records": records,
                 "pages": pages, **matching,
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    dump(args.output / "completed.json", completed)
    dump(args.output / "status.json", {"state": "completed", "pid": os.getpid()} | completed)
    log(stage="completed", steps=len(records), **matching, elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
