#!/usr/bin/env python3
"""Compare privileged GT-assisted VACE denoising with causal pure-noise generation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from evaluate_vace_pure_noise_future import (
    ROOT, UniPCMultistepScheduler, choose_window, dump, labelled,
    load_transformer, load_vae, log, read_prefix, read_video, text_conditions, write_video,
)


@torch.no_grad()
def prepare_latents(vae, prefix, truth, args, device):
    def pixels(video):
        x = torch.from_numpy(video.copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
        return F.interpolate(x.permute(1, 0, 2, 3), size=(args.size, args.size),
                             mode="bilinear", align_corners=False).permute(1, 0, 2, 3)

    control_pixels = torch.zeros((3, args.frames, args.size, args.size), device=device)
    control_pixels[:, :args.observed] = pixels(prefix)
    assert not control_pixels[:, args.observed:].count_nonzero()
    inactive = vae.encode([control_pixels])[0]
    reactive = vae.encode([torch.zeros_like(control_pixels)])[0]
    expected_shape = (16, (args.frames - 1) // 4 + 1, args.size // 8, args.size // 8)
    if inactive.shape != expected_shape:
        raise RuntimeError("Native VAE latent grid does not match the output timeline")
    mask = torch.ones((args.frames, args.size, args.size), device=device)
    mask[:args.observed] = 0
    mask = mask.view(args.frames, args.size // 8, 8, args.size // 8, 8)
    mask = mask.permute(2, 4, 0, 1, 3).flatten(0, 1)
    mask = F.interpolate(mask[None], size=inactive.shape[1:], mode="nearest-exact")[0]
    control = torch.cat((inactive, reactive, mask), dim=0)[None].bfloat16()
    clean = vae.encode([pixels(truth)])[0][None].float()
    if clean.shape != (1, *expected_shape):
        raise RuntimeError("GT latent grid does not match generation grid")
    reconstruction = vae.decode([clean[0]])[0].permute(1, 2, 3, 0).cpu().numpy()
    reconstruction = np.round(np.clip((reconstruction + 1) / 2, 0, 1) * 255).astype(np.uint8)
    return control, clean, reconstruction


@torch.no_grad()
def denoise(model, vae, control, clean, noise, positive, negative, args, sigma, output, started):
    scheduler = UniPCMultistepScheduler.from_pretrained(
        args.text_model / "scheduler", local_files_only=True, flow_shift=args.flow_shift)
    if not scheduler.config.use_flow_sigmas:
        raise ValueError("GT initialization requires a flow-matching scheduler")
    scheduler.set_timesteps(args.steps, device=positive.device)
    # Enter the existing schedule at the actual sigma used to corrupt GT.
    begin = int((scheduler.sigmas[:-1] - sigma).abs().argmin())
    scheduler.set_begin_index(begin)
    timesteps = scheduler.timesteps[begin:]
    latents = scheduler.add_noise(clean, noise, timesteps[:1])
    sampling = {"requested_noise_sigma": sigma, "actual_noise_sigma": float(scheduler.sigmas[begin]),
                "begin_index": begin, "denoise_steps": len(timesteps), "schedule_steps": args.steps,
                "initialization": "(1-sigma)*VAE(full GT video) + sigma*Gaussian noise",
                "uses_future_gt": True}
    dump(output / "sampling.json", sampling)
    log(stage="gt_denoising_ready", **sampling)
    scales = torch.ones(len(model.vace_blocks), device=positive.device, dtype=torch.bfloat16)
    for index, timestep in enumerate(timesteps):
        inputs = {"hidden_states": latents.bfloat16(), "timestep": timestep.reshape(1),
                  "control_hidden_states": control, "control_hidden_states_scale": scales,
                  "return_dict": False}
        conditional = model(**inputs, encoder_hidden_states=positive)[0]
        unconditional = model(**inputs, encoder_hidden_states=negative)[0]
        update = unconditional.float() + args.guidance * (conditional.float() - unconditional.float())
        latents = scheduler.step(update, timestep, latents, return_dict=False)[0]
        if not torch.isfinite(latents).all():
            raise RuntimeError("Nonfinite generated latent")
        status = {"requested_noise_sigma": sigma, "step": index + 1, "steps": len(timesteps),
                  "timestep": float(timestep), "elapsed_seconds": time.monotonic() - started,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "pid": os.getpid(), "state": "generating"}
        dump(args.output / "status.json", status)
        log(stage="gt_denoise", **status)
    torch.save({"latents": latents.detach().cpu(), "seed": args.seed, **sampling},
               output / "generated_latents.pt")
    decoded = vae.decode([latents[0]])[0].permute(1, 2, 3, 0).cpu().numpy()
    if len(decoded) != args.frames or not np.isfinite(decoded).all():
        raise RuntimeError("Invalid GT-assisted output")
    prediction = np.round(np.clip((decoded + 1) / 2, 0, 1) * 255).astype(np.uint8)
    write_video(output / "generated_full.mp4", prediction, args.fps)
    write_video(output / "generated_future.mp4", prediction[args.observed:], args.fps)
    return prediction, sampling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--pure-noise-run", type=Path, default=ROOT / "runs/vace_pure_noise_future_20261006")
    parser.add_argument("--camera", choices=["agentview", "robot0_eye_in_hand"], default="agentview")
    parser.add_argument("--episode", type=int, default=39)
    parser.add_argument("--start", type=int, default=48)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--observed", type=int, default=33)
    parser.add_argument("--frames", type=int, default=65)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=5)
    parser.add_argument("--flow-shift", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    args = parser.parse_args()
    if (args.frames - 1) % 4 or (args.observed - 1) % 4 or not 1 <= args.observed < args.frames:
        parser.error("Total and observed frame counts must be 4n+1, with future frames")
    if not args.noise_levels or any(not 0 < sigma < 1 for sigma in args.noise_levels):
        parser.error("Noise coefficients must lie strictly between 0 and 1")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    case = choose_window(args)
    args.fps = case["fps"]
    source = args.data / "videos" / f"episode_{case['episode']:05d}" / f"{args.camera}.mp4"
    prefix = read_prefix(source, case["start"], args.observed)
    truth = read_video(source)[case["start"]:case["start"] + args.frames]
    if len(truth) != args.frames:
        raise ValueError("Incomplete reference video")
    protocol = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()} | case | {
        "experiment": "Privileged future-GT teacher diagnostic, not causal prediction",
        "control_input": "Observed prefix only; future control pixels zero; observed mask0 / future mask1",
        "main_input": "Full GT video latent mixed with noise at the matching flow timestep",
        "future_gt_used": True, "lora_loaded": False,
        "distillation": "No student trained here. A later student must receive past observations only.",
    }
    pure_protocol = json.loads((args.pure_noise_run / "protocol.json").read_text())
    match_keys = ["episode", "start", "camera", "observed", "frames", "seed", "size",
                  "steps", "guidance", "flow_shift", "model", "text_model", "data"]
    if any(pure_protocol[key] != protocol[key] for key in match_keys):
        raise ValueError("Pure-noise comparison has a different experiment protocol")
    pure = read_video(args.pure_noise_run / "generated_full.mp4")
    if len(pure) != args.frames:
        raise ValueError("Pure-noise comparison has a different timeline")
    dump(args.output / "protocol.json", protocol)
    log(stage="selected_gt_teacher", **case)
    positive, negative, _ = text_conditions(args.text_model, case["task"], device)
    model = load_transformer(args.model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    control, clean, reconstruction = prepare_latents(vae, prefix, truth, args, device)
    write_video(args.output / "vae_reconstruction.mp4", reconstruction, args.fps)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(clean.shape, dtype=torch.float32, device=device, generator=generator)
    videos = [("Recorded GT", truth)]
    summaries = []
    for sigma in args.noise_levels:
        output = args.output / f"noise_{sigma:.2f}".replace(".", "p")
        output.mkdir(parents=True, exist_ok=True)
        dump(output / "protocol.json", protocol | {"requested_noise_sigma": sigma})
        prediction, sampling = denoise(model, vae, control, clean, noise, positive, negative,
                                       args, sigma, output, started)
        videos.append((f"GT + noise sigma={sampling['actual_noise_sigma']:.2f}", prediction))
        summaries.append(sampling | {"output": str(output)})
    videos.append(("Pure noise / no future GT", pure))
    comparison = [np.concatenate([labelled(video[t], name, width=256)
                                  for name, video in videos], axis=1)
                  for t in range(args.observed, args.frames)]
    write_video(args.output / "future_comparison.mp4", np.stack(comparison), args.fps)
    columns = [args.observed - 1, args.observed + 7, args.observed + 15, args.frames - 1]
    rows = []
    for name, video in [videos[0], ("VAE roundtrip / no denoising", reconstruction), *videos[1:]]:
        rows.append(np.concatenate([labelled(video[t], f"{name}: +{(t-args.observed+1)/args.fps:.2f}s")
                                    for t in columns], axis=1))
    cv2.imwrite(str(args.output / "keyframes.jpg"), cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR))
    completed = {"elapsed_seconds": time.monotonic() - started, "sampling": summaries,
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "comparison": str(args.output / "future_comparison.mp4"), "future_gt_used": True}
    dump(args.output / "completed.json", completed)
    dump(args.output / "status.json", {"state": "completed", "pid": os.getpid()} | completed)
    log(stage="completed", **completed)


if __name__ == "__main__":
    main()
