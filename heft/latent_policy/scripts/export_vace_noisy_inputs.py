#!/usr/bin/env python3
"""Decode the exact seeded GT-assisted VACE inputs without diffusion denoising."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_vace_gt_denoising import prepare_latents
from evaluate_vace_pure_noise_future import (
    ROOT, UniPCMultistepScheduler, dump, labelled, load_vae, log, read_prefix, read_video, write_video,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/vace_gt_denoising_20261006")
    options = parser.parse_args()
    protocol = json.loads((options.run / "protocol.json").read_text())
    args = argparse.Namespace(**protocol)
    args.output = options.run
    args.data = Path(args.data)
    args.model = Path(args.model)
    args.text_model = Path(args.text_model)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    source = args.data / "videos" / f"episode_{args.episode:05d}" / f"{args.camera}.mp4"
    truth = read_video(source)[args.start:args.start + args.frames]
    prefix = read_prefix(source, args.start, args.observed)
    if len(truth) != args.frames:
        raise ValueError("Incomplete GT input timeline")
    started = time.monotonic()
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    control, clean, _ = prepare_latents(vae, prefix, truth, args, device)
    del control
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(clean.shape, dtype=torch.float32, device=device, generator=generator)
    videos = [("Recorded GT", truth)]
    summaries = []
    for requested_sigma in args.noise_levels:
        output = args.output / f"noise_{requested_sigma:.2f}".replace(".", "p")
        sampling = json.loads((output / "sampling.json").read_text())
        scheduler = UniPCMultistepScheduler.from_pretrained(
            args.text_model / "scheduler", local_files_only=True, flow_shift=args.flow_shift)
        if not scheduler.config.use_flow_sigmas:
            raise ValueError("Expected a flow-matching schedule")
        scheduler.set_timesteps(args.steps, device=device)
        begin = sampling["begin_index"]
        scheduler.set_begin_index(begin)
        sigma = float(scheduler.sigmas[begin])
        if sigma != sampling["actual_noise_sigma"]:
            raise ValueError("Saved noise coefficient differs from reconstructed schedule")
        noisy = scheduler.add_noise(clean, noise, scheduler.timesteps[begin:begin + 1])
        if not torch.equal(noisy, (1 - sigma) * clean + sigma * noise):
            raise RuntimeError("Flow corruption differs from the original initialization")
        # The transformer receives this bf16 tensor at its first denoising call.
        model_input = noisy.bfloat16()
        torch.save({"latents": model_input.cpu(), "seed": args.seed, "sigma": sigma,
                    "timestep": int(scheduler.timesteps[begin]), "denoising_steps": 0},
                   output / "noisy_input_latents.pt")
        decoded = vae.decode([model_input[0]])[0].permute(1, 2, 3, 0).float().cpu().numpy()
        if len(decoded) != args.frames or not np.isfinite(decoded).all():
            raise RuntimeError("Invalid noisy latent reconstruction")
        video = np.round(np.clip((decoded + 1) / 2, 0, 1) * 255).astype(np.uint8)
        full_path = output / "noisy_input_full.mp4"
        future_path = output / "noisy_input_future.mp4"
        write_video(full_path, video, args.fps)
        write_video(future_path, video[args.observed:], args.fps)
        for path, count in ((full_path, args.frames), (future_path, args.frames - args.observed)):
            encoded = read_video(path)
            if encoded.shape != (count, args.size, args.size, 3):
                raise RuntimeError(f"Unexpected encoded video dimensions for {path}")
        videos.append((f"Noisy input sigma={sigma:.2f}", video))
        summary = {"requested_noise_sigma": requested_sigma, "actual_noise_sigma": sigma,
                   "full_video": str(full_path), "future_video": str(future_path),
                   "display": "Direct VAE decode of seeded initial bf16 latent; no diffusion denoising",
                   "brightness": "Fixed [-1,1] to [0,255] mapping with clipping; no per-frame rescaling"}
        dump(output / "noisy_input_protocol.json", summary | {
            "latent_shape": list(model_input.shape), "seed": args.seed,
            "future_frames": args.frames - args.observed, "fps": args.fps,
            "initialization": sampling["initialization"], "denoising_steps": 0,
        })
        summaries.append(summary)
        log(stage="noisy_input_exported", **summary)
        del noisy, model_input, decoded
    columns = [args.observed, args.observed + 7, args.observed + 15, args.frames - 1]
    rows = [np.concatenate([labelled(video[t], f"{name}: +{(t-args.observed+1)/args.fps:.2f}s")
                            for t in columns], axis=1) for name, video in videos]
    cv2.imwrite(str(args.output / "noisy_input_keyframes.jpg"),
                cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR))
    comparison = np.stack([np.concatenate([labelled(video[t], name, width=256)
                                           for name, video in videos], axis=1)
                           for t in range(args.observed, args.frames)])
    write_video(args.output / "noisy_input_comparison.mp4", comparison, args.fps)
    dump(args.output / "noisy_input_completed.json", {
        "elapsed_seconds": time.monotonic() - started, "outputs": summaries, "denoising_steps": 0,
        "comparison": str(args.output / "noisy_input_comparison.mp4"),
    })
    log(stage="completed", elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
