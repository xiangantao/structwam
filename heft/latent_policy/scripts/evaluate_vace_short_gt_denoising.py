#!/usr/bin/env python3
"""GT-assisted short continuation with noise matched to a longer diagnostic."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_vace_gt_denoising import denoise, prepare_latents
from evaluate_vace_pure_noise_future import (
    ROOT, UniPCMultistepScheduler, choose_window, dump, labelled, load_transformer, load_vae,
    log, read_prefix, read_video, text_conditions, write_video,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, default=ROOT / "runs/vace_gt_denoising_20261006")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--future-frames", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.8)
    options = parser.parse_args()
    reference = json.loads((options.reference_run / "protocol.json").read_text())
    args = argparse.Namespace(**reference)
    args.output = options.output
    args.data, args.model, args.text_model = Path(args.data), Path(args.model), Path(args.text_model)
    args.frames = args.observed + options.future_frames
    args.noise_levels = [options.noise]
    if options.future_frames <= 0 or options.future_frames % 4 or args.frames >= reference["frames"]:
        parser.error("Use a positive future length divisible by4 and shorter than the reference")
    if not 0 < options.noise < 1:
        parser.error("Noise coefficient must be in (0,1)")
    reference_output = options.reference_run / f"noise_{options.noise:.2f}".replace(".", "p")
    reference_sampling = json.loads((reference_output / "sampling.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"noise_{options.noise:.2f}".replace(".", "p")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    case = choose_window(args)
    source = args.data / "videos" / f"episode_{case['episode']:05d}" / f"{args.camera}.mp4"
    prefix = read_prefix(source, case["start"], args.observed)
    truth = read_video(source)[case["start"]:case["start"] + args.frames]
    longer = read_video(reference_output / "generated_full.mp4")
    if len(truth) != args.frames or len(longer) != reference["frames"]:
        raise ValueError("Incomplete comparison timeline")
    protocol = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()} | case | {
        "experiment": "Short-horizon GT-assisted teacher diagnostic; not causal prediction",
        "reference_run": str(options.reference_run), "reference_total_frames": reference["frames"],
        "future_frames": options.future_frames, "future_gt_used": True, "lora_loaded": False,
        "control_input": "Observed prefix only; future zero; observed mask0 / future mask1",
        "main_input": "Full short GT latent mixed with Gaussian noise; includes future GT",
        "noise_matching": "Generate reference-shaped Gaussian noise with the reference seed, then crop time",
        "noise_source_shape": [1, 16, (reference["frames"] - 1) // 4 + 1, args.size // 8, args.size // 8],
        "known_frame_indices": list(range(case["start"], case["start"] + args.observed)),
        "predicted_frame_indices": list(range(case["start"] + args.observed, case["start"] + args.frames)),
    }
    dump(args.output / "protocol.json", protocol)
    dump(output / "protocol.json", protocol)
    write_video(args.output / "observed_prefix.mp4", prefix, args.fps)
    log(stage="selected_short_teacher", **case, observed=args.observed, future=options.future_frames)
    positive, negative, _ = text_conditions(args.text_model, case["task"], device)
    model = load_transformer(args.model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    control, clean, reconstruction = prepare_latents(vae, prefix, truth, args, device)
    write_video(args.output / "vae_reconstruction.mp4", reconstruction, args.fps)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(protocol["noise_source_shape"], dtype=torch.float32, device=device, generator=generator)
    noise = noise[:, :, :clean.shape[2]].contiguous()
    scheduler = UniPCMultistepScheduler.from_pretrained(
        args.text_model / "scheduler", local_files_only=True, flow_shift=args.flow_shift)
    scheduler.set_timesteps(args.steps, device=device)
    begin = int((scheduler.sigmas[:-1] - options.noise).abs().argmin())
    scheduler.set_begin_index(begin)
    actual_sigma = float(scheduler.sigmas[begin])
    if actual_sigma != reference_sampling["actual_noise_sigma"]:
        raise ValueError("Short and long runs use different actual noise coefficients")
    initial = scheduler.add_noise(clean, noise, scheduler.timesteps[begin:begin + 1]).bfloat16()
    reference_initial = torch.load(reference_output / "noisy_input_latents.pt", map_location="cpu",
                                   weights_only=True)["latents"][:, :, :clean.shape[2]]
    difference = initial.cpu().float() - reference_initial.float()
    matching = {"initial_bf16_exact_match": bool(torch.equal(initial.cpu(), reference_initial)),
                "initial_latent_max_difference": float(difference.abs().max()),
                "initial_latent_rms_difference": float(difference.square().mean().sqrt())}
    log(stage="matched_noise", **matching)
    torch.save({"latents": initial.cpu(), "seed": args.seed, "sigma": actual_sigma,
                "noise_source_shape": protocol["noise_source_shape"], "denoising_steps": 0},
               output / "noisy_input_latents.pt")
    decoded = vae.decode([initial[0]])[0].permute(1, 2, 3, 0).float().cpu().numpy()
    if len(decoded) != args.frames or not np.isfinite(decoded).all():
        raise RuntimeError("Invalid noisy input decode")
    noisy_video = np.round(np.clip((decoded + 1) / 2, 0, 1) * 255).astype(np.uint8)
    write_video(output / "noisy_input_full.mp4", noisy_video, args.fps)
    write_video(output / "noisy_input_future.mp4", noisy_video[args.observed:], args.fps)
    del initial, reference_initial, decoded, difference
    prediction, sampling = denoise(model, vae, control, clean, noise, positive, negative,
                                   args, options.noise, output, started)
    panels = [("Recorded GT", truth), ("16-frame run", prediction), ("32-frame run / first16", longer)]
    comparison = np.stack([np.concatenate([labelled(video[t], name) for name, video in panels], axis=1)
                           for t in range(args.observed, args.frames)])
    write_video(args.output / "short_vs_long_future.mp4", comparison, args.fps)
    panels = [("Recorded GT", truth), ("Noisy input", noisy_video),
              ("Short-horizon denoised", prediction), ("Long-horizon denoised", longer)]
    columns = np.linspace(args.observed, args.frames - 1, 4).round().astype(int)
    rows = [np.concatenate([labelled(video[t], f"{name}: +{(t-args.observed+1)/args.fps:.2f}s")
                            for t in columns], axis=1) for name, video in panels]
    cv2.imwrite(str(args.output / "keyframes.jpg"), cv2.cvtColor(np.concatenate(rows), cv2.COLOR_RGB2BGR))
    for path in (output / "noisy_input_future.mp4", output / "generated_future.mp4"):
        if read_video(path).shape != (options.future_frames, args.size, args.size, 3):
            raise RuntimeError(f"Unexpected output dimensions: {path}")
    completed = {"elapsed_seconds": time.monotonic() - started, "sampling": sampling, **matching,
                 "future_frames": options.future_frames, "fps": args.fps,
                 "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                 "noisy_video": str(output / "noisy_input_future.mp4"),
                 "denoised_video": str(output / "generated_future.mp4"),
                 "comparison": str(args.output / "short_vs_long_future.mp4")}
    dump(args.output / "completed.json", completed)
    log(stage="completed", **completed)


if __name__ == "__main__":
    main()
