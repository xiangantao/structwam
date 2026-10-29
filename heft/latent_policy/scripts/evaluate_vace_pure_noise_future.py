#!/usr/bin/env python3
"""Native VACE continuation from an observed prefix and pure Gaussian noise."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import av
import cv2
import numpy as np
from safetensors.torch import load_file
import torch

from search_vace_tracking_heads import ROOT, load_vae, read_video, write_video
from diffusers import UniPCMultistepScheduler
from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel


def log(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def dump(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False))


def choose_window(args):
    split = json.loads((args.data / "split.json").read_text())
    allowed = set(split["val_episodes"])
    choices = []
    for row in split["episodes"]:
        if row["episode"] not in allowed or (args.episode is not None and row["episode"] != args.episode):
            continue
        directory = args.data / "videos" / f"episode_{row['episode']:05d}"
        meta = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "sim_state.npz") as z:
            positions = z["body_xpos"]
        roots = [i for i, name in enumerate(meta["body_names"]) if name.endswith("_main")]
        hand = meta["body_names"].index("gripper0_eef")
        for start in range(16, len(positions) - args.frames + 1, 4):
            if args.start is not None and start != args.start:
                continue
            now = start + args.observed - 1
            future = positions[now:start + args.frames, roots]
            delta = np.linalg.norm(future - future[0], axis=-1).max(0)
            near = np.linalg.norm(positions[now, roots] - positions[now, hand], axis=-1)
            past = positions[start:now + 1, roots]
            past_motion = np.linalg.norm(past - past[-1], axis=-1).max(0)
            target = int(np.argmax(delta * (near < 0.25) * (past_motion > 0.01)))
            score = float(delta[target] * (near[target] < 0.25) * (past_motion[target] > 0.01))
            choices.append((score, row, start, meta, meta["body_names"][roots[target]]))
    if not choices:
        raise ValueError("No held-out window satisfies the requested frame bounds")
    score, row, start, meta, target = max(choices, key=lambda x: x[0])
    if score < 0.02:
        raise ValueError("No selected window has meaningful future object motion")
    return {k: row[k] for k in ("episode", "suite", "task_id", "task", "source_episode")} | {
        "start": start, "fps": meta.get("fps", 20), "diagnostic_target_body": target,
        "future_object_excursion_m": score,
        "selection": "Held-out diagnostic window chosen by recorded simulator motion; this is not an unbiased benchmark.",
    }


def read_prefix(path, start, count):
    frames = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError("Not enough observed video frames")
    return np.stack(frames)


@torch.no_grad()
def text_conditions(path, prompt, device):
    from transformers import AutoTokenizer, UMT5EncoderModel
    negative = "overexposed, blurry, subtitles, low quality, deformed, extra fingers, camera cut"
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(path / "text_encoder", torch_dtype=torch.bfloat16,
                                             local_files_only=True).to(device).eval()
    tokens = tokenizer([prompt, negative], padding="max_length", max_length=512, truncation=True,
                       return_tensors="pt").to(device)
    embeddings = encoder(**tokens).last_hidden_state.detach()
    for index in range(2):
        embeddings[index, int(tokens.attention_mask[index].sum()):] = 0
    del encoder, tokenizer, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings[:1], embeddings[1:], negative


def load_transformer(path, device):
    config = json.loads((path / "config.json").read_text())
    with torch.device("meta"):
        model = WanVACETransformer3DModel(
            num_attention_heads=config["num_heads"], attention_head_dim=config["dim"] // config["num_heads"],
            in_channels=config["in_dim"], out_channels=config["out_dim"], ffn_dim=config["ffn_dim"],
            num_layers=config["num_layers"], vace_layers=config["vace_layers"],
            vace_in_channels=config["vace_in_dim"], eps=config["eps"],
        )
    state = convert_wan_transformer_to_diffusers(load_file(str(path / "diffusion_pytorch_model.safetensors")))
    state = {k.replace(".before_proj.", ".proj_in.").replace(".after_proj.", ".proj_out."): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
    model.rope = WanRotaryPosEmbed(config["dim"] // config["num_heads"], (1, 2, 2), 1024)
    model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    for module in model.modules():
        if hasattr(module, "scale_shift_table"):
            module.scale_shift_table.data = module.scale_shift_table.data.float()
    return model


@torch.no_grad()
def generate(model, vae, scheduler, prefix, positive, negative, args, progress):
    from torch.nn import functional as F
    device = positive.device
    pixels = torch.zeros((3, args.frames, args.size, args.size), device=device)
    observed_pixels = torch.from_numpy(prefix.copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
    observed_pixels = F.interpolate(observed_pixels.permute(1, 0, 2, 3), size=(args.size, args.size),
                                   mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    pixels[:, :args.observed] = observed_pixels
    assert not pixels[:, args.observed:].count_nonzero()
    inactive = vae.encode([pixels])[0]
    reactive = vae.encode([torch.zeros_like(pixels)])[0]
    expected_time = (args.frames - 1) // 4 + 1
    if inactive.shape != (16, expected_time, args.size // 8, args.size // 8):
        raise RuntimeError("Native VAE latent grid does not match the output timeline")
    # Use the native VACE8x8 mask packing and temporal nearest-exact sampling.
    mask = torch.ones((args.frames, args.size, args.size), device=device)
    mask[:args.observed] = 0
    mask = mask.view(args.frames, args.size // 8, 8, args.size // 8, 8)
    mask = mask.permute(2, 4, 0, 1, 3).flatten(0, 1)
    mask = F.interpolate(mask[None], size=inactive.shape[1:], mode="nearest-exact")[0]
    control = torch.cat((inactive, reactive, mask), dim=0)[None].bfloat16()
    del pixels, observed_pixels, inactive, reactive, mask
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn((1, 16, expected_time, args.size // 8, args.size // 8),
                          dtype=torch.float32, device=device, generator=generator)
    scheduler.set_timesteps(args.steps, device=device)
    scales = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
    log(stage="pure_noise_ready", latent_shape=list(latents.shape), control_shape=list(control.shape),
        first_timestep=float(scheduler.timesteps[0]), last_timestep=float(scheduler.timesteps[-1]))
    for index, timestep in enumerate(scheduler.timesteps):
        input_latents = latents.bfloat16()
        time_batch = timestep.reshape(1)
        conditional = model(hidden_states=input_latents, timestep=time_batch, encoder_hidden_states=positive,
                            control_hidden_states=control, control_hidden_states_scale=scales,
                            return_dict=False)[0]
        unconditional = model(hidden_states=input_latents, timestep=time_batch, encoder_hidden_states=negative,
                              control_hidden_states=control, control_hidden_states_scale=scales,
                              return_dict=False)[0]
        update = unconditional.float() + args.guidance * (conditional.float() - unconditional.float())
        latents = scheduler.step(update, timestep, latents, return_dict=False)[0]
        progress(None, index, timestep, {"latents": latents})
    torch.save({"latents": latents.detach().cpu(), "seed": args.seed}, args.output / "generated_latents.pt")
    result = vae.decode([latents[0]])[0].permute(1, 2, 3, 0).cpu().numpy()
    return (result + 1) / 2


def labelled(frame, label, width=384):
    frame = cv2.resize(frame, (width, width), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((width + 30, width, 3), np.uint8)
    canvas[30:] = frame
    cv2.putText(canvas, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--camera", choices=["agentview", "robot0_eye_in_hand"], default="agentview")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--observed", type=int, default=33)
    parser.add_argument("--frames", type=int, default=65)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=5)
    parser.add_argument("--flow-shift", type=float, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.frames - 1) % 4 or (args.observed - 1) % 4 or not 1 <= args.observed < args.frames:
        parser.error("Total and observed frame counts must be4n+1, with some unobserved frames")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    case = choose_window(args)
    source = args.data / "videos" / f"episode_{case['episode']:05d}" / f"{args.camera}.mp4"
    prefix = read_prefix(source, case["start"], args.observed)
    write_video(args.output / "observed_prefix.mp4", prefix, case["fps"])
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | case | {
        "input": "Observed RGB prefix and original task instruction only; future pixels are zero in normalized coordinates.",
        "latent_initialization": "Pure independent Gaussian noise over the entire output timeline.",
        "mask": "Observed prefix0; unknown future1.", "lora_loaded": False,
        "ground_truth_future": "Read only after generation, for comparison.",
        "known_frame_indices": list(range(case["start"], case["start"] + args.observed)),
        "predicted_frame_indices": list(range(case["start"] + args.observed, case["start"] + args.frames)),
    }
    dump(args.output / "protocol.json", protocol)
    log(stage="selected", **case, camera=args.camera, observed=args.observed, total=args.frames)
    positive, negative, negative_text = text_conditions(args.text_model, case["task"], device)
    protocol["negative_prompt"] = negative_text
    dump(args.output / "protocol.json", protocol)
    model = load_transformer(args.model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    scheduler = UniPCMultistepScheduler.from_pretrained(args.text_model / "scheduler", local_files_only=True,
                                                       flow_shift=args.flow_shift)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    def progress(pipeline, index, timestep, kwargs):
        info = {"step": index + 1, "steps": args.steps, "timestep": float(timestep),
                "elapsed_seconds": time.monotonic() - started,
                "latent_std": float(kwargs["latents"].std()),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        if not torch.isfinite(kwargs["latents"]).all():
            raise RuntimeError("Nonfinite generated latent")
        log(stage="denoise", **info)
        dump(args.output / "status.json", info | {"state": "generating", "pid": os.getpid()})
        return kwargs
    result = generate(model, vae, scheduler, prefix, positive, negative, args, progress)
    if result.shape[0] != args.frames or not np.isfinite(result).all():
        raise RuntimeError("Invalid generated video")
    prediction = np.round(np.clip(result, 0, 1) * 255).astype(np.uint8)
    write_video(args.output / "generated_full.mp4", prediction, case["fps"])
    write_video(args.output / "generated_future.mp4", prediction[args.observed:], case["fps"])
    truth = read_video(source)[case["start"]:case["start"] + args.frames]
    write_video(args.output / "ground_truth_future.mp4", truth[args.observed:], case["fps"])
    comparison = []
    for index in range(args.frames):
        known = prefix[min(index, args.observed - 1)]
        comparison.append(np.concatenate((labelled(known, "Observed prefix / last known frame"),
                                          labelled(prediction[index], "VACE: pure-noise continuation"),
                                          labelled(truth[index], "Recorded reference")), axis=1))
    write_video(args.output / "comparison.mp4", np.stack(comparison), case["fps"])
    columns = [args.observed - 1, args.observed + 7, args.observed + 15, args.frames - 1]
    rows = []
    for name, video in (("Recorded", truth), ("VACE", prediction),
                        ("Hold position", np.repeat(prefix[-1:], args.frames, axis=0))):
        rows.append(np.concatenate([labelled(video[t], f"{name}: +{(t-args.observed+1)/case['fps']:.2f}s")
                                    for t in columns], axis=1))
    montage = np.concatenate(rows)
    cv2.imwrite(str(args.output / "keyframes.jpg"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    dump(args.output / "completed.json", {"elapsed_seconds": time.monotonic() - started,
                                          "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                          "case": case, "comparison": str(args.output / "comparison.mp4")})
    log(stage="completed", output=str(args.output), seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
