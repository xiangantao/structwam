#!/usr/bin/env python3
"""Replay the observed video and regenerate sparse/13-update future readout chains.

The RGB background is shared from the legacy fully denoised reference. Future
prompt/K regions, seeds, and QK tracks are independently recomputed for each branch.
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import demo_main_prompt_k_tracking as demo
from vace_latent_distill.schedules import sparse_schedule
from evaluate_vace_gt_denoising import prepare_latents
from evaluate_vace_pure_noise_future import read_video
from scan_vace_main_prompt_attention import find_phrase_positions
from vace_latent_distill.config import NEGATIVE_PROMPT, PipelineConfig
from vace_latent_distill.teacher import VaceTeacher


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def progress(**values):
    print(json.dumps(values), flush=True)


def write_h264(path, frames, fps=10):
    height, width = frames.shape[1:3]
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(command, input=np.ascontiguousarray(frames).tobytes(), check=True)


@torch.no_grad()
def capture_future(teacher, latent, control, text, timestep, positions):
    heat, q, k, fields = demo.capture(
        teacher.model, positions, latent.bfloat16(), control, text,
        torch.tensor([timestep], device=teacher.device), teacher.scales,
        latent.shape[2], teacher.config.image_size // 16, 22, 4, {22: 9, 15: 7}, 15, 4,
    )
    return {"heat": heat.numpy().reshape(13, 32, 32), "q": q.numpy(), "k": k.numpy(),
            "fields": {layer: values.numpy() for layer, values in fields.items()}}


def decode_tracks(values, anchors, config):
    seeds, tracks, raw_tracks, scores, proposals, regions = [], [], [], [], [], []
    temporal = values["heat"].shape[0]
    grid = config.image_size // 16
    for anchor in anchors:
        score, selected = demo.region_from_prompt_and_k(
            values["heat"], values["fields"], int(anchor), grid,
        )
        count = min(config.anchors_per_time, len(selected))
        chosen = selected[np.argsort(score.reshape(-1)[selected])[-count:]]
        end = min(int(anchor) + 3, temporal)
        region = demo.k_region_over_time(values["fields"], chosen, int(anchor), grid)
        raw = demo.track_dense(
            values["q"][:end * grid**2], values["k"][:end * grid**2], chosen,
            int(anchor), grid, region[:end], topk=config.beam_topk, beam_size=config.beam_size,
            max_step=config.beam_max_step, motion_weight=config.motion_weight,
            local_weight=config.local_weight, region_weight=config.region_weight,
            acceleration_weight=config.acceleration_weight, descriptor_ema=config.descriptor_ema,
            direction="q_to_k",
        )
        seeds.append(chosen)
        scores.append(score)
        proposals.append(selected)
        regions.append(region)
        raw_tracks.append(raw)
        tracks.append(demo.regularize_paths(raw, config.regularize_max_step,
                                           config.regularize_max_acceleration, passes=2))
    return {"seeds": np.stack(seeds), "paths": np.stack(tracks),
            "raw_paths": np.stack(raw_tracks), "anchor_scores": np.stack(scores),
            "proposals": np.stack(proposals), "region_scores": np.stack(regions)}


def overlay_heat(rgb, heat, bounds):
    lo, hi = bounds
    values = np.clip((heat - lo) / max(float(hi - lo), 1e-8), 0, 1)
    values = cv2.resize(values.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(np.uint8(values * 255), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    alpha = (0.45 * values)[..., None]
    return np.uint8(rgb * (1 - alpha) + color * alpha)


def render_selections(background, values, decoded, anchors, label):
    rows = []
    # One color scale for both future anchors; no per-frame percentile stretching.
    bounds = np.percentile(values["heat"][anchors[0]:], (1, 99))
    for index, (anchor, rgb_index) in enumerate(zip(anchors, (0, 8), strict=True)):
        rgb = cv2.resize(background[rgb_index], (512, 512))
        prompt = overlay_heat(rgb, values["heat"][anchor], bounds)
        region = overlay_heat(rgb, decoded["anchor_scores"][index], (0, 1))
        for cell in decoded["seeds"][index]:
            xy = ((int(cell) % 32) * 16 + 8, (int(cell) // 32) * 16 + 8)
            cv2.circle(region, xy, 2, (90, 255, 255), -1, cv2.LINE_AA)
        panels = []
        seed_count = len(decoded["seeds"][index])
        for image, title in ((rgb, "Shared reference RGB"), (prompt, "New prompt heat"),
                             (region, f"New prompt/K score + {seed_count} seeds")):
            image = image.copy()
            cv2.rectangle(image, (0, 0), (512, 34), (0, 0, 0), -1)
            cv2.putText(image, f"{label} updates | future anchor {rgb_index} | {title}",
                        (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(image)
        rows.append(np.concatenate(panels, axis=1))
    return np.concatenate(rows, axis=0)


def render_future(background, tracks, anchors_rgb, label):
    frames = []
    for frame_index, rgb in enumerate(background):
        image = cv2.resize(rgb, (512, 512)).copy()
        for path, anchor_rgb in zip(tracks, anchors_rgb, strict=True):
            if frame_index < anchor_rgb:
                continue
            elapsed = min((frame_index - anchor_rgb) / 4, path.shape[1] - 1)
            left = int(np.floor(elapsed))
            right = min(left + 1, path.shape[1] - 1)
            xy = (1 - (elapsed - left)) * path[:, left] + (elapsed - left) * path[:, right]
            for point in range(len(path)):
                history = np.concatenate((path[point, :left + 1], xy[point:point + 1]), axis=0)
                pixels = np.rint((history + 0.5) * 16).astype(np.int32)
                cv2.polylines(image, [pixels], False, (255, 65, 240), 1, cv2.LINE_AA)
                cv2.circle(image, tuple(pixels[-1]), 2, (255, 100, 245), -1, cv2.LINE_AA)
        cv2.rectangle(image, (0, 0), (512, 34), (0, 0, 0), -1)
        cv2.putText(image, f"GT+0.8 FUTURE | {label} updates | L15H4 QK + beam",
                    (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f"future frame {frame_index:02d} | anchors 0,8 | shared reference RGB",
                    (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        frames.append(image)
    return np.stack(frames)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=WORKSPACE / "runs/vace_qk_multi_anchor8_20261008_ep39")
    parser.add_argument("--states", type=Path, default=WORKSPACE / "runs/vace_future16_sigma080_steps_20261006")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vace_structured.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new output directory to preserve earlier previews")
    args.output.mkdir(parents=True)
    frozen = json.loads((args.reference / "protocol.json").read_text())
    source = Path(frozen["future_source"])
    original = json.loads((source / "protocol.json").read_text())
    config = PipelineConfig.load(args.config)
    if (original["frames"], original["observed"], original["size"]) != (49, 33, 512):
        raise ValueError("This renderer expects the frozen 33+16, 512px case")
    if config.teacher.denoise_reference_updates != 13 or not 1 <= args.updates < 13:
        raise ValueError("Compare sparse updates with the existing 13-update endpoint")
    config.teacher = dataclasses.replace(config.teacher, denoise_updates=args.updates)
    for field, expected in (("steps", config.teacher.scheduler_steps), ("guidance", config.teacher.guidance),
                            ("flow_shift", config.teacher.flow_shift), ("model", config.teacher.model_root),
                            ("text_model", config.teacher.text_root)):
        if original[field] != expected:
            raise ValueError(f"Current teacher disagrees with reference: {field}")
    source_video = Path(original["data"]) / "videos" / f"episode_{original['episode']:05d}" / f"{original['camera']}.mp4"
    truth = read_video(source_video)[original["start"]:original["start"] + 49]
    observed_preview = read_video(args.reference / "observed_multi_anchor8_h264.mp4")
    future_background = read_video(source / "generated_full.mp4")[33:]
    if len(truth) != 49 or len(observed_preview) != 33 or len(future_background) != 16:
        raise ValueError("Incomplete reference timelines")
    with np.load(args.reference / "multi_anchor_readout.npz", allow_pickle=True) as archive:
        anchors = archive["future_anchor_latents"].copy()
    if anchors.tolist() != [8, 10]:
        raise ValueError("Expected future anchors at latent8 and10")
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.teacher.text_root + "/tokenizer", local_files_only=True)
    tokens = tokenizer(original["task"], padding="max_length", max_length=512, truncation=True)
    phrase = tokenizer("black bowl", add_special_tokens=False)["input_ids"]
    positions = find_phrase_positions(tokens["input_ids"], phrase)
    device = torch.device("cuda:0")
    progress(stage="load_teacher", episode=original["episode"], start=original["start"], phrase="black bowl")
    teacher = VaceTeacher(config.teacher, "shared", device, text_cache[NEGATIVE_PROMPT])
    started = time.monotonic()
    try:
        prep_args = argparse.Namespace(**original)
        control, clean, _ = prepare_latents(teacher.vae, truth[:33], truth, prep_args, device)
        generator = torch.Generator(device=device).manual_seed(original["seed"])
        noise = torch.randn(original["noise_source_shape"], device=device, generator=generator)[:, :, :13].contiguous()
        scheduler, begin, nodes = sparse_schedule(teacher.scheduler, config.teacher, args.updates, device)
        latent = scheduler.add_noise(clean, noise, scheduler.timesteps[begin:begin + 1])
        saved_initial = torch.load(source / "noisy_input_latents.pt", map_location="cpu", weights_only=True)["latents"]
        if not torch.equal(latent.bfloat16().cpu(), saved_initial):
            raise RuntimeError("New initialization does not reproduce the frozen noisy input")
        progress(stage="initial_noise_verified", exact_bf16_match=True, sigma=nodes[0]["sigma"])
        text = text_cache[original["task"]][None].to(device, dtype=torch.bfloat16)
        pair_text = torch.cat((text, teacher.negative), 0)
        pair_control = torch.cat((control, control), 0)
        for index in range(args.updates):
            velocity = teacher._forward(torch.cat((latent, latent), 0), pair_control, pair_text,
                                        scheduler.timesteps[begin + index])
            conditional, unconditional = velocity.chunk(2)
            guided = unconditional.float() + config.teacher.guidance * (conditional.float() - unconditional.float())
            latent = scheduler.step(guided, scheduler.timesteps[begin + index], latent, return_dict=False)[0]
            progress(stage="sparse_update", step=index + 1, sigma=float(scheduler.sigmas[begin + index + 1]))
        sparse_values = capture_future(teacher, latent, control, text, nodes[-1]["timestep"], positions)
        reference = torch.load(args.states / "step_014/latents.pt", map_location="cpu", weights_only=True)
        if reference["input_sigma"] != nodes[-1]["sigma"] or reference["timestep"] != nodes[-1]["timestep"]:
            raise RuntimeError("Saved 13-update reference has a different readout noise level")
        reference_values = capture_future(teacher, reference["model_input"].to(device), control, text,
                                         reference["timestep"], positions)
        progress(stage="both_readouts_captured", sigma=nodes[-1]["sigma"])
    finally:
        teacher.close()
    sparse_decoded = decode_tracks(sparse_values, anchors, config.teacher)
    reference_decoded = decode_tracks(reference_values, anchors, config.teacher)
    overlap = [len(set(left) & set(right)) for left, right in
               zip(sparse_decoded["seeds"], reference_decoded["seeds"], strict=True)]
    progress(stage="future_chains_regenerated", sparse_seed_counts=list(map(len, sparse_decoded["seeds"])),
             reference_seed_counts=list(map(len, reference_decoded["seeds"])), shared_seed_counts=overlap)
    sparse_preview = render_future(future_background, sparse_decoded["paths"], (0, 8), args.updates)
    reference_preview = render_future(future_background, reference_decoded["paths"], (0, 8), 13)
    sparse_full = np.concatenate((observed_preview, sparse_preview))
    reference_full = np.concatenate((observed_preview, reference_preview))
    combined = args.output / f"observed_then_future_trajectory_{args.updates}step.mp4"
    side_by_side = args.output / f"observed_then_future_trajectory_{args.updates}_vs_13.mp4"
    write_h264(combined, sparse_full)
    write_h264(args.output / "observed_then_future_trajectory_13step.mp4", reference_full)
    write_h264(side_by_side, np.concatenate((sparse_full, reference_full), axis=2))
    arrays = {"anchor_latents": anchors}
    for name, values, decoded in (("sparse", sparse_values, sparse_decoded),
                                  ("reference", reference_values, reference_decoded)):
        arrays.update({f"{name}_{key}": array for key, array in decoded.items()})
        arrays.update({f"{name}_prompt_heat": values["heat"], f"{name}_q": values["q"],
                       f"{name}_k": values["k"]})
        arrays.update({f"{name}_k{layer}_fields": field for layer, field in values["fields"].items()})
        selection_image = render_selections(future_background, values, decoded, anchors,
                                            args.updates if name == "sparse" else 13)
        cv2.imwrite(str(args.output / f"future_prompt_k_selections_{name}.jpg"),
                    cv2.cvtColor(selection_image, cv2.COLOR_RGB2BGR))
    np.savez_compressed(args.output / "future_track_comparison.npz", **arrays)
    tiles = [np.concatenate((sparse_full[time_index], reference_full[time_index]), axis=1)
             for time_index in (33, 41, 48)]
    cv2.imwrite(str(args.output / "comparison_keyframes.jpg"), cv2.cvtColor(np.concatenate(tiles), cv2.COLOR_RGB2BGR))
    dump(args.output / "protocol.json", {
        "frozen_reference": str(args.reference.resolve()), "episode": original["episode"],
        "camera": original["camera"], "source_frames": [original["start"], original["start"] + 48],
        "prompt": original["task"], "phrase": "black bowl", "observed": "Frozen 33-frame preview, including 15-copy warmup in its original readout",
        "future_input": "Full 49-frame GT + sigma0.8 noise; control sees first33 only",
        "future_seed": original["seed"], "noise_source_shape": original["noise_source_shape"],
        "initial_noise_bf16_exact_match": True, "sparse_schedule": nodes, "reference_updates": 13,
        "reference_state": str(args.states / "step_014/latents.pt"), "readout_head": "L15H4 post-RoPE QK",
        "semantic_head": "L22H4 phrase attention", "k_heads": ["L22H9", "L15H7"],
        "future_anchor_rgb": [33, 41], "future_anchor_latents": anchors.tolist(),
        "seeds": "Independently regenerated prompt -> K region -> dense seeds at both future anchors; no GT masks",
        "future_chain_regenerated": ["prompt_heat", "k_fields", "prompt_k_regions", "seeds", "qk", "beam_tracks"],
        "seed_overlap_sparse_vs_reference": overlap,
        "selection": {"prompt_seed_fraction": 0.05, "region_keep_fraction": 0.15,
                      "dense_seed_cap": config.teacher.anchors_per_time},
        "comparison": "Independent seed sets; point-index path differences are not an agreement metric",
        "selection_color_scale": "Prompt p1-p99 over future timeline; combined prompt/K score fixed0..1",
        "decoder": "q_t -> k_(t+1), beam4/topk32, original motion/local/region/acceleration weights",
        "temporal_rendering": "Interpolate each short native trajectory to RGB frames, anchors at future0 and8",
        "background": f"Shared old fully denoised RGB, 23 updates to sigma0; it is not the {args.updates}-update latent decode",
        "optical_flow_used": False, "fps": 10, "frames": 49,
    })
    progress(stage="completed", video=str(combined), comparison=str(side_by_side),
             elapsed_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
