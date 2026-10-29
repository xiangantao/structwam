#!/usr/bin/env python3
"""Compare VAE-only warm-up against original direct K on held-out LIBERO videos."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_vace_k_temporal_boundary import field, temporal_metrics
from search_vace_tracking_heads import CAMERAS, ControlTower, load_empty_text, load_vae, read_video, write_video
from train_vace_k_vector_lora import capture_k, install_k_lora, split_episodes
from train_vace_tracking_lora import make_case
from vace_vae_warmup import observation_control, observation_window_indices
from visualize_vace_k_time_normalization import geometry

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def row_image(rgb, outputs, time_index, frame_index, display, title, warmups):
    width, gap = 224, 8
    canvas = Image.new("RGB", ((len(warmups) + 1) * width + (len(warmups) + 2) * gap, width + 100), (244, 246, 248))
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.truetype(FONT, 16), ImageFont.truetype(FONT, 13)
    draw.text((gap, 8), f"{title} | RGB frame {frame_index}", font=font, fill="black")
    panels = [("RGB", Image.fromarray(rgb))]
    panels.extend(("Original" if warmup == 0 else f"VAE warm-up {warmup} frames",
                   field(outputs[warmup][time_index], display)) for warmup in warmups)
    for col, (caption, panel) in enumerate(panels):
        x = gap + col * (width + gap)
        draw.text((x, 38), caption, font=small, fill="black")
        canvas.paste(panel.resize((width, width), Image.Resampling.NEAREST), (x, 60))
    draw.text((gap, width + 70), "Same checkpoint, noise and retained times; fixed train-fitted PCA3/color limits; full 128-D K retained.", font=small, fill="black")
    return canvas


def stack_images(images):
    output = Image.new("RGB", (images[0].width, sum(image.height for image in images)), "white")
    offset = 0
    for image in images:
        output.paste(image, (0, offset))
        offset += image.height
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-frames", nargs="+", type=int, default=[0, 16, 64, 128])
    parser.add_argument("--starts", nargs="+", type=int, default=[0, 8, 32])
    parser.add_argument("--max-frames", type=int, default=129)
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/tracks")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--display-projection", type=Path, default=ROOT / "runs/vace_k_time_normalization_visual_20261006/fixed_display_projection.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if 0 not in args.warmup_frames or any(w < 0 or w % 4 for w in args.warmup_frames):
        parser.error("Include the zero-warm-up baseline; all warm-ups must be nonnegative multiples of four")
    if any(start < 0 or start % 4 for start in args.starts):
        parser.error("Starts must be nonnegative multiples of four")
    args.warmup_frames = sorted(set(args.warmup_frames))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def log(**values):
        values["elapsed_seconds"] = round(time.monotonic() - started, 2)
        line = json.dumps(values)
        print(line, flush=True)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layer, head = int(payload["layer"]), int(payload["head"])
    _, _, held_out = split_episodes(args.video_root, args.seed)
    by_suite = {}
    for episode in held_out:
        by_suite.setdefault(episode["suite"], episode)
    episodes = [by_suite[suite] for suite in sorted(by_suite)]
    display = dict(np.load(args.display_projection))
    report = {"checkpoint": str(args.checkpoint.resolve()), "step": payload["step"], "layer": layer, "head": head,
              "warmup_frames": args.warmup_frames, "starts": args.starts,
              "protocol": "clean observed RGB, empty text, VAE-only warm-up then crop observed AND reactive latent channels before tower",
              "alignment": "prepend 4m RGB frames, discard m latent positions; retained endpoints stay start+4t",
              "history": "actual preceding frames, repeat earliest available RGB to pad missing prefix; no future history",
              "noise": "identical independent spatial-temporal noise for all warm-up lengths within each retained window",
              "metrics": "full direct pre-RoPE K; unique unmixed visible GT cells only for evaluation; cross/within larger-is-better",
              "display_projection": str(args.display_projection.resolve()),
              "scope": "four held-out episodes, one per suite, both cameras, three window starts; not the full task set",
              "cases": []}
    total = len(episodes) * len(CAMERAS) * len(args.starts)
    log(stage="loading", total=total, layer=layer, head=head)
    text = load_empty_text(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    install_k_lora(tower, layer, head, int(payload["rank"]), float(payload["alpha"]))
    expected = {key for key in tower.state_dict() if "lora_" in key}
    if expected != set(payload["lora"]):
        raise RuntimeError("Checkpoint does not match the selected K head")
    tower.load_state_dict(payload["lora"], strict=False)
    tower.eval().requires_grad_(False)
    zero_latents = {}
    overview = {camera: [] for camera in CAMERAS}
    for episode in episodes:
        for camera in CAMERAS:
            video = read_video(args.video_root / f"episode_{episode['episode']:05d}" / f"{camera}.mp4")
            for start in args.starts:
                case = make_case(args.video_root, args.tracks_root, args.model, episode["episode"], camera,
                                 512, start, args.max_frames, 24, .02, device, args.seed,
                                 video=video, points_only=True, include_static=True)
                name = case["name"]
                directory = args.output / name
                directory.mkdir()
                frame_ids = np.asarray(case["times"])
                length = int(frame_ids[-1] - start + 1)
                generator = torch.Generator(device=device).manual_seed(args.seed + episode["episode"] * 37 + start * 10 + CAMERAS.index(camera))
                noise = torch.randn((1, 16, len(frame_ids), 64, 64), generator=generator, device=device).bfloat16()
                entry = {"name": name, "suite": episode["suite"], "camera": camera, "start": start,
                         "frame_indices": frame_ids.tolist(), "metrics": {}, "warmup_metadata": {}}
                outputs = {}
                for warmup in args.warmup_frames:
                    indices, metadata = observation_window_indices(len(video), start, length, warmup)
                    if not np.array_equal(metadata["frame_indices"], frame_ids):
                        raise RuntimeError("Warm-up endpoints shifted relative to GT")
                    pixels = torch.from_numpy(video[indices].copy()).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
                    pixels = F.interpolate(pixels, size=(512, 512), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
                    with torch.inference_mode():
                        control = observation_control(vae, pixels, warmup, zero_latents)
                        if control.shape != (1, 96, len(frame_ids), 64, 64):
                            raise RuntimeError("Warm-up changed tower input shape")
                        keys = capture_k(tower, layer, len(frame_ids), 32, noise, control, text, 1000, device)[head].float().cpu().numpy()
                    if not np.isfinite(keys).all():
                        raise RuntimeError("Non-finite K")
                    outputs[warmup] = keys
                    metric = {"grid": temporal_metrics(keys), "objects": geometry(keys, case["points"])}
                    entry["metrics"][str(warmup)] = metric
                    entry["warmup_metadata"][str(warmup)] = {key: value for key, value in metadata.items() if key != "frame_indices"}
                    np.savez_compressed(directory / f"warmup_{warmup:03d}_full_K.npz", k=keys.astype(np.float16),
                                        frame_indices=frame_ids, warmup_frames=warmup,
                                        cells=case["points"]["cells"], visible=case["points"]["visible"], labels=case["points"]["labels"])
                    log(stage="captured", case=name, warmup=warmup,
                        first_point_drift=metric["objects"]["first_pair_cosine_drift"],
                        first_group_ratio=metric["objects"]["first_frame_ratio_median"],
                        group_ratio=metric["objects"]["cross_to_within_median"])
                    del pixels, control, keys

                def row(t):
                    return row_image(video[frame_ids[t]], outputs, t, frame_ids[t], display,
                                     f"{episode['suite']} | {name}", args.warmup_frames)

                stack_images([row(t) for t in range(min(4, len(frame_ids)))]).save(directory / "early_frames.jpg", quality=95)
                stack_images([row(t) for t in (0, len(frame_ids) // 2, len(frame_ids) - 1)]).save(directory / "start_middle_end.jpg", quality=95)
                if start == 8:
                    overview[camera].append(row(0))
                    write_video(directory / "fixed_scale_comparison.mp4", np.stack([np.asarray(row(t)) for t in range(len(frame_ids))]), fps=5)
                (directory / "metrics.json").write_text(json.dumps(entry, indent=2))
                report["cases"].append(entry)
                (args.output / "report_partial.json").write_text(json.dumps(report, indent=2))
                log(stage="case_done", case=name, completed=len(report["cases"]), total=total)
                del outputs, noise
            del video

    metric_paths = (("grid", "first_pair_cosine_drift"), ("grid", "later_pair_cosine_drift"),
                    ("objects", "first_pair_cosine_drift"), ("objects", "visible_point_cosine_drift"),
                    ("objects", "first_frame_ratio_median"), ("objects", "cross_to_within_median"),
                    ("objects", "within_rms"), ("objects", "cross_rms"))

    def aggregate(cases):
        values = {}
        for warmup in args.warmup_frames:
            values[str(warmup)] = {}
            for group, metric in metric_paths:
                samples = [case["metrics"][str(warmup)][group][metric] for case in cases]
                samples = [value for value in samples if value is not None]
                values[str(warmup)][f"{group}/{metric}"] = float(np.mean(samples)) if samples else None
        return values

    report["aggregate"] = aggregate(report["cases"])
    report["aggregate_by_start"] = {str(start): aggregate([case for case in report["cases"] if case["start"] == start])
                                    for start in args.starts}
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    for camera, images in overview.items():
        if images:
            stack_images(images).save(args.output / f"all_suites_{camera}_frame8.jpg", quality=95)
    lines = ["# VAE-only warm-up comparison", "", report["protocol"], "", report["history"], "",
             "Same checkpoint and noise; GT only evaluates grouping, never controls extraction.", "",
             "| Warm-up RGB frames | First visible-point drift (lower) | All visible-point drift (lower) | First cross/within (higher) | All cross/within (higher) |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    for warmup in args.warmup_frames:
        metric = report["aggregate"][str(warmup)]
        numbers = [metric[key] for key in ("objects/first_pair_cosine_drift", "objects/visible_point_cosine_drift",
                                            "objects/first_frame_ratio_median", "objects/cross_to_within_median")]
        lines.append("| " + " | ".join([str(warmup)] + [f"{number:.6f}" if number is not None else "n/a" for number in numbers]) + " |")
    lines.extend(["", f"Cases: {len(report['cases'])}, four episodes / four suites / two cameras / three window starts.",
                  "", "The VAE sees extra past RGB frames. Its cropped output and tower noise retain their original time grid.",
                  "", "PCA3 and color limits are fixed from the earlier training-only display projection; K stays 128-D.",
                  "", "No training was launched. Existing checkpoints and original extraction scripts are untouched."])
    (args.output / "current_report.md").write_text("\n".join(lines) + "\n")
    log(stage="done", aggregate=report["aggregate"], output=str(args.output), peak_memory_gib=report["peak_memory_gib"])


if __name__ == "__main__":
    main()
