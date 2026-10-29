#!/usr/bin/env python3
"""Separate VAE boundaries, diffusion noise, and tower position effects in K."""

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

from search_vace_tracking_heads import CAMERAS, ControlTower, load_empty_text, load_vae, read_video
from train_vace_k_vector_lora import capture_k, install_k_lora, split_episodes

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CONDITIONS = (("native", "independent"), ("native", "repeated"),
              ("tiled", "independent"), ("tiled", "repeated"))


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def temporal_metrics(k):
    z = unit(k.astype(np.float32))
    drift = np.maximum(0, 1 - np.clip((z[:-1] * z[1:]).sum(-1), -1, 1)).mean(-1)
    rms = np.sqrt(np.square(k[1:].astype(np.float32) - k[:-1]).mean((1, 2)))
    grid = z.reshape(len(z), 32, 32, -1)
    horizontal = np.maximum(0, 1 - (grid[:, :, 1:] * grid[:, :, :-1]).sum(-1)).mean((1, 2))
    vertical = np.maximum(0, 1 - (grid[:, 1:] * grid[:, :-1]).sum(-1)).mean((1, 2))
    roughness = (horizontal + vertical) / 2
    return {"first_pair_cosine_drift": float(drift[0]),
            "later_pair_cosine_drift": float(drift[1:].mean()),
            "first_pair_raw_rms": float(rms[0]),
            "later_pair_raw_rms": float(rms[1:].mean()),
            "first_spatial_neighbor_distance": float(roughness[0]),
            "later_spatial_neighbor_distance": float(roughness[1:].mean()),
            "adjacent_cosine_drift": drift.tolist(),
            "adjacent_raw_rms": rms.tolist(),
            "spatial_neighbor_distance": roughness.tolist()}


def latent_metrics(latent):
    x = latent.float().cpu().numpy().transpose(1, 2, 3, 0)
    z = unit(x)
    drift = np.maximum(0, 1 - np.clip((z[:-1] * z[1:]).sum(-1), -1, 1)).mean((1, 2))
    rms = np.sqrt(np.square(x[1:] - x[:-1]).mean((1, 2, 3)))
    return {"first_pair_cosine_drift": float(drift[0]),
            "later_pair_cosine_drift": float(drift[1:].mean()),
            "adjacent_cosine_drift": drift.tolist(), "adjacent_raw_rms": rms.tolist()}


def pair_metrics(a, b):
    az, bz = unit(a), unit(b)
    distance = np.maximum(0, 1 - np.clip((az * bz).sum(-1), -1, 1))
    return {"cosine_distance_mean": float(distance.mean()),
            "cosine_distance_median": float(np.median(distance)),
            "raw_rms": float(np.sqrt(np.square(a - b).mean()))}


def field(k, display):
    x = (unit(k).reshape(-1, 128) - display["mean"]) @ display["components"].T
    color = np.clip((x - display["color_low"]) /
                    np.maximum(display["color_high"] - display["color_low"], 1e-8), 0, 1)
    return Image.fromarray((color.reshape(32, 32, 3) * 255).astype(np.uint8))


def static_figure(rgb, outputs, adapter, display, title):
    width, gap, header = 224, 8, 80
    sample_times = (0, 1, 8, len(next(iter(outputs.values()))) - 1)
    height = header + 4 * (width + 56) + 28
    canvas = Image.new("RGB", (5 * width + 6 * gap, height), (244, 246, 248))
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 13)
    draw.text((gap, 8), title + " | " + adapter, font=font, fill="black")
    for col, text in enumerate(("Identical RGB", *(f"K latent position {t}" for t in sample_times))):
        draw.text((gap + col * (width + gap), 44), text, font=small, fill="black")
    for row, (control, noise) in enumerate(CONDITIONS):
        k = outputs[(control, noise)]
        metric = temporal_metrics(k)
        y = header + row * (width + 56)
        caption = (f"{control} control / {noise} noise | first drift={metric['first_pair_cosine_drift']:.5f}"
                   f" | later={metric['later_pair_cosine_drift']:.5f}")
        draw.text((gap, y), caption, font=small, fill="black")
        panels = [Image.fromarray(rgb), *(field(k[t], display) for t in sample_times)]
        for col, panel in enumerate(panels):
            canvas.paste(panel.resize((width, width), Image.Resampling.NEAREST),
                         (gap + col * (width + gap), y + 24))
    draw.text((gap, height - 23), "Direct pre-RoPE K, 128 dimensions retained; shared train-fitted PCA3 and color limits for display only.",
              font=small, fill="black")
    return canvas


def window_figure(rgb, outputs, display, title, metric):
    width, gap = 280, 8
    canvas = Image.new("RGB", (4 * width + 5 * gap, width + 118), (244, 246, 248))
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.truetype(FONT, 16), ImageFont.truetype(FONT, 13)
    draw.text((gap, 8), title, font=font, fill="black")
    captions = ("Same RGB frame 32", "Window start32 / position0", "Window start16 / position4", "Window start0 / position8")
    panels = [Image.fromarray(rgb), *(field(outputs[start], display) for start in (32, 16, 0))]
    for col, (caption, panel) in enumerate(zip(captions, panels)):
        x = gap + col * (width + gap)
        draw.text((x, 40), caption, font=small, fill="black")
        canvas.paste(panel.resize((width, width), Image.Resampling.NEAREST), (x, 62))
    draw.text((gap, width + 72), f"K cosine distance: position0 vs4={metric['0_vs_4']['cosine_distance_mean']:.5f}, "
              f"position4 vs8={metric['4_vs_8']['cosine_distance_mean']:.5f}", font=small, fill="black")
    draw.text((gap, width + 94), "Repeated noise; clean RGB; contexts and VAE histories differ. This is not a pure position-only intervention.",
              font=small, fill="black")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--display-projection", type=Path, default=ROOT / "runs/vace_k_time_normalization_visual_20261006/fixed_display_projection.npz")
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--static-frames", type=int, default=129)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.static_frames < 37 or (args.static_frames - 1) % 4:
        raise ValueError("Static clip length must be 4n+1 and at least 37")
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
    for row in held_out:
        by_suite.setdefault(row["suite"], row)
    rows = [by_suite[suite] for suite in sorted(by_suite)]
    display = dict(np.load(args.display_projection))
    report = {"checkpoint": str(args.checkpoint.resolve()), "step": payload["step"],
              "layer": layer, "head": head, "static_frames": args.static_frames,
              "protocol": "clean observed-only RGB, empty text, direct norm_k output before RoPE, diffusion timestep1000",
              "static_control": "native repeated-RGB VAE versus tile latent8 of BOTH observed and zero-normalized-pixel reactive channels",
              "noise": "independent seeded spatial-temporal noise versus same first spatial noise repeated over time",
              "adapter": "zero lora_b in memory for original; restore exact checkpoint for trained; saved checkpoints untouched",
              "display_projection": str(args.display_projection.resolve()),
              "window_test": "same RGB frame32 in starts32/16/0 of 65-frame windows; latent positions0/4/8; changed contexts",
              "scope": "one held-out episode per LIBERO suite, both cameras; all grid cells, no point-label supervision",
              "cases": []}
    log(stage="loading", cases=len(rows) * len(CAMERAS), layer=layer, head=head)
    text = load_empty_text(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    install_k_lora(tower, layer, head, int(payload["rank"]), float(payload["alpha"]))
    expected = {key for key in tower.state_dict() if "lora_" in key}
    if set(payload["lora"]) != expected:
        raise RuntimeError("Checkpoint does not match K LoRA")
    tower.load_state_dict(payload["lora"], strict=False)
    tower.eval().requires_grad_(False)
    adapter = tower.vace_blocks[layer].attn1.to_k
    saved_b = adapter.lora_b.detach().clone()
    zeros = {}

    def pixels(rgb):
        x = torch.from_numpy(rgb.copy()).to(device).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        return x.permute(1, 0, 2, 3) / 127.5 - 1

    @torch.inference_mode()
    def encode(rgb):
        x = pixels(rgb)
        observed = vae.encode([x])[0]
        if len(rgb) not in zeros:
            zeros[len(rgb)] = vae.encode([torch.zeros_like(x)])[0].detach().cpu()
        reactive = zeros[len(rgb)].to(device)
        return observed, reactive

    def make_control(observed, reactive):
        mask = torch.zeros((64, *observed.shape[1:]), device=device, dtype=observed.dtype)
        return torch.cat((observed, reactive, mask), dim=0).unsqueeze(0).bfloat16()

    @torch.inference_mode()
    def extract(noise, control, trained):
        adapter.lora_b.copy_(saved_b if trained else torch.zeros_like(saved_b))
        k = capture_k(tower, layer, noise.shape[2], 32, noise, control, text, 1000, device)[head]
        x = k.float().cpu().numpy()
        if not np.isfinite(x).all():
            raise RuntimeError("Non-finite K")
        return x

    for row in rows:
        for camera in CAMERAS:
            name = f"episode_{row['episode']:05d}_{camera}"
            directory = args.output / name
            directory.mkdir()
            video = read_video(args.video_root / f"episode_{row['episode']:05d}" / f"{camera}.mp4")
            if len(video) < 97:
                raise ValueError(f"{name} has too few frames for controlled window test")
            static = np.repeat(video[8:9], args.static_frames, axis=0)
            observed, reactive = encode(static)
            length = observed.shape[1]
            controls = {"native": make_control(observed, reactive),
                        "tiled": make_control(observed[:, 8:9].expand_as(observed),
                                              reactive[:, 8:9].expand_as(reactive))}
            generator = torch.Generator(device=device).manual_seed(args.seed + row["episode"] * 37 + CAMERAS.index(camera))
            independent = torch.randn((1, *observed.shape), generator=generator, device=device).bfloat16()
            noises = {"independent": independent, "repeated": independent[:, :, :1].expand_as(independent).contiguous()}
            case = {"name": name, "suite": row["suite"],
                    "vae_observed": latent_metrics(observed), "vae_reactive": latent_metrics(reactive),
                    "static": {}, "window": {}}
            del static, observed, reactive
            for trained in (False, True):
                mode = "trained" if trained else "original"
                captured = {}
                for control, noise in CONDITIONS:
                    k = extract(noises[noise], controls[control], trained)
                    captured[(control, noise)] = k
                    metric = temporal_metrics(k)
                    case["static"][f"{mode}/{control}/{noise}"] = metric
                    np.savez_compressed(directory / f"static_{mode}_{control}_{noise}_full_K.npz",
                                        k=k.astype(np.float16), rgb_frame=8,
                                        latent_position=np.arange(length), control=control, noise=noise)
                    log(stage="static", case=name, mode=mode, control=control, noise=noise,
                        first_drift=metric["first_pair_cosine_drift"], later_drift=metric["later_pair_cosine_drift"])
                static_figure(video[8], captured, mode, display, f"{row['suite']} | {name} | static repeated frame8").save(
                    directory / f"static_factorial_{mode}.jpg", quality=95)
                del captured
            window_outputs = {}
            repeated_noise = noises["repeated"][:, :, :17].contiguous()
            del controls, noises, independent
            for start in (32, 16, 0):
                observed, reactive = encode(video[start:start + 65])
                k = extract(repeated_noise, make_control(observed, reactive), True)
                target_position = (32 - start) // 4
                window_outputs[start] = k[target_position]
                np.savez_compressed(directory / f"real_start{start:04d}_full_K.npz", k=k.astype(np.float16),
                                    frame_indices=start + 4 * np.arange(len(k)), target_frame=32,
                                    target_position=target_position)
                case["window"][f"start{start}"] = {"target_position": target_position, "metrics": temporal_metrics(k)}
                log(stage="window", case=name, start=start, target_position=target_position)
                del observed, reactive, k
            distances = {"0_vs_4": pair_metrics(window_outputs[32], window_outputs[16]),
                         "0_vs_8": pair_metrics(window_outputs[32], window_outputs[0]),
                         "4_vs_8": pair_metrics(window_outputs[16], window_outputs[0])}
            case["window"]["same_frame_distances"] = distances
            window_figure(video[32], window_outputs, display, f"{row['suite']} | {name} | trained K | same RGB frame32", distances).save(
                directory / "same_frame_window_positions.jpg", quality=95)
            (directory / "metrics.json").write_text(json.dumps(case, indent=2))
            report["cases"].append(case)
            (args.output / "report_partial.json").write_text(json.dumps(report, indent=2))
            log(stage="case_done", case=name, completed=len(report["cases"]), total=len(rows) * len(CAMERAS))
            del repeated_noise, video, window_outputs

    keys = list(report["cases"][0]["static"])
    aggregate_keys = ("first_pair_cosine_drift", "later_pair_cosine_drift", "first_pair_raw_rms",
                      "later_pair_raw_rms", "first_spatial_neighbor_distance", "later_spatial_neighbor_distance")
    report["aggregate_static"] = {key: {metric: float(np.mean([case["static"][key][metric] for case in report["cases"]]))
                                       for metric in aggregate_keys} for key in keys}
    report["aggregate_window"] = {pair: {metric: float(np.mean([case["window"]["same_frame_distances"][pair][metric]
                                                               for case in report["cases"]]))
                                       for metric in ("cosine_distance_mean", "raw_rms")}
                                  for pair in ("0_vs_4", "0_vs_8", "4_vs_8")}
    report["aggregate_vae"] = {channel: {metric: float(np.mean([case[channel][metric] for case in report["cases"]]))
                                       for metric in ("first_pair_cosine_drift", "later_pair_cosine_drift")}
                               for channel in ("vae_observed", "vae_reactive")}
    report["elapsed_seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, layout="constrained")
    ticks = np.arange(4)
    for ax, mode in zip(axes, ("original", "trained")):
        aggregates = [report["aggregate_static"][f"{mode}/{c}/{n}"] for c, n in CONDITIONS]
        ax.bar(ticks - .17, [x["first_pair_cosine_drift"] for x in aggregates], width=.34, label="Position0 -> 1", color="#c54b40")
        ax.bar(ticks + .17, [x["later_pair_cosine_drift"] for x in aggregates], width=.34, label="Later adjacent pairs", color="#25827c")
        ax.set_xticks(ticks, [f"{c} control\n{n} noise" for c, n in CONDITIONS], fontsize=9)
        ax.set_title(f"T{layer}H{head} | {mode} | 8 held-out camera/episode cases")
        ax.set_ylabel("Mean 1-cos(K at same grid cell)")
        ax.grid(axis="y", alpha=.2)
        ax.legend()
    fig.savefig(args.output / "static_factorial_summary.png", dpi=160)
    plt.close(fig)
    log(stage="done", output=str(args.output), aggregate_static=report["aggregate_static"],
        aggregate_window=report["aggregate_window"], peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    main()
