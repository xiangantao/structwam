#!/usr/bin/env python3
"""Visualize native K vectors before/after the K-vector LoRA on held-out LIBERO.

The feature itself remains the full K vector. PCA3 is used only to render the
32x32 vector field, with one projection and color scale shared by both models.
No QK matrix, query point, clustering, or preset object count is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from search_vace_tracking_heads import CAMERAS, ControlTower, load_empty_text, load_vae  # noqa: E402
from train_vace_k_vector_lora import (  # noqa: E402
    capture_k,
    install_k_lora,
    make_controls,
    make_case_index,
    split_episodes,
)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def read_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def render_field(rgb: np.ndarray, values: np.ndarray, pca: PCA, low: np.ndarray,
                 high: np.ndarray, alpha: float = 0.66) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    projected = pca.transform(centered) / np.sqrt(np.maximum(pca.explained_variance_, 1e-8))
    colors = np.clip((projected - low) / np.maximum(high - low, 1e-6), 0, 1)
    colors = (colors.reshape(32, 32, 3) * 255).astype(np.uint8)
    image = Image.fromarray(rgb).resize((512, 512), Image.Resampling.BILINEAR)
    field = Image.fromarray(colors).resize((512, 512), Image.Resampling.NEAREST)
    return np.asarray(Image.blend(image, field, alpha=alpha))


def sheet(case_name: str, rgb: np.ndarray, frame_index: int, base: np.ndarray,
          tuned: np.ndarray, pca: PCA, low: np.ndarray, high: np.ndarray, step: int) -> Image.Image:
    font = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 14)
    scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-8))
    base_proj = pca.transform(base - base.mean(axis=0, keepdims=True)) / scale
    tuned_proj = pca.transform(tuned - tuned.mean(axis=0, keepdims=True)) / scale
    frame_low, frame_high = np.percentile(np.concatenate([base_proj, tuned_proj]), [2, 98], axis=0)
    panels = [
        ("RGB", rgb),
        ("base K / frame-local PCA3", render_field(rgb, base, pca, frame_low, frame_high)),
        (f"step{step} K-LoRA / frame-local PCA3", render_field(rgb, tuned, pca, frame_low, frame_high)),
    ]
    width, height, gap = 512, 512, 10
    output = Image.new("RGB", (3 * width + 4 * gap, height + 92), (238, 240, 242))
    draw = ImageDraw.Draw(output)
    draw.text((10, 10), f"{case_name} | RGB frame {frame_index} | direct K vector field", font=font, fill="black")
    for column, (caption, pixels) in enumerate(panels):
        x = gap + column * (width + gap)
        draw.rectangle((x, 42, x + width, 42 + height + 30), fill="white")
        draw.text((x + 7, 50), caption, font=small, fill="black")
        output.paste(Image.fromarray(pixels), (x, 42 + 30))
    draw.text((10, output.height - 16), "Full K retained; frame-local contrast is shared between base and LoRA; colors are not comparable across frames", font=small, fill="black")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/tracks")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--head", type=int, default=11)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--max-cases-per-suite", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--skip-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=129)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _, _, val_rows = split_episodes(args.video_root, args.seed)
    index_args = argparse.Namespace(
        size=args.size, skip_frames=args.skip_frames, max_frames=args.max_frames,
        windows_per_episode=1, points_per_subject=24, min_motion=.02, model=args.model,
    )
    text = load_empty_text(args.text_model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    cases = make_case_index(args.video_root, args.tracks_root, val_rows, list(CAMERAS),
                            index_args, device, args.seed)
    by_suite = {}
    for case in cases:
        by_suite.setdefault(case["suite"], [])
        if len(by_suite[case["suite"]]) < args.max_cases_per_suite:
            by_suite[case["suite"]].append(case)
    cases = [case for suite in sorted(by_suite) for case in by_suite[suite]]
    if not cases:
        raise RuntimeError("No held-out LIBERO cases found")

    zero_latents = {}
    base_vectors, metadata = [], []
    for case in cases:
        case.update(make_controls(case, args.video_root, args.model, vae, zero_latents, device, args.size,
                                  args.seed + case["episode"] * 37 + case["start_frame"]))
        k = capture_k(tower, args.layer, len(case["times"]), args.size // 16,
                      case["noise"], case["control"], text, 1000, device)[args.head]
        base_vectors.append(k.detach().float().cpu().numpy())
        metadata.append(case)
        print(json.dumps({"stage": "base", "case": case["name"], "suite": case["suite"]}), flush=True)

    # A single display-only projection is fitted across both versions.
    install_k_lora(tower, args.layer, args.head, args.rank, args.alpha)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tower.load_state_dict(payload["lora"], strict=False)
    tuned_vectors = []
    for case in metadata:
        k = capture_k(tower, args.layer, len(case["times"]), args.size // 16,
                      case["noise"], case["control"], text, 1000, device)[args.head]
        tuned_vectors.append(k.detach().float().cpu().numpy())
        print(json.dumps({"stage": "lora", "case": case["name"], "suite": case["suite"]}), flush=True)

    all_values = np.concatenate([
        (x - x.mean(axis=tuple(range(x.ndim - 1)), keepdims=True)).reshape(-1, x.shape[-1])
        for x in base_vectors + tuned_vectors
    ], axis=0)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(all_values)
    projected = pca.transform(all_values) / np.sqrt(np.maximum(pca.explained_variance_, 1e-8))
    low, high = np.percentile(projected, [.5, 99.5], axis=0)
    report = {"checkpoint": str(args.checkpoint), "step": payload.get("step"),
              "layer": args.layer, "head": args.head,
              "cases": [case["name"] for case in metadata],
              "suites": sorted(by_suite),
              "shared_pca_explained_variance": pca.explained_variance_ratio_.tolist(),
              "display": "frame-centered PCA3 whitened by component variance; per-frame 2-98% contrast",
              "display_only": True}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    np.savez(args.output / "shared_display_pca.npz", components=pca.components_, mean=pca.mean_,
             color_low=low, color_high=high)
    for case, base, tuned in zip(metadata, base_vectors, tuned_vectors):
        video = read_video(args.video_root / f"episode_{case['episode']:05d}" / f"{case['camera']}.mp4")
        frame_ids = case["times"]
        # Render first, middle, and last observed latent frame as one contact sheet.
        frames = [0, len(frame_ids) // 2, len(frame_ids) - 1]
        columns = [sheet(case["name"], video[int(frame_ids[i])], int(frame_ids[i]),
                         base[i], tuned[i], pca, low, high, int(payload.get("step", 0))) for i in frames]
        out = Image.new("RGB", (columns[0].width, sum(x.height for x in columns) + 8 * (len(columns) - 1)), "white")
        y = 0
        for image in columns:
            out.paste(image, (0, y))
            y += image.height + 8
        path = args.output / f"{case['name']}_base_vs_step{payload.get('step', 'unknown'):04d}.jpg"
        out.save(path, quality=95)
        print(json.dumps({"stage": "rendered", "path": str(path)}), flush=True)
    print(json.dumps({"stage": "done", "output": str(args.output), "cases": len(metadata)}), flush=True)


if __name__ == "__main__":
    main()
