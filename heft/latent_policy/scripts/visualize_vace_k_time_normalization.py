#!/usr/bin/env python3
"""Render train-fitted time normalization of direct K vectors on held-out clips."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from search_vace_tracking_heads import CAMERAS, ControlTower, load_empty_text, load_vae, read_video, write_video
from train_vace_k_vector_lora import capture_k, install_k_lora, make_case_index, make_controls, split_episodes

MODES = ("raw", "center", "zscore")
CAPTIONS = ("RGB", "Original K", "K - train time mean", "Time channel z-score")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def transformed(values, means, stds, mode):
    if mode == "raw":
        return values
    centered = values - means[:len(values), None]
    return centered if mode == "center" else centered / stds[:len(values), None]


def select_rows(rows, per_suite):
    groups = {}
    for row in rows:
        groups.setdefault(row["suite"], []).append(row)
    return [row for suite in sorted(groups) for row in groups[suite][:per_suite]]


def fit_statistics(arrays):
    max_t, dim = max(len(x) for x in arrays), arrays[0].shape[-1]
    sums, squares = np.zeros((max_t, dim)), np.zeros((max_t, dim))
    counts = np.zeros(max_t, dtype=np.int64)
    for values in arrays:
        sums[:len(values)] += values.sum(axis=1, dtype=np.float64)
        squares[:len(values)] += np.square(values.astype(np.float64)).sum(axis=1)
        counts[:len(values)] += values.shape[1]
    means = sums / counts[:, None]
    stds = np.sqrt(np.maximum(squares / counts[:, None] - means ** 2, 1e-5))
    return means.astype(np.float32), stds.astype(np.float32), counts


def geometry(values, points):
    """Score unique, unambiguous cells; retain separate within and cross distances."""
    vectors = unit(values)
    labels = np.asarray([
        "gripper" if str(x) in {"gripper base", "left finger", "right finger"} else str(x)
        for x in points["labels"]
    ])
    rows = []
    for t in range(len(values)):
        visible_ids = np.flatnonzero(points["visible"][:, t])
        cells = points["cells"][visible_ids, t]
        clean = {}
        for cell in np.unique(cells):
            occupants = np.unique(labels[visible_ids[cells == cell]])
            if len(occupants) == 1:
                clean.setdefault(str(occupants[0]), []).append(int(cell))
        moments = {}
        for label, ids in clean.items():
            if len(ids) < 2:
                continue
            z = vectors[t, ids].astype(np.float64)
            mean = z.mean(0)
            variance = np.square(z - mean).sum(1).mean()
            moments[label] = (mean, variance, len(ids))
        names = sorted(moments)
        for i, left in enumerate(names):
            a, va, na = moments[left]
            for right in names[i + 1:]:
                b, vb, nb = moments[right]
                within_sq = (2 * na / (na - 1) * va + 2 * nb / (nb - 1) * vb) / 2
                cross_sq = np.square(a - b).sum() + va + vb
                if within_sq <= 1e-8:
                    continue
                rows.append({"time": t, "left": left, "right": right,
                             "within_rms": float(np.sqrt(within_sq)),
                             "cross_rms": float(np.sqrt(cross_sq)),
                             "cross_to_within": float(np.sqrt(cross_sq / within_sq))})
    point_z = vectors[np.arange(len(values))[None, :], points["cells"]]
    pair_mask = points["visible"][:, :-1] & points["visible"][:, 1:]
    drift = 1 - np.clip((point_z[:, :-1] * point_z[:, 1:]).sum(-1), -1, 1)
    return {"instance_pairs": len(rows),
            "within_rms": float(np.mean([x["within_rms"] for x in rows])) if rows else None,
            "cross_rms": float(np.mean([x["cross_rms"] for x in rows])) if rows else None,
            "cross_to_within_median": float(np.median([x["cross_to_within"] for x in rows])) if rows else None,
            "first_frame_ratio_median": float(np.median([x["cross_to_within"] for x in rows if x["time"] == 0]))
            if any(x["time"] == 0 for x in rows) else None,
            "visible_point_cosine_drift": float(drift[pair_mask].mean()) if pair_mask.any() else None,
            "first_pair_cosine_drift": float(drift[:, 0][pair_mask[:, 0]].mean()) if pair_mask[:, 0].any() else None,
            "first_pair_point_count": int(pair_mask[:, 0].sum())}


def render_row(images, title, width=320):
    font, small = ImageFont.truetype(FONT, 16), ImageFont.truetype(FONT, 13)
    gap = 8
    canvas = Image.new("RGB", (4 * width + 5 * gap, width + 100), (240, 242, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 7), title, font=font, fill="black")
    for col, (caption, pixels) in enumerate(zip(CAPTIONS, images)):
        x = gap + col * (width + gap)
        draw.text((x, 34), caption, font=small, fill="black")
        canvas.paste(Image.fromarray(pixels).resize((width, width), Image.Resampling.NEAREST), (x, 56))
    draw.text((gap, width + 66), "Unit 128-D K -> shared PCA3; one fixed color scale for every time/mode/camera; no clusters or GT colors.",
              font=small, fill="black")
    return canvas


def stack_rows(rows):
    canvas = Image.new("RGB", (rows[0].width, sum(x.height for x in rows)), "white")
    offset = 0
    for row in rows:
        canvas.paste(row, (0, offset))
        offset += row.height
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/tracks")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--train-episodes-per-suite", type=int, default=3)
    parser.add_argument("--val-episodes-per-suite", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layer, head = int(payload["layer"]), int(payload["head"])
    _, train_rows, val_rows = split_episodes(args.video_root, args.seed)
    index_args = SimpleNamespace(model=args.model, size=512, skip_frames=8, max_frames=129,
                                 windows_per_episode=1, points_per_subject=24, min_motion=.02)
    train_rows = select_rows(train_rows, args.train_episodes_per_suite)
    val_rows = select_rows(val_rows, args.val_episodes_per_suite)
    if {x["episode"] for x in train_rows} & {x["episode"] for x in val_rows}:
        raise RuntimeError("Train and validation episodes overlap")
    train_cases = make_case_index(args.video_root, args.tracks_root, train_rows, list(CAMERAS), index_args, device, args.seed)
    val_cases = make_case_index(args.video_root, args.tracks_root, val_rows, list(CAMERAS), index_args, device, args.seed)
    text = load_empty_text(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    install_k_lora(tower, layer, head, int(payload["rank"]), float(payload["alpha"]))
    expected = {k for k in tower.state_dict() if "lora_" in k}
    if set(payload["lora"]) != expected:
        raise RuntimeError("Checkpoint LoRA keys do not match the selected head")
    tower.load_state_dict(payload["lora"], strict=False)
    tower.eval().requires_grad_(False)
    zero_latents = {}

    def extract(case):
        controls = make_controls(case, args.video_root, args.model, vae, zero_latents, device, 512,
                                 args.seed + case["episode"] * 37 + case["start_frame"])
        with torch.inference_mode():
            keys = capture_k(tower, layer, len(case["times"]), 32, controls["noise"],
                             controls["control"], text, 1000, device)[head].float().cpu().numpy()
        if not np.isfinite(keys).all():
            raise RuntimeError(f"Non-finite K: {case['name']}")
        print(json.dumps({"stage": "captured", "case": case["name"], "shape": list(keys.shape)}), flush=True)
        return keys

    train_keys = [extract(c) for c in train_cases]
    means, stds, counts = fit_statistics(train_keys)
    np.savez(args.output / "time_channel_calibration.npz", means=means, stds=stds, counts=counts,
             checkpoint=str(args.checkpoint), train_cases=np.asarray([c["name"] for c in train_cases]))
    # Display projection and range use training data only, with no frame-wise rescaling.
    samples = np.concatenate([
        unit(transformed(keys, means, stds, mode))[:, ::32].reshape(-1, keys.shape[-1])
        for keys in train_keys for mode in MODES
    ])
    pca = PCA(n_components=3, svd_solver="randomized", random_state=args.seed).fit(samples)
    low, high = np.percentile(pca.transform(samples), [1, 99], axis=0)
    np.savez(args.output / "fixed_display_projection.npz", components=pca.components_, mean=pca.mean_,
             color_low=low, color_high=high, explained_variance=pca.explained_variance_ratio_)
    report = {"checkpoint": str(args.checkpoint), "step": payload["step"], "layer": layer, "head": head,
              "train_cases": [c["name"] for c in train_cases], "val_cases": [c["name"] for c in val_cases],
              "protocol": "train-only per-time mean/diagonal std; no full covariance whitening; 128-D retained",
              "display": "unit K; PCA3 and 1-99% limits fitted on training cases, fixed across all modes and times",
              "metrics": "unique unambiguous token cells; pairwise RMS; cross/within is larger-is-better",
              "scope": "same calibration subset as the previous numerical experiment; four suites, limited tasks",
              "per_case": []}
    overview = {camera: [] for camera in CAMERAS}
    for case in val_cases:
        keys = extract(case)
        frame_ids = np.asarray(case["times"])
        original_video = read_video(args.video_root / f"episode_{case['episode']:05d}" / f"{case['camera']}.mp4")
        video = original_video[frame_ids]
        modes = {mode: transformed(keys, means, stds, mode) for mode in MODES}
        metrics = {mode: geometry(values, case["points"]) for mode, values in modes.items()}
        report["per_case"].append({"name": case["name"], "suite": case["suite"], "metrics": metrics})
        np.savez_compressed(args.output / f"{case['name']}_full_K.npz", k=keys.astype(np.float16),
                            frame_indices=frame_ids, cells=case["points"]["cells"],
                            visible=case["points"]["visible"], labels=case["points"]["labels"])
        fields = {}
        for mode, values in modes.items():
            projection = pca.transform(unit(values).reshape(-1, values.shape[-1]))
            colors = np.clip((projection - low) / np.maximum(high - low, 1e-8), 0, 1)
            fields[mode] = (255 * colors.reshape(len(values), 32, 32, 3)).astype(np.uint8)

        def row(t, overlay):
            rgb = Image.fromarray(video[t]).resize((320, 320), Image.Resampling.BILINEAR)
            panels = [np.asarray(rgb)]
            for mode in MODES:
                field = Image.fromarray(fields[mode][t]).resize((320, 320), Image.Resampling.NEAREST)
                panels.append(np.asarray(Image.blend(rgb, field, .72)) if overlay else np.asarray(field))
            return render_row(panels, f"{case['suite']} | {case['name']} | frame {frame_ids[t]} | step {payload['step']}")

        selected = [0, len(keys) // 2, len(keys) - 1]
        for overlay, suffix in [(True, "overlay"), (False, "pure_K")]:
            stack_rows([row(t, overlay) for t in selected]).save(args.output / f"{case['name']}_{suffix}.jpg", quality=95)
        early = list(range(min(4, len(keys))))
        stack_rows([row(t, False) for t in early]).save(args.output / f"{case['name']}_early_frames.jpg", quality=95)
        overview[case["camera"]].append(row(0, False))
        write_video(args.output / f"{case['name']}_fixed_scale.mp4",
                    np.stack([np.asarray(row(t, True)) for t in range(len(keys))]), fps=5)
        print(json.dumps({"stage": "rendered", "case": case["name"], "metrics": metrics}), flush=True)
    for camera, rows in overview.items():
        stack_rows(rows).save(args.output / f"all_suites_{camera}_frame8.jpg", quality=95)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"stage": "done", "output": str(args.output), "cases": len(val_cases)}), flush=True)


if __name__ == "__main__":
    main()
