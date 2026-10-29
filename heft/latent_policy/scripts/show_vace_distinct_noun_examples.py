#!/usr/bin/env python3
"""Inspect fixed-head noun maps and render same-frame object-pair examples."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score

from evaluate_vace_prompt_k_guidance import CASE_PHRASES, FONT, ROOT, evaluation_masks, read_video


def heat_rgb(values, limits, shape):
    lo, hi = limits
    normalized = np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1)
    normalized = cv2.resize(normalized, shape, interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]


def render(video_frame, heatmaps, time_index, phrases, pair, limits, title, row):
    size, gap, width = 320, 16, 1024
    canvas = Image.new("RGB", (width, 806), "#f5f6f8")
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 14)
    draw.text((gap, 10), title, fill="black", font=font)
    draw.text((gap, 38), "Fixed heads: T6H10 + T3H1 + T6H0; empty-prompt K: T13H11; no masks used to produce heatmaps.", fill="black", font=small)
    rgb = cv2.resize(video_frame, (size, size), interpolation=cv2.INTER_LINEAR)
    draw.text((gap, 70), "Observed RGB", fill="black", font=font)
    canvas.paste(Image.fromarray(rgb), (gap, 100))
    for col, p in enumerate(pair, 1):
        x = gap + col * (size + gap)
        heat = heat_rgb(heatmaps[p, time_index], limits, (size, size))
        overlay = np.round(.5 * rgb + .5 * heat).astype(np.uint8)
        draw.text((x, 70), f'"{phrases[p]}" + K guidance', fill="black", font=font)
        canvas.paste(Image.fromarray(overlay), (x, 100))
        draw.text((x, 434), "Same heatmap, without RGB", fill="black", font=small)
        canvas.paste(Image.fromarray(heat), (x, 458))
    draw.text((gap, 434), "Relative noun score", fill="black", font=font)
    for y in range(160):
        value = 1 - y / 159
        color = cv2.applyColorMap(np.array([[round(value * 255)]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0, ::-1]
        draw.line((gap, 472 + y, gap + 24, 472 + y), fill=tuple(int(c) for c in color))
    draw.text((gap + 34, 468), f"{limits[1]:.3f}", fill="black", font=small)
    draw.text((gap + 34, 613), f"{limits[0]:.3f}", fill="black", font=small)
    text = ["One shared color scale for all nouns", "and all frames in this episode.", "Brighter = higher noun score.", "", "This is not a segmentation mask.", "Example selected after evaluation;"]
    for i, line in enumerate(text):
        draw.text((gap, 648 + i * 18), line, fill="black", font=small)
    draw.text((gap, 784), f"Surface-cell mean scores: A on A={row['aa']:.3f}, A on B={row['ab']:.3f}; B on B={row['bb']:.3f}, B on A={row['ba']:.3f}.", fill="black", font=small)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "runs/vace_prompt_k_guidance_20261006_relative")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    candidates, fields = [], {}
    for directory in sorted(args.input.glob("episode_*_agentview")):
        episode = int(directory.name.split("_")[1])
        with np.load(directory / "heatmaps.npz") as z:
            heatmaps, times, phrases = z["k_guided"], z["times"], z["phrases"].astype(str).tolist()
        masks = evaluation_masks(ROOT / f"runs/vace_tracking_all_suites_20261004/tracks/episode_{episode:05d}_agentview_visible_surface.npz", times, 32, 256)
        objects = CASE_PHRASES[episode]
        fields[episode] = (heatmaps, times, phrases)
        for ti, frame in enumerate(times):
            for noun_a, noun_b in itertools.combinations(objects, 2):
                a, b = phrases.index(noun_a), phrases.index(noun_b)
                mask_a, mask_b = masks[objects[noun_a]][ti], masks[objects[noun_b]][ti]
                if mask_a.sum() < 3 or mask_b.sum() < 3:
                    continue
                aa, ab = float(heatmaps[a, ti][mask_a].mean()), float(heatmaps[a, ti][mask_b].mean())
                bb, ba = float(heatmaps[b, ti][mask_b].mean()), float(heatmaps[b, ti][mask_a].mean())
                target = np.r_[np.ones(mask_a.sum()), np.zeros(mask_b.sum())]
                auc_a = float(roc_auc_score(target, np.r_[heatmaps[a, ti][mask_a], heatmaps[a, ti][mask_b]]))
                auc_b = float(roc_auc_score(1 - target, np.r_[heatmaps[b, ti][mask_a], heatmaps[b, ti][mask_b]]))
                # Select qualitative examples only; inference heads/maps remain unchanged.
                candidates.append({"episode": episode, "frame": int(frame), "time_index": ti,
                                   "pair": [a, b], "phrases": [noun_a, noun_b],
                                   "aa": aa, "ab": ab, "bb": bb, "ba": ba,
                                   "auc_a": auc_a, "auc_b": auc_b,
                                   "mutual_auc": min(auc_a, auc_b),
                                   "mutual_margin": min(aa - ab, bb - ba)})
    candidates.sort(key=lambda r: (r["mutual_auc"], r["mutual_margin"]), reverse=True)
    (args.output / "ranked_examples.json").write_text(json.dumps(candidates, indent=2))
    selected = []
    for episode in fields:
        best = next((row for row in candidates if row["episode"] == episode), None)
        if best is None:
            continue
        heatmaps, times, phrases = fields[episode]
        video = read_video(ROOT / f"runs/vace_tracking_all_suites_20261004/videos/episode_{episode:05d}/agentview.mp4")
        # Shared full-window scale exposes variation without rescaling each noun/frame.
        limits = tuple(float(v) for v in np.percentile(heatmaps, [2, 98]))
        image = render(video[best["frame"]], heatmaps, best["time_index"], phrases,
                       best["pair"], limits, f"episode{episode:05d} agentview | observed frame {best['frame']}", best)
        path = args.output / f"episode_{episode:05d}_object_pair.jpg"
        image.save(path, quality=95)
        best = {**best, "image": str(path.resolve()), "display_limits": limits}
        selected.append(best)
        print(json.dumps(best), flush=True)
    (args.output / "selected_examples.json").write_text(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
