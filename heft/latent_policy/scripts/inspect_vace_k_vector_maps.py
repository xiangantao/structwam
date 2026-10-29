#!/usr/bin/env python3
"""Inspect full-dimensional K instance geometry and directly color its vectors."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA


def token_labels(pixels, count, grid):
    height, width = pixels.shape
    cells = pixels.reshape(grid, height // grid, grid, width // grid).transpose(0, 2, 1, 3)
    frequencies = np.stack([(cells == label).mean((2, 3)) for label in range(count)])
    labels = frequencies.argmax(0).astype(int)
    labels[frequencies.max(0) < .75] = -1
    return labels.ravel()


def prepare(values, mode):
    if mode == "unit":
        return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-6)
    return values


def measure_vectors(features, labels, names):
    moments = {}
    for label in np.unique(labels[labels >= 0]):
        points = features[labels == label].astype(np.float64)
        if len(points) < 2:
            continue
        mean = points.mean(0)
        variance = np.square(points - mean).sum(1).mean()
        moments[int(label)] = {"mean": mean, "variance": float(variance), "count": len(points)}
    foreground = [label for label in moments if names[label] not in {"background", "robot_arm"}]
    pairs, object_rows = [], []
    for left, right in itertools.combinations(moments, 2):
        a, b = moments[left], moments[right]
        center_sq = float(np.square(a["mean"] - b["mean"]).sum())
        same_sq = (2 * a["count"] / (a["count"] - 1) * a["variance"]
                   + 2 * b["count"] / (b["count"] - 1) * b["variance"]) / 2
        cross_sq = center_sq + a["variance"] + b["variance"]
        pairs.append({
            "left": str(names[left]), "right": str(names[right]),
            "both_foreground": left in foreground and right in foreground,
            "same_instance_rms_distance": float(np.sqrt(same_sq)),
            "cross_instance_rms_distance": float(np.sqrt(cross_sq)),
            "cross_to_within_ratio": float(np.sqrt(cross_sq / max(same_sq, 1e-12))),
            "center_to_radii_ratio": float(np.sqrt(center_sq) / max(np.sqrt(a["variance"]) + np.sqrt(b["variance"]), 1e-6)),
        })
    for label in foreground:
        local = [row for row in pairs if str(names[label]) in {row["left"], row["right"]} and row["both_foreground"]]
        closest = min(local, key=lambda row: row["cross_to_within_ratio"]) if local else None
        obj = moments[label]
        object_rows.append({"name": str(names[label]), "tokens": obj["count"],
                            "same_instance_rms_distance": float(np.sqrt(2 * obj["count"] / (obj["count"] - 1) * obj["variance"])),
                            "closest_foreground": closest})
    valid = [row for row in object_rows if row["closest_foreground"] is not None]
    return {
        "nearest_foreground_cross_to_within": float(np.mean([row["closest_foreground"]["cross_to_within_ratio"] for row in valid])) if valid else None,
        "nearest_foreground_center_to_radii": float(np.mean([row["closest_foreground"]["center_to_radii_ratio"] for row in valid])) if valid else None,
        "objects": object_rows, "pairs": pairs,
    }


def rank(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row["candidate"], {}).setdefault(row["case"], []).append(row)
    rankings = []
    for candidate, cases in grouped.items():
        scores, centers = {}, {}
        for name, rows in cases.items():
            valid = [row for row in rows if row["metrics"]["nearest_foreground_cross_to_within"] is not None]
            if valid:
                scores[name] = float(np.mean([row["metrics"]["nearest_foreground_cross_to_within"] for row in valid]))
                centers[name] = float(np.mean([row["metrics"]["nearest_foreground_center_to_radii"] for row in valid]))
        rankings.append({"candidate": candidate, "distance_ratio": float(np.mean(list(scores.values()))),
                         "center_to_radii_ratio": float(np.mean(list(centers.values()))), "by_case": scores})
    return sorted(rankings, key=lambda row: -row["distance_ratio"])


def parse_candidate(candidate):
    stem, mode = candidate.split("/")
    layer = int(stem.split("H" if "H" in stem else "_")[0][1:])
    head = int(stem.split("H")[1]) if "H" in stem else None
    return layer, head, mode


def features_for(keys, frame, head, mode):
    values = keys[:, frame].astype(np.float32)
    values = values[head] if head is not None else values.transpose(1, 0, 2).reshape(values.shape[1], -1)
    return prepare(values, mode)


def render_candidates(root, output, cases, candidates):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    maps = {case["name"]: [] for case in cases}
    for candidate in candidates:
        layer, head, mode = parse_candidate(candidate)
        arrays, references = [], []
        for case in cases:
            with np.load(root / f"{case['name']}.npz") as source:
                keys = source[f"layer_{layer}_k"]
                arrays.append([features_for(keys, frame, head, mode) for frame in range(3)])
                references.extend(values - values.mean(0) for values in arrays[-1])
        # One unsupervised color projection shared across cases and all times.
        reference = np.concatenate(references)
        pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(reference)
        color_reference = pca.transform(reference)
        low, high = np.percentile(color_reference, [.5, 99.5], axis=0)
        np.savez(output / f"{candidate.replace('/', '_')}_display_only.npz",
                 projection=pca.components_, mean=pca.mean_, color_low=low, color_high=high,
                 subtract_frame_mean=True,
                 explained_variance_ratio=pca.explained_variance_ratio_)
        print(json.dumps({"color_projection": candidate, "explained_variance": pca.explained_variance_ratio_.tolist()}), flush=True)
        for case, frames in zip(cases, arrays):
            color_frames = []
            for values in frames:
                projected = pca.transform(values - values.mean(0))
                colors = np.clip((projected - low) / np.maximum(high - low, 1e-6), 0, 1)
                color_frames.append((colors.reshape(case["grid"], case["grid"], 3) * 255).astype(np.uint8))
            maps[case["name"]].append((candidate, color_frames))
            if candidate in candidates[:2]:
                np.savez(output / f"{case['name']}_{candidate.replace('/', '_')}_K_map.npz",
                         k=np.stack(frames).reshape(3, case["grid"], case["grid"], -1).astype(np.float16),
                         frame_indices=np.asarray(case["frame_indices"]))
    for case in cases:
        with np.load(root / f"{case['name']}.npz") as source:
            rgb, frames = source["rgb"], source["frame_indices"]
        width, gap = 320, 8
        rows = 1 + len(candidates)
        sheet = Image.new("RGB", (3 * width + 4 * gap, rows * (width + 40 + gap) + 100), (238, 240, 242))
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), f"{case['name']} | direct K vector colors | no clustering", font=font, fill="black")
        contents = [("RGB", list(rgb))]
        for candidate, colors in maps[case["name"]]:
            contents.append((candidate, colors))
        for row, (name, images) in enumerate(contents):
            for column, pixels in enumerate(images):
                x, y = gap + column * (width + gap), 42 + row * (width + 40 + gap)
                draw.rectangle((x, y, x + width, y + width + 40), fill="white")
                draw.text((x + 7, y + 10), f"{name} | RGB frame {frames[column]}", font=small, fill="black")
                sheet.paste(Image.fromarray(pixels).resize((width, width), Image.Resampling.NEAREST), (x, y + 40))
        draw.text((10, sheet.height - 42), "Full D K retained. Display: frame-mean translation (distances unchanged), then fixed PCA3.", font=small, fill="black")
        draw.text((10, sheet.height - 22), "No GT, RGB, centers or preset object count enter vector colors. Similar colors alone do not prove grouping.", font=small, fill="black")
        sheet.save(output / f"{case['name']}_direct_K.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.render_only)
    cases = json.loads((args.input / "manifest.json").read_text())["cases"]
    if args.render_only:
        path = args.output / "ranking.json"
        report = json.loads(path.read_text())
        render_candidates(args.input, args.output, cases, report["selected"])
        report["color_display"] = "Display removes frame mean without changing within-frame distances; fixed label-free PCA3 and color limits fitted on all 18 frames; original K retained"
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps({"done": True, "render_only": True}), flush=True)
        return
    records = []
    invariance_error = 0.
    for case in cases:
        with np.load(args.input / f"{case['name']}.npz") as source:
            names, pixels, frames = source["label_names"], source["instance_labels"], source["frame_indices"]
            labels = [token_labels(p, len(names), case["grid"]) for p in pixels]
            for layer in range(case["layers"]):
                keys = source[f"layer_{layer}_k"]
                for frame in range(3):
                    for head in [*range(12), None]:
                        stem = f"T{layer}H{head}" if head is not None else f"T{layer}_all12"
                        for mode in ("native", "unit"):
                            values = features_for(keys, frame, head, mode)
                            metrics = measure_vectors(values, labels[frame], names)
                            records.append({"candidate": f"{stem}/{mode}", "case": case["name"],
                                            "frame": int(frames[frame]), "metrics": metrics})
                        if layer == 13 and head == 8:
                            values = features_for(keys, frame, head, "native").astype(np.float64)
                            centered = values - values.mean(0)
                            original = measure_vectors(values, labels[frame], names)
                            shifted = measure_vectors(centered, labels[frame], names)
                            for a, b in zip(original["pairs"], shifted["pairs"]):
                                invariance_error = max(invariance_error, abs(a["cross_instance_rms_distance"] - b["cross_instance_rms_distance"]))
                print(json.dumps({"case": case["name"], "layer": layer, "direct_vectors_scored": True}), flush=True)
    rankings = rank(records)
    native = [row for row in rankings if row["candidate"].endswith("/native")]
    single = [row for row in native if "H" in row["candidate"]][0]
    whole = [row for row in native if "_all12" in row["candidate"]][0]
    selected = list(dict.fromkeys([single["candidate"], whole["candidate"], "T13H8/native", "T5_all12/native"]))
    (args.output / "scores.json").write_text(json.dumps(records, indent=2))
    (args.output / "ranking.json").write_text(json.dumps({
        "protocol": "Direct full-dimensional K vectors; no K-K or Q-K token matrices, no clustering, no preset cluster centers/count",
        "metrics": "Exact RMS Euclidean distances from per-instance sufficient statistics; GT used only for geometry scoring. Ratio >1 means cross-instance RMS exceeds within-instance RMS; it is not a segmentation score.",
        "color_display": "Display removes frame mean without changing within-frame distances; fixed label-free PCA3 and color limits fitted on all 18 frames; original K retained",
        "centering_distance_invariance_max_error": invariance_error,
        "selected": selected, "ranking": rankings,
    }, indent=2))
    print(json.dumps({"best_single_native": single, "best_all_native": whole,
                      "centering_distance_invariance_max_error": invariance_error, "render": selected}), flush=True)
    render_candidates(args.input, args.output, cases, selected)
    print(json.dumps({"done": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
