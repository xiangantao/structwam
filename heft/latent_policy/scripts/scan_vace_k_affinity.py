#!/usr/bin/env python3
"""Rank and render train-free K affinity without clustering or fitted thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def token_labels(pixels, count, grid):
    height, width = pixels.shape
    if height % grid or width % grid:
        raise ValueError("Image resolution must be divisible by token grid")
    blocks = pixels.reshape(grid, height // grid, grid, width // grid).transpose(0, 2, 1, 3)
    frequencies = np.stack([(blocks == label).mean((2, 3)) for label in range(count)])
    labels = frequencies.argmax(0).astype(int)
    labels[frequencies.max(0) < .75] = -1
    return labels.ravel()


def queries_for_labels(labels, per_object):
    queries, objects = [], []
    for label in np.unique(labels[labels >= 0]):
        cells = np.flatnonzero(labels == label)
        if len(cells) < 2:
            continue
        selected = cells[np.unique(np.linspace(0, len(cells) - 1, min(per_object, len(cells))).astype(int))]
        queries.extend(selected.tolist())
        objects.extend([int(label)] * len(selected))
    return np.asarray(queries, int), np.asarray(objects, int)


def score_maps(maps, labels, queries, objects, names):
    result = {}
    for label in np.unique(objects):
        rows = []
        for row in np.flatnonzero(objects == label):
            valid = (labels >= 0) & (np.arange(len(labels)) != queries[row])
            target = labels[valid] == label
            scores = maps[row, valid]
            if not target.any() or target.all():
                continue
            positives = maps[row, (labels == label) & (np.arange(len(labels)) != queries[row])]
            other = (labels >= 0) & (labels != label)
            other_objects = other & (labels != 0)
            background = (labels == 0) & (labels != label)
            rows.append({
                "ap": float(average_precision_score(target, scores)),
                "positive_fraction": float(target.mean()),
                "top1_same_object": float(target[scores.argmax()]),
                "same_mean_cosine": float(positives.mean()),
                "other_mean_cosine": float(maps[row, other].mean()),
                "background_mean_cosine": float(maps[row, background].mean()) if background.any() else None,
                "other_objects_mean_cosine": float(maps[row, other_objects].mean()) if other_objects.any() else None,
            })
        if rows:
            result[str(names[label])] = {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
                                       if any(row[key] is not None for row in rows) else None
                                       for key in rows[0]}
            result[str(names[label])]["query_count"] = len(rows)
    return result


def affinity_features(keys, frame, mode, head=None):
    values = keys[:, frame].astype(np.float32)
    if mode == "centered":
        values -= values.mean(1, keepdims=True)
    if head is not None:
        return normalize(values[head])
    return normalize(values.transpose(1, 0, 2).reshape(values.shape[1], -1))


def summarize(records):
    grouped = {}
    for row in records:
        grouped.setdefault(row["candidate"], []).append(row)
    ranking = []
    for candidate, values in grouped.items():
        case_values, per_object = {}, {}
        for row in values:
            foreground = [metrics["ap"] for name, metrics in row["objects"].items()
                          if name not in {"background", "robot_arm"}]
            if foreground:
                case_values.setdefault(row["case"], []).append(float(np.mean(foreground)))
            for name, metrics in row["objects"].items():
                per_object.setdefault(name, []).append(metrics)
        groups = {}
        for group in ("objects", "gripper", "background", "robot_arm"):
            metrics = [row for name, rows in per_object.items()
                       if (name == group if group != "objects"
                           else name not in {"background", "gripper", "robot_arm"})
                       for row in rows]
            def mean_metric(key):
                values = [row[key] for row in metrics if row.get(key) is not None]
                return float(np.mean(values)) if values else None
            groups[group] = {key: mean_metric(key) for key in (
                "ap", "same_mean_cosine", "other_mean_cosine",
                "background_mean_cosine", "other_objects_mean_cosine",
            )}
        ranking.append({
            "candidate": candidate,
            "foreground_ap": float(np.mean([np.mean(v) for v in case_values.values()])) if case_values else None,
            "by_case": {name: float(np.mean(v)) for name, v in case_values.items()},
            "groups": groups,
        })
    return sorted(ranking, key=lambda row: -(row["foreground_ap"] or 0))


def title_panel(title, pixels, width=384):
    image = Image.new("RGB", (width, width + 42), "white")
    image.paste(Image.fromarray(pixels).resize((width, width), Image.Resampling.NEAREST), (0, 42))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    ImageDraw.Draw(image).text((9, 11), title, font=font, fill=(25, 25, 25))
    return image


def heat_panel(rgb, scores, cell, grid, label, candidate):
    resized = cv2.resize(scores.reshape(grid, grid), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    intensity = np.clip((resized + 1) * 127.5, 0, 255).astype(np.uint8)
    colors = cv2.cvtColor(cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    overlay = (rgb * .50 + colors * .50).astype(np.uint8)
    point = ((int(cell) % grid + .5) * rgb.shape[1] / grid,
             (int(cell) // grid + .5) * rgb.shape[0] / grid)
    cv2.drawMarker(overlay, tuple(np.rint(point).astype(int)), (0, 0, 0), cv2.MARKER_CROSS, 13, 3)
    cv2.drawMarker(overlay, tuple(np.rint(point).astype(int)), (255, 255, 255), cv2.MARKER_CROSS, 11, 1)
    return title_panel(f"{label[:22]} | {candidate}", overlay)


def render_case(path, candidate, output):
    stem, mode = candidate.split("/")
    head = int(stem.split("H")[1]) if "H" in stem else None
    layer = int(stem.split("H" if head is not None else "_")[0].removeprefix("T"))
    mean_heads = stem.endswith("_mean12")
    with np.load(path) as source:
        keys = source[f"layer_{layer}_k"]
        names, frames, rgb = source["label_names"], source["frame_indices"], source["rgb"]
        pixels = source["instance_labels"]
    grid = int(round(keys.shape[-2] ** .5))
    matrices = []
    for frame, frame_index in enumerate(frames):
        features = affinity_features(keys, frame, mode, head)
        matrix = features @ features.T
        if mean_heads:
            matrix = np.mean([affinity_features(keys, frame, mode, h) @ affinity_features(keys, frame, mode, h).T
                              for h in range(keys.shape[0])], axis=0)
        matrices.append(matrix)
        labels = token_labels(pixels[frame], len(names), grid)
        targets = []
        for category in ("gripper", "background"):
            if category in names:
                targets.append(list(names).index(category))
        targets += [i for i, name in enumerate(names) if name not in {"gripper", "background", "robot_arm"}]
        targets = [label for label in targets if np.count_nonzero(labels == label) >= 2][:5]
        marked = rgb[frame].copy()
        selected = []
        for index, label in enumerate(targets):
            mask = (labels.reshape(grid, grid) == label).astype(np.uint8)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            cell = int(distance.argmax())
            selected.append((label, cell))
            point = (int((cell % grid + .5) * 256 / grid), int((cell // grid + .5) * 256 / grid))
            cv2.drawMarker(marked, point, (255, 255, 255), cv2.MARKER_CROSS, 9, 1)
            cv2.putText(marked, str(index + 1), point, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        panels = [title_panel(f"RGB | frame {frame_index} | query locations", marked)]
        panels += [heat_panel(rgb[frame], matrix[cell], cell, grid, f"{index+1}: {names[label]}", candidate)
                   for index, (label, cell) in enumerate(selected)]
        while len(panels) < 6:
            panels.append(title_panel("Not visible", rgb[frame]))
        width, height, gap = panels[0].width, panels[0].height, 10
        sheet = Image.new("RGB", (3 * width + 4 * gap, 2 * height + 3 * gap + 48), (235, 237, 240))
        for index, panel in enumerate(panels):
            sheet.paste(panel, (gap + (index % 3) * (width + gap), gap + (index // 3) * (height + gap)))
        bar = np.tile(np.arange(256, dtype=np.uint8), (12, 1))
        bar = cv2.cvtColor(cv2.applyColorMap(bar, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(bar).resize((350, 12)), (24, sheet.height - 36))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        ImageDraw.Draw(sheet).text((390, sheet.height - 37), "Cosine: -1 (blue) to +1 (red); fixed scale; GT selects display queries only",
                                  font=font, fill=(30, 30, 30))
        sheet.save(output / f"{path.stem}_{candidate.replace('/', '_')}_frame{frame_index:04d}.jpg", quality=95)
    np.savez(output / f"{path.stem}_{candidate.replace('/', '_')}_affinity.npz",
             frame_indices=frames, cosine_affinity=np.asarray(matrices, np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries-per-object", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    records = []
    cases = json.loads((args.input / "manifest.json").read_text())["cases"]
    for case in cases:
        path = args.input / f"{case['name']}.npz"
        with np.load(path) as source:
            names, frames = source["label_names"], source["frame_indices"]
            pixels = source["instance_labels"]
            for layer in range(case["layers"]):
                keys = source[f"layer_{layer}_k"]
                grid = case["grid"]
                for frame, frame_index in enumerate(frames):
                    labels = token_labels(pixels[frame], len(names), grid)
                    queries, objects = queries_for_labels(labels, args.queries_per_object)
                    for mode in ("raw", "centered"):
                        values = keys[:, frame].astype(np.float32)
                        if mode == "centered":
                            values -= values.mean(1, keepdims=True)
                        tensor = torch.as_tensor(normalize(values), device=device)
                        head_maps = (tensor[:, queries] @ tensor.transpose(1, 2)).cpu().numpy()
                        whole = normalize(values.transpose(1, 0, 2).reshape(grid * grid, -1))
                        whole = torch.as_tensor(whole, device=device)
                        candidates = [(f"T{layer}H{h}/{mode}", head_maps[h]) for h in range(keys.shape[0])]
                        candidates += [(f"T{layer}_all12/{mode}", (whole[queries] @ whole.T).cpu().numpy()),
                                       (f"T{layer}_mean12/{mode}", head_maps.mean(0))]
                        for candidate, maps in candidates:
                            records.append({"case": case["name"], "suite": case["suite"], "frame": int(frame_index),
                                            "candidate": candidate,
                                            "objects": score_maps(maps, labels, queries, objects, names)})
                print(json.dumps({"case": case["name"], "layer": layer, "scored": True}), flush=True)
        (args.output / "per_frame_scores.json").write_text(json.dumps(records, indent=2))
    ranking = summarize(records)
    report = {
        "protocol": "Frozen empty-prompt control-tower pre-RoPE K; K-K cosine; no clustering, sigmoid fit, mask threshold or per-heatmap rescaling",
        "cases": cases, "query_selection": "Up to four spatially distributed GT-labelled tokens per visible instance for diagnostic scoring; self matches excluded",
        "gt_scope": "Complete simulator labels, including background; mixed token boundaries below 75% purity ignored for scoring only",
        "normalization": "raw=L2 only; centered=subtract full-frame per-channel mean then L2",
        "ranking": ranking,
    }
    (args.output / "ranking.json").write_text(json.dumps(report, indent=2))
    singles = [row for row in ranking if "H" in row["candidate"]]
    whole = [row for row in ranking if "_all12" in row["candidate"]]
    selected = list(dict.fromkeys([singles[0]["candidate"], whole[0]["candidate"], "T5_all12/raw"]))
    print(json.dumps({"top": ranking[:10], "render": selected}), flush=True)
    for case in cases:
        for candidate in selected:
            render_case(args.input / f"{case['name']}.npz", candidate, args.output)
    print(json.dumps({"done": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
