#!/usr/bin/env python3
"""Compare fixed, training-free processing of a frozen regional camera-depth cue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

from validate_vace_relative_depth_holdout import (
    CHANNEL, GROUP, SIGN, ROOT, mean_defined, pair_summary, region_kind, rho,
)


METHODS = {
    "raw_mean": {"map": "raw", "aggregation": "mean", "alpha": None},
    "region_median": {"map": "raw", "aggregation": "median", "alpha": None},
    "region_trim20": {"map": "raw", "aggregation": "trim20", "alpha": None},
    "spatial_median3": {"map": "median3", "aggregation": "mean", "alpha": None},
    "rgb_bilateral": {"map": "rgb_bilateral", "aggregation": "mean", "alpha": None},
    "rgb_bilateral_ema": {"map": "rgb_bilateral", "aggregation": "mean", "alpha": 2 / 3},
}


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def rgb_joint_bilateral(field, rgb):
    """Smooth a 2D scalar field using uint8 RGB and no segmentation masks."""
    field = np.asarray(field, np.float32)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError("Scalar field must be finite and two-dimensional")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("RGB guide must be an HxWx3 uint8 array")
    rgb = cv2.resize(rgb, (field.shape[1], field.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    radius, sigma_space, sigma_rgb = 2, 1.0, 0.10
    padded = np.pad(field, radius, mode="reflect")
    colors = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    numerator, denominator = np.zeros_like(field), np.zeros_like(field)
    height, width = field.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y, x = radius + dy, radius + dx
            neighbor = colors[y:y + height, x:x + width]
            weight = np.exp(-(dx * dx + dy * dy) / (2 * sigma_space**2)
                            - np.sum((neighbor - rgb)**2, axis=-1) / (2 * sigma_rgb**2))
            numerator += weight * padded[y:y + height, x:x + width]
            denominator += weight
    return numerator / denominator


def aggregate_region(field, mask, mode):
    values = np.asarray(field, np.float32)[np.asarray(mask, bool)]
    if values.size < 3 or not np.isfinite(values).all():
        return None
    if mode == "mean":
        value = float(values.mean())
    elif mode == "median":
        value = float(np.median(values))
    elif mode == "trim20":
        ordered = np.sort(values)
        cut = int(len(ordered) * 0.20)
        value = float(ordered[cut:len(ordered) - cut].mean())
    else:
        raise ValueError(f"Unknown region aggregation: {mode}")
    return {"value": value, "cells": int(values.size),
            "mad": float(np.median(np.abs(values - np.median(values))))}


def discovery_scale(directory):
    rows = json.loads((directory / "inspected_channels.json").read_text())
    channels = []
    for row in rows:
        if row["group"] == GROUP and row["channel"] not in channels:
            channels.append(row["channel"])
    index = channels.index(CHANNEL)
    values = []
    for case in json.loads((directory / "cases.json").read_text()):
        if case["role"] != "discovery" or case["camera"] != "robot0_eye_in_hand":
            continue
        with np.load(directory / "preview_fields" / f"{case['name']}.npz") as archive:
            values.append((archive[GROUP][..., index] * SIGN).ravel())
    values = np.concatenate(values)
    low, center, high = np.percentile(values, [25, 50, 75])
    return {"center": float(center), "iqr": float(high - low),
            "display_range": np.percentile(values, [2, 98]).tolist(),
            "source": "Four original wrist discovery clips; native scalar statistics, no GT labels or depth fit."}


def load_case(source, case):
    with np.load(source / "preview_fields" / f"{case['name']}.npz") as archive:
        raw = archive[GROUP][..., 0].reshape(4, 32, 32) * SIGN
    with np.load(source / "annotations" / f"{case['name']}.npz") as archive:
        annotation = {key: archive[key] for key in archive.files}
    fields = {"raw": raw, "median3": median_filter(raw, size=(1, 3, 3), mode="reflect"),
              "rgb_bilateral": np.stack([rgb_joint_bilateral(field, rgb)
                                          for field, rgb in zip(raw, annotation["rgb"], strict=True)])}
    return annotation, fields


def process(source, cases, scale):
    original = json.loads((source / "region_points.json").read_text())
    by_case = {}
    for row in original:
        by_case.setdefault(row["case"], []).append(row)
    processed, dense = {name: {} for name in METHODS}, []
    for case in cases:
        annotation, fields = load_case(source, case)
        rows = by_case.get(case["name"], [])
        for method, spec in METHODS.items():
            field = fields[spec["map"]]
            own = []
            for row in rows:
                t = row["anchor"] // 8
                label = np.flatnonzero(annotation["label_names"] == row["object"])[0]
                result = aggregate_region(field[t], annotation["labels"][t] == label, spec["aggregation"])
                if result is None:
                    raise ValueError("Processing must retain the original region coverage")
                own.append({**row, "raw_signal": row["signal"], "signal": result["value"],
                            "regional_mad": result["mad"]})
            for name in sorted({r["object"] for r in own}):
                sequence = sorted((r for r in own if r["object"] == name), key=lambda r: r["anchor"])
                previous, first = None, None
                for row in sequence:
                    if spec["alpha"] is not None and previous is not None and row["anchor"] == previous["anchor"] + 8:
                        row["signal"] = spec["alpha"] * row["signal"] + (1 - spec["alpha"]) * previous["signal"]
                    if first is None:
                        first = row["signal"]
                    row["normalized_signal"] = (row["signal"] - scale["center"]) / scale["iqr"]
                    row["change_from_first"] = (row["signal"] - first) / scale["iqr"]
                    row["change_from_previous"] = ((row["signal"] - previous["signal"]) / scale["iqr"]
                                                    if previous is not None and row["anchor"] == previous["anchor"] + 8 else None)
                    row["normalized_mad"] = row["regional_mad"] / scale["iqr"]
                    key = (row["case"], row["object"], row["anchor"])
                    processed[method][key] = row
                    if method == "raw_mean" and abs(row["signal"] - row["raw_signal"]) > 1e-6:
                        raise ValueError("Raw mean baseline changed")
                    previous = row
            if spec["alpha"] is not None:
                continue
            for kind in ("object", "gripper", "wrist"):
                correlations = []
                for t in range(4):
                    mask = np.zeros((32, 32), bool)
                    for label, name in enumerate(annotation["label_names"]):
                        region = annotation["labels"][t] == label
                        if region_kind(str(name)) == kind and region.sum() >= 3:
                            mask |= region
                    correlations.append(rho(field[t, mask], annotation["depth"][t, mask]))
                dense.append({**case, "method": method, "kind": kind,
                              "dense_mean_rho": mean_defined(correlations)})
    return processed, dense


def paired_temporal_gain(rows):
    temporal = [row for row in rows if row["kind"] == "temporal_object"]
    episodes = sorted({row["episode"] for row in temporal})
    if not episodes:
        return {"episodes": 0, "episode_macro_gain_percentage_points": None, "paired_ci95_percentage_points": None}
    gains = np.array([np.mean([row["scores"][0] - row["raw_score"] for row in temporal if row["episode"] == episode])
                      for episode in episodes])
    rng = np.random.default_rng(23441)
    bootstrap = gains[rng.integers(0, len(gains), (5000, len(gains)))].mean(1)
    return {"episodes": len(episodes), "episode_macro_gain_percentage_points": float(gains.mean() * 100),
            "paired_ci95_percentage_points": (np.percentile(bootstrap, [2.5, 97.5]) * 100).tolist()}


def evaluate(source, cases, processed, dense):
    original_pairs = json.loads((source / "pairs.json").read_text())
    summary = {}
    for method, points in processed.items():
        pairs = []
        for row in original_pairs:
            left = points[(row["case"], row["left_object"], row["left_anchor"])]
            right = points[(row["case"], row["right_object"], row["right_anchor"])]
            delta = right["signal"] - left["signal"]
            score = float((np.sign(delta) * np.sign(row["depth_delta_m"]) + 1) / 2)
            pairs.append({**row, "signal_delta": delta, "raw_score": row["scores"][0],
                          "scores": [score, *row["scores"][1:]]})
        summary[method] = {"definition": METHODS[method], "camera": {}}
        for camera in sorted({case["camera"] for case in cases}):
            own = [row for row in points.values() if row["camera"] == camera]
            grouped = {}
            for row in own:
                grouped.setdefault((row["case"], row["kind"]), []).append(row)
            region_stats = {}
            for kind in ("object", "gripper", "wrist"):
                subset = [row for row in own if row["kind"] == kind]
                correlations = [rho([r["signal"] for r in group], [r["camera_z_m"] for r in group])
                                for (_case, category), group in grouped.items() if category == kind]
                region_stats[kind] = {"observations": len(subset),
                    "pooled_rho": rho([r["signal"] for r in subset], [r["camera_z_m"] for r in subset]),
                    "mean_window_rho": mean_defined(correlations),
                    "mean_dense_anchor_rho": mean_defined([r["dense_mean_rho"] for r in dense
                                                            if r["camera"] == camera and r["method"] == method and r["kind"] == kind])}
            eligible = [row for row in pairs if row["camera"] == camera]
            orders = {kind: pair_summary([row for row in eligible if row["kind"] == kind])
                      for kind in ("temporal_object", "temporal_gripper", "temporal_wrist",
                                   "same_anchor_objects", "same_anchor_object_gripper")}
            for name, sign in (("approach", -1), ("retreat", 1)):
                orders["temporal_object_" + name] = pair_summary([row for row in eligible
                    if row["kind"] == "temporal_object" and np.sign(row["depth_delta_m"]) == sign])
            summary[method]["camera"][camera] = {"regions": region_stats, "pair_order": orders,
                "temporal_object_paired_gain_vs_raw": paired_temporal_gain(eligible),
                "by_window": {window: pair_summary([r for r in eligible if r["kind"] == "temporal_object" and r["window"] == window])
                              for window in ("early", "middle", "late")},
                "by_suite": {suite: pair_summary([r for r in eligible if r["kind"] == "temporal_object" and r["suite"] == suite])
                             for suite in sorted({case["suite"] for case in cases})}}
    return summary


def render(source, output, cases, scale):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    for case in cases:
        if case["task_id"] != 2 or case["window"] != "middle" or case["camera"] != "robot0_eye_in_hand":
            continue
        annotation, fields = load_case(source, case)
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
        depth_bounds = np.percentile(annotation["depth"], [2, 90])
        for t in range(4):
            axes[0, t].imshow(annotation["rgb"][t])
            axes[0, t].set_title(f"RGB frame {case['start'] + t * 8}", fontsize=10)
            axes[1, t].imshow(annotation["depth"][t], cmap="viridis", vmin=depth_bounds[0], vmax=depth_bounds[1])
            axes[1, t].set_title("GT camera Z", fontsize=10)
            for row, name in ((2, "raw"), (3, "rgb_bilateral")):
                image = axes[row, t].imshow(fields[name][t], cmap="viridis", vmin=scale["display_range"][0],
                                            vmax=scale["display_range"][1], interpolation="nearest")
                axes[row, t].set_title("Raw native scalar" if name == "raw" else "RGB-guided scalar", fontsize=10)
            for ax in axes[:, t]:
                ax.axis("off")
        fig.colorbar(image, ax=axes[2:, :].ravel(), shrink=0.5, label="Native scalar; one discovery color range")
        fig.suptitle(f"{case['suite']} | episode {case['episode']} | wrist camera\nFixed 5x5 RGB-guided filter, no depth labels in processing; original 32x32 grid.", fontsize=11)
        fig.savefig(figures / f"{case['name']}_before_after.jpg", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/vace_relative_depth_expanded_20261009")
    parser.add_argument("--discovery", type=Path, default=ROOT / "runs/vace_relative_depth_native_20261009")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads((args.source / "protocol_seed42.json").read_text())
    if source_protocol["candidate"] != {"group": GROUP, "channel": CHANNEL, "sign": SIGN}:
        raise ValueError("Source must use the fixed candidate and sign")
    cases = json.loads((args.source / "cases.json").read_text())
    scale = discovery_scale(args.discovery)
    if scale["iqr"] <= 1e-6:
        raise ValueError("Discovery scale has no usable variation")
    protocol = {"source": str(args.source), "channel": CHANNEL, "sign": SIGN, "methods": METHODS,
                "rgb_filter": {"radius_grid_cells": 2, "sigma_space": 1.0, "sigma_rgb": 0.10},
                "selection": "Fixed small processing menu; no new channel/sign search or fitted depth readout.",
                "scale": scale, "inference": "Spatial filter uses scalar and RGB only. Aggregation accepts caller-supplied regions. EMA uses preceding visible anchors only and resets after gaps.",
                "evaluation": "Same 192 previously evaluated windows and GT visibility regions; all original pair denominators retained. This is exploratory processing comparison, not a fresh independent holdout.",
                "output": "Normalized scalar, change from first/previous anchor, and regional MAD. These are unitless signals, not meters or calibrated probabilities.",
                "paired_gain_bootstrap": "5000 paired episode resamples, seed23441; intervals are unadjusted for the processing menu.",
                "dense_ema": "Temporal EMA acts on regional scalars only; no dense-map score is assigned to it."}
    dump(args.output / "protocol.json", protocol)
    processed, dense = process(args.source, cases, scale)
    summary = evaluate(args.source, cases, processed, dense)
    original_summary = json.loads((args.source / "summary.json").read_text())
    for camera in original_summary["camera"]:
        for kind in ("temporal_object", "temporal_gripper", "temporal_wrist", "same_anchor_objects"):
            old = original_summary["camera"][camera]["pair_order"][kind]
            new = summary["raw_mean"]["camera"][camera]["pair_order"][kind]
            if old["comparisons"] != new["comparisons"] or old["accuracy"] != new["accuracy"]:
                raise ValueError(f"Baseline/pair coverage changed: {camera} {kind}")
    dump(args.output / "summary.json", summary)
    dump(args.output / "regional_readouts.json", {method: list(points.values()) for method, points in processed.items()})
    render(args.source, args.output, cases, scale)
    for method, row in summary.items():
        wrist = row["camera"]["robot0_eye_in_hand"]
        print(json.dumps({"method": method, "object_regions": wrist["regions"]["object"],
                          "temporal_object": wrist["pair_order"]["temporal_object"],
                          "same_anchor_objects": wrist["pair_order"]["same_anchor_objects"]}), flush=True)


if __name__ == "__main__":
    main()
