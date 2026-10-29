#!/usr/bin/env python3
"""Validate one frozen native depth scalar on new tasks, windows, and robot regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

from scan_vace_relative_depth import ANCHORS, CAMERAS, LATENTS, ROOT, dump, log, prepare, scan


GROUP = "main.L16.k_pre_rope"
CHANNEL = 1204
SIGN = -1
HAND_NAMES = {"gripper base", "left finger", "right finger"}
BASELINES = ("minus_image_x", "minus_image_y", "minus_brightness", "anchor_time")


def region_kind(name):
    if name in HAND_NAMES:
        return "gripper"
    if name == "wrist":
        return "wrist"
    if name == "background" or name.startswith(("wooden_cabinet", "flat_stove")):
        return "excluded"
    return "object"


def build_cases(args):
    cases = []
    for suite_offset in (0, 40, 80, 120):
        for task_id in range(2, 10):
            episode = suite_offset + task_id * 4 + 2
            directory = args.data / "videos" / f"episode_{episode:05d}"
            meta = json.loads((directory / "metadata.json").read_text())
            if meta["task_id"] != task_id or meta["num_frames"] < 33:
                raise ValueError(f"Unexpected task or short recording: {directory}")
            last = meta["num_frames"] - 33
            for window, start in (("early", 0), ("middle", last // 2), ("late", last)):
                for camera in args.cameras:
                    cases.append({"episode": episode, "camera": camera, "role": "holdout",
                                  "suite": meta["suite"], "task_id": task_id, "task": meta["task"],
                                  "start": start, "window": window,
                                  "name": f"ep{episode:05d}_{camera}_s{start:04d}"})
    if len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Window cache names must be unique")
    return cases


def rho(x, y, minimum=8):
    if len(x) < minimum or np.ptp(x) <= 1e-7 or np.ptp(y) <= 1e-7:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def pair(left, right, threshold, pair_kind):
    dz = right["camera_z_m"] - left["camera_z_m"]
    if abs(dz) < threshold:
        return None
    differences = [right["signal"] - left["signal"]]
    differences.extend(right[name] - left[name] for name in BASELINES)
    # Exact ties receive half credit; an undefined constant field is not perfect.
    scores = ((np.sign(differences) * np.sign(dz) + 1) / 2).tolist()
    return {"case": left["case"], "episode": left["episode"], "camera": left["camera"],
            "suite": left["suite"], "window": left["window"], "kind": pair_kind,
            "left_object": left["object"], "right_object": right["object"],
            "left_anchor": left["anchor"], "right_anchor": right["anchor"],
            "depth_delta_m": dz, "signal_delta": differences[0], "scores": scores}


def collect(args, cases):
    points, pairs, details, spans = [], [], [], []
    yy, xx = np.mgrid[:32, :32]
    suffix = "" if args.noise_seed == 42 else f"_seed{args.noise_seed}"
    for case in cases:
        with np.load(args.output / "annotations" / f"{case['name']}.npz") as archive:
            annotation = {name: archive[name] for name in archive.files}
        with np.load(args.output / ("preview_fields" + suffix) / f"{case['name']}.npz") as archive:
            signal = archive[GROUP][..., 0].reshape(4, 32, 32) * SIGN
        if not np.isfinite(signal).all():
            raise ValueError(f"Nonfinite scalar for {case['name']}")
        own, dense = [], {kind: [] for kind in ("object", "gripper", "wrist")}
        for t, anchor in enumerate(ANCHORS):
            labels = annotation["labels"][t]
            depth = annotation["depth"][t]
            rgb = cv2.resize(annotation["rgb"][t], (32, 32), interpolation=cv2.INTER_AREA)
            category_pixels = {kind: np.zeros((32, 32), bool) for kind in dense}
            for label in np.unique(labels[labels >= 0]):
                name = str(annotation["label_names"][label])
                kind = region_kind(name)
                mask = labels == label
                if kind == "excluded" or mask.sum() < 3:
                    continue
                category_pixels[kind] |= mask
                row = {**case, "case": case["name"], "object": name, "kind": kind,
                       "anchor": int(anchor), "recorded_frame": case["start"] + int(anchor),
                       "cells": int(mask.sum()), "camera_z_m": float(depth[mask].mean()),
                       "signal": float(signal[t, mask].mean()),
                       "minus_image_x": -float(xx[mask].mean()),
                       "minus_image_y": -float(yy[mask].mean()),
                       "minus_brightness": -float(rgb[mask].mean()) / 255,
                       "anchor_time": int(anchor)}
                own.append(row)
            for kind, mask in category_pixels.items():
                dense[kind].append(rho(signal[t, mask], depth[mask]))
        points.extend(own)
        for kind in dense:
            rows = [p for p in own if p["kind"] == kind]
            details.append({**case, "kind": kind, "region_observations": len(rows),
                            "region_rho": rho([p["signal"] for p in rows], [p["camera_z_m"] for p in rows]),
                            "dense_mean_rho": mean_defined(dense[kind]),
                            "dense_eligible_anchors": sum(v is not None for v in dense[kind]),
                            "baseline_rho": {name: rho([p[name] for p in rows], [p["camera_z_m"] for p in rows])
                                             for name in BASELINES},
                            "replay_mae_max": float(annotation["replay_mae"].max())})
        for name in sorted({p["object"] for p in own}):
            rows = [p for p in own if p["object"] == name]
            depths = [p["camera_z_m"] for p in rows]
            spans.append({**case, "object": name, "kind": rows[0]["kind"],
                          "visible_anchors": len(rows), "depth_span_m": float(np.ptp(depths)),
                          "min_camera_z_m": min(depths), "max_camera_z_m": max(depths)})
        for i, left in enumerate(own):
            for right in own[i + 1:]:
                if left["object"] == right["object"]:
                    row = pair(left, right, 0.02, "temporal_" + left["kind"])
                elif left["anchor"] == right["anchor"]:
                    kinds = {left["kind"], right["kind"]}
                    if kinds == {"object"}:
                        row = pair(left, right, 0.03, "same_anchor_objects")
                    elif kinds == {"object", "gripper"}:
                        row = pair(left, right, 0.03, "same_anchor_object_gripper")
                    elif kinds == {"gripper"}:
                        row = pair(left, right, 0.03, "same_anchor_gripper_parts")
                    else:
                        row = None
                else:
                    row = None
                if row is not None:
                    pairs.append(row)
    return points, pairs, details, spans


def pair_summary(rows):
    if not rows:
        return {"comparisons": 0, "episodes": 0, "accuracy": None, "episode_macro_accuracy": None}
    scores = np.array([row["scores"] for row in rows])
    episodes = sorted({row["episode"] for row in rows})
    episode_scores = np.array([scores[[row["episode"] == ep for row in rows]].mean(0) for ep in episodes])
    rng = np.random.default_rng(9823)
    resampled = episode_scores[rng.integers(0, len(episodes), (2000, len(episodes)))].mean(1)[:, 0]
    return {"comparisons": len(rows), "episodes": len(episodes),
            "correct": int(np.sum(scores[:, 0] == 1)), "incorrect": int(np.sum(scores[:, 0] == 0)),
            "ties": int(np.sum(scores[:, 0] == 0.5)), "accuracy": float(scores[:, 0].mean()),
            "episode_macro_accuracy": float(episode_scores[:, 0].mean()),
            "episode_bootstrap_macro_ci95": np.percentile(resampled, [2.5, 97.5]).tolist(),
            "baseline_accuracy": dict(zip(BASELINES, scores[:, 1:].mean(0).tolist(), strict=True))}


def scope_summary(camera, points, pairs, details, spans):
    points = [p for p in points if p["camera"] == camera]
    pairs = [p for p in pairs if p["camera"] == camera]
    details = [p for p in details if p["camera"] == camera]
    spans = [p for p in spans if p["camera"] == camera]
    result = {"episodes": len({p["episode"] for p in details}),
              "windows": len({p["name"] for p in details}), "regions": {}, "pair_order": {}}
    for kind in ("object", "gripper", "wrist"):
        own = [p for p in points if p["kind"] == kind]
        eligible = [p for p in details if p["kind"] == kind and p["region_rho"] is not None]
        ranges = [p["depth_span_m"] for p in spans if p["kind"] == kind and p["visible_anchors"] >= 2]
        result["regions"][kind] = {
            "observations": len(own), "windows_with_regions": len({p["case"] for p in own}),
            "eligible_rho_windows": len(eligible), "pooled_rho": rho([p["signal"] for p in own], [p["camera_z_m"] for p in own]),
            "mean_window_rho": mean_defined([p["region_rho"] for p in eligible]),
            "median_window_rho": float(np.median([p["region_rho"] for p in eligible])) if eligible else None,
            "mean_dense_anchor_rho": mean_defined([p["dense_mean_rho"] for p in details if p["kind"] == kind]),
            "depth_span_m": {"region_windows": len(ranges),
                             "median": float(np.median(ranges)) if ranges else None,
                             "maximum": max(ranges) if ranges else None,
                             "at_least_2cm": int(np.sum(np.array(ranges) >= 0.02))}}
    for kind in ("temporal_object", "temporal_gripper", "temporal_wrist", "same_anchor_objects",
                 "same_anchor_object_gripper", "same_anchor_gripper_parts"):
        result["pair_order"][kind] = pair_summary([p for p in pairs if p["kind"] == kind])
    for direction, predicate in (("approach", lambda p: p["depth_delta_m"] < 0),
                                 ("retreat", lambda p: p["depth_delta_m"] > 0)):
        result["pair_order"]["temporal_object_" + direction] = pair_summary(
            [p for p in pairs if p["kind"] == "temporal_object" and predicate(p)])
    return result


def evaluate(args, cases):
    points, pairs, details, spans = collect(args, cases)
    summary = {"group": GROUP, "channel": CHANNEL, "head": 9, "head_channel": 52, "sign": SIGN,
               "noise_seed": args.noise_seed, "cases": len(cases), "episodes": len({c["episode"] for c in cases}),
               "tasks": len({(c["suite"], c["task_id"]) for c in cases}), "camera": {}, "by_suite": {}, "by_window": {},
               "baseline_signs": "Image-x/image-y/brightness signs fixed from the previous discovery summary; anchor-time positive. No selection on new data.",
               "limitation": "Visible region means, not material-point depths. Window pairs overlap and are correlated; bootstrap units are episodes. The wrist-selected channel is used unchanged in agentview. No metric calibration or fitted decoder."}
    for camera in args.cameras:
        summary["camera"][camera] = scope_summary(camera, points, pairs, details, spans)
        summary["by_suite"][camera] = {}
        summary["by_window"][camera] = {}
        for key, groups in (("suite", summary["by_suite"][camera]), ("window", summary["by_window"][camera])):
            for value in sorted({c[key] for c in cases}):
                groups[value] = scope_summary(camera, *([p for p in rows if p[key] == value]
                                                        for rows in (points, pairs, details, spans)))
    suffix = "" if args.noise_seed == 42 else f"_seed{args.noise_seed}"
    for name, value in (("summary", summary), ("region_points", points), ("pairs", pairs),
                        ("per_case", details), ("depth_spans", spans)):
        dump(args.output / f"{name}{suffix}.json", value)
    log(stage="validation", cases=len(cases), episodes=summary["episodes"], results=summary["camera"])
    return summary, points, pairs


def render(args, cases, summary, points, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output / "figures"
    output.mkdir(exist_ok=True)
    for camera in args.cameras:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
        for ax, suite in zip(axes.flat, sorted({c["suite"] for c in cases}), strict=True):
            for kind, color, marker in (("object", "#009A8C", "o"), ("gripper", "#D54E56", "+"),
                                        ("wrist", "#7658A2", "x")):
                rows = [p for p in points if p["camera"] == camera and p["suite"] == suite and p["kind"] == kind]
                ax.scatter([p["camera_z_m"] for p in rows], [p["signal"] for p in rows],
                           s=15, alpha=0.5, color=color, marker=marker, label=f"{kind} ({len(rows)})")
            order = summary["by_suite"][camera][suite]["pair_order"]["temporal_object"]
            accuracy = order["accuracy"]
            acc = f"{accuracy:.1%}" if accuracy is not None else "n/a"
            ax.set_title(f"{suite} | temporal objects {acc}, {order['comparisons']} pairs", fontsize=11)
            ax.set_xlabel("GT camera Z (m)")
            ax.set_ylabel("Fixed native scalar: -K[L16,H9,52]")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"New tasks / three fixed windows | {camera}\nNo fitted projection, no per-window normalization", fontsize=12)
        fig.savefig(output / f"{camera}_regional_depth.png", dpi=150)
        plt.close(fig)

    hand_cases = [c for c in cases if c["camera"] == "agentview" and c["task_id"] == 2 and c["window"] == "middle"]
    if hand_cases:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True, constrained_layout=True)
        for ax, case in zip(axes.flat, hand_cases, strict=False):
            for kind, color in (("gripper", "#D54E56"), ("wrist", "#7658A2")):
                own = sorted((p for p in points if p["case"] == case["name"] and p["kind"] == kind),
                             key=lambda p: p["anchor"])
                ax.plot([p["camera_z_m"] for p in own], [p["signal"] for p in own], "o-", color=color, label=kind)
                for p in own:
                    ax.annotate(str(p["recorded_frame"]), (p["camera_z_m"], p["signal"]),
                                xytext=(4, 4), textcoords="offset points", fontsize=8, color=color)
            ax.set_title(f"{case['suite']} | episode {case['episode']}", fontsize=11)
            ax.set_xlabel("GT camera Z (m)")
            ax.set_ylabel("Raw scalar: -K[L16,H9,52]")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
        fig.suptitle("Hand regions in the external camera | fixed task2/middle examples\nLines connect observed anchors; labels are recorded frame numbers. No depth fit.", fontsize=11)
        fig.savefig(output / "agentview_hand_depth_curves.png", dpi=150)
        plt.close(fig)

    # Examples are predetermined task2/middle, not selected by their performance.
    old = args.discovery_output
    inspected = json.loads((old / "inspected_channels.json").read_text())
    channels = []
    for row in inspected:
        if row["group"] == GROUP and row["channel"] not in channels:
            channels.append(row["channel"])
    ci = channels.index(CHANNEL)
    discovered = json.loads((old / "cases.json").read_text())
    fields = []
    for case in discovered:
        if case["role"] == "discovery" and case["camera"] == "robot0_eye_in_hand":
            with np.load(old / "preview_fields" / f"{case['name']}.npz") as archive:
                fields.extend((archive[GROUP][..., ci] * SIGN).ravel().tolist())
    bounds = np.percentile(fields, [2, 98])
    for case in cases:
        if case["task_id"] != 2 or case["window"] != "middle":
            continue
        with np.load(args.output / "annotations" / f"{case['name']}.npz") as archive:
            annotation = {name: archive[name] for name in archive.files}
        with np.load(args.output / "preview_fields" / f"{case['name']}.npz") as archive:
            signal = archive[GROUP][..., 0].reshape(4, 32, 32) * SIGN
        fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
        z_bounds = np.percentile(annotation["depth"], [2, 90])
        for t, anchor in enumerate(ANCHORS):
            ax = axes[0, t]
            ax.imshow(annotation["rgb"][t])
            ax.set_title(f"RGB | frame {case['start'] + anchor}")
            image = axes[1, t].imshow(annotation["depth"][t], cmap="viridis", vmin=z_bounds[0], vmax=z_bounds[1])
            axes[1, t].set_title("GT camera Z")
            feature = axes[2, t].imshow(signal[t], cmap="viridis", vmin=bounds[0], vmax=bounds[1])
            axes[2, t].set_title("Native -K[L16,H9,52]\nFixed discovery scale", fontsize=10)
            for ax in axes[:, t]:
                ax.axis("off")
        fig.colorbar(image, ax=axes[1, :], shrink=0.5, label="Camera Z (m)")
        fig.colorbar(feature, ax=axes[2, :], shrink=0.5, label="Raw scalar r")
        fig.suptitle(f"{case['suite']} | {case['camera']} | episode {case['episode']}\n{case['task']}", fontsize=11)
        fig.savefig(output / f"{case['name']}.jpg", dpi=140)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    parser.add_argument("--discovery-output", type=Path, default=ROOT / "runs/vace_relative_depth_native_20261009")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS))
    parser.add_argument("--size", type=int, choices=(512,), default=512)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--stage", choices=("all", "prepare", "scan", "evaluate"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cases = build_cases(args)
    protocol = {"cases": cases, "candidate": {"group": GROUP, "channel": CHANNEL, "sign": SIGN},
                "selection": "Frozen before expanded validation; new episodes never select channels, signs or thresholds.",
                "sample_selection": "Episode index 2 of tasks2-9 in all suites; early=0, middle=floor((N-33)/2), late=N-33. No motion/score selection.",
                "novelty": "All 32 episodes absent from discovery/previous holdout. Spatial task9 occurred in the old holdout, with a different episode.",
                "observed_rgb_frames": 33, "warmup_copies": 15, "anchor_rgb_frames": ANCHORS.tolist(),
                "anchor_latents": LATENTS.tolist(), "noise_seed": args.noise_seed,
                "input": "Full dataset prompt, real RGB via control, pure main noise at t1000, one frozen forward; no future frames.",
                "temporal_alignment": "Native VAE receptive fields span RGB frames; existing anchor convention, not per-frame samples.",
                "gt_usage": "Simulator depth and visibility define evaluation regions only; never supplied to VACE.",
                "region_min_cells": 3, "label_patch_min_purity": 0.75,
                "temporal_min_depth_delta_m": 0.02, "same_anchor_min_depth_delta_m": 0.03,
                "readout": "Negative mean of native channel1204 in a region, with no calibration or normalization."}
    protocol_path = args.output / f"protocol_seed{args.noise_seed}.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing output has a different extraction protocol; use a fresh directory")
    dump(protocol_path, protocol)
    if args.stage in {"all", "prepare"}:
        prepare(args, cases)
    if args.stage in {"all", "scan"}:
        scan(args, cases, {GROUP: [CHANNEL]})
    if args.stage in {"all", "evaluate"}:
        summary, points, pairs = evaluate(args, cases)
        if args.noise_seed == 42:
            render(args, cases, summary, points, pairs)


if __name__ == "__main__":
    main()
