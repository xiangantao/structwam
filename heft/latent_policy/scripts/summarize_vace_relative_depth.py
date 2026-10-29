#!/usr/bin/env python3
"""Summarize the fixed wrist readout and plot raw region values against camera Z."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def weighted_accuracy(rows, key, count_key):
    eligible = [row for row in rows if row.get(count_key, 0) and row.get(key) is not None]
    count = sum(row[count_key] for row in eligible)
    correct = sum((row[key] + 1) * 0.5 * row[count_key] for row in eligible)
    return {"correct": int(round(correct)), "comparisons": count,
            "accuracy": correct / count if count else None}


def metrics_for(path, identifier):
    rows = [row for row in json.loads(path.read_text())
            if row["role"] == "holdout" and row["candidate"] == identifier]
    rho = [row["metrics"]["region_rho"] for row in rows if row["metrics"]["region_rho"] is not None]
    temporal = [row | {"order": row["metrics"]["temporal_order"]} for row in rows]
    return {"episodes": len(rows), "mean_object_region_rho": float(np.mean(rho)),
            "episodes_with_region_rho": len(rho),
            "temporal_object_order": weighted_accuracy(temporal, "order", "temporal_pairs"),
            "same_anchor_object_order": weighted_accuracy(rows, "same_anchor_region_order", "same_anchor_region_pairs"),
            "per_case": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.output
    camera = "robot0_eye_in_hand"
    selected = json.loads((path / "per_view_ranking.json").read_text())[camera][0]
    group, channel, sign = selected["group"], selected["channel"], selected["sign"]
    identifier = f"{group}:{channel}:{sign}:{camera}"
    candidates = json.loads((path / "inspected_channels.json").read_text())
    channels = {}
    for row in candidates:
        target = channels.setdefault(row["group"], [])
        if row["channel"] not in target:
            target.append(row["channel"])
    ci = channels[group].index(channel)
    cases = [case for case in json.loads((path / "cases.json").read_text())
             if case["role"] == "holdout" and case["camera"] == camera]
    points = []
    for case in cases:
        with np.load(path / "annotations" / f"{case['name']}.npz") as archive:
            depth, labels = archive["depth"].reshape(4, -1), archive["labels"].reshape(4, -1)
            foreground, names = archive["foreground"].reshape(4, -1), archive["label_names"]
        with np.load(path / "preview_fields" / f"{case['name']}.npz") as archive:
            values = archive[group][..., ci] * sign
        with np.load(path / "preview_fields_seed99" / f"{case['name']}.npz") as archive:
            other = archive[group][..., ci] * sign
        for t, anchor in enumerate((0, 8, 16, 24)):
            for label in np.unique(labels[t, foreground[t]]):
                name = str(names[label])
                if name in {"gripper base", "left finger", "right finger", "wrist", "background"}:
                    continue
                mask = (labels[t] == label) & foreground[t]
                if mask.sum() < 3:
                    continue
                points.append({"case": case["name"], "suite": case["suite"], "object": name,
                               "anchor": anchor, "camera_z_m": float(depth[t, mask].mean()),
                               "raw_signal_seed42": float(values[t, mask].mean()),
                               "raw_signal_seed99": float(other[t, mask].mean()), "cells": int(mask.sum())})
    base = metrics_for(path / "object_only_validation.json", identifier)
    other = metrics_for(path / "object_only_validation_seed99.json", identifier)
    z = [point["camera_z_m"] for point in points]
    r = [point["raw_signal_seed42"] for point in points]
    r_other = [point["raw_signal_seed99"] for point in points]
    summary = {"candidate": selected, "raw_readout": "r(region,t) = -mean(K_pre_RoPE[L16,H9,d52,t,region])",
               "seed42": base, "seed99": other, "object_region_observations": len(points),
               "pooled_object_region_rho_seed42": float(spearmanr(z, r).statistic),
               "pooled_object_region_rho_seed99": float(spearmanr(z, r_other).statistic),
               "paired_signal_noise_seed_rho": float(spearmanr(r, r_other).statistic),
               "input": "33 observed RGB, 15 frame0 copies, full task prompt; pure main noise at t1000",
               "readout_selection": "Channel/sign fixed using four discovery episodes in four suites; no weights fitted",
               "gt_usage": "Simulator visible-object masks and camera depth score the readout only; not inference inputs",
               "limitation": "Five held-out wrist clips, four anchors each. Pair comparisons within an episode are correlated. Regional motion-depth ordering is stronger than dense depth or same-frame object ordering. The signal is not metric depth."}
    (path / "wrist_readout_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    (path / "wrist_region_points.json").write_text(json.dumps(points, indent=2, allow_nan=False))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    bounds = (min(r + r_other) - 0.08, max(r + r_other) + 0.08)
    for ax, case in zip(axes.ravel(), cases, strict=False):
        case_points = [point for point in points if point["case"] == case["name"]]
        for i, name in enumerate(sorted({point["object"] for point in case_points})):
            own = sorted((point for point in case_points if point["object"] == name), key=lambda point: point["anchor"])
            color = f"C{i}"
            x = [point["camera_z_m"] for point in own]
            y = [point["raw_signal_seed42"] for point in own]
            ax.plot(x, y, "o-", color=color, label=name.replace("akita_", "").replace("_", " "), markersize=5)
            ax.scatter(x, [point["raw_signal_seed99"] for point in own], marker="x", color=color, s=30)
            for point in own:
                ax.annotate(str(point["anchor"]), (point["camera_z_m"], point["raw_signal_seed42"]),
                            xytext=(3, 4), textcoords="offset points", fontsize=8)
        ax.set_title(f"{case['suite']} | episode {case['episode']}", fontsize=10)
        ax.set_xlabel("GT camera Z (m): farther to the right")
        ax.set_ylabel("Raw native signal r: fixed sign, no depth fit")
        ax.set_ylim(*bounds)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")
    axes.ravel()[-1].scatter(z, r, s=18, alpha=0.7, label="seed42")
    axes.ravel()[-1].scatter(z, r_other, s=22, marker="x", alpha=0.7, label="seed99")
    axes.ravel()[-1].set_title(f"All {len(points)} object-region observations", fontsize=10)
    axes.ravel()[-1].set_xlabel("GT camera Z (m)")
    axes.ravel()[-1].set_ylabel("Raw native signal r")
    axes.ravel()[-1].set_ylim(*bounds)
    axes.ravel()[-1].grid(alpha=0.2)
    axes.ravel()[-1].legend(fontsize=8)
    fig.suptitle("Wrist L16H9 channel52 before RoPE | circles: seed42, crosses: seed99\nLines connect anchors of the same GT object; labels are anchor frames. No regression or per-frame rescaling.", fontsize=11)
    fig.savefig(path / "wrist_region_depth_curves.png", dpi=150)
    plt.close(fig)
    print(json.dumps({key: value for key, value in summary.items() if key not in {"candidate", "seed42", "seed99"}}, indent=2))
    print(json.dumps({"seed42": {key: value for key, value in base.items() if key != "per_case"},
                      "seed99": {key: value for key, value in other.items() if key != "per_case"}}, indent=2))


if __name__ == "__main__":
    main()
