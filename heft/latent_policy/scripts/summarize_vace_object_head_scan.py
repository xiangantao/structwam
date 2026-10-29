#!/usr/bin/env python3
"""Summarize full-trajectory consistency and matched no-motion baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scan_vace_object_tracking_heads import dump, measure, prepare_points, rank_cases
from search_vace_tracking_heads import ROOT, cell_to_xy


def full_track_quality(predictions, points):
    error = np.linalg.norm(predictions - points["xy"][None], axis=-1)
    valid, moving = points["valid"], points["phases"]["moving"]
    count, moving_count = valid.sum(1), moving.sum(1)
    eligible = (count >= 6) & (moving_count >= 3)
    total_ratio = ((error <= 8) & valid[None]).sum(2) / np.maximum(count, 1)[None]
    moving_ratio = ((error <= 8) & moving[None]).sum(2) / np.maximum(moving_count, 1)[None]
    whole, joint = [], []
    for label in np.unique(points["labels"]):
        ids = eligible & (points["labels"] == label)
        if not ids.any():
            continue
        whole.append((total_ratio[:, ids] >= 0.8).mean(1))
        joint.append(((total_ratio[:, ids] >= 0.8) & (moving_ratio[:, ids] >= 0.8)).mean(1))
    if not whole:
        return [{"full_track80": None, "joint_track80": None, "eligible_tracks": 0}
                for _ in predictions]
    whole, joint = np.stack(whole).mean(0), np.stack(joint).mean(0)
    return [{"full_track80": float(a), "joint_track80": float(b),
             "eligible_tracks": int(eligible.sum())} for a, b in zip(whole, joint)]


def aggregate_quality(cases, field):
    methods = sorted({row["method"] for case in cases for row in case[field]})
    result = []
    for method in methods:
        selected = [(case, row) for case in cases for row in case[field] if row["method"] == method]
        strata = sorted({(case["suite"], case["camera"]) for case, _ in selected})
        summary = {"method": method, "cases": len(selected), "strata": {}}
        for metric in ("full_track80", "joint_track80"):
            values = []
            for suite, camera in strata:
                items = [row[metric] for case, row in selected
                         if case["suite"] == suite and case["camera"] == camera and row[metric] is not None]
                if items:
                    value = float(np.mean(items))
                    summary["strata"].setdefault(f"{suite}/{camera}", {})[metric] = value
                    values.append(value)
            summary[metric] = float(np.mean(values)) if values else None
        result.append(summary)
    return sorted(result, key=lambda row: (-(row["joint_track80"] or 0), -(row["full_track80"] or 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = ROOT / "runs/vace_tracking_all_suites_20261004"
    protocol = json.loads((args.output / "protocol.json").read_text())
    prepared_args = SimpleNamespace(video_root=base / "videos", tracks_root=base / "tracks",
                                    size=protocol["size"], points_per_subject=12)
    manifest = json.loads((args.output / "manifest.json").read_text())
    candidates_path = args.output / "beam_candidates.json"
    candidates = json.loads(candidates_path.read_text()) if candidates_path.exists() else []
    cases = []
    for case in manifest:
        directory = args.output / case["name"]
        if not (directory / "native_trajectories.npz").exists():
            continue
        points, times = prepare_points(prepared_args, case)
        with np.load(directory / "native_trajectories.npz") as stored:
            if not np.array_equal(stored["times"], times):
                raise ValueError("Stored prediction times differ from annotations")
            values = stored["predicted_xy"].reshape(360, len(points["ids"]), len(times), 2)
        quality = full_track_quality(values, points)
        native = []
        for layer in range(15):
            for mode, decoder in enumerate(("anchor_q_to_k", "rolling_q_to_k")):
                for head in range(12):
                    native.append({"method": f"T{layer}H{head}/{decoder}",
                                   **quality[layer * 24 + mode * 12 + head]})
        anchors = points["anchor"]
        continuous = points["xy"][np.arange(len(anchors)), anchors]
        quantized = cell_to_xy(points["cells"][np.arange(len(anchors)), anchors], protocol["grid"], 256)
        oracle = cell_to_xy(points["cells"], protocol["grid"], 256)
        baselines = [np.repeat(x[:, None], len(times), axis=1) for x in (continuous, quantized)] + [oracle]
        baseline_quality = full_track_quality(np.stack(baselines), points)
        baseline_rows, baseline_full = [], []
        for index, (name, prediction, full) in enumerate(zip(
                ("no_motion_pixel_anchor", "no_motion_grid_anchor", "grid_quantization_oracle"),
                baselines, baseline_quality)):
            baseline_rows.append({"layer": -1, "head": index, "decoder": name,
                                  "metrics": measure(prediction, points, protocol["grid"])})
            baseline_full.append({"method": name, **full})
        beam = []
        if (directory / "beam_trajectories.npz").exists():
            with np.load(directory / "beam_trajectories.npz") as stored:
                beam_quality = full_track_quality(stored["predicted_xy"], points)
            beam = [{"method": f"{candidate['head_name']}/fixed_beam", **full}
                    for candidate, full in zip(candidates, beam_quality)]
        cases.append(case | {"native_quality": native, "beam_quality": beam,
                             "baseline_rows": baseline_rows, "baseline_quality": baseline_full})
    result = {"definition": "At least6 visible point-times and3 moving point-times. Full-track80 requires >=80% within8px across all visible times. Joint-track80 additionally requires >=80% within8px during movement. Objects, cases, suite/camera strata receive equal weight.",
              "cases": len(cases), "native": aggregate_quality(cases, "native_quality"),
              "beam": aggregate_quality(cases, "beam_quality"),
              "baselines": rank_cases(cases, "baseline_rows"),
              "baseline_full": aggregate_quality(cases, "baseline_quality"),
              "per_case": [{"name": case["name"], "suite": case["suite"], "camera": case["camera"],
                            "beam": case["beam_quality"], "baselines": case["baseline_quality"]} for case in cases]}
    dump(args.output / "full_track_report.json", result)
    print(json.dumps({"cases": len(cases), "native_top": result["native"][:8],
                      "beam": result["beam"], "baseline_full": result["baseline_full"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
