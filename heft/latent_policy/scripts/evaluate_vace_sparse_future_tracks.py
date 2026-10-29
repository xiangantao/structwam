#!/usr/bin/env python3
"""Compare saved sparse/reference future tracks against simulator surface GT."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import demo_main_prompt_k_tracking as demo
from search_vace_tracking_heads import GRIPPER, cell_to_xy, xy_to_cell
from vace_latent_distill.config import PipelineConfig


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def select_points(data, frame, labels, grid, cap):
    eligible = np.flatnonzero(data["visible"][frame] & np.isin(data["point_label"], labels))
    if not len(eligible):
        return eligible
    xy = data["xy"][frame, eligible]
    cells = xy_to_cell(xy, grid, 256)
    distance = np.linalg.norm(xy - cell_to_xy(cells, grid, 256), axis=-1)
    # One surface point nearest the center of each occupied token cell.
    candidates = np.asarray([eligible[np.flatnonzero(cells == cell)[np.argmin(distance[cells == cell])]]
                             for cell in np.unique(cells)])
    if len(candidates) <= cap:
        return candidates
    positions = data["xy"][frame, candidates]
    chosen = [int(np.argmin(np.linalg.norm(positions - positions.mean(0), axis=-1)))]
    while len(chosen) < cap:
        distances = np.linalg.norm(positions[:, None] - positions[chosen][None], axis=-1).min(1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return candidates[chosen]


def measure(records, mode, group, displaced_only=False):
    absolute, motion, endpoint = [], [], []
    points = 0
    for row in records:
        mask = row["valid"].copy()
        if displaced_only:
            mask &= row["displaced"]
        mask &= row["groups"][:, None] == group
        if not mask.any():
            continue
        predicted, truth = row[mode], row["xy"]
        error = np.linalg.norm(predicted - truth, axis=-1)
        delta_error = np.linalg.norm((predicted - predicted[:, :1]) - (truth - truth[:, :1]), axis=-1)
        absolute.extend(error[mask].tolist())
        motion.extend(delta_error[mask].tolist())
        endpoint.extend(error[:, -1][mask[:, -1]].tolist())
        points += int(mask.any(1).sum())
    if not absolute:
        return {"epe_px": None, "within8": None, "count": 0, "anchor_points": 0}
    return {"epe_px": float(np.mean(absolute)), "within8": float(np.mean(np.asarray(absolute) <= 8)),
            "motion_epe_px": float(np.mean(motion)),
            "endpoint_epe_px": float(np.mean(endpoint)) if endpoint else None,
            "count": len(absolute), "anchor_points": points}


def summarize(records, modes):
    return {mode: {group + ("_displaced" if displaced else "_visible"):
                   measure(records, mode, group, displaced)
                   for group in ("moving_object", "gripper") for displaced in (False, True)}
            for mode in modes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vace_structured.json")
    parser.add_argument("--points-per-group", type=int, default=64)
    args = parser.parse_args()
    if args.points_per_group < 1:
        parser.error("--points-per-group must be positive")
    protocol = json.loads((args.run / "protocol.json").read_text())
    frozen = json.loads((Path(protocol["frozen_reference"]) / "protocol.json").read_text())
    original = json.loads((Path(frozen["future_source"]) / "protocol.json").read_text())
    config = PipelineConfig.load(args.config).teacher
    grid = config.image_size // 16
    tracks_file = Path(original["data"]) / "tracks" / f"episode_{original['episode']:05d}_{original['camera']}_visible_surface.npz"
    with np.load(tracks_file) as archive:
        data = {key: archive[key] for key in ("xy", "visible", "world_xyz", "point_label", "annotated_frame_indices")}
    with np.load(args.run / "future_track_comparison.npz") as archive:
        values = {key: archive[key] for key in archive.files}
    target = original["diagnostic_target_body"].removesuffix("_main")
    records, matched_seed_records, selection_coverage = [], [], []
    saved = {}
    for index, anchor in enumerate(values["anchor_latents"]):
        end = min(int(anchor) + 3, 13)
        times = original["start"] + np.arange(int(anchor), end) * 4
        if not np.isin(times, data["annotated_frame_indices"]).all():
            raise ValueError("Evaluation requires real GT at every native readout time")
        ids = np.concatenate((select_points(data, times[0], [target], grid, args.points_per_group),
                              select_points(data, times[0], sorted(GRIPPER), grid, args.points_per_group)))
        xy = data["xy"][times][:, ids].transpose(1, 0, 2)
        world = data["world_xyz"][times][:, ids].transpose(1, 0, 2)
        valid = data["visible"][times][:, ids].T.copy()
        valid[:, 0] = False
        groups = np.where(np.isin(data["point_label"][ids], list(GRIPPER)), "gripper", "moving_object")
        displaced = (np.linalg.norm(xy - xy[:, :1], axis=-1) > 8)
        displaced &= np.linalg.norm(world - world[:, :1], axis=-1) > 0.01
        chosen = xy_to_cell(xy[:, 0], grid, 256)
        row = {"ids": ids, "groups": groups, "xy": xy, "valid": valid, "displaced": displaced,
               "hold": np.repeat(xy[:, :1], len(times), axis=1)}
        recent = times[0] - np.asarray([8, 4, 0])
        velocity_valid = data["visible"][recent][:, ids].all(0)
        velocity = (data["xy"][recent[-1], ids] - data["xy"][recent[0], ids]) / 8
        velocity[~velocity_valid] = 0
        row["velocity"] = xy[:, :1] + velocity[:, None] * (times - times[0])[None, :, None]
        row["quantization"] = cell_to_xy(xy_to_cell(xy, grid, 256), grid, 256)
        saved.update({f"anchor{index}_ids": ids, f"anchor{index}_times": times,
                      f"anchor{index}_gt": xy, f"anchor{index}_valid": valid,
                      f"anchor{index}_groups": groups, f"anchor{index}_displaced": displaced})
        common = np.intersect1d(values["sparse_seeds"][index], values["reference_seeds"][index])
        selected_mask = np.isin(chosen, common)
        matched = {key: array[selected_mask] for key, array in row.items()}
        coverage = {"native_anchor": int(anchor), "source_anchor_frame": int(times[0]),
                    "common_prompt_seeds": len(common), "source_evaluation_frames": times[1:].tolist()}
        for group in ("moving_object", "gripper"):
            coverage[group + "_gt_points"] = int((groups == group).sum())
            coverage[group + "_common_selected_points"] = int((selected_mask & (groups == group)).sum())
        selection_coverage.append(coverage)
        for branch in ("sparse", "reference"):
            raw = demo.track_dense(values[f"{branch}_q"][:end * grid**2],
                                   values[f"{branch}_k"][:end * grid**2], chosen, int(anchor), grid,
                                   values[f"{branch}_region_scores"][index, :end],
                                   topk=config.beam_topk, beam_size=config.beam_size,
                                   max_step=config.beam_max_step, motion_weight=config.motion_weight,
                                   local_weight=config.local_weight, region_weight=config.region_weight,
                                   acceleration_weight=config.acceleration_weight, descriptor_ema=config.descriptor_ema,
                                   direction="q_to_k")
            paths = demo.regularize_paths(raw, config.regularize_max_step,
                                          config.regularize_max_acceleration, passes=2)
            prediction = (paths + 0.5) * (256 / grid) - 0.5
            row[branch] = prediction
            saved[f"anchor{index}_{branch}_prediction"] = prediction
            positions = {int(cell): offset for offset, cell in enumerate(values[f"{branch}_seeds"][index])}
            selected_paths = values[f"{branch}_paths"][index, [positions[int(cell)] for cell in chosen[selected_mask]]]
            selected_prediction = (selected_paths + 0.5) * (256 / grid) - 0.5
            np.testing.assert_allclose(prediction[selected_mask], selected_prediction, atol=1e-5)
            matched[branch] = selected_prediction
        records.append(row)
        matched_seed_records.append(matched)
    result = {"episode": original["episode"], "camera": original["camera"],
              "pixel_resolution": 256, "target_object": target,
              "updates": {"sparse": len(protocol["sparse_schedule"]) - 1, "reference": protocol["reference_updates"]},
              "scope": "Single privileged-GT clip; comparison with simulator trajectories, not causal prediction accuracy",
              "alignment": "Native time t maps to source_start +4*t; evaluate annotated times only, exclude anchors",
              "scoring": "Visible points only; displaced requires >8px and >1cm from its anchor; static objects excluded",
              "tracking_comparison": "Identical GT-visible anchor query cells for both modes, original per-mode prompt/K gates and frozen decoder; GT only initializes/scorers queries",
              "same_seed_pipeline": "Subset of actual prompt-selected cells shared by both modes; stored rendered paths checked against rerun",
              "selection_coverage": selection_coverage,
              "fixed_gt_queries": summarize(records, ("sparse", "reference", "hold", "velocity", "quantization")),
              "shared_actual_prompt_queries": summarize(matched_seed_records, ("sparse", "reference")),
              "trajectory_error_definition": "Euclidean predicted-vs-GT point distance; motion_epe subtracts each path's initial position"}
    dump(args.run / "future_gt_accuracy.json", result)
    np.savez_compressed(args.run / "future_gt_accuracy_paths.npz", **saved)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
