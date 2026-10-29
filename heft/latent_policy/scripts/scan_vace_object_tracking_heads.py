#!/usr/bin/env python3
"""Compare frozen, native control-tower heads on observed object trajectories."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from heft_postprocess_vace_tracking import beam_track
from search_vace_tracking_heads import (
    CAMERAS, GRIPPER, ROOT, ControlTower, cell_to_xy, load_empty_text,
    load_vae, read_video, select_points, write_video, xy_to_cell,
)
from vace_vae_warmup import observation_control, observation_window_indices


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def log(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def select_cases(args):
    split = json.loads(args.split.read_text())
    allowed = set(split["train_episodes"])
    candidates = []
    for record in split["episodes"]:
        ep = record["episode"]
        if ep not in allowed:
            continue
        directory = args.video_root / f"episode_{ep:05d}"
        meta = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "sim_state.npz") as z:
            positions = z["body_xpos"]
        names = meta["body_names"]
        hand = names.index("gripper0_eef")
        roots = [i for i, name in enumerate(names) if name.endswith("_main")]
        bodies = [i for i, name in enumerate(names)
                  if any(name.startswith(names[root].removesuffix("_main")) for root in roots)]
        options = []
        last = max(16, 4 * ((len(positions) - args.max_frames) // 4))
        for start in sorted(set([16, 32, last] + list(range(16, last + 1, 32)))):
            n = 1 + 4 * ((min(args.max_frames, len(positions) - start) - 1) // 4)
            if n < 49:
                continue
            times = np.arange(start, start + n, 4)
            world = positions[times]
            excursions = np.linalg.norm(world[:, bodies] - world[0, bodies], axis=-1).max(0)
            root_excursions = np.linalg.norm(world[:, roots] - world[0, roots], axis=-1).max(0)
            hand_motion = float(np.linalg.norm(world[:, hand] - world[0, hand], axis=-1).max())
            distances = np.linalg.norm(world[:, roots] - world[:, hand, None], axis=-1).min(0)
            nearest = int(np.argmin(distances))
            # This chooses diagnostic clips, never inference-time objects or matches.
            options.append(dict(start_frame=int(start), rgb_frames=int(n),
                                object_excursion_m=float(excursions.max()),
                                hand_excursion_m=hand_motion,
                                nearest_object=names[roots[nearest]],
                                nearest_object_distance_m=float(distances[nearest]),
                                nearest_object_excursion_m=float(root_excursions[nearest])))
        if not options:
            continue
        moving = max(options, key=lambda x: x["object_excursion_m"])
        still = max(options, key=lambda x: (
            x["hand_excursion_m"] > 0.05 and x["nearest_object_excursion_m"] <= 0.02,
            -x["nearest_object_distance_m"], x["hand_excursion_m"]))
        candidates.append(record | {"moving_window": moving, "still_window": still})
    selected = []
    for suite in sorted({row["suite"] for row in candidates}):
        pool = [row for row in candidates if row["suite"] == suite]
        used_tasks = set()
        for kind in ("moving", "moving", "still", "coverage"):
            eligible = [row for row in pool if row["task_id"] not in used_tasks]
            if kind == "still":
                row = max(eligible, key=lambda r: (
                    r["still_window"]["hand_excursion_m"] > 0.05
                    and r["still_window"]["nearest_object_excursion_m"] <= 0.02,
                    -r["still_window"]["nearest_object_distance_m"]))
                window = row["still_window"]
            elif kind == "coverage":
                row = min(eligible, key=lambda r: (abs(r["task_id"] - 5), r["episode"]))
                window = row["moving_window"]
            else:
                row = max(eligible, key=lambda r: r["moving_window"]["object_excursion_m"])
                window = row["moving_window"]
            used_tasks.add(row["task_id"])
            for camera in CAMERAS:
                selected.append({key: row[key] for key in
                                 ("episode", "suite", "task_id", "task", "source_episode")}
                                | window | {"camera": camera, "selection_kind": kind,
                                            "name": f"episode_{row['episode']:05d}_{camera}_start{window['start_frame']:04d}"})
    return selected


def prepare_points(args, case):
    ep, camera, start, n = (case[k] for k in ("episode", "camera", "start_frame", "rgb_frames"))
    times = np.arange(start, start + n, 4)
    with np.load(args.tracks_root / f"episode_{ep:05d}_{camera}_visible_surface.npz") as z:
        data = {key: z[key] for key in
                ("xy", "visible", "world_xyz", "point_label", "local_xyz", "annotated_frame_indices")}
    if not np.isin(times, data["annotated_frame_indices"]).all():
        raise ValueError(f"Missing point annotations: {case['name']}")
    points = select_points(data, times, args.size // 16, args.points_per_subject, 0.02,
                           surface_motion=True, include_static=True)
    keep = ~np.isin(points["labels"], list(GRIPPER))
    for key in ("ids", "anchor", "groups", "xy", "valid", "displaced", "cells", "labels"):
        points[key] = points[key][keep]
    points["visible"] = data["visible"][times][:, points["ids"]].T
    labels = np.asarray(["gripper" if label in GRIPPER else str(label)
                         for label in data["point_label"]])
    identities = sorted(set(labels))
    point_owners = np.asarray([identities.index(str(label)) for label in points["labels"]])
    grid = args.size // 16
    occupancy = np.full((len(times), grid * grid), -1, np.int32)
    all_cells = xy_to_cell(data["xy"][times], grid, 256)
    all_owners = np.asarray([identities.index(label) for label in labels])
    for ti in range(len(times)):
        visible = data["visible"][times[ti]]
        cells, owners = all_cells[ti, visible], all_owners[visible]
        low, high = np.full(grid * grid, len(identities), int), np.full(grid * grid, -1, int)
        np.minimum.at(low, cells, owners)
        np.maximum.at(high, cells, owners)
        clean = high == low
        occupancy[ti, clean] = high[clean]
        occupancy[ti, (high >= 0) & ~clean] = -2
    meta = json.loads((args.video_root / f"episode_{ep:05d}" / "metadata.json").read_text())
    with np.load(args.video_root / f"episode_{ep:05d}" / "sim_state.npz") as z:
        hand = z["body_xpos"][times, meta["body_names"].index("gripper0_eef")]
    world = data["world_xyz"][times][:, points["ids"]].transpose(1, 0, 2)
    initial_world = world[np.arange(len(world)), points["anchor"]]
    world_delta = np.linalg.norm(world - initial_world[:, None], axis=-1)
    initial_hand = hand[points["anchor"]]
    hand_delta = np.linalg.norm(hand[None] - initial_hand[:, None], axis=-1)
    near_hand = np.linalg.norm(world - hand[None], axis=-1) <= 0.15
    moving_labels = [row["label"] for row in points["motion"] if row["surface_excursion_m"] >= 0.02]
    points["phases"] = {
        "moving": points["displaced"] & np.isin(points["labels"], moving_labels)[:, None],
        "near_stationary": points["valid"] & (world_delta <= 0.01) & near_hand & (hand_delta > 0.02),
        "all_visible": points["valid"],
    }
    active_labels = set(points["labels"][(points["phases"]["moving"]
                                        | points["phases"]["near_stationary"]).any(1)])
    active = np.isin(points["labels"], list(active_labels))
    for phase in points["phases"]:
        points["phases"][phase] &= active[:, None]
    points.update(occupancy=occupancy, point_owners=point_owners,
                  gripper_owner=identities.index("gripper") if "gripper" in identities else -3)
    return points, times


def measure(prediction, points, grid):
    error = np.linalg.norm(prediction - points["xy"], axis=-1)
    cells = xy_to_cell(prediction, grid, 256)
    owner = points["occupancy"][np.arange(cells.shape[1])[None], cells]
    true_owner = points["occupancy"][np.arange(cells.shape[1])[None], points["cells"]]
    clean_true = true_owner == points["point_owners"][:, None]
    wrong = (owner >= 0) & (owner != points["point_owners"][:, None]) & clean_true
    swapped = wrong & (owner == points["gripper_owner"])
    result = {}
    for phase, mask in points["phases"].items():
        objects = []
        for label in np.unique(points["labels"]):
            selected = mask & (points["labels"] == label)[:, None]
            if not selected.any():
                continue
            count = selected.sum(1)
            trajectories = count >= 3
            ratio = ((error <= 8) & selected).sum(1) / np.maximum(count, 1)
            objects.append({"label": str(label), "count": int(selected.sum()),
                            "epe_px": float(error[selected].mean()),
                            "within8": float((error[selected] <= 8).mean()),
                            "trajectory80_at8": float((ratio[trajectories] >= 0.8).mean()) if trajectories.any() else None,
                            "gripper_swap": float(swapped[selected].mean()),
                            "known_wrong_identity": float(wrong[selected].mean()),
                            "known_owner_coverage": float((owner[selected] >= 0).mean())})
        result[phase] = {"objects": objects, "count": sum(row["count"] for row in objects)}
        for key in ("epe_px", "within8", "trajectory80_at8", "gripper_swap",
                    "known_wrong_identity", "known_owner_coverage"):
            values = [row[key] for row in objects if row[key] is not None]
            result[phase][key] = float(np.mean(values)) if values else None
    return result


def decode_native(q, k, points, grid):
    heads, frames, spatial, dim = q.shape
    anchors, cells = points["anchor"], points["cells"]
    index = torch.as_tensor(anchors * spatial + cells[np.arange(len(anchors)), anchors], device=q.device)
    query = q.flatten(1, 2)[:, index].float()
    keys = k.float()
    fixed = []
    for start in range(0, len(anchors), 16):
        scores = torch.einsum("hpd,htsd->hpts", query[:, start:start + 16], keys) / math.sqrt(dim)
        fixed.append(scores.argmax(-1).cpu().numpy())
    fixed = np.concatenate(fixed, 1)
    rolling = np.zeros_like(fixed)
    head_index = torch.arange(heads, device=q.device)[:, None]
    current = torch.as_tensor(cells[np.arange(len(anchors)), anchors], device=q.device).expand(heads, -1).clone()
    for ti in range(frames):
        if ti:
            previous = q[:, ti - 1][head_index, current].float()
            scores = torch.einsum("hpd,hsd->hps", previous, keys[:, ti]) / math.sqrt(dim)
            active = torch.as_tensor(anchors < ti, device=q.device)[None]
            current = torch.where(active, scores.argmax(-1), current)
        rolling[:, :, ti] = current.cpu().numpy()
    return cell_to_xy(fixed, grid, 256), cell_to_xy(rolling, grid, 256)


def rank_cases(cases, key="rows"):
    identities = sorted({(row["layer"], row["head"], row["decoder"])
                         for case in cases for row in case.get(key, [])})
    ranking = []
    for layer, head, decoder in identities:
        observations = [(case, row) for case in cases for row in case.get(key, [])
                        if (row["layer"], row["head"], row["decoder"]) == (layer, head, decoder)]
        strata = sorted({(case["suite"], case["camera"]) for case, _ in observations})
        summary = {"layer": layer, "head": head, "decoder": decoder,
                   "head_name": f"T{layer}H{head}", "cases": len(observations), "strata": {}}
        for phase in ("moving", "near_stationary", "all_visible"):
            summary[phase] = {}
            for metric in ("epe_px", "within8", "trajectory80_at8", "gripper_swap", "known_wrong_identity"):
                values = []
                for suite, camera in strata:
                    grouped = [row["metrics"][phase][metric] for case, row in observations
                               if case["suite"] == suite and case["camera"] == camera
                               and row["metrics"][phase][metric] is not None]
                    if grouped:
                        value = float(np.mean(grouped))
                        summary["strata"].setdefault(f"{suite}/{camera}", {}).setdefault(phase, {})[metric] = value
                        values.append(value)
                summary[phase][metric] = float(np.mean(values)) if values else None
        ranking.append(summary)
    # Moving trajectories lead the ranking; stationary coverage cannot win by itself.
    ranking.sort(key=lambda r: (-(r["moving"]["trajectory80_at8"] or 0),
                                -(r["moving"]["within8"] or 0),
                                -(r["near_stationary"]["within8"] or 0),
                                r["all_visible"]["gripper_swap"] or 0))
    return ranking


def render_case(args, case, points, times, candidates, predictions):
    video = read_video(args.video_root / f"episode_{case['episode']:05d}/{case['camera']}.mp4")
    # Show the most moving object, not a fixture with the largest point count.
    labels = np.unique(points["labels"])
    label = max(labels, key=lambda x: int((points["phases"]["moving"]
                                         & (points["labels"] == x)[:, None]).sum()))
    ids = np.flatnonzero(points["labels"] == label)[:12]
    images = []
    for ti, frame in enumerate(times):
        panels = []
        for prediction, title in [(points["xy"], "GT object points")] + [
                (predictions[i], f"{row['head_name']} fixed beam") for i, row in enumerate(candidates)]:
            panel = video[frame].copy()
            for pid in ids:
                if not points["visible"][pid, ti]:
                    continue
                xy = prediction[pid, ti]
                if not np.isfinite(xy).all():
                    continue
                color = (40, 240, 80) if title.startswith("GT") else (245, 70, 210)
                cv2.circle(panel, tuple(np.round(xy).astype(int)), 2, color, -1)
                for previous in range(max(int(points["anchor"][pid]), ti - 4), ti):
                    path = prediction[pid, previous:previous + 2]
                    if points["visible"][pid, previous:previous + 2].all() and np.isfinite(path).all():
                        cv2.line(panel, tuple(np.round(path[0]).astype(int)), tuple(np.round(path[1]).astype(int)), color, 1)
            panel = cv2.copyMakeBorder(panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cv2.putText(panel, title, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
            cv2.putText(panel, f"RGB {frame}: {label}"[:48], (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (255, 255, 255), 1)
            panels.append(panel)
        images.append(np.concatenate(panels, 1))
    directory = args.output / case["name"]
    write_video(directory / "object_head_comparison.mp4", np.stack(images), 5)
    sample = sorted({0, len(images) // 2, len(images) - 1})
    sheet = np.concatenate([images[i] for i in sample], 0)
    cv2.imwrite(str(directory / "object_head_comparison.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "runs/vace_tracking_all_suites_20261004"
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=base / "split.json")
    parser.add_argument("--video-root", type=Path, default=base / "videos")
    parser.add_argument("--tracks-root", type=Path, default=base / "tracks")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=129)
    parser.add_argument("--points-per-subject", type=int, default=12)
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--timestep", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam-heads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.size % 16 or args.warmup_frames % 4 or args.beam_heads < 1:
        parser.error("Invalid grid, warmup, or candidate count")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else select_cases(args)
    dump(manifest_path, manifest)
    protocol = {"scope": "native frozen VACE control tower attn1; 15 layers x 12 heads",
                "text": "empty prompt", "weights_updated": False,
                "input": "all recorded frames in each observation window; no generated future",
                "size": args.size, "grid": args.size // 16, "max_rgb_frames": args.max_frames,
                "warmup_frames": args.warmup_frames, "timestep": args.timestep, "seed": args.seed,
                "cases": len(manifest), "episodes": len({c['episode'] for c in manifest}),
                "split": "training source episodes only; validation episodes excluded",
                "initial_points": "GT surface point first-visible cells for offline head diagnosis only",
                "reanchoring": "none; no future GT is supplied to trajectory decoding",
                "moving": "object surface excursion >=2cm in window; visible point displaced >1cm world and >8px image from anchor",
                "near_stationary": "point <=1cm from anchor, within15cm of eef, eef displaced >2cm",
                "aggregation": "equal objects within a case, equal cases within suite/camera, equal suite/camera strata",
                "ranking": "moving trajectory80_at8, then moving within8, stationary within8, gripper swap",
                "identity_swap": "lower bound from annotated clean token ownership; unknown/mixed cells are not background labels",
                "beam": {"topk": 32, "width": 8, "motion_weight": 0.45, "max_step_cells": 4,
                         "local_weight": 0, "descriptor_ema": 0}}
    dump(args.output / "protocol.json", protocol)
    indexed = []
    for case in manifest:
        points, times = prepare_points(args, case)
        indexed.append(case | {"points": len(points["ids"]),
                               "active_objects": sorted(set(points["labels"][points["phases"]["all_visible"].any(1)])),
                               "phase_counts": {k: int(v.sum()) for k, v in points["phases"].items()}})
    dump(args.output / "data_summary.json", indexed)
    log(stage="indexed", cases=len(manifest), episodes=protocol["episodes"],
        phase_counts={phase: sum(case["phase_counts"][phase] for case in indexed)
                      for phase in ("moving", "near_stationary", "all_visible")})
    if args.prepare_only:
        return
    device = torch.device("cuda:0")
    log(stage="empty_text")
    text = load_empty_text(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", device)
    log(stage="tower_load")
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    log(stage="vae_load")
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    zero_latents, completed = {}, []
    for case_index, case in enumerate(manifest):
        started = time.monotonic()
        directory = args.output / case["name"]
        directory.mkdir(exist_ok=True)
        points, times = prepare_points(args, case)
        result_path, cache_path = directory / "native_heads.json", directory / "control.pt"
        if result_path.exists() and cache_path.exists():
            rows = []
            with np.load(directory / "native_trajectories.npz") as stored:
                predictions = stored["predicted_xy"]
            for layer in range(15):
                for mode, decoder in enumerate(("anchor_q_to_k", "rolling_q_to_k")):
                    for head in range(12):
                        rows.append({"layer": layer, "head": head, "decoder": decoder,
                                     "metrics": measure(predictions[layer, mode, head], points, args.size // 16)})
            dump(result_path, rows)
            completed.append(case | {"rows": rows})
            log(stage="reused", progress=f"{len(completed)}/{len(manifest)}", case=case["name"])
            continue
        log(stage="encode", case=case["name"], progress=f"{case_index + 1}/{len(manifest)}")
        video = read_video(args.video_root / f"episode_{case['episode']:05d}/{case['camera']}.mp4")
        indices, _ = observation_window_indices(len(video), case["start_frame"], case["rgb_frames"], args.warmup_frames)
        pixels = torch.from_numpy(video[indices].copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
        pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=(args.size, args.size),
                               mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
        control = observation_control(vae, pixels, args.warmup_frames, zero_latents)
        generator = torch.Generator(device=device).manual_seed(args.seed + case["episode"] * 10000
                    + case["start_frame"] * 10 + CAMERAS.index(case["camera"]))
        noise = torch.randn((1, 16, *control.shape[2:]), generator=generator, device=device).bfloat16()
        torch.save({"control": control.cpu(), "noise": noise.cpu(), "times": times}, cache_path)
        del pixels, video
        rows, paths = [], []
        grid = args.size // 16

        def score(layer, q, k):
            q, k = (x[0].reshape(12, len(times), grid * grid, 128) for x in (q, k))
            predictions = decode_native(q, k, points, grid)
            paths.append(np.stack(predictions))
            for decoder, predicted in zip(("anchor_q_to_k", "rolling_q_to_k"), predictions):
                for head in range(12):
                    rows.append({"layer": layer, "head": head, "decoder": decoder,
                                 "metrics": measure(predicted[head], points, grid)})
            log(stage="layer", case=case["name"], layer=layer,
                moving_best=max((r["metrics"]["moving"]["within8"] or 0) for r in rows[-24:]))

        with torch.autocast("cuda", dtype=torch.bfloat16):
            tower(noise, control, text, torch.tensor([args.timestep], device=device), score)
        dump(result_path, rows)
        np.savez_compressed(directory / "native_trajectories.npz", predicted_xy=np.stack(paths),
                            gt_xy=points["xy"], valid=points["valid"], times=times,
                            labels=points["labels"], anchor=points["anchor"],
                            moving=points["phases"]["moving"], near_stationary=points["phases"]["near_stationary"])
        completed.append(case | {"rows": rows})
        ranking = rank_cases(completed)
        dump(args.output / "native_ranking.json", ranking)
        log(stage="case_done", progress=f"{len(completed)}/{len(manifest)}", case=case["name"],
            seconds=round(time.monotonic() - started, 1),
            allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            top=[(r["head_name"], r["decoder"], r["moving"]["within8"]) for r in ranking[:3]])
        del control, noise, paths
        gc.collect()
        torch.cuda.empty_cache()
    del vae, zero_latents
    gc.collect()
    torch.cuda.empty_cache()
    ranking = rank_cases(completed)
    candidates, used = [], set()
    for row in ranking:
        identity = row["layer"], row["head"]
        if identity not in used:
            candidates.append(row)
            used.add(identity)
        if len(candidates) == args.beam_heads:
            break
    dump(args.output / "beam_candidates.json", candidates)
    beam_cases = []
    for case_index, case in enumerate(manifest):
        started = time.monotonic()
        directory = args.output / case["name"]
        points, times = prepare_points(args, case)
        saved = torch.load(directory / "control.pt", map_location=device, weights_only=False)
        rows, paths = [], [None] * len(candidates)

        def score_beam(layer, q, k):
            for index, candidate in enumerate(candidates):
                if candidate["layer"] != layer:
                    continue
                grid = args.size // 16
                queries, keys = (x[0].reshape(12, len(times), grid * grid, 128) for x in (q, k))
                predicted = beam_track(queries, keys, points, candidate["head"], grid)
                paths[index] = predicted
                rows.append({"layer": layer, "head": candidate["head"], "decoder": "fixed_beam",
                             "metrics": measure(predicted, points, grid)})

        log(stage="beam", case=case["name"], progress=f"{case_index + 1}/{len(manifest)}",
            candidates=[r["head_name"] for r in candidates])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tower(saved["noise"], saved["control"], text,
                  torch.tensor([args.timestep], device=device), score_beam)
        dump(directory / "beam_heads.json", rows)
        np.savez_compressed(directory / "beam_trajectories.npz", predicted_xy=np.stack(paths),
                            gt_xy=points["xy"], valid=points["valid"], times=times,
                            labels=points["labels"], anchor=points["anchor"],
                            heads=np.asarray([r["head_name"] for r in candidates]))
        render_case(args, case, points, times, candidates, paths)
        beam_cases.append(case | {"beam_rows": rows})
        dump(args.output / "beam_ranking.json", rank_cases(beam_cases, "beam_rows"))
        log(stage="beam_done", progress=f"{len(beam_cases)}/{len(manifest)}",
            case=case["name"], seconds=round(time.monotonic() - started, 1))
        del saved, paths
        torch.cuda.empty_cache()
    result = {"protocol": protocol, "native_top": ranking[:12],
              "beam_ranking": rank_cases(beam_cases, "beam_rows"), "cases": indexed}
    dump(args.output / "report.json", result)
    # These are temporary encoded inputs created by this scan, not prior experiment caches.
    for case in manifest:
        (args.output / case["name"] / "control.pt").unlink()
    log(stage="complete", output=str(args.output), top=[r["head_name"] for r in result["beam_ranking"]])


if __name__ == "__main__":
    main()
