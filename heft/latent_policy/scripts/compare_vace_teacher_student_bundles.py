#!/usr/bin/env python3
"""Run the VACE teacher on the student-visualization windows and compare all heads.

Input is an output directory of ``render_vace_student_region_bundles.py``.  The
teacher sees exactly the same 49 training-domain frames (the student saw only
the first 33).  Teacher and student fields are rendered by the same region-bundle
code with one shared K color projection, then compared:

* semantic: per-phrase spatial heat, teacher vs student, and phrase separation;
* tracking: QK self-match rate and simulator-GT point error (fixed GT queries,
  each source's own prompt/K gate, frozen beam decoder);
* depth: Spearman against simulator depth, full scene and object cells.

Simulator tracks/depth come from the horizontally mirrored tracking renders and
are flipped into training-video coordinates.  They are used for scoring only.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import render_vace_teacher_region_bundles as rb
from render_vace_student_region_bundles import depth_sheet, write_h264
from vace_latent_distill.bundles import phrase_positions, read_region_bundles, word_heats
from vace_latent_distill.config import NEGATIVE_PROMPT, PipelineConfig
from vace_latent_distill.data import decode_indices, stable_seed
from vace_latent_distill.teacher import content_token_masks, local_scripts

GRID = 32
FIELDS = ("seg_k_l22h9", "seg_k_l15h7", "track_q", "track_k", "text_attention", "depth")
TRACKING_CAMERA = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}
TARGET_BODY = {1653: "akita_black_bowl_1", 850: "alphabet_soup_1"}
GRIPPER = ["gripper base", "left finger", "right finger"]
ANCHORS = {"observed": (4, 6, 8, 10), "future": (8, 10)}


def source_frame(start, branch, native):
    """Recorded frame of a native readout position (existing evaluator conventions)."""
    return start + (native - 4) * 4 if branch == "observed" else start + native * 4


def decode_case(job):
    torch.set_num_threads(4)
    local_scripts()
    target, phrases, content, config = job
    return {branch: read_region_bundles(target[branch], branch, phrases, content, config)
            for branch in ("observed", "future")}


def stack_media(teacher_dir, student_dir, output):
    for name in ("public_regions.jpg", "phrase_response.jpg", "phrase_regions.jpg",
                 "trajectory_fields.jpg", "depth.jpg"):
        images = [cv2.imread(str(d / name)) for d in (teacher_dir, student_dir)]
        width = max(i.shape[1] for i in images)
        images = [cv2.copyMakeBorder(i, 0, 0, 0, width - i.shape[1], cv2.BORDER_CONSTANT,
                                     value=(255, 255, 255)) for i in images]
        cv2.imwrite(str(output / f"compare_{name}"), np.concatenate(images, 0),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    import av

    for name in ("named_regions_and_tracks.mp4", "public_qk_tracks.mp4", "public_regions.mp4"):
        clips = []
        for directory in (teacher_dir, student_dir):
            with av.open(str(directory / name)) as container:
                clips.append(np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)]))
        write_h264(output / f"compare_{name}", np.concatenate(clips, axis=1))


def semantic_metrics(fields, phrases, content):
    result = {}
    for branch in ("observed", "future"):
        heats = {source: word_heats(fields[source][branch]["text_attention"].float().numpy(), content, phrases)
                 for source in fields}
        names = ["Full instruction", *phrases]
        rows = {}
        for index, name in enumerate(names):
            teacher, student = heats["teacher"][index], heats["student"][index]
            rows[name] = {
                "teacher_student_corr": float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(teacher, student)])),
                "contrast_max_over_mean": {s: float(np.mean(h[index].max(-1) / np.maximum(h[index].mean(-1), 1e-8)))
                                           for s, h in heats.items()},
            }
        separation = {}
        for source, heat in heats.items():
            pairs = [float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(heat[i], heat[j])]))
                     for i in range(1, len(names)) for j in range(i + 1, len(names))]
            separation[source] = float(np.mean(pairs)) if pairs else None
        noun_mass = {}
        for source in fields:
            attention = fields[source][branch]["text_attention"].float().numpy()
            noun_mass[source] = {name: float(attention[..., mask].sum(-1).mean()) for name, mask in phrases.items()}
        result[branch] = {"phrases": rows, "mean_pairwise_phrase_heat_corr": separation,
                          "mean_phrase_mass_per_cell": noun_mass}
    return result


def self_match(fields, branch):
    q = fields[branch]["track_q"].float().numpy()
    k = fields[branch]["track_k"].float().numpy()
    q, k = (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8) for x in (q, k))
    times = range(4, 11) if branch == "observed" else range(8, 12)
    return float(np.mean([((q[t] @ k[t + 1].T).argmax(1) == np.arange(GRID * GRID)).mean() for t in times]))


def tracking_metrics(fields, phrase_mask, case, tracks, config):
    import demo_main_prompt_k_tracking as demo
    import render_vace_sparse_future_demo as future_demo
    from evaluate_vace_sparse_future_tracks import select_points, summarize
    from search_vace_tracking_heads import xy_to_cell

    xy_all = tracks["xy"].copy()
    xy_all[..., 0] = 255 - xy_all[..., 0]  # tracking render -> training-video coordinates
    data = {**tracks, "xy": xy_all}
    result = {}
    for branch in ("observed", "future"):
        temporal = 12 if branch == "observed" else 13
        values = {}
        for source, target in fields.items():
            branch_fields = {key: value.float().numpy() for key, value in target[branch].items() if key in FIELDS}
            values[source] = {
                "heat": branch_fields["text_attention"][..., phrase_mask].sum(-1).reshape(-1, GRID, GRID),
                "q": branch_fields["track_q"].reshape(-1, branch_fields["track_q"].shape[-1]),
                "k": branch_fields["track_k"].reshape(-1, branch_fields["track_k"].shape[-1]),
                "fields": {22: branch_fields["seg_k_l22h9"], 15: branch_fields["seg_k_l15h7"]}}
        records = []
        for anchor in ANCHORS[branch]:
            end = min(anchor + 3, temporal)
            times = np.asarray([source_frame(case["start"], branch, t) for t in range(anchor, end)])
            if times.max() >= len(data["xy"]) or not np.isin(times, data["annotated_frame_indices"]).all():
                continue
            ids = np.concatenate((select_points(data, times[0], [TARGET_BODY[case["episode"]]], GRID, 64),
                                  select_points(data, times[0], GRIPPER, GRID, 64)))
            if not len(ids):
                continue
            xy = data["xy"][times][:, ids].transpose(1, 0, 2)
            world = data["world_xyz"][times][:, ids].transpose(1, 0, 2)
            valid = data["visible"][times][:, ids].T.copy()
            valid[:, 0] = False
            groups = np.where(np.isin(data["point_label"][ids], GRIPPER), "gripper", "moving_object")
            displaced = (np.linalg.norm(xy - xy[:, :1], axis=-1) > 8) & (np.linalg.norm(world - world[:, :1], axis=-1) > 0.01)
            row = {"groups": groups, "xy": xy, "valid": valid, "displaced": displaced,
                   "hold": np.repeat(xy[:, :1], len(times), axis=1)}
            chosen = xy_to_cell(xy[:, 0], GRID, 256)
            for source, value in values.items():
                gate = future_demo.decode_tracks(value, (anchor,), config)["region_scores"][0, :end]
                raw = demo.track_dense(value["q"][:end * GRID**2], value["k"][:end * GRID**2], chosen, anchor, GRID, gate,
                                       topk=config.beam_topk, beam_size=config.beam_size, max_step=config.beam_max_step,
                                       motion_weight=config.motion_weight, local_weight=config.local_weight,
                                       region_weight=config.region_weight, acceleration_weight=config.acceleration_weight,
                                       descriptor_ema=config.descriptor_ema, direction="q_to_k")
                paths = demo.regularize_paths(raw, config.regularize_max_step, config.regularize_max_acceleration, passes=2)
                row[source] = (paths + 0.5) * (256 / GRID) - 0.5
            records.append(row)
        summary = summarize(records, (*fields, "hold")) if records else {}
        result[branch] = {
            "self_match": {source: self_match(target, branch) for source, target in fields.items()},
            "gt_displaced": {mode: {group: {k: summary[mode][f"{group}_displaced"][k] for k in ("epe_px", "within8", "count")}
                                    for group in ("moving_object", "gripper")} for mode in summary},
        }
    return result


def depth_metrics(fields, case, depth, tracks):
    from scipy.stats import spearmanr

    depth = depth[:, :, ::-1]
    xy = tracks["xy"].copy()
    xy[..., 0] = 255 - xy[..., 0]
    result = {}
    for branch, natives in (("observed", range(4, 12)), ("future", range(8, 13))):
        rows = {source: {"full": [], "object_cells": []} for source in fields}
        for native in natives:
            frame = source_frame(case["start"], branch, native)
            if frame >= len(depth):
                continue
            truth = cv2.resize(depth[frame].astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA).ravel()
            visible = tracks["visible"][frame]
            cells = np.unique(np.clip(np.floor((xy[frame, visible] + 0.5) * GRID / 256).astype(int), 0, GRID - 1)
                              @ np.array([1, GRID]))
            for source, target in fields.items():
                value = target[branch]["depth"].float().numpy()[native].ravel()
                rows[source]["full"].append(spearmanr(value, truth).correlation)
                if len(cells) > 8:
                    rows[source]["object_cells"].append(spearmanr(value[cells], truth[cells]).correlation)
        result[branch] = {source: {k: (float(np.nanmean(v)) if v else None) for k, v in r.items()}
                          for source, r in rows.items()}
        result[branch]["teacher_student_corr"] = float(np.mean([
            np.corrcoef(fields["teacher"][branch]["depth"].float().numpy()[t].ravel(),
                        fields["student"][branch]["depth"].float().numpy()[t].ravel())[0, 1] for t in natives]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="pipeline.json of the student run")
    parser.add_argument("--tracking-root", type=Path, required=True,
                        help="Extracted runs/vace_tracking_all_suites_20261004 (videos/ and tracks/)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)
    rb.FONT = str(Path(cv2.__file__).parent / "qt/fonts/DejaVuSans.ttf")
    rb.write_h264 = write_h264
    started = time.monotonic()
    config = PipelineConfig.load(args.config)
    summary = json.loads((args.student_run / "summary.json").read_text())
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(config.teacher.text_root) / "tokenizer", local_files_only=True)
    content = content_token_masks(config.teacher.text_root, sorted({row["prompt"] for row in summary}))
    cases = []
    for row in summary:
        first, last = row["frame_indices"]
        video = decode_indices(row["video"], range(first, last + 1), 10)
        cases.append({**row, "rgb": video.transpose(0, 2, 3, 1).copy(), "teacher_rgb": torch.from_numpy(video),
                      "seed": stable_seed(42, row["episode"], row["camera"], row["start"]),
                      "phrase_masks": phrase_positions(tokenizer, row["prompt"], row["phrases"]),
                      "student": torch.load(str(Path(row["directory"]) / "student_fields.pt"),
                                            map_location="cpu", weights_only=True)})

    from vace_latent_distill.teacher import VaceTeacher

    rb.progress(stage="load_teacher", device=args.device)
    teacher = VaceTeacher(config.teacher, "shared", args.device, text_cache[NEGATIVE_PROMPT])
    try:
        for begin in range(0, len(cases), args.batch_size):
            batch = cases[begin:begin + args.batch_size]
            tick = time.monotonic()
            embeddings = [text_cache[case["prompt"]] for case in batch]
            targets = teacher.extract_batch([case["teacher_rgb"] for case in batch], embeddings,
                                            [e.abs().sum(-1) != 0 for e in embeddings],
                                            [case["seed"] for case in batch],
                                            [content[case["prompt"]] for case in batch])
            for case, target in zip(batch, targets, strict=True):
                case["teacher"] = {"target": {b: {k: v.cpu() if torch.is_tensor(v) else v for k, v in target[b].items()}
                                              for b in ("observed", "future")},
                                   "protocol": dict(teacher.protocol)}
            rb.progress(stage="teacher_batch", cases=[f"{c['episode']}-{c['camera']}" for c in batch],
                        seconds=round(time.monotonic() - tick, 1))
    finally:
        teacher.close()
        del teacher

    renders = []
    for case in cases:
        for source in ("teacher", "student"):
            target = case[source]["target"]
            directory = args.output / source / f"episode_{case['episode']:05d}_{case['camera']}"
            directory.mkdir(parents=True)
            if source == "teacher":
                torch.save(case["teacher"], directory / "teacher_fields.pt")
            renders.append({**{k: case[k] for k in ("episode", "camera", "rgb", "phrase_masks", "prompt")},
                            "source": source, "directory": directory,
                            "target": {b: {k: target[b][k].float() for k in FIELDS} for b in ("observed", "future")}})
    rb.fit_display(renders)  # One K color projection across teacher and student.
    jobs = [(r["target"], r["phrase_masks"], content[r["prompt"]], config.teacher) for r in renders]
    import multiprocessing

    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for render, decoded in zip(renders, pool.map(decode_case, jobs), strict=True):
            render["decoded"] = {"names": ["Full instruction", *render["phrase_masks"]], **decoded}
    for render in renders:
        label = "teacher" if render["source"] == "teacher" else f"student step{cases[0]['student'].get('step')}"
        rb.render_case(render, render["directory"], source=label)
        depth_sheet(render, render["directory"], source=render["source"])
        rb.progress(stage="rendered", source=render["source"], episode=render["episode"], camera=render["camera"])

    metrics = []
    for case in cases:
        name = f"episode_{case['episode']:05d}_{case['camera']}"
        compare = args.output / "compare" / name
        compare.mkdir(parents=True)
        stack_media(args.output / "teacher" / name, args.output / "student" / name, compare)
        fields = {"teacher": case["teacher"]["target"], "student": case["student"]["target"]}
        stem = f"episode_{case['tracking_episode']:05d}_{TRACKING_CAMERA[case['camera']]}"
        with np.load(args.tracking_root / "tracks" / f"{stem}_visible_surface.npz") as archive:
            tracks = {k: archive[k] for k in ("xy", "visible", "world_xyz", "point_label", "annotated_frame_indices")}
        with np.load(args.tracking_root / "videos" / f"episode_{case['tracking_episode']:05d}" /
                      f"{TRACKING_CAMERA[case['camera']]}_depth.npz") as archive:
            depth = archive["depth"].astype(np.float32)
        first_phrase = case["phrase_masks"][case["phrases"][0]]
        metrics.append({"case": name, "prompt": case["prompt"],
                        "semantic": semantic_metrics(fields, case["phrase_masks"], content[case["prompt"]]),
                        "tracking": tracking_metrics(fields, first_phrase, case, tracks, config.teacher),
                        "depth": depth_metrics(fields, case, depth, tracks)})
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
        rb.progress(stage="metrics", case=name)
    (args.output / "protocol.json").write_text(json.dumps({
        "student_run": str(args.student_run), "config": str(args.config), "device": args.device,
        "teacher_protocol": cases[0]["teacher"]["protocol"],
        "teacher_input": "Same 49 training-domain frames; observed 33 + future 16 GT with noise (existing protocol)",
        "display": "Shared region-bundle renderer and one K PCA projection over teacher and student",
        "scoring": "Simulator tracks/depth from tracking renders, mirrored x->255-x; evaluation only",
        "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
    rb.progress(stage="completed", output=str(args.output), seconds=round(time.monotonic() - started, 1))


if __name__ == "__main__":
    main()
