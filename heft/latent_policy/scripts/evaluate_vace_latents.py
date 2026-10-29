#!/usr/bin/env python3
"""Unified latent evaluation on cached held-out windows (teacher vs students).

For every held-out window of a ``cache_vace_teacher_local.py`` cache, students named
on the command line are run on the same past 33 frames; their fields are saved for
reuse (e.g. by the action/state probe).  Per source (teacher and each student):

* is the readout right (simulator GT, evaluation only): GT point error of QK + beam
  tracks on displaced object/gripper points, depth Spearman on object cells, and the
  share of each phrase's top heat cells that land on labeled objects;
* is it learned (students vs teacher): K cosine, per-cell word KL, word spatial KL,
  phrase-map contrast and correlation, QK self-match, depth correlation.

Phrases are contiguous content-token runs without a leading verb; that split exists
only here, for scoring.  GT comes from horizontally mirrored tracking renders and is
flipped into training-video coordinates.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vace_latent_distill.config import PipelineConfig, ModelConfig, timeline
from vace_latent_distill.data import decode_indices, resize_rgb

GRID = 32
FIELDS = ("seg_k_l22h9", "seg_k_l15h7", "track_q", "track_k", "text_attention", "depth")
CAMERAS = {"observation.images.image": "agentview", "observation.images.image2": "robot0_eye_in_hand"}
GRIPPER = ["gripper base", "left finger", "right finger"]
NOT_OBJECTS = {*GRIPPER, "wrist"}
VERBS = {"pick", "place", "put", "open", "close", "turn", "push", "stack", "move", "lift", "take", "set"}
ANCHORS = {"observed": (4, 6, 8, 10), "future": (8, 10)}


def source_frame(start, branch, native):
    return start + (native - 4) * 4 if branch == "observed" else start + native * 4


def phrases_of(tokenizer, prompt, content):
    pieces = tokenizer.convert_ids_to_tokens(
        tokenizer(prompt, padding="max_length", max_length=512, truncation=True)["input_ids"])
    groups, current = [], []
    for index, flag in enumerate(content):
        if flag:
            current.append(index)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    phrases = {}
    for group in groups:
        words = [pieces[i].lstrip(chr(9601)) for i in group]
        if words[0].lower() in VERBS:
            group, words = group[1:], words[1:]
        if group:
            name = " ".join(words)
            mask = phrases.get(name, np.zeros(len(content), bool)).copy()
            mask[group] = True                      # repeated phrases share one map
            phrases[name] = mask
    return phrases


def cosine(a, b):
    a, b = (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8) for x in (a, b))
    return float((a * b).sum(-1).mean())


def window_metrics(job):
    """CPU worker: one held-out window, all sources."""
    torch.set_num_threads(2)
    import cv2
    from scipy.stats import spearmanr

    from vace_latent_distill.offline import unpack_target
    from vace_latent_distill.losses import word_spatial_loss
    from vace_latent_distill.teacher import local_scripts, seed_heat

    local_scripts()
    import demo_main_prompt_k_tracking as demo
    import render_vace_sparse_future_demo as future_demo
    from evaluate_vace_sparse_future_tracks import select_points, summarize
    from search_vace_tracking_heads import xy_to_cell

    row, students, tracking_root, config, content, phrases = job
    payload = torch.load(row["target_path"], map_location="cpu", weights_only=True)
    fields = {"teacher": unpack_target(payload, identity=row["provenance_id"])}
    for name, path in students.items():
        fields[name] = torch.load(path, map_location="cpu", weights_only=True)
    np_fields = {s: {b: {k: v[k].float().numpy() for k in FIELDS} for b, v in f.items() if b in ANCHORS}
                 for s, f in fields.items()}
    camera = CAMERAS[row["camera"]]
    has_gt = "tracking_episode" in row  # train windows: learnability only, no simulator GT
    if has_gt:
        stem = f"episode_{row['tracking_episode']:05d}_{camera}"
        with np.load(tracking_root / "tracks" / f"{stem}_visible_surface.npz") as archive:
            tracks = {k: archive[k] for k in ("xy", "visible", "world_xyz", "point_label",
                                              "annotated_frame_indices")}
        tracks["xy"] = tracks["xy"].copy()
        tracks["xy"][..., 0] = 255 - tracks["xy"][..., 0]
        with np.load(tracking_root / "videos" / f"episode_{row['tracking_episode']:05d}"
                     / f"{camera}_depth.npz") as a:
            depth = a["depth"].astype(np.float32)[:, :, ::-1]
        objects = sorted(set(tracks["point_label"].tolist()) - NOT_OBJECTS)
    else:
        tracks = {"visible": np.zeros((0, 0), bool)}
        depth = np.zeros((0, GRID, GRID), np.float32)
        objects = []
    result = {"sample_id": row["sample_id"], "camera": camera, "task": row["task"], "phrases": list(phrases)}
    times = timeline()
    for branch in ("observed", "future"):
        valid = times[branch]["valid"].numpy()
        natives = np.flatnonzero(valid)
        out = {}
        teacher = np_fields["teacher"][branch]
        for source, by_branch in np_fields.items():
            values = by_branch[branch]
            entry = {}
            # Learnability against the teacher.
            if source != "teacher":
                for key in ("seg_k_l22h9", "seg_k_l15h7"):
                    entry[f"{key}_cosine"] = cosine(values[key][valid], teacher[key][valid])
                p, t = values["text_attention"][valid], teacher["text_attention"][valid]
                entry["word_kl"] = float((t * (np.log(np.maximum(t, 1e-12)) - np.log(np.maximum(p, 1e-7)))).sum(-1).mean())
                entry["word_spatial_kl"] = float(word_spatial_loss(
                    torch.from_numpy(values["text_attention"])[None], torch.from_numpy(teacher["text_attention"])[None],
                    torch.from_numpy(content)[None], torch.from_numpy(valid)[None]))
                entry["depth_teacher_corr"] = float(np.mean([np.corrcoef(values["depth"][n], teacher["depth"][n])[0, 1]
                                                             for n in natives]))
            # Phrase maps: contrast, separation, landing on objects (GT cells).
            heats = {name: values["text_attention"][..., mask].sum(-1) for name, mask in phrases.items()}
            entry["phrase_contrast"] = float(np.mean([(h[valid].max(-1) / np.maximum(h[valid].mean(-1), 1e-8)).mean()
                                                      for h in heats.values()])) if heats else None
            names = list(heats)
            pairs = [np.mean([np.corrcoef(heats[a][n], heats[b][n])[0, 1] for n in natives])
                     for i, a in enumerate(names) for b in names[i + 1:]]
            entry["phrase_pair_corr"] = float(np.mean(pairs)) if pairs else None
            hits, coverage = [], []
            for native in natives:
                frame = source_frame(row["start"], branch, int(native))
                if frame >= len(tracks["visible"]):
                    continue
                visible = tracks["visible"][frame] & np.isin(tracks["point_label"], objects + GRIPPER)
                cells = set(xy_to_cell(tracks["xy"][frame, visible], GRID, 256).tolist())
                coverage.append(len(cells) / GRID**2)
                for heat in heats.values():
                    top = np.argsort(heat[native])[-51:]
                    hits.append(np.mean([cell in cells for cell in top]))
            entry["phrase_top5pct_on_objects"] = float(np.mean(hits)) if hits else None
            entry["object_cell_fraction"] = float(np.mean(coverage)) if coverage else None  # chance level
            # QK self-match.
            q = values["track_q"] / np.maximum(np.linalg.norm(values["track_q"], axis=-1, keepdims=True), 1e-8)
            k = values["track_k"] / np.maximum(np.linalg.norm(values["track_k"], axis=-1, keepdims=True), 1e-8)
            entry["self_match"] = float(np.mean([((q[n] @ k[n + 1].T).argmax(1) == np.arange(GRID * GRID)).mean()
                                                 for n in natives[:-1]]))
            # Depth against simulator depth.
            full, obj = [], []
            for native in natives:
                frame = source_frame(row["start"], branch, int(native))
                if frame >= len(depth):
                    continue
                truth = cv2.resize(depth[frame], (GRID, GRID), interpolation=cv2.INTER_AREA).ravel()
                visible = tracks["visible"][frame]
                cells = np.unique(xy_to_cell(tracks["xy"][frame, visible], GRID, 256))
                full.append(spearmanr(values["depth"][native], truth).correlation)
                if len(cells) > 8:
                    obj.append(spearmanr(values["depth"][native][cells], truth[cells]).correlation)
            entry["depth_spearman_full"] = float(np.nanmean(full)) if full else None
            entry["depth_spearman_objects"] = float(np.nanmean(obj)) if obj else None
            out[source] = entry
        # GT tracks: identical queries for all sources, each source's own prompt/K gate.
        records = []
        for anchor in (ANCHORS[branch] if has_gt else ()):
            end = min(anchor + 3, len(valid))
            frames = np.asarray([source_frame(row["start"], branch, t) for t in range(anchor, end)])
            if frames.max() >= len(tracks["xy"]) or not np.isin(frames, tracks["annotated_frame_indices"]).all():
                continue
            ids = np.concatenate((select_points(tracks, frames[0], objects, GRID, 64),
                                  select_points(tracks, frames[0], GRIPPER, GRID, 64)))
            if not len(ids):
                continue
            xy = tracks["xy"][frames][:, ids].transpose(1, 0, 2)
            world = tracks["world_xyz"][frames][:, ids].transpose(1, 0, 2)
            gt_valid = tracks["visible"][frames][:, ids].T.copy()
            gt_valid[:, 0] = False
            groups = np.where(np.isin(tracks["point_label"][ids], GRIPPER), "gripper", "moving_object")
            displaced = (np.linalg.norm(xy - xy[:, :1], axis=-1) > 8) & (np.linalg.norm(world - world[:, :1], axis=-1) > 0.01)
            record = {"groups": groups, "xy": xy, "valid": gt_valid, "displaced": displaced,
                      "hold": np.repeat(xy[:, :1], len(frames), axis=1)}
            chosen = xy_to_cell(xy[:, 0], GRID, 256)
            for source, values in np_fields.items():
                heat = seed_heat(values[branch]["text_attention"], content, "spatial_normalized_max")
                decoder_values = {"heat": heat.reshape(-1, GRID, GRID),
                                  "q": values[branch]["track_q"].reshape(-1, 128),
                                  "k": values[branch]["track_k"].reshape(-1, 128),
                                  "fields": {22: values[branch]["seg_k_l22h9"], 15: values[branch]["seg_k_l15h7"]}}
                gate = future_demo.decode_tracks(decoder_values, (anchor,), config)["region_scores"][0, :end]
                raw = demo.track_dense(decoder_values["q"][:end * GRID**2], decoder_values["k"][:end * GRID**2],
                                       chosen, anchor, GRID, gate, topk=config.beam_topk, beam_size=config.beam_size,
                                       max_step=config.beam_max_step, motion_weight=config.motion_weight,
                                       local_weight=config.local_weight, region_weight=config.region_weight,
                                       acceleration_weight=config.acceleration_weight,
                                       descriptor_ema=config.descriptor_ema, direction="q_to_k")
                paths = demo.regularize_paths(raw, config.regularize_max_step, config.regularize_max_acceleration, passes=2)
                record[source] = (paths + 0.5) * (256 / GRID) - 0.5
            records.append(record)
        if records:
            gt = summarize(records, (*np_fields, "hold"))
            for source in (*np_fields, "hold"):
                out.setdefault(source, {})
                for group in ("moving_object", "gripper"):
                    value = gt[source][f"{group}_displaced"]
                    out[source][f"gt_{group}_epe"] = value["epe_px"]
                    out[source][f"gt_{group}_count"] = value["count"]
        result[branch] = out
    return result


def aggregate(results, sources):
    """Mean over windows, overall and per camera; GT errors weighted by point count."""
    summary = {}
    for scope in ("all", "agentview", "robot0_eye_in_hand"):
        chosen = [r for r in results if scope == "all" or r["camera"] == scope]
        for branch in ("observed", "future"):
            for source in (*sources, "hold"):
                entries = [r[branch].get(source, {}) for r in chosen]
                keys = sorted({k for e in entries for k in e if not k.endswith("_count")})
                values = {}
                for key in keys:
                    if key.startswith("gt_"):
                        counts = [e.get(key.replace("_epe", "_count"), 0) for e in entries]
                        pairs = [(e[key], c) for e, c in zip(entries, counts) if e.get(key) is not None and c]
                        values[key] = (sum(v * c for v, c in pairs) / sum(c for _, c in pairs)) if pairs else None
                    else:
                        series = [e[key] for e in entries if e.get(key) is not None and np.isfinite(e[key])]
                        values[key] = float(np.mean(series)) if series else None
                summary.setdefault(scope, {}).setdefault(branch, {})[source] = values
    return summary


def write_markdown(summary, sources, path):
    rows = [("K L22H9 cosine (S vs T)", "seg_k_l22h9_cosine", False), ("K L15H7 cosine", "seg_k_l15h7_cosine", False),
            ("word KL", "word_kl", False), ("word spatial KL", "word_spatial_kl", False),
            ("phrase contrast (max/mean)", "phrase_contrast", True), ("phrase pair corr (lower=sharper)", "phrase_pair_corr", True),
            ("phrase top5% on objects", "phrase_top5pct_on_objects", True),
            ("object cell fraction (chance)", "object_cell_fraction", True), ("QK self-match", "self_match", True),
            ("GT object EPE px", "gt_moving_object_epe", True), ("GT gripper EPE px", "gt_gripper_epe", True),
            ("depth Spearman objects", "depth_spearman_objects", True), ("depth corr with teacher", "depth_teacher_corr", False)]
    columns = ["teacher", *sources, "hold"]
    lines = ["# Latent evaluation on held-out windows", ""]
    for scope, branches in summary.items():
        for branch, values in branches.items():
            lines += [f"## {scope} / {branch}", "", "| metric | " + " | ".join(columns) + " |",
                      "|---|" + "---|" * len(columns)]
            for label, key, with_teacher in rows:
                cells = []
                for column in columns:
                    value = values.get(column, {}).get(key)
                    if column == "teacher" and not with_teacher or (column == "hold" and not key.startswith("gt_")):
                        cells.append("")
                    else:
                        cells.append("–" if value is None else f"{value:.3f}")
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="name=path, repeatable")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("heldout", "train"), default="heldout")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig.load(args.cache_dir / "pipeline.json")
    rows = [json.loads(l) for l in (args.cache_dir / f"manifest_{args.split}.jsonl").read_text().splitlines()
            if l.strip()]
    rows = rows[:args.limit] if args.limit else rows
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    from transformers import AutoTokenizer

    from vace_latent_distill.model import StructuredStudent
    from vace_latent_distill.teacher import content_token_masks

    tokenizer = AutoTokenizer.from_pretrained(Path(config.teacher.text_root) / "tokenizer", local_files_only=True)
    content = content_token_masks(config.teacher.text_root, sorted({r["task"] for r in rows}))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    students = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        directory = args.output / "fields" / name
        directory.mkdir(parents=True, exist_ok=True)
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        model = StructuredStudent(ModelConfig(**saved["config"]["model"]), pretrained=False)
        model.load_state_dict(saved["model"], strict=True)
        del saved
        model = model.float().eval().to(device)
        tick = time.monotonic()
        for row in rows:
            path_out = directory / f"{row['sample_id']}.pt"
            if path_out.is_file():
                continue
            video = decode_indices(row["video"], row["frame_indices"][:33], row["fps"])
            rgb = resize_rgb(torch.from_numpy(video), 256)[None].to(device)
            text = text_cache[row["task"]].float()[None].to(device)
            output = model(rgb, text, text.abs().sum(-1) != 0)
            torch.save({b: {k: output[b][k][0].half().cpu() for k in FIELDS} for b in ("observed", "future")},
                       path_out)
        print(json.dumps({"stage": "student_fields", "name": name, "windows": len(rows),
                          "seconds": round(time.monotonic() - tick, 1)}), flush=True)
        students[name] = directory
        del model
    import multiprocessing

    jobs = [(row, {n: d / f"{row['sample_id']}.pt" for n, d in students.items()}, args.tracking_root,
             config.teacher, content[row["task"]], phrases_of(tokenizer, row["task"], content[row["task"]]))
            for row in rows]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(window_metrics, jobs))
    (args.output / "per_window.json").write_text(json.dumps(results, indent=1) + "\n")
    summary = aggregate(results, ["teacher", *students])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, list(students), args.output / "summary.md")
    print((args.output / "summary.md").read_text())


if __name__ == "__main__":
    main()
