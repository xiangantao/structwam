#!/usr/bin/env python3
"""Compare native main/control QK over all LIBERO suites with fixed heads."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from evaluate_vace_pure_noise_future import (
    ROOT, UniPCMultistepScheduler, load_transformer, load_vae, log, read_video, write_video,
)
from search_vace_tracking_heads import CAMERAS, GRIPPER, cell_to_xy, xy_to_cell

PHASES = ("moving_object_displaced", "moving_object_visible", "static_object", "gripper_displaced", "gripper_visible")
FIXED = (("main", 15, 4, "anchor_qk"), ("main", 15, 4, "rolling_qk"),
         ("control", 7, 11, "anchor_qk"), ("control", 7, 11, "rolling_qk"),
         ("control", 11, 1, "anchor_qk"), ("control", 11, 1, "rolling_qk"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def select_cases(args):
    split = json.loads((args.data / "split.json").read_text())
    allowed = set(split["val_episodes"])
    cases = []
    for record in split["episodes"]:
        if record["episode"] not in allowed:
            continue
        directory = args.data / "videos" / f"episode_{record['episode']:05d}"
        meta = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "sim_state.npz") as archive:
            positions, rotations = archive["body_xpos"], archive["body_xmat"]
        names = meta["body_names"]
        roots = [name.removesuffix("_main") for name in names if name.endswith("_main")]
        bodies = [i for i, name in enumerate(names) if any(name == root or name.startswith(root + "_") for root in roots)]
        hand = names.index("gripper0_eef")
        starts = range(16 if len(positions) >= 65 else 0, len(positions) - args.frames + 1, 4)
        choices = []
        for start in starts:
            anchor = start + args.observed - 1
            times = np.arange(anchor, start + args.frames, 4)
            translation = np.linalg.norm(positions[times][:, bodies] - positions[anchor, bodies], axis=-1)
            rotation = np.linalg.norm(rotations[times][:, bodies] - rotations[anchor, bodies], axis=-1) * 0.05
            object_motion = float(np.maximum(translation, rotation).max())
            hand_motion = float(np.linalg.norm(positions[times, hand] - positions[anchor, hand], axis=-1).max())
            choices.append((object_motion + 0.01 * hand_motion, start, object_motion, hand_motion))
        if not choices:
            raise ValueError(f"Episode {record['episode']} is too short for the fixed timeline")
        chosen = max(choices, key=lambda row: (row[0], -row[1]))
        if record["episode"] == 39:
            chosen = next(row for row in choices if row[1] == 48)
        for camera in CAMERAS:
            cases.append(record | {"camera": camera, "start": chosen[1], "fps": meta["fps"],
                                   "object_motion_selection_score_m": chosen[2], "hand_motion_m": chosen[3],
                                   "calibration_episode": record["episode"] == 39,
                                   "name": f"episode_{record['episode']:05d}_{camera}"})
    return cases


def seed_ids(data, anchor, grid, per_body):
    selected = []
    for label in np.unique(data["point_label"]):
        if label == "wrist":
            continue
        subjects = np.flatnonzero((data["point_label"] == label) & data["visible"][anchor])
        used = set()
        for body in np.unique(data["body_id"][subjects]):
            eligible = subjects[data["body_id"][subjects] == body]
            xy = data["xy"][anchor, eligible]
            cells = xy_to_cell(xy, grid, 256)
            remaining, chosen = set(range(len(eligible))), []
            distances = np.linalg.norm(xy - xy.mean(0), axis=-1)
            choice = int(distances.argmin())
            while remaining and len(chosen) < per_body:
                if int(cells[choice]) not in used:
                    selected.append(int(eligible[choice]))
                    chosen.append(choice)
                    used.add(int(cells[choice]))
                remaining.remove(choice)
                if not remaining:
                    break
                if chosen:
                    distances = np.linalg.norm(xy[:, None] - xy[chosen][None], axis=-1).min(1)
                choice = max(remaining, key=lambda i: distances[i])
    if not selected:
        raise ValueError("No annotated surfaces are visible at the query anchor")
    return np.asarray(selected, dtype=np.int64)


def prepare_points(args, case):
    times = np.arange(case["start"] + args.observed - 1, case["start"] + args.frames, 4)
    path = args.data / "tracks" / f"episode_{case['episode']:05d}_{case['camera']}_visible_surface.npz"
    with np.load(path) as archive:
        data = {key: archive[key] for key in ("xy", "visible", "world_xyz", "point_label", "body_id", "annotated_frame_indices")}
    if not np.isin(times, data["annotated_frame_indices"]).all():
        raise ValueError(f"Missing annotation times in {case['name']}")
    ids = seed_ids(data, times[0], args.size // 16, args.points_per_body)
    xy = data["xy"][times][:, ids].transpose(1, 0, 2)
    valid = data["visible"][times][:, ids].T.copy()
    valid[:, 0] = False
    world = data["world_xyz"][times][:, ids].transpose(1, 0, 2)
    world_delta = np.linalg.norm(world - world[:, :1], axis=-1)
    displaced = valid & (world_delta > 0.01) & (np.linalg.norm(xy - xy[:, :1], axis=-1) > 8)
    labels = data["point_label"][ids]
    gripper = np.isin(labels, list(GRIPPER))
    moving_labels = []
    for label in np.unique(data["point_label"]):
        all_ids = np.flatnonzero(data["point_label"] == label)
        excursion = np.linalg.norm(data["world_xyz"][times][:, all_ids] - data["world_xyz"][times[0], all_ids], axis=-1).max()
        if excursion >= 0.02:
            moving_labels.append(str(label))
    moving = np.isin(labels, moving_labels) & ~gripper
    phase_masks = {
        "moving_object_displaced": displaced & moving[:, None],
        "moving_object_visible": valid & moving[:, None],
        "static_object": valid & (~moving & ~gripper)[:, None] & (world_delta <= 0.01),
        "gripper_displaced": displaced & gripper[:, None],
        "gripper_visible": valid & gripper[:, None],
    }
    recent = np.asarray([times[0] - 8, times[0] - 4, times[0]])
    recent_valid = data["visible"][recent][:, ids].all(0)
    velocity = (data["xy"][recent[-1], ids] - data["xy"][recent[0], ids]) / 8
    velocity[~recent_valid] = 0
    static = np.repeat(xy[:, :1], len(times), axis=1)
    points = {"ids": ids, "labels": labels, "xy": xy, "valid": valid, "displaced": displaced,
              "cells": xy_to_cell(xy, args.size // 16, 256), "times": times,
              "anchor_time": (args.observed - 1) // 4,
              "latent_times": np.arange(case["start"], case["start"] + args.frames, 4),
              "velocity_available": recent_valid, "phase_masks": phase_masks,
              "score_labels": np.where(gripper, "gripper", labels)}
    baselines = {"hold_position": static,
                 "recent_velocity": xy[:, :1] + velocity[:, None] * (times - times[0])[None, :, None],
                 "grid_quantization_oracle": cell_to_xy(points["cells"], args.size // 16, 256)}
    return points, baselines


def measure_many(predictions, points):
    error = np.linalg.norm(predictions - points["xy"][None], axis=-1)
    output = [{} for _ in range(len(predictions))]
    for phase, mask in points["phase_masks"].items():
        epe, within, trajectory, endpoint = [], [], [], []
        subjects = []
        for label in np.unique(points["score_labels"]):
            selected = mask & (points["score_labels"] == label)[:, None]
            if not selected.any():
                continue
            subjects.append(str(label))
            epe.append(error[:, selected].mean(1))
            within.append((error[:, selected] <= 8).mean(1))
            count = selected.sum(1)
            eligible = count >= 3
            if eligible.any():
                ratio = ((error <= 8) & selected[None]).sum(2) / np.maximum(count, 1)[None]
                trajectory.append((ratio[:, eligible] >= 0.8).mean(1))
            if selected[:, -1].any():
                endpoint.append(error[:, :, -1][:, selected[:, -1]].mean(1))
        vectors = {"epe_px": np.mean(epe, axis=0) if epe else None,
                   "within8": np.mean(within, axis=0) if within else None,
                   "trajectory80_at8": np.mean(trajectory, axis=0) if trajectory else None,
                   "endpoint_epe_px": np.mean(endpoint, axis=0) if endpoint else None}
        for hi in range(len(predictions)):
            output[hi][phase] = {key: float(value[hi]) if value is not None else None for key, value in vectors.items()}
            output[hi][phase].update(count=int(mask.sum()), objects=len(subjects))
    return output


class Capture:
    def __init__(self, points, grid):
        self.points, self.grid = points, grid
        self.enabled = False
        self.experiment, self.state = None, None
        self.rows, self.fixed_predictions = [], {}

    def __call__(self, branch, layer, query, key):
        if not self.enabled:
            return
        p, spatial, heads = self.points, self.grid ** 2, query.shape[1]
        temporal, dim = len(p["latent_times"]), query.shape[-1]
        if query.shape != key.shape or query.shape[2] != temporal * spatial:
            raise ValueError("Native attention shape does not match annotated times")
        q = query[0].reshape(heads, temporal, spatial, dim)
        k = key[0].reshape(heads, temporal, spatial, dim)
        anchor = int(p["anchor_time"])
        initial = torch.as_tensor(p["cells"][:, 0], device=q.device)
        scores = torch.einsum("hpd,htsd->hpts", q[:, anchor, initial].float(), k[:, anchor + 1:].float()) / math.sqrt(dim)
        fixed = torch.cat((initial[None, :, None].expand(heads, -1, -1), scores.argmax(-1)), -1)
        head_index = torch.arange(heads, device=q.device)[:, None]
        current = initial[None].expand(heads, -1).clone()
        paths = [current]
        for ti in range(anchor + 1, temporal):
            scores = torch.einsum("hpd,hsd->hps", q[:, ti - 1][head_index, current].float(), k[:, ti].float()) / math.sqrt(dim)
            current = scores.argmax(-1)
            paths.append(current)
        for decoder, cells in (("anchor_qk", fixed), ("rolling_qk", torch.stack(paths, -1))):
            predictions = cell_to_xy(cells.cpu().numpy(), self.grid, 256)
            predictions[:, :, 0] = p["xy"][:, 0]
            measurements = measure_many(predictions, p)
            for hi in range(heads):
                row = {"experiment": self.experiment, "state_after_steps": self.state,
                       "branch": branch, "layer": layer, "head": hi, "decoder": decoder,
                       "metrics": measurements[hi]}
                self.rows.append(row)
                if (branch, layer, hi, decoder) in FIXED:
                    name = f"{self.experiment}_after{self.state:02d}_{branch}_L{layer}H{hi}_{decoder}"
                    self.fixed_predictions[name] = predictions[hi]


@torch.no_grad()
def encode_texts(path, prompts, device):
    from transformers import AutoTokenizer, UMT5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(path / "text_encoder", torch_dtype=torch.bfloat16,
                                             local_files_only=True).to(device).eval()
    embeddings = {}
    unique = sorted(set(prompts))
    for start in range(0, len(unique), 4):
        batch = unique[start:start + 4]
        tokens = tokenizer(batch, padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
        values = encoder(**tokens).last_hidden_state
        for i, text in enumerate(batch):
            values[i, int(tokens.attention_mask[i].sum()):] = 0
            embeddings[text] = values[i:i + 1].detach().cpu()
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings


@torch.no_grad()
def video_inputs(vae, video, args, device, zero_cache):
    pixels = torch.from_numpy(video.copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
    pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=(args.size, args.size), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    prefix = pixels.clone()
    prefix[:, args.observed:] = 0
    inactive = vae.encode([prefix])[0]
    clean = vae.encode([pixels])[0][None].float()
    shape = (16, (args.frames - 1) // 4 + 1, args.size // 8, args.size // 8)
    if inactive.shape != shape or clean.shape != (1, *shape):
        raise ValueError("Unexpected native VAE timeline")
    if "zero" not in zero_cache:
        zero_cache["zero"] = vae.encode([torch.zeros_like(pixels)])[0]
    reactive = zero_cache["zero"]
    mask = torch.ones((args.frames, args.size, args.size), device=device)
    mask[:args.observed] = 0
    mask = mask.view(args.frames, args.size // 8, 8, args.size // 8, 8).permute(2, 4, 0, 1, 3).flatten(0, 1)
    mask = F.interpolate(mask[None], size=inactive.shape[1:], mode="nearest-exact")[0]
    control = torch.cat((inactive, reactive, mask), 0)[None].bfloat16()
    observed_all = torch.cat((clean[0].bfloat16(), reactive, torch.zeros_like(mask)), 0)[None].bfloat16()
    return control, observed_all, clean


def overlay(video, points, prediction, phase, title, ti):
    frame = cv2.resize(video[points["times"][ti]], (512, 512))
    cv2.rectangle(frame, (0, 0), (512, 44), (0, 0, 0), -1)
    cv2.putText(frame, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (255, 255, 255), 1)
    cv2.putText(frame, "GT green / QK magenta", (6, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    active = points["phase_masks"][phase].any(1)
    selected = []
    for label in np.unique(points["score_labels"][active]):
        selected.extend(np.flatnonzero(active & (points["score_labels"] == label))[:4].tolist())
    for pid in selected:
        if ti and not points["valid"][pid, ti]:
            continue
        gt, pred = np.round(points["xy"][pid] * 2).astype(int), np.round(prediction[pid] * 2).astype(int)
        for previous in range(ti):
            if (previous == 0 or points["valid"][pid, previous]) and points["valid"][pid, previous + 1]:
                cv2.line(frame, tuple(gt[previous]), tuple(gt[previous + 1]), (40, 240, 60), 1)
                cv2.line(frame, tuple(pred[previous]), tuple(pred[previous + 1]), (240, 40, 210), 1)
        cv2.circle(frame, tuple(gt[ti]), 3, (40, 240, 60), -1)
        cv2.circle(frame, tuple(pred[ti]), 3, (240, 40, 210), -1)
    return frame


def previews(directory, video, points, capture, case):
    images = []
    for phase in ("moving_object_displaced", "gripper_visible"):
        if not points["phase_masks"][phase].any():
            continue
        control_layer, control_head = (11, 1) if phase.startswith("moving") else (7, 11)
        main = capture.fixed_predictions["gt_assisted_after13_main_L15H4_anchor_qk"]
        control = capture.fixed_predictions[f"gt_assisted_after13_control_L{control_layer}H{control_head}_anchor_qk"]
        frames = []
        for ti in range(len(points["times"])):
            left = overlay(video, points, main, phase, f"{case['name']} after13 main L15H4", ti)
            right = overlay(video, points, control, phase, f"after13 control L{control_layer}H{control_head}", ti)
            frames.append(np.concatenate((left, right), axis=1))
        write_video(directory / f"{phase}.mp4", np.stack(frames * 3), 5)
        images.append(frames[-1])
    if images:
        cv2.imwrite(str(directory / "endpoint_comparison.jpg"), cv2.cvtColor(np.concatenate(images, 0), cv2.COLOR_RGB2BGR))


@torch.no_grad()
def run_worker(args, cases):
    output = args.output / "cases"
    output.mkdir(parents=True, exist_ok=True)
    assigned = [case for i, case in enumerate(cases) if i % args.shards == args.shard_index]
    pending = [case for case in assigned if not (output / case["name"] / "completed.json").exists()]
    status_path = args.output / f"shard_{args.shard_index}_status.json"
    started = time.monotonic()
    dump(status_path, {"state": "loading", "assigned": len(assigned), "pending": len(pending)})
    if not pending:
        dump(status_path, {"state": "complete", "cases": len(assigned)})
        return
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    negative_prompt = "overexposed, blurry, subtitles, low quality, deformed, extra fingers, camera cut"
    texts = encode_texts(args.text_model, [case["task"] for case in pending] + ["", negative_prompt], device)
    negative, empty = texts[negative_prompt].to(device), texts[""].to(device)
    model, vae = load_transformer(args.model, device), load_vae(args.model / "Wan2.1_VAE.pth", device)
    scales = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
    active_capture = [None]
    for branch, blocks in (("main", model.blocks), ("control", model.vace_blocks)):
        for layer, block in enumerate(blocks):
            block.attn1._heft_capture = lambda branch=branch, layer=layer, **kw: active_capture[0](branch, layer, kw["query"], kw["key"])
    zero_cache = {}
    for completed, case in enumerate(pending):
        directory = output / case["name"]
        directory.mkdir(parents=True, exist_ok=True)
        points, base_predictions = prepare_points(args, case)
        baselines = {name: measure_many(prediction[None], points)[0] for name, prediction in base_predictions.items()}
        video = read_video(args.data / "videos" / f"episode_{case['episode']:05d}" / f"{case['camera']}.mp4")
        truth = video[case["start"]:case["start"] + args.frames]
        if len(truth) != args.frames:
            raise ValueError(f"Incomplete RGB window in {case['name']}")
        control, observed_all, clean = video_inputs(vae, truth, args, device, zero_cache)
        capture = Capture(points, args.size // 16)
        active_capture[0] = capture
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn((1, 16, 17, args.size // 8, args.size // 8), generator=generator, device=device)[:, :, :clean.shape[2]]
        capture.experiment, capture.state, capture.enabled = "observed_all_noise", 0, True
        prediction = model(hidden_states=noise.bfloat16(), timestep=torch.tensor([1000], device=device),
                           encoder_hidden_states=empty, control_hidden_states=observed_all,
                           control_hidden_states_scale=scales, return_dict=False)[0]
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Nonfinite all-observed model prediction")
        capture.enabled = False
        scheduler = UniPCMultistepScheduler.from_pretrained(args.text_model / "scheduler", local_files_only=True, flow_shift=3)
        scheduler.set_timesteps(40, device=device)
        begin = int((scheduler.sigmas[:-1] - 0.8).abs().argmin())
        scheduler.set_begin_index(begin)
        timesteps = scheduler.timesteps[begin:]
        latents = scheduler.add_noise(clean, noise, timesteps[:1])
        positive = texts[case["task"]].to(device)
        states = []
        for index, timestep in enumerate(timesteps):
            if index > max(args.states):
                break
            capture.experiment, capture.state = "gt_assisted", index
            capture.enabled = index in args.states
            inputs = {"hidden_states": latents.bfloat16(), "timestep": timestep.reshape(1),
                      "control_hidden_states": control, "control_hidden_states_scale": scales, "return_dict": False}
            conditional = model(**inputs, encoder_hidden_states=positive)[0]
            if capture.enabled:
                states.append({"after_steps": index, "sigma": float(scheduler.sigmas[begin + index])})
            capture.enabled = False
            if index == max(args.states):
                break
            unconditional = model(**inputs, encoder_hidden_states=negative)[0]
            velocity = unconditional.float() + 5 * (conditional.float() - unconditional.float())
            latents = scheduler.step(velocity, timestep, latents, return_dict=False)[0]
            if not torch.isfinite(latents).all():
                raise RuntimeError("Nonfinite denoising state")
            if index % 5 == 0:
                dump(status_path, {"state": "running", "case": case["name"], "after_steps": index + 1,
                                   "finished_this_worker": completed, "pending_at_start": len(pending),
                                   "elapsed_seconds": time.monotonic() - started})
        baselines = {name: measure_many(prediction[None], points)[0] for name, prediction in base_predictions.items()}
        dump(directory / "rows.json", capture.rows)
        np.savez_compressed(directory / "fixed_trajectories.npz", **capture.fixed_predictions)
        np.savez_compressed(directory / "points.npz", **{k: v for k, v in points.items() if k != "phase_masks"},
                            **{"mask_" + key: value for key, value in points["phase_masks"].items()})
        previews(directory, video, points, capture, case)
        dump(directory / "completed.json", {"case": case, "states": states, "rows": len(capture.rows),
                                             "seeds": len(points["ids"]), "baselines": baselines})
        chosen = next(row for row in capture.rows if row["experiment"] == "gt_assisted" and row["state_after_steps"] == 13
                      and (row["branch"], row["layer"], row["head"], row["decoder"]) == FIXED[0])
        log(stage="case_complete", shard=args.shard_index, case=case["name"], finished=completed + 1,
            pending_at_start=len(pending), metrics=chosen["metrics"], elapsed_seconds=time.monotonic() - started)
        del capture, latents, clean, control, observed_all, conditional, unconditional, inputs, prediction
        active_capture[0] = None
    for blocks in (model.blocks, model.vace_blocks):
        for block in blocks:
            del block.attn1._heft_capture
    dump(status_path, {"state": "complete", "cases": len(assigned), "elapsed_seconds": time.monotonic() - started})


def aggregate(observations):
    output = {}
    for phase in PHASES:
        strata = {}
        for suite, camera in sorted({(case["suite"], case["camera"]) for case, _ in observations}):
            eligible = [metrics[phase] for case, metrics in observations
                        if case["suite"] == suite and case["camera"] == camera and metrics[phase]["epe_px"] is not None]
            stratum = {"cases": len(eligible), "point_times": sum(row["count"] for row in eligible)}
            for metric in ("epe_px", "within8", "trajectory80_at8", "endpoint_epe_px"):
                values = [row[metric] for row in eligible if row[metric] is not None]
                stratum[metric] = float(np.mean(values)) if values else None
            strata[f"{suite}/{camera}"] = stratum
        output[phase] = {"strata": strata, "eligible_cases": sum(row["cases"] for row in strata.values())}
        for metric in ("epe_px", "within8", "trajectory80_at8", "endpoint_epe_px"):
            values = [row[metric] for row in strata.values() if row[metric] is not None]
            output[phase][metric] = float(np.mean(values)) if values else None
    return output


def summarize(args, cases):
    all_rows, finished, missing, baseline_groups = {}, [], [], {}
    oracle_groups = {}
    for case in cases:
        directory = args.output / "cases" / case["name"]
        if not (directory / "completed.json").exists():
            missing.append(case["name"])
            continue
        completed = json.loads((directory / "completed.json").read_text())
        finished.append(case)
        if case["calibration_episode"]:
            continue
        for name, metrics in completed["baselines"].items():
            baseline_groups.setdefault(name, []).append((case, metrics))
        rows = json.loads((directory / "rows.json").read_text())
        for row in rows:
            identity = (row["experiment"], row["state_after_steps"], row["branch"], row["layer"], row["head"], row["decoder"])
            all_rows.setdefault(identity, []).append((case, row["metrics"]))
        for experiment, state, branch in sorted({(row["experiment"], row["state_after_steps"], row["branch"]) for row in rows}):
            candidates = [row for row in rows if (row["experiment"], row["state_after_steps"], row["branch"]) == (experiment, state, branch)]
            metrics = {}
            for phase in PHASES:
                eligible = [row for row in candidates if row["metrics"][phase]["epe_px"] is not None]
                if eligible:
                    best = min(eligible, key=lambda row: row["metrics"][phase]["epe_px"])
                    metrics[phase] = best["metrics"][phase]
                else:
                    metrics[phase] = {"epe_px": None, "within8": None, "trajectory80_at8": None,
                                      "endpoint_epe_px": None, "count": 0, "objects": 0}
            oracle_groups.setdefault((experiment, state, branch), []).append((case, metrics))
    fixed, ranking = [], []
    for identity, observations in all_rows.items():
        experiment, state, branch, layer, head, decoder = identity
        record = {"experiment": experiment, "state_after_steps": state, "branch": branch,
                  "layer": layer, "head": head, "decoder": decoder, "metrics": aggregate(observations)}
        ranking.append(record)
        if (branch, layer, head, decoder) in FIXED:
            fixed.append(record)
    dump(args.output / "fixed_head_summary.json", fixed)
    dump(args.output / "global_ranking_exploratory.json", ranking)
    dump(args.output / "oracle_per_case_summary.json", [{"experiment": key[0], "state_after_steps": key[1],
                                                          "branch": key[2], "metrics": aggregate(value)}
                                                         for key, value in oracle_groups.items()])
    dump(args.output / "baseline_summary.json", {key: aggregate(value) for key, value in baseline_groups.items()})
    dump(args.output / "completion.json", {"planned_cases": len(cases), "finished_cases": len(finished),
                                           "missing": missing, "independent_cases": sum(not case["calibration_episode"] for case in finished)})
    log(stage="summary", planned=len(cases), finished=len(finished), missing=len(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "summarize"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--observed", type=int, default=33)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--states", type=int, nargs="+", default=[0, 10, 13, 18, 22])
    parser.add_argument("--points-per-body", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.observed != 33 or args.frames != 49 or args.size != 512 or 13 not in args.states:
        parser.error("This benchmark fixes33 observed +16 future frames at512, including state13")
    if not 0 <= args.shard_index < args.shards:
        parser.error("Invalid shard index")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    if args.mode == "prepare":
        cases = select_cases(args)
        dump(manifest, cases)
        dump(args.output / "protocol.json", {
            "scope": "Four suites,40 source-episode-held-out tasks, both cameras, one matched short window per episode",
            "selection": "Select the highest simulator object-motion window including articulated-body rotation; independent of head scores",
            "calibration_exclusion": "Episode39 was used to select L15H4 and is excluded from primary aggregates in both cameras",
            "gt_assisted": "33 real prefix control frames;16 zero future control frames. Main full GT latent mixed with sigma0.8 noise, task prompt, CFG5,40-point UniPC flow_shift3",
            "observed_all_noise": "Same49 RGB frames supplied to control with mask0; main pure noise at timestep1000, empty prompt, one forward, no generated video",
            "future_information": "Both protocols contain recorded later frames; this is trajectory readout and privileged teacher evaluation, not causal prediction",
            "states_after_updates": args.states, "grid": [13, 32, 32], "fps": 20,
            "query": "Anchor-visible simulator surface seeds sampled by spatial spread per rigid body; no later GT query refresh",
            "metrics": "Original256-pixel coordinates; moving points require8px and1cm displacement and object excursion2cm; static scored separately",
            "aggregation": "Equal objects per case, equal cases per suite/camera, equal nonempty suite/camera strata. Gripper components are one object",
            "fixed_candidates": FIXED, "head_selection": "L15H4 selected on previous single episode39; control7H11/11H1 fixed from earlier experiments; no per-test head tuning",
            "exploratory": "Global test rankings and per-case GT-selected best heads are diagnostics, not primary generalization results",
            "noise": "Seed42 Gaussian reference17-time noise cropped to13 temporal positions, same realization for compared branches",
            "lora_loaded": False, "training": False, "compression": False,
            "model": str(args.model), "data": str(args.data), "points_per_body": args.points_per_body,
        })
        log(stage="manifest", cases=len(cases), tasks=len({(case["suite"], case["task_id"]) for case in cases}),
            independent_cases=sum(not case["calibration_episode"] for case in cases))
    else:
        cases = json.loads(manifest.read_text())
        if args.mode == "run":
            run_worker(args, cases)
        else:
            summarize(args, cases)


if __name__ == "__main__":
    main()
