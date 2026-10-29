#!/usr/bin/env python3
"""Search frozen, native VACE scalars for camera-depth ordering, without fitting."""

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
from scipy.stats import rankdata
from torch.nn import functional as F

from search_vace_tracking_heads import ROOT, load_vae, read_video
from benchmark_vace_motion_heads_all_suites import encode_texts
from evaluate_vace_pure_noise_future import load_transformer


ANCHORS = np.array([0, 8, 16, 24])
LATENTS = np.array([4, 6, 8, 10])
METRICS = ("dense_rho", "foreground_rho", "region_rho", "same_row_order", "temporal_order")
CAMERAS = ("agentview", "robot0_eye_in_hand")
MAIN_MLP = {4, 7, 15, 22, 26}
CONTROL_MLP = {3, 7, 11}


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def log(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def cases_for(args):
    cases = []
    for role, episodes in (("discovery", args.discovery), ("holdout", args.holdout)):
        for episode in episodes:
            directory = args.data / "videos" / f"episode_{episode:05d}"
            meta = json.loads((directory / "metadata.json").read_text())
            if meta["num_frames"] < args.start + 33:
                raise ValueError(f"Episode {episode} is too short for the fixed observation window")
            for camera in args.cameras:
                cases.append({"episode": episode, "camera": camera, "role": role,
                              "start": args.start, "task": meta["task"], "task_id": meta["task_id"],
                              "suite": meta["suite"], "name": f"ep{episode:05d}_{camera}"})
    return cases


def prepare(args, cases):
    from collect_libero_long_videos import make_env
    from render_libero_surface_tracks import render_buffers, restore, restore_fixed_body_poses, select_subjects

    annotations = args.output / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    grid = args.size // 16
    for episode in sorted({case["episode"] for case in cases}):
        episode_cases = [case for case in cases if case["episode"] == episode]
        if all((annotations / f"{case['name']}.npz").exists() for case in episode_cases):
            continue
        directory = args.data / "videos" / f"episode_{episode:05d}"
        meta = json.loads((directory / "metadata.json").read_text())
        with np.load(directory / "sim_state.npz") as archive:
            state = {name: archive[name] for name in archive.files}
        videos = {case["camera"]: read_video(directory / f"{case['camera']}.mp4") for case in episode_cases}
        height, width = next(iter(videos.values())).shape[1:3]
        if height != width or height % grid:
            raise ValueError("Recorded image must divide evenly into the native feature grid")
        _, _, env, _ = make_env(meta["suite"], meta["task_id"], height, width, meta["fps"], 1000)
        try:
            env.reset()
            sim = env.env.sim
            pose_error = restore_fixed_body_poses(sim, state)
            if pose_error > 0.001:
                raise RuntimeError(f"Simulator pose mismatch: {pose_error}")
            names = list(sim.model.body_names)
            _, subjects = select_subjects(sim, names, include_objects=True)
            # These masks are evaluation annotations only, never encoder inputs.
            semantic = np.array([subjects.get(i, "background") for i in range(sim.model.nbody)])
            foreground = np.array([name not in {"background", "wrist"}
                                   and not name.startswith(("wooden_cabinet", "flat_stove"))
                                   for name in semantic])
            semantic_names = sorted(set(semantic))
            semantic_ids = np.array([semantic_names.index(name) for name in semantic])
            for case in episode_cases:
                depths, labels, fg, rgbs, errors = [], [], [], [], []
                for relative_time in ANCHORS:
                    native_time = case["start"] + int(relative_time)
                    restore(sim, state, native_time)
                    rgb, depth, geometry = render_buffers(sim, case["camera"], height, width)
                    recorded = videos[case["camera"]][native_time]
                    errors.append(float(np.abs(rgb.astype(float) - recorded).mean()))
                    scale = height // grid
                    depth_cells = depth.reshape(grid, scale, grid, scale).transpose(0, 2, 1, 3)
                    body = np.zeros_like(geometry)
                    valid = geometry >= 0
                    body[valid] = np.asarray(sim.model.geom_bodyid)[geometry[valid]]
                    label_pixels = semantic_ids[body]
                    patch_labels = label_pixels.reshape(grid, scale, grid, scale).transpose(0, 2, 1, 3)
                    label = np.empty((grid, grid), np.int32)
                    purity = np.empty((grid, grid), np.float32)
                    for y in range(grid):
                        for x in range(grid):
                            counts = np.bincount(patch_labels[y, x].ravel(), minlength=len(semantic_names))
                            label[y, x] = counts.argmax()
                            purity[y, x] = counts.max() / (scale * scale)
                    depth_map = np.median(depth_cells, axis=(-1, -2)).astype(np.float32)
                    foreground_map = foreground[body].reshape(grid, scale, grid, scale).mean((1, 3)) >= 0.75
                    label[purity < 0.75] = -1
                    foreground_map &= label >= 0
                    depths.append(depth_map)
                    labels.append(label)
                    fg.append(foreground_map)
                    rgbs.append(recorded)
                if max(errors) > 10:
                    raise RuntimeError(f"RGB replay mismatch for {case['name']}: {errors}")
                np.savez_compressed(annotations / f"{case['name']}.npz", depth=np.stack(depths),
                                    labels=np.stack(labels), foreground=np.stack(fg), rgb=np.stack(rgbs),
                                    label_names=np.array(semantic_names), replay_mae=np.array(errors))
                log(stage="annotate", case=case["name"], replay_mae=max(errors),
                    foreground_cells=int(np.sum(fg)))
        finally:
            env.close()
    dump(args.output / "cases.json", cases)


def rank_correlation(values, target, minimum=8):
    if len(target) < minimum or np.ptp(target) < 1e-5:
        return np.full(values.shape[-1], np.nan, np.float32)
    ranks = rankdata(values, axis=0, method="average").astype(np.float32)
    target_rank = rankdata(target, method="average").astype(np.float32)
    ranks -= ranks.mean(0)
    target_rank -= target_rank.mean()
    denominator = np.sqrt(np.sum(ranks * ranks, 0) * np.sum(target_rank * target_rank))
    result = np.full(values.shape[-1], np.nan, np.float32)
    np.divide(target_rank @ ranks, denominator, out=result, where=denominator > 1e-8)
    return result


def average(values):
    values = np.asarray(values, np.float32)
    count = np.sum(np.isfinite(values), axis=0)
    result = np.full(count.shape, np.nan, np.float32)
    np.divide(np.sum(np.nan_to_num(values), axis=0), count, out=result, where=count > 0)
    return result


class Evaluator:
    def __init__(self, annotation, grid, seed):
        self.grid = grid
        self.depth = annotation["depth"].reshape(4, -1)
        self.labels = annotation["labels"].reshape(4, -1)
        self.foreground = annotation["foreground"].reshape(4, -1)
        rng = np.random.default_rng(seed)
        self.samples = rng.choice(grid * grid, min(192, grid * grid), replace=False)
        self.regions, self.region_time, self.region_label, self.region_depth = [], [], [], []
        for t in range(4):
            for label in np.unique(self.labels[t, self.foreground[t]]):
                mask = (self.labels[t] == label) & self.foreground[t]
                if mask.sum() >= 3:
                    weight = np.zeros((4, grid * grid), np.float32)
                    weight[t, mask] = 1 / mask.sum()
                    self.regions.append(weight.reshape(-1))
                    self.region_time.append(t)
                    self.region_label.append(int(label))
                    self.region_depth.append(float(self.depth[t, mask].mean()))
        self.weights = np.array(self.regions, np.float32).reshape(-1, 4 * grid * grid)
        self.region_depth = np.array(self.region_depth, np.float32)
        self.row_pairs = []
        for t in range(4):
            for y in range(grid):
                ids = np.flatnonzero(self.foreground[t] & (np.arange(grid * grid) // grid == y))
                if len(ids) < 2:
                    continue
                left, right = np.meshgrid(ids, ids)
                dz = self.depth[t, right] - self.depth[t, left]
                valid = (right > left) & (np.abs(dz) >= 0.03)
                valid &= self.labels[t, right] != self.labels[t, left]
                for a, b, sign in zip(left[valid], right[valid], np.sign(dz[valid]), strict=True):
                    self.row_pairs.append((t * grid * grid + int(a), t * grid * grid + int(b), float(sign)))
        if len(self.row_pairs) > 1024:
            ids = rng.choice(len(self.row_pairs), 1024, replace=False)
            self.row_pairs = [self.row_pairs[i] for i in ids]
        self.temporal_pairs = []
        for left in range(len(self.region_depth)):
            for right in range(left + 1, len(self.region_depth)):
                if self.region_label[left] != self.region_label[right]:
                    continue
                dz = self.region_depth[right] - self.region_depth[left]
                if abs(dz) >= 0.02:
                    self.temporal_pairs.append((left, right, float(np.sign(dz))))

    @staticmethod
    def order(values, pairs):
        if not len(pairs):
            return np.full(values.shape[-1], np.nan, np.float32)
        pairs = np.asarray(pairs)
        a, b = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
        return np.mean(np.sign(values[b] - values[a]) * pairs[:, 2, None], axis=0).astype(np.float32)

    def score(self, values):
        values = np.asarray(values, np.float32).reshape(4, self.grid * self.grid, -1)
        result = np.empty((values.shape[-1], len(METRICS)), np.float32)
        sampled = values[:, self.samples]
        result[:, 0] = average([rank_correlation(sampled[t], self.depth[t, self.samples]) for t in range(4)])
        fg_scores = []
        for t in range(4):
            mask = self.foreground[t, self.samples]
            fg_scores.append(rank_correlation(sampled[t, mask], self.depth[t, self.samples][mask]))
        result[:, 1] = average(fg_scores)
        flat = values.reshape(-1, values.shape[-1])
        regions = self.weights @ flat
        result[:, 2] = rank_correlation(regions, self.region_depth)
        result[:, 3] = self.order(flat, self.row_pairs)
        result[:, 4] = self.order(regions, self.temporal_pairs)
        return result


class Collector:
    def __init__(self, annotation, grid, case, preview_groups=None):
        self.grid, self.case = grid, case
        self.evaluator = Evaluator(annotation, grid, 42 + case["episode"])
        self.groups, self.scores = [], []
        self.handles, self.attentions = [], []
        self.preview_groups = preview_groups or {}
        self.maps = {}

    def consume(self, group, values):
        if isinstance(values, tuple):
            values = values[0]
        if values.ndim == 3 and values.shape[:2] != (4, self.grid * self.grid):
            values = values[0].reshape(-1, self.grid * self.grid, values.shape[-1])[LATENTS]
        if values.shape[:2] != (4, self.grid * self.grid):
            raise ValueError(f"Unexpected native field shape {group}: {values.shape}")
        if group in self.preview_groups:
            self.maps[group] = values[..., self.preview_groups[group]].float().cpu().numpy()
            return
        if self.preview_groups:
            return
        width = values.shape[-1]
        metrics = []
        for start in range(0, width, 512):
            features = values[..., start:start + 512].float().cpu().numpy()
            metrics.append(self.evaluator.score(features))
        self.groups.append({"name": group, "width": width})
        self.scores.append(np.concatenate(metrics, axis=0))

    def hook(self, module, name, element=None):
        if self.preview_groups and name not in self.preview_groups:
            return
        def callback(_module, _inputs, output):
            if element is not None:
                output = output[element]
            self.consume(name, output)
        self.handles.append(module.register_forward_hook(callback))

    def attention_stats(self, name):
        def callback(query, key, **_):
            grid, spatial = self.grid, self.grid * self.grid
            query, key = query[0], key[0]
            frames, heads = query.shape[1] // spatial, query.shape[0]
            statistics = []
            for t in LATENTS:
                q = query[:, t * spatial:(t + 1) * spatial].float()
                k = key[:, t * spatial:(t + 1) * spatial].float()
                # Same-time native logits are normalized within this spatial slice.
                # They are explicitly not the full time-space attention probabilities.
                probability = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]), -1)
                entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1) / math.log(spatial)
                maximum = probability.max(-1).values
                diagonal = probability.diagonal(dim1=-2, dim2=-1)
                received = probability.mean(-2)
                field = torch.stack((entropy, maximum, diagonal, received), -1).permute(1, 0, 2).reshape(spatial, heads * 4)
                statistics.append(field)
            self.consume(name, torch.stack(statistics))
        return callback

    def install(self, model):
        for branch, blocks, inner in (("main", model.blocks, MAIN_MLP),
                                       ("control", model.vace_blocks, CONTROL_MLP)):
            for layer, block in enumerate(blocks):
                prefix = f"{branch}.L{layer}"
                self.hook(block, f"{prefix}.hidden", 1 if branch == "control" else None)
                self.hook(block.ffn, f"{prefix}.ffn")
                self.hook(block.attn1.norm_k, f"{prefix}.k_pre_rope")
                self.hook(block.attn1.to_v, f"{prefix}.v")
                if layer in inner:
                    self.hook(block.ffn.net[0], f"{prefix}.mlp_inner")
                name = f"{prefix}.spatial_attention"
                if not self.preview_groups or name in self.preview_groups:
                    attention = block.attn1
                    attention._heft_capture = self.attention_stats(name)
                    self.attentions.append(attention)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        for attention in self.attentions:
            del attention._heft_capture


@torch.inference_mode()
def inputs(vae, video, args, device, zero_cache):
    padded = np.concatenate((np.repeat(video[:1], 15, axis=0), video), axis=0)
    pixels = torch.from_numpy(padded.copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
    pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=(args.size, args.size), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
    encoded = vae.encode([pixels])[0]
    if "zero" not in zero_cache:
        zero_cache["zero"] = vae.encode([torch.zeros_like(pixels)])[0]
    control = torch.cat((encoded, zero_cache["zero"],
                         torch.zeros((64, *encoded.shape[1:]), dtype=encoded.dtype, device=device)), 0)[None].bfloat16()
    if encoded.shape[1] != 12:
        raise RuntimeError(f"Unexpected 48-slot native VAE length: {encoded.shape}")
    noise = torch.randn((1, *encoded.shape), device=device, dtype=torch.bfloat16,
                        generator=torch.Generator(device=device).manual_seed(args.noise_seed))
    return control, encoded, noise


def baseline_fields(annotation, grid):
    yy, xx = np.mgrid[:grid, :grid]
    rgb = np.stack([cv2.resize(image, (grid, grid), interpolation=cv2.INTER_AREA)
                    for image in annotation["rgb"]]).astype(np.float32) / 255
    return np.concatenate((np.repeat(xx[None, ..., None], 4, axis=0),
                           np.repeat(yy[None, ..., None], 4, axis=0), rgb,
                           rgb.mean(-1, keepdims=True)), axis=-1).reshape(4, grid * grid, 6)


@torch.inference_mode()
def scan(args, cases, preview_groups=None):
    grid = args.size // 16
    suffix = "" if args.noise_seed == 42 else f"_seed{args.noise_seed}"
    results = args.output / (("preview_fields" if preview_groups else "scores") + suffix)
    if all((results / f"{case['name']}.npz").exists() for case in cases):
        return
    device = torch.device("cuda:0")
    texts = encode_texts(args.text_model, [case["task"] for case in cases], device)
    model = load_transformer(args.model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    scales = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
    zero_cache = {}
    results.mkdir(parents=True, exist_ok=True)
    for case in cases:
        path = results / f"{case['name']}.npz"
        if path.exists():
            continue
        began = time.monotonic()
        log(stage="start", case=case["name"], role=case["role"])
        with np.load(args.output / "annotations" / f"{case['name']}.npz") as archive:
            annotation = {name: archive[name] for name in archive.files}
        collector = Collector(annotation, grid, case, preview_groups)
        video = read_video(args.data / "videos" / f"episode_{case['episode']:05d}" / f"{case['camera']}.mp4")
        video = video[case["start"]:case["start"] + 33]
        control, encoded, noise = inputs(vae, video, args, device, zero_cache)
        latent = F.avg_pool2d(encoded[:, LATENTS].permute(1, 0, 2, 3).float(), 2)
        collector.consume("vae.observed", latent.permute(0, 2, 3, 1).reshape(4, grid * grid, 16))
        if not preview_groups:
            baseline = collector.evaluator.score(baseline_fields(annotation, grid))
        collector.install(model)
        try:
            output = model(hidden_states=noise, timestep=torch.tensor([1000], device=device),
                           encoder_hidden_states=texts[case["task"]].to(device),
                           control_hidden_states=control, control_hidden_states_scale=scales,
                           return_dict=False)[0]
            if not torch.isfinite(output).all():
                raise RuntimeError("Nonfinite native VACE output")
        finally:
            collector.remove()
        if preview_groups:
            if set(collector.maps) != set(preview_groups):
                raise RuntimeError("A selected native field was not captured")
            np.savez_compressed(path, **collector.maps)
        else:
            index = args.output / "groups.json"
            if index.exists():
                if json.loads(index.read_text()) != collector.groups:
                    raise RuntimeError("Native channel order changed between cases")
            else:
                dump(index, collector.groups)
            np.savez_compressed(path, scores=np.concatenate(collector.scores), baseline=baseline,
                                counts=np.array([len(collector.evaluator.row_pairs), len(collector.evaluator.temporal_pairs)]))
        log(stage="finish", case=case["name"], seconds=round(time.monotonic() - began, 2),
            native_channels=sum(group["width"] for group in collector.groups),
            row_pairs=len(collector.evaluator.row_pairs), temporal_pairs=len(collector.evaluator.temporal_pairs),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        del collector, output, control, encoded, noise, latent
        gc.collect()
        torch.cuda.empty_cache()
    del model, vae, texts
    gc.collect()
    torch.cuda.empty_cache()


def candidate(groups, index):
    offset = 0
    for group in groups:
        if index < offset + group["width"]:
            channel = index - offset
            result = {"index": int(index), "group": group["name"], "channel": int(channel)}
            if group["name"].endswith("spatial_attention"):
                result.update(head=int(channel // 4), statistic=("entropy", "max_probability", "self_probability", "received_mass")[channel % 4])
            elif group["name"].endswith(("k_pre_rope", ".v")):
                result.update(head=int(channel // 128), head_channel=int(channel % 128))
            return result
        offset += group["width"]
    raise IndexError(index)


def finite(value):
    return None if not np.isfinite(value) else float(value)


def report(args, cases):
    groups = json.loads((args.output / "groups.json").read_text())
    raw, baselines, counts = [], [], []
    for case in cases:
        with np.load(args.output / "scores" / f"{case['name']}.npz") as archive:
            raw.append(archive["scores"])
            baselines.append(archive["baseline"])
            counts.append(archive["counts"].tolist())
    raw, baselines = np.array(raw), np.array(baselines)
    discover = np.array([case["role"] == "discovery" for case in cases])
    discovery = average(raw[discover])
    signs = np.where(discovery[:, 1] >= 0, 1, -1)
    # Selection uses only discovery scores. No feature directions or weights are fitted.
    quality = signs * average(discovery[:, 1:4].T)
    quality = np.nan_to_num(quality, nan=-np.inf)
    top = np.argsort(quality)[-args.top:][::-1]
    rows = []
    for index in top:
        sign = signs[index]
        row = candidate(groups, int(index)) | {"sign": int(sign), "discovery_selection_score": float(quality[index])}
        for name, mask in (("discovery", discover), ("holdout", ~discover)):
            mean = average(raw[mask, index]) * sign
            row[name] = {metric: finite(value) for metric, value in zip(METRICS, mean, strict=True)}
        row["cases"] = [case | {"metrics": {metric: finite(value * sign)
                           for metric, value in zip(METRICS, raw[i, index], strict=True)},
                               "same_row_pairs": counts[i][0], "temporal_pairs": counts[i][1]}
                          for i, case in enumerate(cases)]
        rows.append(row)
    baseline_rows = []
    for index, name in enumerate(("image_x", "image_y", "rgb_r", "rgb_g", "rgb_b", "brightness")):
        mean = average(baselines[discover, index])
        sign = 1 if mean[1] >= 0 else -1
        baseline_rows.append({"name": name, "sign": sign,
                              "discovery": {metric: finite(value * sign) for metric, value in zip(METRICS, mean, strict=True)},
                              "holdout": {metric: finite(value * sign) for metric, value in zip(METRICS, average(baselines[~discover, index]), strict=True)}})
    dump(args.output / "ranking.json", rows)
    family_best = {}
    offset = 0
    for group in groups:
        family = group["name"].split(".")[0] + "." + group["name"].split(".")[-1]
        local = offset + int(np.argmax(quality[offset:offset + group["width"]]))
        if family not in family_best or quality[local] > quality[family_best[family]]:
            family_best[family] = local
        offset += group["width"]
    families = []
    for family, index in family_best.items():
        sign = signs[index]
        families.append(candidate(groups, index) | {"family": family, "sign": int(sign),
                        "discovery_selection_score": float(quality[index]),
                        "discovery": {metric: finite(value * sign) for metric, value in zip(METRICS, discovery[index], strict=True)},
                        "holdout": {metric: finite(value * sign) for metric, value in zip(METRICS, average(raw[~discover, index]), strict=True)}})
    families.sort(key=lambda row: -row["discovery_selection_score"])
    views = {}
    for camera in args.cameras:
        camera_mask = np.array([case["camera"] == camera for case in cases])
        explored = average(raw[discover & camera_mask])
        oriented = np.where(explored[:, 1] >= 0, 1, -1)
        objective = oriented * average(explored[:, 1:4].T)
        indices = np.argsort(np.nan_to_num(objective, nan=-np.inf))[-5:][::-1]
        views[camera] = [candidate(groups, int(index)) | {"sign": int(oriented[index]),
                          "discovery_selection_score": float(objective[index]),
                          "discovery": {metric: finite(value * oriented[index]) for metric, value in zip(METRICS, explored[index], strict=True)},
                          "holdout": {metric: finite(value * oriented[index]) for metric, value in zip(METRICS, average(raw[~discover & camera_mask, index]), strict=True)}}
                         for index in indices]
    dump(args.output / "families.json", families)
    dump(args.output / "per_view_ranking.json", views)
    dump(args.output / "summary.json", {"metrics": METRICS, "cases": len(cases), "native_channels": raw.shape[1],
                                       "discovery_cases": int(discover.sum()), "holdout_cases": int((~discover).sum()),
                                       "top": [{key: value for key, value in row.items() if key != "cases"} for row in rows[:10]],
                                       "best_by_family": families, "best_by_camera": views,
                                       "baselines": baseline_rows,
                                       "interpretation": "Native-channel/sign selection is depth-informed discovery, not training. Holdout does not select candidates. Order scores are 2*accuracy-1; chance is zero. Temporal scores compare visible-region means, not tracked physical surface points."})
    log(stage="report", cases=len(cases), channels=raw.shape[1], top=rows[0]["group"], channel=rows[0]["channel"],
        discovery=rows[0]["discovery"], holdout=rows[0]["holdout"])
    return rows


def render(args, cases, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = rows[:3]
    inspection = rows[:10] + json.loads((args.output / "families.json").read_text())
    per_view = json.loads((args.output / "per_view_ranking.json").read_text())
    for camera, candidates in per_view.items():
        inspection.extend(row | {"readout_camera": camera} for row in candidates[:3])
    fields = {}
    for row in inspection:
        channels = fields.setdefault(row["group"], [])
        if row["channel"] not in channels:
            channels.append(row["channel"])
    scan(args, cases, fields)
    evaluate_selected(args, cases, inspection, fields)
    for camera in args.cameras:
        selected = per_view[camera][:3]
        bounds = []
        for row in selected:
            values = []
            for case in cases:
                if case["role"] != "discovery" or case["camera"] != camera:
                    continue
                with np.load(args.output / "preview_fields" / f"{case['name']}.npz") as archive:
                    ci = fields[row["group"]].index(row["channel"])
                    values.append(archive[row["group"]][..., ci].ravel() * row["sign"])
            bounds.append(np.percentile(np.concatenate(values), [2, 98]).tolist())
        for case in cases:
            if case["camera"] != camera or case["role"] != "holdout":
                continue
            with np.load(args.output / "annotations" / f"{case['name']}.npz") as archive:
                annotation = {name: archive[name] for name in archive.files}
            with np.load(args.output / "preview_fields" / f"{case['name']}.npz") as archive:
                feature_maps = {name: archive[name] for name in archive.files}
            fig, axes = plt.subplots(4, 5, figsize=(15, 12), constrained_layout=True)
            depth_bounds = np.percentile(annotation["depth"], [2, 98])
            for t in range(4):
                axes[t, 0].imshow(annotation["rgb"][t])
                axes[t, 0].set_title(f"RGB anchor {ANCHORS[t]}")
                axes[t, 1].imshow(annotation["depth"][t], cmap="viridis", vmin=depth_bounds[0], vmax=depth_bounds[1])
                axes[t, 1].set_title("GT camera Z (m)")
                for index, row in enumerate(selected):
                    ci = fields[row["group"]].index(row["channel"])
                    value = feature_maps[row["group"]][t, :, ci].reshape(args.size // 16, -1) * row["sign"]
                    axes[t, index + 2].imshow(value, cmap="viridis", vmin=bounds[index][0], vmax=bounds[index][1])
                    axes[t, index + 2].set_title(f"{row['group']} c{row['channel']}\nfixed discovery color range", fontsize=9)
                for axis in axes[t]:
                    axis.axis("off")
            fig.suptitle(f"{case['suite']} | {camera} | {case['task']}\nNative readouts; no fitted depth decoder", fontsize=11)
            fig.savefig(args.output / f"{case['name']}_depth_comparison.jpg", dpi=130)
            plt.close(fig)
            if case["episode"] == 39 and camera == "agentview":
                anchor_example(args, case, annotation, feature_maps, fields, selected)
    dump(args.output / "preview_channels.json", {camera: per_view[camera][:3] for camera in args.cameras})


def evaluate_selected(args, cases, candidates, fields, suffix=""):
    detailed = []
    grid = args.size // 16
    for case in cases:
        with np.load(args.output / "annotations" / f"{case['name']}.npz") as archive:
            annotation = {name: archive[name] for name in archive.files}
        # Foreground already excludes background and fixtures. This extra check
        # rejects a readout that merely separates the near gripper from objects.
        hand_labels = [i for i, name in enumerate(annotation["label_names"])
                       if name in {"gripper base", "left finger", "right finger", "wrist"}]
        objects = annotation | {"foreground": annotation["foreground"] & ~np.isin(annotation["labels"], hand_labels)}
        evaluator = Evaluator(objects, grid, 42 + case["episode"])
        spatial_pairs = []
        for left in range(len(evaluator.region_depth)):
            for right in range(left + 1, len(evaluator.region_depth)):
                if evaluator.region_time[left] != evaluator.region_time[right]:
                    continue
                dz = evaluator.region_depth[right] - evaluator.region_depth[left]
                if abs(dz) >= 0.03:
                    spatial_pairs.append((left, right, float(np.sign(dz))))
        with np.load(args.output / ("preview_fields" + suffix) / f"{case['name']}.npz") as archive:
            values = {name: archive[name] for name in archive.files}
        for row in candidates:
            if row.get("readout_camera", case["camera"]) != case["camera"]:
                continue
            ci = fields[row["group"]].index(row["channel"])
            metrics = evaluator.score(values[row["group"]][..., ci:ci + 1])[0] * row["sign"]
            regions = evaluator.weights @ values[row["group"]][..., ci].reshape(-1, 1)
            spatial_score = evaluator.order(regions, spatial_pairs)[0] * row["sign"]
            detailed.append({"case": case["name"], "role": case["role"], "camera": case["camera"],
                             "suite": case["suite"], "candidate": f"{row['group']}:{row['channel']}:{row['sign']}:{row.get('readout_camera', 'both')}",
                             "metrics": {name: finite(value) for name, value in zip(METRICS, metrics, strict=True)},
                             "object_cells": int(objects["foreground"].sum()),
                             "object_regions": len(evaluator.region_depth), "same_row_pairs": len(evaluator.row_pairs),
                             "temporal_pairs": len(evaluator.temporal_pairs), "same_anchor_region_pairs": len(spatial_pairs),
                             "same_anchor_region_order": finite(spatial_score)})
        metrics = evaluator.score(baseline_fields(annotation, grid))
        for index, name in enumerate(("image_x", "image_y", "rgb_r", "rgb_g", "rgb_b", "brightness")):
            detailed.append({"case": case["name"], "role": case["role"], "camera": case["camera"],
                             "suite": case["suite"], "candidate": f"baseline.{name}",
                             "metrics": {metric: finite(value) for metric, value in zip(METRICS, metrics[index], strict=True)},
                             "object_cells": int(objects["foreground"].sum()),
                             "object_regions": len(evaluator.region_depth), "same_row_pairs": len(evaluator.row_pairs),
                             "temporal_pairs": len(evaluator.temporal_pairs)})
        time_field = np.broadcast_to(ANCHORS[:, None, None], (4, grid * grid, 1))
        metrics = evaluator.score(time_field)[0]
        detailed.append({"case": case["name"], "role": case["role"], "camera": case["camera"],
                         "suite": case["suite"], "candidate": "baseline.anchor_time",
                         "metrics": {metric: finite(value) for metric, value in zip(METRICS, metrics, strict=True)},
                         "object_cells": int(objects["foreground"].sum()),
                         "object_regions": len(evaluator.region_depth), "same_row_pairs": len(evaluator.row_pairs),
                         "temporal_pairs": len(evaluator.temporal_pairs)})
    dump(args.output / ("object_only_validation" + suffix + ".json"), detailed)
    if not suffix:
        dump(args.output / "inspected_channels.json", candidates)


def stability(args, cases):
    if args.noise_seed == 42:
        raise ValueError("Stability check requires a different noise seed")
    candidates = json.loads((args.output / "inspected_channels.json").read_text())
    fields = {}
    for row in candidates:
        channels = fields.setdefault(row["group"], [])
        if row["channel"] not in channels:
            channels.append(row["channel"])
    heldout = [case for case in cases if case["role"] == "holdout"]
    scan(args, heldout, fields)
    evaluate_selected(args, heldout, candidates, fields, f"_seed{args.noise_seed}")


def anchor_example(args, case, annotation, feature_maps, fields, selected):
    import matplotlib.pyplot as plt

    archive_path = ROOT / "runs/vace_qk_multi_anchor8_20261008_ep39/multi_anchor_readout.npz"
    if case["start"] != 48 or not archive_path.exists():
        return
    with np.load(archive_path, allow_pickle=True) as archive:
        seeds = archive["observed_seeds"]
    grid = args.size // 16
    if seeds.shape[0] != 4 or grid != 32:
        raise ValueError("Existing prompt/K anchor regions require the frozen 32x32 grid")
    first = selected[0]
    ci = fields[first["group"]].index(first["channel"])
    values = feature_maps[first["group"]][..., ci] * first["sign"]
    records, all_depth, all_value, all_time = [], [], [], []
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    for t in range(4):
        ids = np.unique(np.asarray(seeds[t], dtype=np.int64))
        ids = ids[(ids >= 0) & (ids < grid * grid)]
        region_depth = annotation["depth"][t].ravel()[ids]
        region_values = values[t, ids]
        records.append({"anchor": int(ANCHORS[t]), "selected_cells": len(ids),
                        "gt_median_camera_z_m": float(np.median(region_depth)),
                        "native_median_signal": float(np.median(region_values)),
                        "within_region_spearman": finite(rank_correlation(region_values[:, None], region_depth)[0])})
        image = cv2.resize(annotation["rgb"][t], (512, 512), interpolation=cv2.INTER_AREA)
        for cell in ids:
            x, y = int(cell % grid), int(cell // grid)
            cv2.rectangle(image, (x * 16, y * 16), (x * 16 + 15, y * 16 + 15), (255, 235, 40), 1)
        axes[0, t].imshow(image)
        axes[0, t].set_title(f"Existing prompt/K anchor {ANCHORS[t]}")
        axes[0, t].axis("off")
        axes[1, t].scatter(region_depth, region_values, s=12, alpha=0.7)
        axes[1, t].set_xlabel("GT camera Z (m)")
        axes[1, t].set_ylabel("Native scalar (fixed sign)")
        axes[1, t].set_title(f"rho={records[-1]['within_region_spearman']}", fontsize=9)
        all_depth.extend(region_depth.tolist())
        all_value.extend(region_values.tolist())
        all_time.extend([int(ANCHORS[t])] * len(ids))
    # Same signal scale in every anchor, rather than per-frame contrast stretching.
    bounds = np.percentile(all_value, [1, 99])
    for axis in axes[1]:
        axis.set_ylim(bounds[0], bounds[1])
    fig.suptitle(f"{first['group']} channel {first['channel']}: frozen anchor regions, no depth fit", fontsize=11)
    fig.savefig(args.output / "ep00039_existing_anchor_regions.jpg", dpi=150)
    plt.close(fig)
    dump(args.output / "ep00039_existing_anchor_regions.json", {"candidate": first,
         "selection": "Existing saved prompt/K seeds; no simulator masks used to choose these cells",
         "anchor_records": records, "camera_z_m": all_depth, "signal": all_value, "anchors": all_time,
         "pooled_cell_spearman": finite(rank_correlation(np.array(all_value)[:, None], np.array(all_depth))[0]),
         "region_median_spearman": finite(rank_correlation(np.array([r['native_median_signal'] for r in records])[:, None],
                                                         np.array([r['gt_median_camera_z_m'] for r in records]), minimum=3)[0])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--discovery", nargs="+", type=int, default=[0, 40, 80, 120])
    parser.add_argument("--holdout", nargs="+", type=int, default=[7, 47, 87, 127])
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS))
    parser.add_argument("--start", type=int, default=48)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--stage", choices=("prepare", "scan", "report", "render", "stability", "all"), default="all")
    args = parser.parse_args()
    if set(args.discovery) & set(args.holdout):
        parser.error("Discovery and holdout episodes must be disjoint")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cases = cases_for(args)
    protocol = {"observed_rgb_frames": 33, "warmup_copies_of_frame0": 15,
                "model_rgb_slots": 48, "anchor_rgb_frames": ANCHORS.tolist(), "anchor_latents": LATENTS.tolist(),
                "temporal_alignment": "Existing pipeline convention: latent 4/6/8/10 reads native anchor 0/8/16/24. The VAE receptive fields span multiple RGB frames; these are not per-RGB depth samples.",
                "main_input": "pure Gaussian noise, timestep 1000, no denoising",
                "noise_seed": args.noise_seed,
                "control_input": "all observed RGB, zero mask, no future RGB", "prompt": "full dataset task instruction",
                "readout": "one native channel or one fixed native attention statistic, optional fixed sign",
                "training": "none; no PCA, regression, LoRA, learned projection or depth-model checkpoint",
                "gt_usage": "Simulator depth/segmentation only score candidates and select on discovery; never encoder inputs. Selection uses GT and is not an entirely label-free discovery method.",
                "attention_statistics": "same-time slice probabilities (softmax over spatial keys), not full time-space softmax",
                "color_scaling": "one discovery range per channel/view, fixed for all holdout anchors",
                "metrics": METRICS, "cases": cases}
    protocol_file = "protocol.json" if args.noise_seed == 42 else f"protocol_seed{args.noise_seed}.json"
    dump(args.output / protocol_file, protocol)
    if args.stage in {"all", "prepare"}:
        prepare(args, cases)
    if args.stage in {"all", "scan"}:
        scan(args, cases)
    if args.stage in {"all", "report"}:
        rows = report(args, cases)
    if args.stage in {"all", "render"}:
        if args.stage == "render":
            rows = json.loads((args.output / "ranking.json").read_text())
        render(args, cases, rows)
    if args.stage == "stability":
        stability(args, cases)


if __name__ == "__main__":
    main()
