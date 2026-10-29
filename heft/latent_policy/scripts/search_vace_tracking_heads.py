#!/usr/bin/env python3
"""Search native VACE control self-attention for observed surface trajectories."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import av
import cv2
import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from diffusers.models.transformers.transformer_wan_vace import WanVACETransformerBlock

CAMERAS = ("agentview", "robot0_eye_in_hand")
GRIPPER = {"gripper base", "left finger", "right finger"}


def log(**values):
    print(json.dumps(values), flush=True)


def read_video(path):
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


def write_video(path, frames, fps):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=int(fps))
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def xy_to_cell(xy, grid, image_size):
    cell = np.floor((np.nan_to_num(xy) + 0.5) * grid / image_size).astype(np.int64)
    return np.clip(cell[..., 1], 0, grid - 1) * grid + np.clip(cell[..., 0], 0, grid - 1)


def cell_to_xy(cell, grid, image_size):
    return np.stack((cell % grid, cell // grid), -1).astype(np.float32) * (image_size / grid) + image_size / (2 * grid) - 0.5


def decode_matches(q, k, anchor, cells, grid):
    heads, frames, spatial, dim = q.shape
    if spatial != grid * grid or k.shape != q.shape:
        raise ValueError("Q/K grids must agree")
    anchor_index = torch.tensor(anchor * spatial + cells[np.arange(len(anchor)), anchor], device=q.device)
    query = q.flatten(1, 2)[:, anchor_index].float()
    anchor_key = k.flatten(1, 2)[:, anchor_index].float()
    decoded, reverse = [], []
    # Only anchor rows are materialized; candidate keys retain the complete grid.
    for start in range(0, len(anchor), 16):
        logits = torch.einsum("hpd,htsd->hpts", query[:, start:start + 16], k.float()) / math.sqrt(dim)
        decoded.append(logits.argmax(-1).cpu().numpy())
        reverse_logits = torch.einsum("hpd,htsd->hpts", anchor_key[:, start:start + 16], q.float()) / math.sqrt(dim)
        reverse.append(reverse_logits.argmax(-1).cpu().numpy())
    decoded = np.concatenate(decoded, 1)
    reverse = np.concatenate(reverse, 1)
    rolling = np.zeros_like(decoded)
    head_index = torch.arange(heads, device=q.device)[:, None]
    current = torch.tensor(cells[np.arange(len(anchor)), anchor], device=q.device).expand(heads, -1).clone()
    for ti in range(frames):
        starting = torch.tensor(anchor == ti, device=q.device)
        initial = torch.tensor(cells[:, ti], device=q.device).expand(heads, -1)
        current = torch.where(starting[None], initial, current)
        if ti:
            # This query comes from the preceding prediction, including across occlusions.
            previous_query = q[:, ti - 1][head_index, current].float()
            step_logits = torch.einsum("hpd,hsd->hps", previous_query, k[:, ti].float()) / math.sqrt(dim)
            proposed = step_logits.argmax(-1)
            active = torch.tensor(anchor < ti, device=q.device)[None]
            current = torch.where(active, proposed, current)
        rolling[:, :, ti] = current.cpu().numpy()
    return decoded, reverse, rolling


def select_points(data, times, grid, max_points, min_motion, surface_motion=False, include_static=False):
    xy, visible, world = data["xy"][times], data["visible"][times], data["world_xyz"][times]
    labels = data["point_label"]
    cells = xy_to_cell(xy, grid, 256)
    selected, anchors, groups, motion_report, static_ids = [], [], [], [], []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        centers = world[:, ids].mean(1)
        center_excursion = float(np.linalg.norm(centers - centers[0], axis=-1).max())
        point_excursion = float(np.linalg.norm(world[:, ids] - world[0, ids], axis=-1).max())
        excursion = point_excursion if surface_motion else center_excursion
        motion_report.append({"label": str(label), "excursion_m": excursion,
                              "center_excursion_m": center_excursion, "surface_excursion_m": point_excursion})
        if excursion < min_motion and label not in GRIPPER and label != "wrist" and label not in {"wooden_cabinet_1", "flat_stove_1"}:
            static_ids.extend(ids.tolist())
        group = "gripper" if label in GRIPPER else "moving_object"
        if label == "wrist" or (excursion < min_motion and not include_static):
            continue
        # A point gets one initial visible query; later GT never updates it.
        eligible = []
        for pid in ids:
            vi = np.flatnonzero(visible[:, pid])
            if len(vi) < 6:
                continue
            anchor = int(vi[0])
            # The label-level excursion above decides whether this object is
            # moving. Keep visible surface points even when a particular point
            # is occluded soon after pickup or has little image displacement;
            # `displaced` remains the metric mask for points that truly moved.
            eligible.append((int(pid), anchor))
        if not eligible:
            continue
        candidates = np.array([x[0] for x in eligible])
        xyz = data["local_xyz"][candidates].astype(np.float64)
        distances = np.full(len(candidates), np.inf)
        choice = int(np.argmin(np.linalg.norm(xyz - xyz.mean(0), axis=-1)))
        used, occupied = set(), set()
        for _ in range(len(candidates)):
            pid, anchor = eligible[choice]
            key = (anchor, int(cells[anchor, pid]))
            if key not in occupied:
                selected.append(pid)
                anchors.append(anchor)
                groups.append(group)
                occupied.add(key)
                if len(occupied) >= max_points:
                    break
            used.add(choice)
            distances = np.minimum(distances, np.sum((xyz - xyz[choice]) ** 2, axis=-1))
            distances[list(used)] = -1
            choice = int(np.argmax(distances))
    if not selected:
        raise ValueError("No visible moving subjects after warmup")
    ids = np.asarray(selected)
    anchor = np.asarray(anchors)
    valid = visible[:, ids].T & (np.arange(len(times))[None] > anchor[:, None])
    gt = xy[:, ids].transpose(1, 0, 2)
    gt_world = world[:, ids].transpose(1, 0, 2)
    initial_xy = gt[np.arange(len(ids)), anchor]
    initial_world = gt_world[np.arange(len(ids)), anchor]
    displaced = valid & (np.linalg.norm(gt - initial_xy[:, None], axis=-1) > 8)
    displaced &= np.linalg.norm(gt_world - initial_world[:, None], axis=-1) > 0.01
    gripper_mask = np.zeros((len(times), grid * grid), dtype=bool)
    reference_cells = []
    for t in range(len(times)):
        pids = np.flatnonzero(visible[t] & np.isin(labels, list(GRIPPER)))
        gripper_mask[t, np.unique(cells[t, pids])] = True
        candidates = np.asarray(static_ids, dtype=int)
        reference = np.unique(cells[t, candidates[visible[t, candidates]]])
        reference_cells.append(reference[np.linspace(0, len(reference) - 1, min(8, len(reference))).astype(int)] if len(reference) else reference)
    return {
        "ids": ids, "anchor": anchor, "groups": np.asarray(groups),
        "xy": gt, "valid": valid, "displaced": displaced,
        "cells": cells[:, ids].T, "gripper_mask": gripper_mask,
        "motion": motion_report, "labels": labels[ids], "reference_cells": reference_cells,
    }


def track_metrics(prediction, points, group, displaced_only=False):
    mask = points["valid"] & (points["groups"][:, None] == group)
    if displaced_only:
        mask &= points["displaced"]
    if not mask.any():
        return {"epe_px": None, "within4": None, "within8": None, "trajectory80_at8": None, "count": 0}
    error = np.linalg.norm(prediction - points["xy"], axis=-1)
    point_ids = np.flatnonzero(mask.any(1))
    consistency = [float((error[p, mask[p]] <= 8).mean() >= 0.8) for p in point_ids]
    return {"epe_px": float(error[mask].mean()), "within4": float((error[mask] <= 4).mean()),
            "within8": float((error[mask] <= 8).mean()), "trajectory80_at8": float(np.mean(consistency)),
            "count": int(mask.sum())}


class ControlTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv3d(16, 1536, (1, 2, 2), stride=(1, 2, 2))
        self.vace_patch_embedding = nn.Conv3d(96, 1536, (1, 2, 2), stride=(1, 2, 2))
        self.rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024)
        self.condition_embedder = WanTimeTextImageEmbedding(1536, 256, 9216, 4096)
        self.vace_blocks = nn.ModuleList([
            WanVACETransformerBlock(1536, 8960, 12, cross_attn_norm=True,
                                    apply_input_projection=i == 0, apply_output_projection=True)
            for i in range(15)
        ])

    @classmethod
    def load(cls, path, device):
        with torch.device("meta"):
            model = cls()
        wanted = ("patch_embedding.", "vace_patch_embedding.", "time_embedding.",
                  "time_projection.", "text_embedding.", "vace_blocks.")
        with safe_open(str(path), framework="pt", device="cpu") as source:
            state = {key: source.get_tensor(key) for key in source.keys() if key.startswith(wanted)}
        state = convert_wan_transformer_to_diffusers(state)
        state = {key.replace(".before_proj.", ".proj_in.").replace(".after_proj.", ".proj_out."): value
                 for key, value in state.items()}
        model.load_state_dict(state, strict=True, assign=True)
        # RoPE buffers constructed on meta need to be regenerated.
        model.rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024)
        model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        for module in model.modules():
            if hasattr(module, "scale_shift_table"):
                module.scale_shift_table.data = module.scale_shift_table.data.float()
        return model

    def forward(self, noise, control, text, timestep, callback):
        rotary = self.rope(noise)
        initial = self.patch_embedding(noise).flatten(2).transpose(1, 2)
        c = self.vace_patch_embedding(control).flatten(2).transpose(1, 2)
        _, temb, text, _ = self.condition_embedder(timestep, text)
        temb = temb.unflatten(1, (6, -1))
        for layer, block in enumerate(self.vace_blocks):
            block.attn1._heft_capture = lambda layer=layer, **kw: callback(layer, kw["query"], kw["key"])
            try:
                _, c = block(initial, text, c, temb, rotary)
            finally:
                del block.attn1._heft_capture
        # Returning the final control state keeps the autograd graph visible to
        # DDP when a training caller consumes captured intermediate Q/K tensors.
        return c


def load_empty_text(model_root, device):
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(model_root / "text_encoder", torch_dtype=torch.bfloat16,
                                             local_files_only=True).to(device).eval()
    tokens = tokenizer([""], padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
    # The text condition is frozen and reused for every optimization step.
    # Do not retain the text encoder autograd graph across steps.
    with torch.no_grad():
        embedding = encoder(**tokens).last_hidden_state
    embedding[:, int(tokens.attention_mask.sum()):] = 0
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return embedding.detach()


def load_vae(path, device):
    source = ROOT / "VEGA-3D/llava/model/multimodal_generative_encoder/wan/modules/vae.py"
    spec = importlib.util.spec_from_file_location("tracking_native_wan_vae", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WanVAE(vae_pth=str(path), device=device, dtype=torch.bfloat16)


class HeadScorer:
    def __init__(self, points, grid, frames, directory):
        self.points, self.grid, self.frames, self.directory = points, grid, frames, directory
        self.rows, self.predictions, self.routing = [], [], []

    @torch.autocast("cuda", enabled=False)
    def __call__(self, layer, q, k):
        started = time.monotonic()
        p, g, t = self.points, self.grid, len(self.frames)
        cells, anchor = p["cells"], p["anchor"]
        if q.shape[2] != t * g * g:
            raise ValueError("Attention token grid does not match annotations")
        keys = k[0].reshape(12, t, g * g, 128)
        queries = q[0].reshape(12, t, g * g, 128)
        decoded, reverse, rolling = decode_matches(queries, keys, anchor, cells, g)
        predictions = np.stack([cell_to_xy(value, g, 256) for value in (decoded, reverse, rolling)])
        modes = ("anchor_q_to_k", "anchor_k_to_q", "rolling_q_to_k")
        for pid, a in enumerate(anchor):
            predictions[:, :, pid, a] = p["xy"][pid, a]
        routing_global = [[] for _ in range(12)]
        routing_same = [[] for _ in range(12)]
        routing_reference = [[] for _ in range(12)]
        heatmaps = np.zeros((12, t, g, g), dtype=np.float16)
        global_mask = torch.tensor(p["gripper_mask"].reshape(-1), device=q.device)
        global_area = float(global_mask.float().mean())
        for ti in range(t):
            ids = np.flatnonzero((p["groups"] == "moving_object") & p["valid"][:, ti])[:8]
            if not len(ids) or not p["gripper_mask"][ti].any():
                continue
            index = torch.tensor(ti * g * g + cells[ids, ti], device=q.device)
            rows = torch.einsum("hpd,hnd->hpn", q[0, :, index].float(), k[0].float()) / math.sqrt(128)
            probs = rows.softmax(-1)
            enrichment = (probs[..., global_mask].sum(-1) / global_area).mean(-1).cpu().numpy()
            within = rows[..., ti * g * g:(ti + 1) * g * g].softmax(-1)
            same_mask = torch.tensor(p["gripper_mask"][ti], device=q.device)
            same = (within[..., same_mask].sum(-1) / float(same_mask.float().mean())).mean(-1).cpu().numpy()
            heatmaps[:, ti] = within.mean(1).reshape(12, g, g).cpu().numpy().astype(np.float16)
            reference = p["reference_cells"][ti]
            if len(reference):
                index = torch.tensor(ti * g * g + reference, device=q.device)
                reference_rows = torch.einsum("hpd,hnd->hpn", q[0, :, index].float(), k[0].float()) / math.sqrt(128)
                reference_mass = (reference_rows.softmax(-1)[..., global_mask].sum(-1) / global_area).mean(-1).cpu().numpy()
            else:
                reference_mass = None
            for head in range(12):
                routing_global[head].append(float(enrichment[head]))
                routing_same[head].append(float(same[head]))
                if reference_mass is not None:
                    routing_reference[head].append(float(reference_mass[head]))
        for mode_id, mode in enumerate(modes):
            for head in range(12):
                row = {"tower_layer": layer, "backbone_injection_layer": layer * 2, "head": head, "decoder": mode}
                for group in ("moving_object", "gripper"):
                    row.update({group + "_" + key: val for key, val in track_metrics(predictions[mode_id, head], p, group).items()})
                    row.update({group + "_displaced_" + key: val for key, val in track_metrics(predictions[mode_id, head], p, group, True).items()})
                row["object_to_gripper_global_enrichment"] = float(np.mean(routing_global[head])) if routing_global[head] else None
                row["object_to_gripper_same_frame_enrichment"] = float(np.mean(routing_same[head])) if routing_same[head] else None
                row["static_object_to_gripper_global_enrichment"] = float(np.mean(routing_reference[head])) if routing_reference[head] else None
                numerator, denominator = row["object_to_gripper_global_enrichment"], row["static_object_to_gripper_global_enrichment"]
                row["moving_vs_static_gripper_ratio"] = numerator / denominator if numerator is not None and denominator else None
                self.rows.append(row)
        self.predictions.append(predictions)
        self.routing.append(heatmaps)
        log(layer=layer, best_displaced_within8=max((row["moving_object_displaced_within8"] or 0) for row in self.rows[-36:]),
            seconds=round(time.monotonic() - started, 2))

    def save(self):
        (self.directory / "heads.json").write_text(json.dumps(self.rows, indent=2))
        p = self.points
        np.savez_compressed(self.directory / "trajectories.npz", predicted_xy=np.asarray(self.predictions),
                            gt_xy=p["xy"], valid=p["valid"], displaced=p["displaced"], point_ids=p["ids"],
                            anchor=p["anchor"], groups=p["groups"], labels=p["labels"], frame_indices=self.frames)
        np.savez_compressed(self.directory / "routing.npz", probability=np.asarray(self.routing),
                            gripper_mask=p["gripper_mask"], frame_indices=self.frames)


def render_preview(video, times, prediction, points, group, title, path):
    ids = np.flatnonzero(points["groups"] == group)[:12]
    images = []
    for ti, frame_id in enumerate(times):
        image = video[frame_id].copy()
        cv2.rectangle(image, (0, 0), (256, 28), (20, 20, 20), -1)
        cv2.putText(image, title, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.putText(image, "GT green / QK magenta", (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        for pid in ids:
            if not points["valid"][pid, ti] and ti != points["anchor"][pid]:
                continue
            gt, pred = points["xy"][pid, ti], prediction[pid, ti]
            cv2.circle(image, tuple(np.round(gt).astype(int)), 2, (30, 240, 60), -1)
            cv2.circle(image, tuple(np.round(pred).astype(int)), 2, (240, 40, 210), -1)
            for prev in range(max(int(points["anchor"][pid]), ti - 5), ti):
                if points["valid"][pid, prev:prev + 2].all():
                    cv2.line(image, tuple(np.round(prediction[pid, prev]).astype(int)),
                             tuple(np.round(prediction[pid, prev + 1]).astype(int)), (240, 40, 210), 1)
        images.append(image)
    write_video(path, np.stack(images), 5)
    sample = np.linspace(0, len(images) - 1, 6).astype(int)
    sheet = np.concatenate([np.concatenate([images[i] for i in sample[:3]], 1),
                            np.concatenate([images[i] for i in sample[3:]], 1)], 0)
    cv2.imwrite(str(path.with_suffix(".jpg")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def render_routing(video, times, heatmaps, gripper_mask, title, path):
    images = []
    grid = heatmaps.shape[-1]
    for frame_id, heatmap, mask in zip(times, heatmaps, gripper_mask, strict=True):
        image = video[frame_id].copy()
        if heatmap.max() > 0:
            resized = cv2.resize(heatmap.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
            normalized = np.clip(resized / max(float(resized.max()), 1e-8), 0, 1)
            color = cv2.cvtColor(cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            alpha = (normalized * 0.7)[..., None]
            image = (image * (1 - alpha) + color * alpha).astype(np.uint8)
        mask = cv2.resize(mask.reshape(grid, grid).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (30, 240, 60), 1)
        cv2.rectangle(image, (0, 0), (256, 26), (20, 20, 20), -1)
        cv2.putText(image, title, (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.putText(image, "Object queries / green: gripper", (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        images.append(image)
    write_video(path, np.stack(images), 5)
    chosen = np.linspace(0, len(images) - 1, 6).astype(int)
    sheet = np.concatenate([np.concatenate([images[i] for i in chosen[:3]], 1),
                            np.concatenate([images[i] for i in chosen[3:]], 1)], 0)
    cv2.imwrite(str(path.with_suffix(".jpg")), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def aggregate(output, cases):
    rows = []
    for index, example in enumerate(cases[0]["heads"]):
        result = {key: example[key] for key in ("tower_layer", "backbone_injection_layer", "head", "decoder")}
        selected = [c["heads"][index] for c in cases]
        for key in selected[0]:
            if key in result:
                continue
            values = [x[key] for x in selected if x[key] is not None]
            result[key] = float(np.mean(values)) if values else None
        rows.append(result)
    rows.sort(key=lambda r: (-(r["moving_object_displaced_within8"] or 0), r["moving_object_displaced_epe_px"] or float("inf")))
    (output / "ranking.json").write_text(json.dumps(rows, indent=2))
    with (output / "ranking.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"protocol": {"scope": "control tower attn1 only, all frames observed", "text": "empty string",
                           "matching": "fixed anchor Q->K, anchor K->Q, and rolling Q->K; full spatial grid",
                           "native_vae_stride": [4, 8, 8], "no_gt_reanchoring": True,
                           "displaced_filter": "visible target >8 original pixels and >1cm from its anchor",
                           "ranking": "unweighted mean across episode-camera cases of displaced moving-object within8"},
              "cases": [{k: v for k, v in c.items() if k != "heads"} for c in cases],
              "top_tracking": rows[:10],
              "top_gripper_routing": sorted([r for r in rows if r["decoder"] == "anchor_q_to_k"],
                                             key=lambda r: -(r["object_to_gripper_global_enrichment"] or 0))[:10]}
    for camera in CAMERAS:
        subset = [c for c in cases if c["name"].endswith(camera)]
        if subset:
            camera_rows = []
            for index, example in enumerate(subset[0]["heads"]):
                row = {key: example[key] for key in ("tower_layer", "backbone_injection_layer", "head", "decoder")}
                for key in example:
                    if key not in row:
                        values = [c["heads"][index][key] for c in subset if c["heads"][index][key] is not None]
                        row[key] = float(np.mean(values)) if values else None
                camera_rows.append(row)
            camera_rows.sort(key=lambda r: (-(r["moving_object_displaced_within8"] or 0), r["moving_object_displaced_epe_px"] or float("inf")))
            (output / f"ranking_{camera}.json").write_text(json.dumps(camera_rows, indent=2))
            report[f"top_tracking_{camera}"] = camera_rows[:10]
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/libero_video_spatial_long_20261003")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/libero_video_spatial_objects_gripper_trails_20261003/annotations")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS), choices=CAMERAS)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--skip-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=129)
    parser.add_argument("--points-per-subject", type=int, default=24)
    parser.add_argument("--min-motion", type=float, default=0.02)
    parser.add_argument("--timestep", type=float, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.size % 16 or args.skip_frames < 0 or args.max_frames < 25:
        parser.error("size must be divisible by16; skip >=0; max-frames >=25")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))
    device = torch.device("cuda:0")
    log(stage="empty_text")
    text = load_empty_text(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", device)
    log(stage="tower_load")
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    log(stage="vae_load")
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    cases = []
    for ep in args.episodes:
        for camera in args.cameras:
            name = f"episode_{ep:05d}_{camera}"
            directory = args.output / name
            directory.mkdir()
            video = read_video(args.video_root / f"episode_{ep:05d}" / f"{camera}.mp4")
            n = min(len(video) - args.skip_frames, args.max_frames)
            n = 1 + 4 * ((n - 1) // 4)
            indices = np.arange(args.skip_frames, args.skip_frames + n)
            times = indices[::4]
            with np.load(args.tracks_root / f"{name}_visible_surface.npz") as source:
                data = {key: source[key] for key in source.files}
            if len(video) != len(data["xy"]):
                raise ValueError("Video/annotation length mismatch")
            points = select_points(data, times, args.size // 16, args.points_per_subject, args.min_motion)
            log(stage="encode", case=name, rgb_frames=n, latent_frames=len(times), points=len(points["ids"]), subjects=points["motion"])
            pixels = torch.from_numpy(video[indices].copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
            pixels = F.interpolate(pixels.permute(1, 0, 2, 3), size=(args.size, args.size), mode="bilinear", align_corners=False).permute(1, 0, 2, 3)
            observed, reactive = vae.encode([pixels, torch.zeros_like(pixels)])
            if observed.shape[1] != len(times):
                raise ValueError("Native VAE compression is not four")
            control = torch.cat((observed, reactive, torch.zeros((64, *observed.shape[1:]), device=device)), 0).unsqueeze(0).bfloat16()
            generator = torch.Generator(device=device).manual_seed(args.seed)
            noise = torch.randn((1, *observed.shape), generator=generator, device=device, dtype=torch.float32).bfloat16()
            del pixels, observed, reactive
            scorer = HeadScorer(points, args.size // 16, times, directory)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tower(noise, control, text, torch.tensor([args.timestep], device=device), scorer)
            scorer.save()
            static = np.stack([np.repeat(points["xy"][pid, a:a + 1], len(times), axis=0)
                               for pid, a in enumerate(points["anchor"])])
            oracle = cell_to_xy(points["cells"], args.size // 16, 256)
            case = {"name": name, "rgb_frames": n, "frame_indices": times.tolist(),
                    "motion": points["motion"], "point_counts": {str(k): int((points["labels"] == k).sum()) for k in np.unique(points["labels"])},
                    "static_baseline": {g: track_metrics(static, points, g) for g in ("moving_object", "gripper")},
                    "static_displaced_baseline": {g: track_metrics(static, points, g, True) for g in ("moving_object", "gripper")},
                    "grid_quantization_oracle": {g: track_metrics(oracle, points, g) for g in ("moving_object", "gripper")},
                    "heads": scorer.rows}
            cases.append(case)
            ranking = aggregate(args.output, cases)
            for group in ("moving_object", "gripper"):
                score = group + ("_displaced_within8" if group == "moving_object" else "_within8")
                best = max(scorer.rows, key=lambda row: row[score] or 0)
                mode_id = ("anchor_q_to_k", "anchor_k_to_q", "rolling_q_to_k").index(best["decoder"])
                pred = scorer.predictions[best["tower_layer"]][mode_id, best["head"]]
                render_preview(video, times, pred, points, group,
                               f"T{best['tower_layer']} H{best['head']} {best['decoder']}", directory / f"best_{group}.mp4")
            log(stage="case_done", case=name, current_top=ranking[0])
            del scorer, noise, control
            gc.collect()
            torch.cuda.empty_cache()
    # Use the same aggregate winner for all case previews.
    best = aggregate(args.output, cases)[0]
    report = json.loads((args.output / "report.json").read_text())
    route_best = report["top_gripper_routing"][0]
    for case in cases:
        directory = args.output / case["name"]
        with np.load(directory / "trajectories.npz") as tracks:
            ep = int(case["name"].split("_")[1])
            camera = case["name"].split("_", 2)[2]
            video = read_video(args.video_root / f"episode_{ep:05d}" / f"{camera}.mp4")
            points = {key: tracks[key] for key in ("valid", "displaced", "anchor", "groups")}
            points["xy"] = tracks["gt_xy"]
            mode_id = ("anchor_q_to_k", "anchor_k_to_q", "rolling_q_to_k").index(best["decoder"])
            render_preview(video, tracks["frame_indices"], tracks["predicted_xy"][best["tower_layer"], mode_id, best["head"]],
                           points, "moving_object", f"shared T{best['tower_layer']} H{best['head']}", directory / "shared_winner.mp4")
            camera_best = report[f"top_tracking_{camera}"][0]
            mode_id = ("anchor_q_to_k", "anchor_k_to_q", "rolling_q_to_k").index(camera_best["decoder"])
            render_preview(video, tracks["frame_indices"], tracks["predicted_xy"][camera_best["tower_layer"], mode_id, camera_best["head"]],
                           points, "moving_object", f"T{camera_best['tower_layer']} H{camera_best['head']} {camera_best['decoder']}",
                           directory / "camera_winner.mp4")
        with np.load(directory / "routing.npz") as routing:
            render_routing(video, routing["frame_indices"], routing["probability"][route_best["tower_layer"], route_best["head"]],
                           routing["gripper_mask"], f"Object to gripper T{route_best['tower_layer']} H{route_best['head']}",
                           directory / "gripper_routing.mp4")
    log(stage="done", cases=len(cases), top=best)


if __name__ == "__main__":
    main()
