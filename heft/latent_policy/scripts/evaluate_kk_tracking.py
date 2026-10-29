#!/usr/bin/env python3
"""HEFT-style K-K descriptor tracking on teacher and student fields (diagnostic).

HEFT tracks points by matching a feature with itself across frames (K-K), on
features upsampled toward image resolution, with a local search window, a sub-cell
argmax neighbourhood and a slow EMA of the point descriptor; for post-RoPE features
it keeps only RoPE pairs in a frequency band.  This script applies that recipe to
the 32x32 fields we already have (cached teacher targets, saved student fields) and
scores tracks against simulator GT on held-out windows, using the same query points
and displaced-point filter as ``evaluate_vace_latents.py``.  GT is for scoring only.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

GRID = 32
CAMERAS = {"observation.images.image": "agentview", "observation.images.image2": "robot0_eye_in_hand"}
GRIPPER = ["gripper base", "left finger", "right finger"]
NOT_OBJECTS = {*GRIPPER, "wrist"}
ANCHORS = {"observed": (4, 6, 8, 10), "future": (8, 10)}
WAN_ROPE_PAIRS = (22, 21, 21)  # head_dim 128: temporal / height / width complex pairs, adjacent channels


def rope_band(lower, upper):
    """HEFT RopeSpec.channel_indices for Wan's adjacent pairing."""
    pairs, start = [], 0
    for count in WAN_ROPE_PAIRS:
        pairs.extend(range(start + int(count * lower), start + int(count * upper)))
        start += count
    return np.asarray([c for pair in pairs for c in (2 * pair, 2 * pair + 1)])


FEATURES = {
    "trackK_all": ("track_k", None),
    "trackK_band": ("track_k", (0.15, 1.0)),
    "trackK_band30": ("track_k", (0.3, 1.0)),
    "trackK_band50": ("track_k", (0.5, 1.0)),
    "trackK_low50": ("track_k", (0.0, 0.5)),
    "segK_l15h7": ("seg_k_l15h7", None),
    "segK_l22h9": ("seg_k_l22h9", None),
    "segK_both": (("seg_k_l15h7", "seg_k_l22h9"), None),
}


def field_volume(branch, spec):
    key, band = spec
    keys = key if isinstance(key, tuple) else (key,)
    parts = []
    for name in keys:
        value = branch[name]
        if band is not None:
            value = value[..., rope_band(*band)]
        parts.append(value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8))
    return np.concatenate(parts, -1).reshape(len(parts[0]), GRID, GRID, -1)


def upsample(frame, scale):
    """[32,32,C] -> unit-norm [32*scale, 32*scale, C] torch tensor."""
    value = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None].float()
    value = F.interpolate(value, scale_factor=scale, mode="bilinear", align_corners=False)[0]
    return F.normalize(value, dim=0)


def kk_track(volume, start_xy, anchor, end, *, scale=4, search_px=32.0, argmax_px=8.0,
             ema=0.05, temperature=0.02):
    """Track pixel positions (256-px frame) from native ``anchor`` through ``end``."""
    pixel = 256 / (GRID * scale)  # image pixels per upsampled cell
    frames = {t: upsample(volume[t], scale) for t in range(anchor, end + 1)}
    side = GRID * scale
    position = torch.as_tensor(start_xy, dtype=torch.float32) / pixel  # upsampled coordinates
    grid = torch.stack(torch.meshgrid(torch.arange(side), torch.arange(side), indexing="xy"), -1).float()

    def sample(frame, xy):
        norm = (xy + 0.5) / side * 2 - 1
        return F.normalize(F.grid_sample(frame[None], norm[None, :, None], align_corners=False)[0, :, :, 0].T, dim=1)

    descriptor = sample(frames[anchor], position)
    path = [position.clone()]
    for time in range(anchor + 1, end + 1):
        frame = frames[time].reshape(frames[time].shape[0], -1)            # [C, side*side]
        similarity = descriptor @ frame                                       # [P, side*side]
        distance = (grid.reshape(-1, 2)[None] - position[:, None]).norm(dim=-1) * pixel
        similarity = similarity.masked_fill(distance > search_px, -2)
        best = similarity.argmax(1)
        peak = grid.reshape(-1, 2)[best]
        near = (grid.reshape(-1, 2)[None] - peak[:, None]).norm(dim=-1) * pixel <= argmax_px
        weights = ((similarity - similarity.max(1, keepdim=True).values) / temperature).exp() * near
        position = (weights[..., None] * grid.reshape(-1, 2)[None]).sum(1) / weights.sum(1, keepdim=True)
        descriptor = F.normalize((1 - ema) * descriptor + ema * sample(frames[time], position), dim=1)
        path.append(position.clone())
    return (torch.stack(path, 1) * pixel).numpy()                             # [P, steps, 2] in pixels


def source_frame(start, branch, native):
    return start + (native - 4) * 4 if branch == "observed" else start + native * 4


def prior_velocity(observed, start_xy, anchor):
    """The student warp prior at query points: px per native step for content at ``start_xy``.

    Soft-argmax K-K velocity between the last two observed times (radius 3 cells,
    temperature 0.02, as ``FutureMotion``), forward-splatted to where content sits at
    future slot ``anchor`` (``anchor - 7`` steps after the last observed time).
    """
    from vace_latent_distill.model import soft_velocity

    field = torch.from_numpy(np.ascontiguousarray(observed[10:12])).float().reshape(2, 1, GRID * GRID, -1)
    velocity = soft_velocity(field[0], field[1], GRID, 3, 0.02)
    return landed_velocity(velocity, start_xy, anchor)


def heft_velocity(observed, start_xy, anchor):
    """As ``prior_velocity`` but the per-cell velocity comes from HEFT K-K tracking
    (x4 upsampling, local search, sub-cell argmax) of every cell from t=11 back to t=10."""
    centres = np.stack(np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="xy"), -1).reshape(-1, 2) * 8 + 3.5
    path = kk_track(np.ascontiguousarray(observed[[11, 10]]), centres, 0, 1)
    velocity = torch.from_numpy((path[:, 0] - path[:, 1]) / 8).float().reshape(1, GRID, GRID, 2)
    return landed_velocity(velocity, start_xy, anchor)


def landed_velocity(velocity, start_xy, anchor):
    """Forward-splat a [1, G, G, 2] cell velocity at t=11 to future slot ``anchor``; px/step at queries."""
    from vace_latent_distill.model import splat_velocity

    landed = splat_velocity(velocity, torch.tensor([anchor - 7]))[0, 0].numpy()        # [G, G, 2] cells
    cells = np.clip(np.round((np.asarray(start_xy) - 3.5) / 8).astype(int), 0, GRID - 1)
    return landed[cells[:, 1], cells[:, 0]] * 8


def step_errors(records, mode, group, horizon):
    """Displaced-point EPE sums and counts by native steps after the anchor (1..horizon)."""
    sums, counts = np.zeros(horizon), np.zeros(horizon, dtype=int)
    for record in records:
        error = np.linalg.norm(record[mode] - record["xy"], axis=-1)
        mask = record["valid"] & record["displaced"] & (record["groups"] == group)[:, None]
        for step in range(1, mask.shape[1]):
            sums[step - 1] += float(error[:, step][mask[:, step]].sum())
            counts[step - 1] += int(mask[:, step].sum())
    return sums.tolist(), counts.tolist()


def evaluate_window(job):
    torch.set_num_threads(2)
    from evaluate_vace_sparse_future_tracks import select_points, summarize
    from search_vace_tracking_heads import cell_to_xy, xy_to_cell
    from vace_latent_distill.offline import unpack_target

    row, students, tracking_root, horizon, names = job
    fields = {"teacher": unpack_target(torch.load(row["target_path"], map_location="cpu", weights_only=True),
                                       identity=row["provenance_id"])}
    for name, path in students.items():
        fields[name] = torch.load(path, map_location="cpu", weights_only=True)
    fields = {s: {b: {k: v.float().numpy() for k, v in f[b].items() if k != "paths" and v.ndim >= 2}
                  for b in ANCHORS} for s, f in fields.items()}
    camera = CAMERAS[row["camera"]]
    with np.load(tracking_root / "tracks" / f"episode_{row['tracking_episode']:05d}_{camera}_visible_surface.npz") as a:
        tracks = {k: a[k] for k in ("xy", "visible", "world_xyz", "point_label", "annotated_frame_indices")}
    tracks["xy"] = tracks["xy"].copy()
    tracks["xy"][..., 0] = 255 - tracks["xy"][..., 0]
    objects = sorted(set(tracks["point_label"].tolist()) - NOT_OBJECTS)
    result = {"sample_id": row["sample_id"], "camera": camera}
    for branch, anchors in ANCHORS.items():
        temporal = 12 if branch == "observed" else 13
        records = []
        for anchor in anchors:
            end = min(anchor + horizon, temporal - 1)
            frames = np.asarray([source_frame(row["start"], branch, t) for t in range(anchor, end + 1)])
            if frames.max() >= len(tracks["xy"]) or not np.isin(frames, tracks["annotated_frame_indices"]).all():
                continue
            ids = np.concatenate((select_points(tracks, frames[0], objects, GRID, 64),
                                  select_points(tracks, frames[0], GRIPPER, GRID, 64)))
            if not len(ids):
                continue
            xy = tracks["xy"][frames][:, ids].transpose(1, 0, 2)
            world = tracks["world_xyz"][frames][:, ids].transpose(1, 0, 2)
            valid = tracks["visible"][frames][:, ids].T.copy()
            valid[:, 0] = False
            record = {"groups": np.where(np.isin(tracks["point_label"][ids], GRIPPER), "gripper", "moving_object"),
                      "xy": xy, "valid": valid,
                      "displaced": (np.linalg.norm(xy - xy[:, :1], axis=-1) > 8)
                      & (np.linalg.norm(world - world[:, :1], axis=-1) > 0.01),
                      "hold": np.repeat(xy[:, :1], len(frames), axis=1)}
            steps = np.arange(len(frames))[None, :, None]
            if branch == "future":
                # Constant-velocity references: GT velocity over the 4 frames before the anchor
                # (oracle ceiling for extrapolation), and each field's own observed-branch K-K step.
                previous = source_frame(row["start"], branch, anchor - 1)
                known = previous in set(tracks["annotated_frame_indices"].tolist())
                velocity = xy[:, 0] - tracks["xy"][previous][ids] if known else np.zeros_like(xy[:, 0])
                record["cv_gt"] = xy[:, :1] + steps * velocity[:, None]
                # Same, but the velocity one step older (frames 24->28 for the present anchor),
                # i.e. what an extrapolator reading only the observed branch can know.
                older = source_frame(row["start"], branch, anchor - 2)
                known = known and older in set(tracks["annotated_frame_indices"].tolist())
                lagged = tracks["xy"][previous][ids] - tracks["xy"][older][ids] if known else np.zeros_like(xy[:, 0])
                record["cv_gt_lag"] = xy[:, :1] + steps * lagged[:, None]
            start = cell_to_xy(xy_to_cell(xy[:, 0], GRID, 256), GRID, 256)  # same cell-center queries
            for source, by_branch in fields.items():
                for feature, spec in FEATURES.items():
                    if feature not in names:
                        continue
                    volume = field_volume(by_branch[branch], spec)
                    record[f"{source}|{feature}"] = kk_track(volume, start, anchor, end)
                    if branch == "future":
                        past = field_volume(by_branch["observed"], spec)
                        record[f"{source}|cv_{feature}"] = start[:, None] + steps * prior_velocity(
                            past, start, anchor)[:, None]
                        record[f"{source}|cvk_{feature}"] = start[:, None] + steps * heft_velocity(
                            past, start, anchor)[:, None]
            records.append(record)
        if records:
            modes = [k for k in records[0] if "|" in k] + [m for m in ("hold", "cv_gt", "cv_gt_lag") if m in records[0]]
            gt = summarize(records, modes)
            result[branch] = {mode: {g: (gt[mode][f"{g}_displaced"]["epe_px"], gt[mode][f"{g}_displaced"]["count"])
                                     for g in ("moving_object", "gripper")} for mode in modes}
            result[f"{branch}_steps"] = {mode: {g: step_errors(records, mode, g, horizon)
                                                for g in ("moving_object", "gripper")} for mode in modes}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--fields", action="append", default=[], help="name=directory of saved student fields")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=2, help="native steps tracked after each anchor")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--features", default="trackK_all,trackK_band,segK_l15h7,segK_l22h9,segK_both",
                        help="comma list of FEATURES keys")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (args.cache_dir / "manifest_heldout.jsonl").read_text().splitlines() if l.strip()]
    students = dict(item.split("=", 1) for item in args.fields)
    jobs = [(row, {n: Path(d) / f"{row['sample_id']}.pt" for n, d in students.items()}, args.tracking_root,
             args.horizon, set(args.features.split(","))) for row in rows]
    import multiprocessing

    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(evaluate_window, jobs))
    (args.output / "per_window.json").write_text(json.dumps(results) + "\n")
    lines = [f"# HEFT-style K-K tracking vs simulator GT (held-out, displaced points, px; horizon {args.horizon})", ""]
    for scope in ("all", "agentview", "robot0_eye_in_hand"):
        for branch in ANCHORS:
            chosen = [r for r in results if branch in r and (scope == "all" or r["camera"] == scope)]
            if not chosen:
                continue
            modes = list(chosen[0][branch])
            lines += [f"## {scope} / {branch}", "", "| tracker | object EPE | gripper EPE |", "|---|---|---|"]
            for mode in modes:
                cells = []
                for group in ("moving_object", "gripper"):
                    pairs = [r[branch][mode][group] for r in chosen if r[branch][mode][group][1]]
                    total = sum(c for _, c in pairs)
                    cells.append(f"{sum(e * c for e, c in pairs) / total:.2f} (n={total})" if total else "–")
                lines.append(f"| {mode} | {cells[0]} | {cells[1]} |")
            lines.append("")
            # Error by step after the anchor: how far ahead the field stays trackable.
            lines += [f"### {scope} / {branch} by step after anchor (object / gripper EPE px)", "",
                      "| tracker | " + " | ".join(f"+{s}" for s in range(1, args.horizon + 1)) + " |",
                      "|---|" + "---|" * args.horizon]
            for mode in modes:
                cells = []
                for step in range(args.horizon):
                    parts = []
                    for group in ("moving_object", "gripper"):
                        total = sum(r[f"{branch}_steps"][mode][group][1][step] for r in chosen)
                        error = sum(r[f"{branch}_steps"][mode][group][0][step] for r in chosen)
                        parts.append(f"{error / total:.1f}" if total else "–")
                    cells.append(" / ".join(parts))
                lines.append(f"| {mode} | " + " | ".join(cells) + " |")
            lines.append("")
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
