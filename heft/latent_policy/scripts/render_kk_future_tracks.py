#!/usr/bin/env python3
"""Filmstrips of future-branch K-K tracks over the recorded future frames (display only).

For held-out agentview windows, simulator GT query points on moving objects and the
gripper (the same selection as ``evaluate_kk_tracking.py``) are taken at the present
frame (future slot 8, window frame 32) and tracked HEFT-style on each source's future
track K through slots 9-12 (frames 36-48).  One row per source, one column per slot;
white is the simulator GT trail, the colour is the source's trail, and a thin line joins
each prediction to its GT point.  Students saw only the first 33 frames; the recorded
future frames and GT are shown for display and scoring only.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_kk_tracking import CAMERAS, GRID, GRIPPER, NOT_OBJECTS, field_volume, kk_track, source_frame

SLOTS = (8, 9, 10, 11, 12)
PALETTE = [(60, 160, 255), (80, 200, 120), (255, 140, 60), (200, 120, 255), (60, 220, 230)]  # BGR


def window_frames(row):
    from vace_latent_distill.data import decode_indices, resize_rgb

    indices = [row["frame_indices"][4 * t] for t in SLOTS]
    video = decode_indices(row["video"], indices, row["fps"])
    frames = resize_rgb(torch.from_numpy(video), 256).numpy().transpose(0, 2, 3, 1)  # [5, 256, 256, 3] RGB
    return np.ascontiguousarray(frames.clip(0, 255).astype(np.uint8))


def query_points(row, tracks, limit):
    from evaluate_vace_sparse_future_tracks import select_points

    frames = np.asarray([source_frame(row["start"], "future", t) for t in SLOTS])
    if frames.max() >= len(tracks["xy"]) or not np.isin(frames, tracks["annotated_frame_indices"]).all():
        return None
    objects = sorted(set(tracks["point_label"].tolist()) - NOT_OBJECTS)
    chosen = []
    for labels, count in ((objects, limit), (GRIPPER, limit // 2)):
        ids = select_points(tracks, frames[0], labels, GRID, 64)
        xy = tracks["xy"][frames][:, ids]
        moved = np.linalg.norm(xy[-1] - xy[0], axis=-1) > 8
        seen = tracks["visible"][frames][:, ids].all(0)
        ids = ids[moved & seen]
        chosen.append(ids[np.linspace(0, len(ids) - 1, min(count, len(ids))).astype(int)] if len(ids) else ids)
    ids = np.concatenate(chosen)
    if len(ids) < 4:
        return None
    groups = np.where(np.isin(tracks["point_label"][ids], GRIPPER), "gripper", "object")
    return ids, tracks["xy"][frames][:, ids].transpose(1, 0, 2), groups  # [P, 5, 2] px


def draw(frame, scale, gt, path, upto, colour):
    image = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
    for points, tint, width in ((gt, (255, 255, 255), 1), (path, colour, 2)):
        for p in points:
            line = np.round(p[: upto + 1] * scale).astype(int)
            if len(line) > 1:
                cv2.polylines(image, [line.reshape(-1, 1, 2)], False, tint, width, cv2.LINE_AA)
    for g, p in zip(gt, path):
        a, b = tuple(np.round(g[upto] * scale).astype(int)), tuple(np.round(p[upto] * scale).astype(int))
        cv2.line(image, a, b, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.circle(image, a, 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, b, 4, colour, -1, cv2.LINE_AA)
    return image


def label(image, text, height=34):
    bar = np.full((height, image.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate((bar, image), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--fields", action="append", default=[], help="name=directory of saved student fields")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4, help="windows with the most moving GT points")
    parser.add_argument("--points", type=int, default=16, help="object points per window (gripper: half)")
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()
    from search_vace_tracking_heads import cell_to_xy, xy_to_cell
    from vace_latent_distill.offline import unpack_target

    args.output.mkdir(parents=True, exist_ok=True)
    students = dict(item.split("=", 1) for item in args.fields)
    rows = [json.loads(l) for l in (args.cache_dir / "manifest_heldout.jsonl").read_text().splitlines() if l.strip()]
    candidates = []
    for row in rows:
        if row["camera"] != "observation.images.image":
            continue
        name = f"episode_{row['tracking_episode']:05d}_{CAMERAS[row['camera']]}_visible_surface.npz"
        with np.load(args.tracking_root / "tracks" / name) as archive:
            tracks = {k: archive[k] for k in ("xy", "visible", "world_xyz", "point_label", "annotated_frame_indices")}
        tracks["xy"] = tracks["xy"].copy()
        tracks["xy"][..., 0] = 255 - tracks["xy"][..., 0]  # tracking renders are mirrored vs training video
        chosen = query_points(row, tracks, args.points)
        if chosen is not None:
            motion = float(np.linalg.norm(chosen[1][:, -1] - chosen[1][:, 0], axis=-1).mean())
            candidates.append((motion * len(chosen[0]), row, chosen))
    candidates.sort(key=lambda item: -item[0])
    index = []
    for _, row, (ids, gt, groups) in candidates[: args.count]:
        sources = {"teacher": unpack_target(torch.load(row["target_path"], map_location="cpu", weights_only=True),
                                            identity=row["provenance_id"])}
        for name, folder in students.items():
            sources[name] = torch.load(Path(folder) / f"{row['sample_id']}.pt", map_location="cpu", weights_only=True)
        start = cell_to_xy(xy_to_cell(gt[:, 0], GRID, 256), GRID, 256)  # same cell-centre queries as the evaluation
        frames = window_frames(row)
        lines, record = [], {"sample_id": row["sample_id"], "task": row["task"], "points": len(ids), "epe_px": {}}
        hold = np.repeat(gt[:, :1], len(SLOTS), 1)
        record["epe_px"]["hold"] = np.linalg.norm(hold - gt, axis=-1).mean(0)[1:].round(1).tolist()
        for (name, fields), colour in zip(sources.items(), PALETTE):
            future = {k: v.float().numpy() for k, v in fields["future"].items() if k == "track_k"}
            path = kk_track(field_volume(future, ("track_k", None)), start, SLOTS[0], SLOTS[-1])
            error = np.linalg.norm(path - gt, axis=-1).mean(0)
            record["epe_px"][name] = error[1:].round(1).tolist()
            panels = [label(draw(frames[j], args.scale, gt, path, j, colour),
                            f"{name}  +{j} (frame {4 * SLOTS[j]})" + (f"  err {error[j]:.1f}px" if j else "  now"))
                      for j in range(len(SLOTS))]
            lines.append(np.concatenate(panels, 1))
        title = (f"{row['sample_id']}  |  {row['task']}  |  white = simulator GT, colour = K-K track on the "
                 f"future branch (students saw frames 0-32 only); hold err +1..+4: "
                 + ", ".join(f"{v:.0f}" for v in record["epe_px"]["hold"]) + " px")
        sheet = label(np.concatenate(lines, 0), title[:220], height=40)
        out = args.output / f"{row['sample_id']}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        record["image"] = out.name
        index.append(record)
        print(json.dumps(record), flush=True)
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
