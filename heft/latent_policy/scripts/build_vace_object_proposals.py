#!/usr/bin/env python3
"""Build prompt-free object proposals from VACE pre-RoPE features.

The proposal stage is deliberately independent of object names.  It combines
pre-RoPE features, low-resolution RGB, spatially connected SLIC regions, and
frame-to-frame descriptor matching.  Prompt attention is only used afterwards
to rank proposals; it is never used to create or split a proposal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import slic
from sklearn.decomposition import PCA


def read_video(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def normalize_rows(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def token_rgb(video: np.ndarray, frame_indices: np.ndarray, grid: int) -> np.ndarray:
    result = []
    for index in frame_indices:
        small = cv2.resize(video[int(index)], (grid, grid), interpolation=cv2.INTER_AREA)
        result.append(small.astype(np.float32) / 255.0)
    return np.asarray(result)


def feature_image(features: np.ndarray, rgb: np.ndarray, fit_frames: int, components: int,
                  rgb_weight: float) -> np.ndarray:
    frames, height, width, channels = features.shape
    flat = features.reshape(frames * height * width, channels).astype(np.float32)
    fit = flat[:fit_frames * height * width]
    fit = fit - fit.mean(0, keepdims=True)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=0)
    pca.fit(fit)
    projected = pca.transform(flat).reshape(frames, height, width, components)
    # Equalize PCA channels so one high-variance direction cannot dominate SLIC.
    projected /= np.maximum(projected[:fit_frames].reshape(-1, components).std(0), 1e-4)
    projected = normalize_rows(projected.reshape(-1, components)).reshape(frames, height, width, components)
    return np.concatenate((projected, rgb * rgb_weight), axis=-1).astype(np.float32)


def region_descriptors(image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    descriptors, centers, areas = [], [], []
    for label in ids:
        mask = labels == label
        y, x = np.where(mask)
        descriptors.append(image[mask].mean(0))
        centers.append((x.mean(), y.mean()))
        areas.append(mask.sum())
    if not ids.size:
        return np.empty((0, image.shape[-1]), np.float32), np.empty((0, 2), np.float32), np.empty(0, np.int32)
    return np.asarray(descriptors, np.float32), np.asarray(centers, np.float32), np.asarray(areas, np.int32)


def temporal_relabel(images: np.ndarray, raw_labels: list[np.ndarray], max_cost: float = 1.45) -> list[np.ndarray]:
    """Match adjacent SLIC regions while allowing splits and new regions."""
    next_id = 1
    output: list[np.ndarray] = []
    previous_ids: np.ndarray | None = None
    previous_desc = np.empty((0, images.shape[-1]), np.float32)
    previous_centers = np.empty((0, 2), np.float32)
    previous_areas = np.empty(0, np.float32)
    for frame, (image, raw) in enumerate(zip(images, raw_labels, strict=True)):
        desc, centers, areas = region_descriptors(image, raw)
        ids = np.zeros_like(raw, dtype=np.int32)
        if frame == 0 or len(desc) == 0 or len(previous_desc) == 0:
            for index in range(len(desc)):
                ids[raw == (index + 1)] = next_id
                next_id += 1
        else:
            d = 1.0 - np.clip(desc @ previous_desc.T, -1.0, 1.0)
            c = np.linalg.norm(centers[:, None] - previous_centers[None], axis=-1) / max(raw.shape)
            a = np.abs(np.log(np.maximum(areas[:, None], 1) / np.maximum(previous_areas[None], 1)))
            cost = d + 0.35 * c + 0.08 * np.minimum(a, 3.0)
            rows, cols = linear_sum_assignment(cost)
            assigned = {}
            for row, col in zip(rows, cols, strict=True):
                if cost[row, col] <= max_cost:
                    assigned[int(row)] = int(previous_ids[col])
            for index in range(len(desc)):
                group_id = assigned.get(index)
                if group_id is None:
                    group_id = next_id
                    next_id += 1
                ids[raw == (index + 1)] = group_id
        output.append(ids)
        previous_ids = np.unique(ids[ids > 0])
        previous_desc, previous_centers, previous_areas = region_descriptors(image, ids)
        # region_descriptors returns descriptors in sorted ID order, matching previous_ids.
    return output


def proposal_records(labels: np.ndarray, prompt_maps: dict[str, np.ndarray] | None,
                     background: np.ndarray) -> list[list[dict]]:
    records = []
    for t, frame_labels in enumerate(labels):
        frame_records = []
        for group_id in np.unique(frame_labels):
            if group_id <= 0:
                continue
            mask = frame_labels == group_id
            y, x = np.where(mask)
            row = {"group_id": int(group_id), "area": int(mask.sum()),
                   "bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())]}
            if prompt_maps:
                row["prompt_score"] = {
                    name: float(values[t][mask].mean()) for name, values in prompt_maps.items()
                }
            frame_records.append(row)
        records.append(frame_records)
    return records


def gt_metrics(labels: np.ndarray, tracks_path: Path, frame_indices: np.ndarray, grid: int) -> dict:
    """Evaluate compactness/purity only after proposals are built."""
    with np.load(tracks_path, allow_pickle=True) as source:
        xy = source["xy"]
        visible = source["visible"]
        point_labels = source["point_label"].astype(str)
    names = [name for name in np.unique(point_labels)
             if name not in {"wrist", "gripper base", "left finger", "right finger"}]
    per_name = {}
    for name in names:
        ious, purities = [], []
        ids = np.flatnonzero(point_labels == name)
        for t, frame_index in enumerate(frame_indices):
            cells = np.floor((np.nan_to_num(xy[frame_index, ids]) + 0.5) * grid / 256.0).astype(int)
            cells = np.clip(cells, 0, grid - 1)
            valid = visible[frame_index, ids]
            gt = np.zeros((grid, grid), bool)
            gt[cells[valid, 1], cells[valid, 0]] = True
            if not gt.any():
                continue
            best_iou, best_purity = 0.0, 0.0
            for group_id in np.unique(labels[t]):
                if group_id <= 0:
                    continue
                proposal = labels[t] == group_id
                inter = np.logical_and(proposal, gt).sum()
                union = np.logical_or(proposal, gt).sum()
                best_iou = max(best_iou, float(inter / max(union, 1)))
                if inter:
                    best_purity = max(best_purity, float(inter / proposal.sum()))
            ious.append(best_iou)
            purities.append(best_purity)
        per_name[name] = {"mean_best_iou": float(np.mean(ious)) if ious else 0.0,
                          "mean_best_purity": float(np.mean(purities)) if purities else 0.0,
                          "frames": len(ious)}
    return {"per_object": per_name,
            "mean_best_iou": float(np.mean([v["mean_best_iou"] for v in per_name.values()])),
            "mean_best_purity": float(np.mean([v["mean_best_purity"] for v in per_name.values()]))}


def draw_overlay(frame: np.ndarray, labels: np.ndarray, records: list[dict], prompt_maps: dict[str, np.ndarray] | None,
                 t: int, selected: dict[str, int] | None = None) -> np.ndarray:
    out = frame.copy()
    rng = np.random.default_rng(1337)
    colors = {int(row["group_id"]): rng.integers(30, 230, 3, dtype=np.uint8) for row in records}
    for row in records:
        group = row["group_id"]
        mask = cv2.resize((labels == group).astype(np.uint8), (out.shape[1], out.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        color = colors[group]
        out[mask] = (0.55 * out[mask] + 0.45 * color).astype(np.uint8)
        x0, y0, x1, y1 = row["bbox_xyxy"]
        cv2.rectangle(out, (x0 * 8, y0 * 8), ((x1 + 1) * 8 - 1, (y1 + 1) * 8 - 1), tuple(int(x) for x in color[::-1]), 1)
    if selected:
        for name, group in selected.items():
            if group <= 0:
                continue
            row = next((x for x in records if x["group_id"] == group), None)
            if row is None:
                continue
            x0, y0, x1, y1 = row["bbox_xyxy"]
            cv2.rectangle(out, (x0 * 8, y0 * 8), ((x1 + 1) * 8 - 1, (y1 + 1) * 8 - 1), (255, 255, 255), 2)
            cv2.putText(out, name, (x0 * 8, max(12, y0 * 8 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", default="layer_5")
    parser.add_argument("--observe-frames", type=int, default=8)
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--rgb-weight", type=float, default=0.75)
    parser.add_argument("--background-mask", type=Path)
    parser.add_argument("--prompt-dir", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.features, allow_pickle=True) as source:
        raw = source[args.layer]
        frame_indices = source["frame_indices"].astype(int)
    if raw.shape[0] != len(frame_indices) * 32 * 32:
        raise ValueError(f"expected 32x32 feature grid, got {raw.shape} and {len(frame_indices)} frames")
    features = raw.reshape(len(frame_indices), 32, 32, raw.shape[-1])
    video = read_video(args.video)
    rgb = token_rgb(video, frame_indices, 32)
    images = feature_image(features, rgb, min(args.observe_frames, len(frame_indices)), args.pca_components, args.rgb_weight)
    if args.background_mask:
        with np.load(args.background_mask) as source:
            background = source["mask"].astype(bool)
        if background.shape != (len(frame_indices), 32, 32):
            raise ValueError("background mask shape does not match feature cache")
    else:
        background = np.zeros((len(frame_indices), 32, 32), bool)
    raw_labels = []
    for t, image in enumerate(images):
        # Keep the proposal stage local and spatially connected.  The mask is
        # only a conservative background prior; it cannot create object labels.
        valid = ~background[t]
        labels = slic(image, n_segments=args.segments, compactness=0.35, sigma=0,
                      start_label=1, channel_axis=-1, enforce_connectivity=True, mask=valid)
        labels[~valid] = 0
        # skimage may leave gaps in labels after masking; compact IDs simplify matching.
        remap = {old: new for new, old in enumerate(np.unique(labels[labels > 0]), start=1)}
        labels = np.vectorize(lambda x: remap.get(int(x), 0), otypes=[np.int32])(labels)
        raw_labels.append(labels.astype(np.int32))
    labels_list = temporal_relabel(images, raw_labels)
    labels = np.asarray(labels_list, dtype=np.int32)
    prompt_maps = None
    if args.prompt_dir:
        prompt_maps = {}
        for path in sorted(args.prompt_dir.glob("layer_05_*.npy")):
            name = path.stem.removeprefix("layer_05_")
            values = np.load(path).mean(0)
            if values.shape == (len(frame_indices), 32, 32):
                prompt_maps[name] = values
    records = proposal_records(labels, prompt_maps, background)
    selected = []
    if prompt_maps:
        for t, frame_records in enumerate(records):
            selected.append({name: max(frame_records, key=lambda row: row["prompt_score"][name])["group_id"]
                             for name in prompt_maps if frame_records})
    else:
        selected = [{} for _ in records]
    metrics = gt_metrics(labels, args.tracks, frame_indices, 32)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "groups.json").write_text(json.dumps({"records": records, "selected": selected}, indent=2))
    np.savez_compressed(args.output / "groups.npz", labels=labels, frame_indices=frame_indices, background=background)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Render all observed frames; the first eight are the primary proposal view.
    writer = av.open(str(args.output / "object_proposals.mp4"), "w")
    stream = writer.add_stream("libx264", rate=5)
    stream.width = stream.height = 256
    stream.pix_fmt = "yuv420p"
    for t, frame in enumerate(frame_indices):
        image = draw_overlay(video[frame], labels[t], records[t], prompt_maps, t, selected[t])
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            writer.mux(packet)
    for packet in stream.encode():
        writer.mux(packet)
    writer.close()
    sample = np.linspace(0, min(args.observe_frames, len(frame_indices)) - 1, min(8, args.observe_frames)).astype(int)
    sheet = np.concatenate([np.concatenate([draw_overlay(video[frame_indices[t]], labels[t], records[t], prompt_maps, t, selected[t])
                                            for t in sample[:4]], axis=1),
                            np.concatenate([draw_overlay(video[frame_indices[t]], labels[t], records[t], prompt_maps, t, selected[t])
                                            for t in sample[4:]], axis=1)], axis=0)
    cv2.imwrite(str(args.output / "object_proposals_first8.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(json.dumps({"output": str(args.output), "frames": len(frame_indices),
                      "observe_frames": args.observe_frames, "groups_first": len(records[0]),
                      "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
