#!/usr/bin/env python3
"""Expand prompt anchors to their initial K regions and track every region cell."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from demo_vace_prompt_seed_tracking import COLORS
from evaluate_vace_prompt_k_guidance import FONT, ROOT, encode_prompts
from heft_postprocess_vace_tracking import beam_track, load_lora
from search_vace_tracking_heads import CAMERAS, ControlTower, read_video, write_video


def region_points(labels, seeds):
    """Use the entire K group touched by each seed, without another mask filter."""
    cells, groups, counts = [], [], []
    for group, seed in enumerate(seeds):
        region = int(labels[int(seed["cell"])])
        if region != seed["region"]:
            raise RuntimeError("Prompt anchor and K region cache disagree")
        ids = np.flatnonzero(labels == region)
        cells.extend(ids.tolist())
        groups.extend([group] * len(ids))
        counts.append(int(len(ids)))
    cells, groups = np.asarray(cells, np.int64), np.asarray(groups, np.int64)
    if len(np.unique(cells)) != len(cells):
        raise RuntimeError("Prompt seeds should touch distinct K regions")
    return cells, groups, counts


def initial_region_overlay(rgb, labels, seeds, show_seeds=True):
    result = rgb.astype(np.float32).copy()
    for group, seed in enumerate(seeds):
        mask = (labels.reshape(32, 32) == seed["region"]).astype(np.uint8)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        color = COLORS[group % len(COLORS)]
        selected = mask.astype(bool)
        result[selected] = result[selected] * 0.76 + np.asarray(color) * 0.24
    result = np.round(result).astype(np.uint8)
    for group, seed in enumerate(seeds):
        mask = (labels.reshape(32, 32) == seed["region"]).astype(np.uint8)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, COLORS[group % len(COLORS)], 1, cv2.LINE_AA)
        if show_seeds:
            cell = seed["cell"]
            xy = (int((cell % 32 + 0.5) * 8 - 0.5), int((cell // 32 + 0.5) * 8 - 0.5))
            cv2.circle(result, xy, 5, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(result, xy, 3, COLORS[group % len(COLORS)], -1, cv2.LINE_AA)
            cv2.putText(result, str(group + 1), (xy[0] + 5, xy[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def dense_overlay(rgb, paths, groups, ti, tail=2, only_group=None):
    result = np.ascontiguousarray(rgb.copy())
    ids = np.arange(len(paths)) if only_group is None else np.flatnonzero(groups == only_group)
    # Show every point but keep histories short so the underlying scene stays visible.
    for point in ids:
        color = COLORS[int(groups[point]) % len(COLORS)]
        if tail and ti:
            segment = np.round(paths[point, max(0, ti - tail):ti + 1]).astype(np.int32)
            cv2.polylines(result, [segment], False, color, 1, cv2.LINE_AA)
    for point in ids:
        xy = tuple(np.round(paths[point, ti]).astype(np.int32))
        cv2.circle(result, xy, 1, COLORS[int(groups[point]) % len(COLORS)], -1, cv2.LINE_AA)
    return result


def render_case(case, output, video_root, tail=0):
    directory = output / case["name"]
    seeds = json.loads((directory / "selected_regions.json").read_text())
    with np.load(directory / "dense_tracks.npz") as z:
        paths, groups, times, labels = z["xy"], z["point_group"], z["times"], z["initial_K_regions"]
    video = read_video(video_root / f"episode_{case['episode']:05d}/{case['camera']}.mp4")
    initial = video[times[0]]
    region_image = initial_region_overlay(initial, labels, seeds)
    font, small = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 14)
    tiny = ImageFont.truetype(FONT, 12)
    frames, snapshots = [], []
    chosen_times = sorted({0, len(times) // 2, len(times) - 1})
    per_group_videos = [[] for _ in seeds]
    for ti, frame in enumerate(times):
        canvas = Image.new("RGB", (1200, 548), "#f4f5f7")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"{case['suite']} | episode {case['episode']} | {case['camera']} | RGB frame {frame}", font=font, fill="black")
        for index, start in enumerate(range(0, len(case["prompt"]), 120)):
            draw.text((12, 32 + index * 18), case["prompt"][start:start + 120], font=small, fill="black")
        images = [video[frame], region_image, dense_overlay(video[frame], paths, groups, ti, tail)]
        captions = ["Observed RGB", f"Prompt anchors -> initial K regions (frame {times[0]})",
                    f"All {len(paths)} region points -> T7H11 beam"]
        for col, (caption, image) in enumerate(zip(captions, images)):
            x = 12 + col * 396
            draw.text((x, 76), caption, font=small, fill="black")
            canvas.paste(Image.fromarray(image).resize((384, 384), Image.Resampling.NEAREST), (x, 100))
        legends = [f"{i + 1}:{seed['word']} {seed['head']} ({seed['point_count']} pts)" for i, seed in enumerate(seeds)]
        draw.text((12, 490), " | ".join(legends[:3]), font=tiny, fill="black")
        draw.text((12, 507), " | ".join(legends[3:]), font=tiny, fill="black")
        draw.text((12, 527), f"Colors identify initial K groups. Every native region cell is tracked; {tail}-step tails. No GT masks or anchors.", font=tiny, fill="black")
        frames.append(np.asarray(canvas))
        if ti in chosen_times:
            canvas.save(directory / f"frame_{frame:03d}.jpg", quality=95)
            snapshots.append(canvas)
            sheet = Image.new("RGB", (1200, 2 * 448), "#f4f5f7")
            d = ImageDraw.Draw(sheet)
            for group, seed in enumerate(seeds):
                x, y = 12 + (group % 3) * 396, (group // 3) * 448
                d.text((x, y + 8), f"Region {group + 1}: {seed['word']} {seed['head']} | {seed['point_count']} pts", font=small, fill="black")
                d.text((x, y + 28), f"{case['camera']} frame {frame}", font=small, fill="black")
                image = dense_overlay(video[frame], paths, groups, ti, tail, group)
                sheet.paste(Image.fromarray(image).resize((384, 384), Image.Resampling.NEAREST), (x, y + 52))
            sheet.save(directory / f"regions_separate_frame_{frame:03d}.jpg", quality=95)
        for group, seed in enumerate(seeds):
            panel = Image.new("RGB", (800, 472), "#f4f5f7")
            pd = ImageDraw.Draw(panel)
            pd.text((12, 8), f"{case['camera']} frame {frame} | region {group + 1}: {seed['word']} {seed['head']} | {seed['point_count']} points", font=small, fill="black")
            # Use the combined-view color for this region in both panels.
            mask = cv2.resize((labels.reshape(32, 32) == seed["region"]).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST).astype(bool)
            left = initial.copy()
            left[mask] = np.round(left[mask] * 0.76 + np.array(COLORS[group % len(COLORS)]) * 0.24).astype(np.uint8)
            right = dense_overlay(video[frame], paths, groups, ti, tail, group)
            for col, (caption, image) in enumerate(((f"Selected K region at frame {times[0]}", left), ("Dense tracked points", right))):
                x = 12 + col * 396
                pd.text((x, 35), caption, font=small, fill="black")
                panel.paste(Image.fromarray(image).resize((384, 384), Image.Resampling.NEAREST), (x, 60))
            pd.text((12, 451), "Initial group membership only; no later GT correction or fixed region-shape constraint.", font=tiny, fill="black")
            per_group_videos[group].append(np.asarray(panel))
    write_video(directory / "prompt_K_regions_dense_tracks.mp4", np.stack(frames), fps=5)
    for group, sequence in enumerate(per_group_videos):
        write_video(directory / f"region_{group + 1:02d}_dense_tracks.mp4", np.stack(sequence), fps=5)
    sheet = Image.new("RGB", (1200, len(snapshots) * 548), "white")
    for index, snapshot in enumerate(snapshots):
        sheet.paste(snapshot, (0, index * 548))
    sheet.save(directory / "start_middle_end.jpg", quality=95)
    displacement = np.linalg.norm(paths - paths[:, :1], axis=-1).max(1)
    diagnostics = []
    for group, seed in enumerate(seeds):
        ids = groups == group
        counts = [len(np.unique(paths[ids, ti], axis=0)) for ti in range(len(times))]
        diagnostics.append({"region": group + 1, "word": seed["word"], "head": seed["head"],
                            "points": int(ids.sum()), "median_max_displacement_px": float(np.median(displacement[ids])),
                            "unique_locations_per_frame": counts,
                            "final_unique_location_fraction": float(counts[-1] / ids.sum())})
    return {"case": case["name"], "points": len(paths), "frames": len(times), "regions": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/vace_prompt_seed_tracking_20261006")
    parser.add_argument("--seed-source", type=Path, default=ROOT / "runs/vace_prompt_seed_tracking_20261006/local_hotspots")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point-batch", type=int, default=64)
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.point_batch < 1 or args.tail < 0:
        parser.error("point-batch must be positive and tail nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    start = time.monotonic()
    protocol = json.loads((args.seed_source / "protocol.json").read_text())
    cases = protocol["cases"]
    protocol.update({"source": str(args.source.resolve()), "seed_source": str(args.seed_source.resolve()),
                     "dense_selection": "every first-frame native32x32 cell in the K groups touched by existing prompt anchors; no additional masks or object-count assumptions",
                     "point_count": "determined by selected region area, one point per native token cell; coordinates are not claimed to be dense pixel ground truth",
                     "region_identity": "initial K group only; no GT identities or frame-local region-ID matching",
                     "tracking": "same empty-prompt T7H11 step10000 beam, top32 width8 motion0.45 bound4 cells, per-point decoding",
                     "render": "all points drawn, short tails; separate region videos show each group without hiding others in the combined output"})
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))

    def log(**values):
        values["elapsed_seconds"] = round(time.monotonic() - start, 1)
        line = json.dumps(values)
        print(line, flush=True)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")

    for case in cases:
        directory = args.output / case["name"]
        directory.mkdir(exist_ok=True)
        seeds = json.loads((args.seed_source / case["name"] / "seeds.json").read_text())
        with np.load(args.source / case["name"] / "semantic_fields.npz") as z:
            labels, times = z["regions"][0], z["times"]
        cells, groups, counts = region_points(labels, seeds)
        for seed, count in zip(seeds, counts):
            seed["point_count"] = count
        (directory / "selected_regions.json").write_text(json.dumps(seeds, indent=2))
        np.savez_compressed(directory / "initial_regions.npz", initial_K_regions=labels,
                            anchor_cells=cells, point_group=groups, times=times)
        log(stage="regions_selected", case=case["name"], points=len(cells), per_region=counts)

    device, texts, tower = None, None, None
    if not args.render_only:
        device = torch.device("cuda:0")
        _, texts = encode_prompts(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", [""], device)
        checkpoint = Path(protocol["tracking_checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        tower = ControlTower.load(ROOT / "models/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors", device)
        load_lora(tower, checkpoint, payload["candidates"], payload["rank"], payload["alpha"])
        tower.eval().requires_grad_(False)
    summaries = []
    for case in cases:
        directory = args.output / case["name"]
        path = directory / "dense_tracks.npz"
        if not path.exists():
            if args.render_only:
                raise FileNotFoundError(path)
            with np.load(directory / "initial_regions.npz") as z:
                cells, groups, labels, times = z["anchor_cells"], z["point_group"], z["initial_K_regions"], z["times"]
            saved = torch.load(args.source / case["name"] / "control.pt", map_location="cpu", weights_only=False)
            if not np.array_equal(times, saved["times"]):
                raise RuntimeError("Observation windows disagree")
            generator = torch.Generator(device=device).manual_seed(42 + case["episode"] * 37 + CAMERAS.index(case["camera"]))
            noise = torch.randn((1, 16, len(times), 64, 64), generator=generator, device=device).bfloat16()
            control = saved["control"].to(device)
            captures = {}

            def capture(layer, q, k):
                if layer == 7:
                    captures["q"] = q[0, 11].detach().reshape(1, len(times), 1024, 128).contiguous()
                    captures["k"] = k[0, 11].detach().reshape(1, len(times), 1024, 128).contiguous()

            log(stage="tracking_forward", case=case["name"])
            with torch.inference_mode():
                tower(noise, control, texts[""][0].to(device), torch.tensor([1000], device=device), capture)
                paths = []
                for begin in range(0, len(cells), args.point_batch):
                    batch = cells[begin:begin + args.point_batch]
                    points = {"ids": np.arange(len(batch)), "anchor": np.zeros(len(batch), dtype=np.int64),
                              "cells": np.repeat(batch[:, None], len(times), axis=1)}
                    paths.append(beam_track(captures["q"], captures["k"], points, 0, 32))
                    log(stage="dense_batch", case=case["name"], finished=min(begin + len(batch), len(cells)), total=len(cells))
            xy = np.concatenate(paths)
            expected = np.stack((cells % 32, cells // 32), -1) * 8 + 3.5
            if not np.isfinite(xy).all() or not np.allclose(xy[:, 0], expected):
                raise RuntimeError("Dense point trajectories are invalid or lost their anchors")
            np.savez_compressed(path, xy=xy, point_group=groups, anchor_cells=cells,
                                times=times, initial_K_regions=labels)
            del noise, control, captures, saved, paths, xy
        summary = render_case(case, args.output,
                              ROOT / "runs/vace_tracking_all_suites_20261004/videos", args.tail)
        summaries.append(summary)
        log(stage="case_complete", case=case["name"], points=summary["points"],
            final_unique_location_fraction=[region["final_unique_location_fraction"] for region in summary["regions"]])
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    del tower, texts
    gc.collect()
    if device is not None:
        torch.cuda.empty_cache()
    for suite in sorted({case["suite"] for case in cases}):
        suite_cases = [case for case in cases if case["suite"] == suite]
        sheet = Image.new("RGB", (1200, len(suite_cases) * 548), "white")
        for index, case in enumerate(suite_cases):
            with Image.open(sorted((args.output / case["name"]).glob("frame_*.jpg"))[-1]) as panel:
                sheet.paste(panel, (0, index * 548))
        sheet.save(args.output / f"{suite}_both_views.jpg", quality=95)
    log(stage="complete", output=str(args.output.resolve()))


if __name__ == "__main__":
    main()
