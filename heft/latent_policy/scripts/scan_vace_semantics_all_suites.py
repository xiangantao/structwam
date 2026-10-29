#!/usr/bin/env python3
"""Frozen tower attn2 head/layer search with disjoint LIBERO calibration tasks."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

from evaluate_vace_instruction_regions import k_regions
from evaluate_vace_prompt_k_guidance import (
    FONT, ROOT, encode_prompts, evaluation_masks, score_heatmap,
)
from scan_vace_prompt_attention import find_phrase_positions, phrase_token_ids
from search_vace_tracking_heads import CAMERAS, ControlTower, load_vae, read_video
from train_vace_k_vector_lora import capture_k, install_k_lora, split_episodes
from vace_vae_warmup import observation_control, observation_window_indices

READOUTS = ("native", "spatial")
ALIASES = {
    "porcelain_mug": ["white mug"],
    "white_yellow_mug": ["yellow and white mug"],
    "cream_cheese": ["cream cheese box"],
    "cookies": ["cookie box"],
}
OLD_HEADS = [(9, 1), (10, 7), (5, 6)]


def annotate_phrases(instruction, labels):
    """Text annotations only; no pixel/trajectory data enters phrase selection."""
    grouped = defaultdict(list)
    for label in labels:
        stem = label.rsplit("_", 1)[0]
        words = stem.replace("_", " ").split()
        candidates = [" ".join(words[i:]) for i in range(len(words))]
        candidates += [phrase + "s" for phrase in candidates]
        candidates += ALIASES.get(stem, [])
        if "cabinet" in stem:
            candidates += ["top drawer", "middle drawer", "bottom drawer", "drawer"]
        if stem == "plate" and "left plate" in instruction and "right plate" in instruction:
            candidates = ["left plate" if label.endswith("_1") else "right plate"]
        matches = [phrase for phrase in candidates if " " + phrase + " " in " " + instruction + " "]
        if not matches:
            raise ValueError(f"No textual annotation for {label}: {instruction}")
        phrase = max(matches, key=lambda value: (len(value.split()), len(value)))
        grouped[phrase].append(label)
    return [{"phrase": phrase, "labels": names,
             "proxy": "whole cabinet, not individual drawer" if "drawer" in phrase else None}
            for phrase, names in grouped.items()]


def make_manifest(video_root, seed, split_seed):
    from bddl.parsing import scan_tokens
    import libero.libero
    from libero.libero import benchmark

    bddl_root = Path(libero.libero.__file__).resolve().parent / "bddl_files"
    _, _, spatial_holdout = split_episodes(video_root, seed)
    suites = {name: benchmark.get_benchmark_dict()[name]() for name in sorted({row["suite"] for row in spatial_holdout})}
    annotations, ignored = {}, {}
    for row in spatial_holdout:
        task = suites[row["suite"]].get_task(row["task_id"])
        path = bddl_root / task.problem_folder / task.bddl_file
        tokens = scan_tokens(filename=str(path))
        sections = {group[0]: group[1:] for group in tokens[1:] if isinstance(group, list)}
        roots = [name.removesuffix("_main") for name in row["body_names"] if name.endswith("_main")]
        labels, missing = [], []
        for item in sections[":obj_of_interest"]:
            matches = [root for root in roots if item == root or item.startswith(root + "_")]
            if matches:
                labels.append(max(matches, key=len))
            else:
                missing.append(item)
        key = row["suite"], row["task_id"]
        annotations[key] = annotate_phrases(row["task"], list(dict.fromkeys(labels)))
        ignored[key] = missing
    rng = np.random.default_rng(split_seed)
    calibration = set()
    for suite in sorted({row["suite"] for row in spatial_holdout}):
        task_ids = sorted(row["task_id"] for row in spatial_holdout if row["suite"] == suite)
        rng.shuffle(task_ids)
        count = len(task_ids) // 2
        if suite == "libero_spatial" and 9 not in task_ids[:count]:
            # Keep the earlier semantic-head calibration task out of this test set.
            index = task_ids.index(9)
            task_ids[count - 1], task_ids[index] = task_ids[index], task_ids[count - 1]
        calibration.update((suite, tid) for tid in task_ids[:count])
    manifest = []
    for row in sorted(spatial_holdout, key=lambda r: (r["suite"], r["task_id"])):
        manifest.append({"episode": row["episode"], "suite": row["suite"], "task_id": row["task_id"],
                         "prompt": row["task"], "phrases": annotations[row["suite"], row["task_id"]],
                         "unscored_region_targets": ignored[row["suite"], row["task_id"]],
                         "split": "calibration" if (row["suite"], row["task_id"]) in calibration else "test"})
    if len(manifest) != 40 or len(calibration) != 20:
        raise RuntimeError("Expected 40 tasks from the four local LIBERO suites")
    return manifest


@torch.inference_mode()
def capture_phrase_maps(tower, positions, noise, control, text):
    frames, cells = noise.shape[2], 32 * 32
    captures = {method: {} for method in READOUTS}

    def callback(layer):
        def capture(query, key, attention_mask, is_causal, **_):
            if attention_mask is not None or is_causal:
                raise RuntimeError("Expected native unmasked tower attn2")
            q, k = query[0].float(), key[0].float()
            native_chunks = []
            for start in range(0, q.shape[1], 1024):
                logits = q[:, start:start + 1024] @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
                log_probability = logits.log_softmax(-1)
                native_chunks.append(torch.stack([torch.logsumexp(log_probability[..., ids], -1) for ids in positions], 1))
            native = torch.cat(native_chunks, -1).reshape(12, len(positions), frames, cells)
            # Log-space normalization preserves very small phrase probabilities.
            native = native.softmax(-1) * cells
            spatial = []
            q_frame = q.reshape(12, frames, cells, -1)
            for ids in positions:
                logits = q_frame @ k[:, ids].transpose(-1, -2)[:, None] / math.sqrt(q.shape[-1])
                spatial.append(logits.softmax(2).mean(-1) * cells)
            captures["native"][layer] = native.cpu().numpy()
            captures["spatial"][layer] = torch.stack(spatial, 1).cpu().numpy()
        return capture

    for layer, block in enumerate(tower.vace_blocks):
        block.attn2._heft_capture = callback(layer)
    try:
        tower(noise, control, text, torch.tensor([1000], device=noise.device), lambda *_: None)
    finally:
        for block in tower.vace_blocks:
            del block.attn2._heft_capture
    return {method: np.stack([captures[method][i] for i in range(15)]).reshape(180, len(positions), frames, 32, 32)
            for method in READOUTS}


def group_maps(values, regions):
    output = np.empty_like(values)
    flat = values.reshape(*values.shape[:-2], -1)
    result = output.reshape(*output.shape[:-2], -1)
    for ti, labels in enumerate(regions):
        for label in np.unique(labels):
            ids = labels == label
            result[:, :, ti, ids] = flat[:, :, ti, ids].mean(-1)[..., None]
    return output


def evaluation_samples(row, density, masks):
    samples, groups = [], []
    for pi, phrase in enumerate(row["phrases"]):
        target = np.logical_or.reduce([masks[label] for label in phrase["labels"]])
        others = [mask for label, mask in masks.items() if label not in phrase["labels"]]
        for ti in range(density.shape[2]):
            if not target[ti].any():
                continue
            negative = [mask[ti] for mask in others if mask[ti].any()]
            if not negative:
                continue
            values = np.stack([density[:, pi, ti, mask].mean(-1) for mask in [target[ti], *negative]], -1)
            padded = np.full((180, 16), np.nan, dtype=np.float32)
            padded[:, :values.shape[1]] = values
            samples.append(padded)
            groups.append(f"{row['suite']}/{row['camera']}/task{row['task_id']}")
    if not samples:
        return np.empty((180, 0, 16), np.float32), []
    return np.stack(samples, 1), groups


def sample_metrics(scores):
    target, negatives = scores[:, :1], scores[:, 1:]
    valid = np.isfinite(negatives)
    count = valid.sum(-1)
    wins = ((target > negatives) + .5 * (target == negatives)) * valid
    pairwin = wins.sum(-1) / count
    best = np.nanmax(scores, -1, keepdims=True)
    top1 = (scores[:, 0] == best[:, 0]) / (scores == best).sum(-1)
    return pairwin, top1


def candidate_metrics(scores, groups):
    pairwin, top1 = sample_metrics(scores)
    by_case, by_group = {}, {}
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        ids = group_array == group
        by_case[group] = {"pairwin": float(pairwin[ids].mean()), "top1": float(top1[ids].mean()), "samples": int(ids.sum())}
    for group in sorted({name.rsplit("/", 1)[0] for name in by_case}):
        rows = [value for name, value in by_case.items() if name.rsplit("/", 1)[0] == group]
        by_group[group] = {"pairwin": float(np.mean([value["pairwin"] for value in rows])),
                           "top1": float(np.mean([value["top1"] for value in rows])),
                           "samples": sum(value["samples"] for value in rows), "tasks": len(rows)}
    mean = float(np.mean([value["pairwin"] for value in by_group.values()]))
    worst = min(value["pairwin"] for value in by_group.values())
    return {"selection_score": .75 * mean + .25 * worst,
            "macro_pairwin": mean, "worst_suite_view_pairwin": worst,
            "macro_top1": float(np.mean([value["top1"] for value in by_group.values()])), "by_suite_view": by_group}


def head_names(heads):
    return "+".join(f"T{head // 12}H{head % 12}" for head in heads)


def candidates(calibration, groups):
    single_rank = sorted(range(180), key=lambda h: candidate_metrics(calibration[h], groups)["selection_score"], reverse=True)
    rows, seen = [], set()

    def add(family, heads):
        heads = tuple(sorted(set(heads)))
        key = family, heads
        if key not in seen:
            rows.append({"family": family, "heads": list(heads), "name": head_names(heads)})
            seen.add(key)

    for head in range(180):
        add("single", [head])
    for layer in range(15):
        heads = list(range(layer * 12, (layer + 1) * 12))
        add("layer", heads)
        within = sorted(heads, key=single_rank.index)
        for count in (2, 3, 4, 6):
            add("head_combination", within[:count])
    for a, b in itertools.combinations(range(15), 2):
        add("layer_combination", [*range(a * 12, (a + 1) * 12), *range(b * 12, (b + 1) * 12)])
    for heads in itertools.combinations(single_rank[:24], 2):
        add("head_combination", heads)
    for heads in itertools.combinations(single_rank[:12], 3):
        add("head_combination", heads)
    for count in (2, 3, 4, 6, 8, 12):
        add("head_combination", single_rank[:count])
    add("all_heads", range(180))
    add("previous", [layer * 12 + head for layer, head in OLD_HEADS])
    return rows


def render_case(row, cache, winners, directory, video_root):
    video = read_video(video_root / f"episode_{row['episode']:05d}/{row['camera']}.mp4")
    times = cache["times"]
    displays = [(family, winners[family]) for family in ("single", "layer", "head_combination")]
    masks = evaluation_masks(ROOT / f"runs/vace_tracking_all_suites_20261004/tracks/episode_{row['episode']:05d}_{row['camera']}_visible_surface.npz", times, 32, 256)
    width, size, gap = 4 * 256 + 5 * 12, 256, 12
    panels = []
    font, small = ImageFont.truetype(FONT, 15), ImageFont.truetype(FONT, 12)
    readouts = {method: cache[method].astype(np.float32) for method in {winner["readout"] for _, winner in displays}}
    selected_maps = {family: group_maps(readouts[winner["readout"]][winner["heads"]].mean(0)[None], cache["regions"])[0]
                     for family, winner in displays}
    for pi, phrase in enumerate(row["phrases"]):
        selected = [selected_maps[family][pi] for family, _ in displays]
        scale = max(float(np.percentile(np.concatenate([np.maximum(value - 1, 0).ravel() for value in selected]), 98)), 1e-4)
        for ti in sorted(set([0, len(times) // 2, len(times) - 1])):
            image = Image.new("RGB", (width, 356), "#f5f6f8")
            draw = ImageDraw.Draw(image)
            draw.text((gap, 7), f"{row['suite']} task{row['task_id']} {row['split']} | {row['camera']} frame{times[ti]}", font=font, fill="black")
            draw.text((gap, 30), f"Phrase: {phrase['phrase']} | Original complete instruction is the text condition", font=small, fill="black")
            rgb = Image.fromarray(video[times[ti]])
            reference = ImageDraw.Draw(rgb)
            target = np.logical_or.reduce([masks[label][ti] for label in phrase["labels"]])
            yy, xx = np.where(target)
            if len(xx):
                reference.rectangle((int(xx.min() * 8), int(yy.min() * 8), int((xx.max() + 1) * 8 - 1), int((yy.max() + 1) * 8 - 1)), outline=(255, 220, 40), width=2)
            image.paste(rgb, (gap, 76))
            draw.text((gap, 53), "RGB + GT reference bounds", font=small, fill="black")
            for col, ((family, winner), maps) in enumerate(zip(displays, selected), 1):
                value = cv2.resize(np.maximum(maps[ti] - 1, 0), (size, size), interpolation=cv2.INTER_NEAREST)
                heat = cv2.applyColorMap(np.round(np.clip(value / scale, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
                alpha = (.65 * np.clip(value / scale, 0, 1))[..., None]
                overlay = np.round(video[times[ti]] * (1 - alpha) + heat * alpha).astype(np.uint8)
                x = gap + col * (size + gap)
                label = f"{family}: {winner['name']}"
                if draw.textlength(label, font=small) > size:
                    label = f"{family}: {len(winner['heads'])} heads"
                draw.text((x, 53), label, font=small, fill="black")
                image.paste(Image.fromarray(overlay), (x, 76))
            draw.text((gap, 338), f"Response above uniform density; fixed scale 0..{scale:.3f} across heads/frames. GT only annotates RGB; never generates heatmaps.", font=small, fill="black")
            safe = phrase["phrase"].replace(" ", "_")
            image.save(directory / f"{safe}_frame_{times[ti]:03d}.jpg", quality=93)
            if ti == len(times) // 2:
                panels.append(image)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/tracks")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/vace_k_vector_lora_all_scenes_20261005_v4/k_vector_lora_step_1400.pt")
    parser.add_argument("--reuse-spatial-cache", type=Path,
                        help="Reuse identical empty-prompt K/regions only; always recompute semantic maps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20261006)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    start = time.monotonic()

    def log(**values):
        values["elapsed_seconds"] = round(time.monotonic() - start, 2)
        line = json.dumps(values)
        print(line, flush=True)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")

    manifest = make_manifest(args.video_root, args.seed, args.split_seed)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(stage="manifest", tasks=len(manifest), calibration_tasks=sum(r["split"] == "calibration" for r in manifest), views=len(CAMERAS))
    device = torch.device("cuda:0")
    tokenizer, texts = encode_prompts(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers", ["", *[row["prompt"] for row in manifest]], device)
    model_root = ROOT / "models/Wan2.1-VACE-1.3B"
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tower = ControlTower.load(model_root / "diffusion_pytorch_model.safetensors", device)
    install_k_lora(tower, int(payload["layer"]), int(payload["head"]), int(payload["rank"]), float(payload["alpha"]))
    tower.load_state_dict(payload["lora"], strict=False)
    tower.eval().requires_grad_(False)
    vae = load_vae(model_root / "Wan2.1_VAE.pth", device)
    zeros, cases = {}, []
    sample_sets = {method: {split: [] for split in ("calibration", "test")} for method in READOUTS}
    sample_groups = {split: [] for split in ("calibration", "test")}
    for task in manifest:
        for camera in CAMERAS:
            row = {**task, "camera": camera}
            name = f"episode_{row['episode']:05d}_{camera}"
            directory = args.output / name
            directory.mkdir(exist_ok=True)
            path = directory / "fields.npz"
            if not path.exists():
                video = read_video(args.video_root / f"episode_{row['episode']:05d}/{camera}.mp4")
                length = min(65, (len(video) - 33) // 4 * 4 + 1)
                indices, alignment = observation_window_indices(len(video), 32, length, 16)
                times = alignment["frame_indices"]
                embedding, ids = texts[row["prompt"]]
                positions = [find_phrase_positions(ids, phrase_token_ids(tokenizer, p["phrase"])) for p in row["phrases"]]
                log(stage="extracting", case=name, split=row["split"], phrases=row["phrases"], retained_frames=length)
                with torch.inference_mode():
                    pixels = torch.from_numpy(video[indices].copy()).to(device).permute(0, 3, 1, 2).float()
                    pixels = F.interpolate(pixels, size=(512, 512), mode="bilinear", align_corners=False).permute(1, 0, 2, 3) / 127.5 - 1
                    control = observation_control(vae, pixels, 16, zeros)
                    generator = torch.Generator(device=device).manual_seed(args.seed + row["episode"] * 37 + CAMERAS.index(camera))
                    noise = torch.randn((1, 16, len(times), 64, 64), generator=generator, device=device).bfloat16()
                    old_path = args.reuse_spatial_cache / name / "fields.npz" if args.reuse_spatial_cache else None
                    if old_path and old_path.exists():
                        with np.load(old_path) as old:
                            if not np.array_equal(old["times"], times) or str(old["checkpoint"]) != str(args.checkpoint.resolve()):
                                raise RuntimeError("Reusable K cache protocol differs")
                            k, regions = old["k"].astype(np.float32), old["regions"]
                    else:
                        k = capture_k(tower, int(payload["layer"]), len(times), 32, noise, control, texts[""][0].to(device), 1000, device)[int(payload["head"])].float().cpu().numpy()
                        regions = None
                    maps = capture_phrase_maps(tower, positions, noise, control, embedding.to(device))
                if regions is None:
                    regions, _ = k_regions(k)
                if any(not np.isfinite(values).all() for values in maps.values()):
                    raise RuntimeError("Non-finite semantic density")
                if any(np.max(np.abs(values.mean((-2, -1)) - 1)) > 1e-4 for values in maps.values()):
                    raise RuntimeError("Semantic density must have per-frame spatial mean1")
                np.savez_compressed(path, **{method: values.astype(np.float16) for method, values in maps.items()},
                                    k=k.astype(np.float16), regions=regions, times=times,
                                    phrases=np.array([p["phrase"] for p in row["phrases"]]),
                                    prompt=np.array(row["prompt"]), checkpoint=np.array(str(args.checkpoint.resolve())),
                                    protocol_version=np.array(2))
                del pixels, control, noise, maps, k, video
                gc.collect()
                torch.cuda.empty_cache()
            with np.load(path) as cache:
                if int(cache["protocol_version"]) != 2:
                    raise RuntimeError("Expected the stable log-space native readout")
                if str(cache["prompt"]) != row["prompt"] or cache["phrases"].tolist() != [p["phrase"] for p in row["phrases"]]:
                    raise RuntimeError("Cached prompt annotations differ")
                masks = evaluation_masks(args.tracks_root / f"{name}_visible_surface.npz", cache["times"], 32, 256)
                for method in READOUTS:
                    density = group_maps(cache[method].astype(np.float32), cache["regions"])
                    samples, groups = evaluation_samples(row, density, masks)
                    sample_sets[method][row["split"]].append(samples)
                    if method == READOUTS[0]:
                        sample_groups[row["split"]].extend(groups)
            cases.append(row)
            log(stage="case_complete", cases=len(cases), total=len(manifest) * len(CAMERAS), case=name, samples=len(groups))
    del tower, vae, texts
    gc.collect()
    torch.cuda.empty_cache()
    report = {"training": False, "observation_only": True, "tower_only": True, "checkpoint": str(args.checkpoint.resolve()),
              "tasks": len(manifest), "cases": len(cases), "split_seed": args.split_seed, "protocol_version": 2,
              "native_numerics": "logsumexp over phrase token log-probabilities, then stable spatial normalization; no probability floor",
              "old_semantic_calibration_task": "libero_spatial task9 forced into calibration",
              "split": "20 calibration tasks / 20 disjoint test tasks; 5 of each per suite; all episodes also held out from spatial K training",
              "readouts": {"native": "sum target phrase columns of native softmax over all512 text keys, normalized to mean1 per head/phrase/frame",
                           "spatial": "each target phrase token softmax over spatial cells per frame, then mean phrase tokens; density mean1"},
              "K_guidance": "broadcast the mean pixel score within fixed empty-prompt K communities; no GT/RGB grouping",
              "selection": "calibration only: .75 * mean suite/view pairwise object-mean win rate + .25 * worst suite/view rate",
              "metric": "visible GT surface-cell object means; average frame/phrase pairs within each task/view, then tasks within each of 8 suite/view groups; top1 shares credit on ties",
              "text_annotations": "official BDDL objects of interest mapped to explicit prompt phrases; labels only define evaluation targets and text columns, never pixel heatmaps",
              "limitations": ["one observed clip per task", "task-disjoint but some scene layouts are shared", "surface-point occupancy is not a full segmentation mask", "drawer phrase evaluation uses whole cabinet geometry"],
              "rankings": {}, "winners": {}}
    global_winners = {}
    for method in READOUTS:
        calibration = np.concatenate(sample_sets[method]["calibration"], 1)
        test = np.concatenate(sample_sets[method]["test"], 1)
        ranking = candidates(calibration, sample_groups["calibration"])
        for candidate in ranking:
            candidate["readout"] = method
            candidate["calibration"] = candidate_metrics(calibration[candidate["heads"]].mean(0), sample_groups["calibration"])
        ranking.sort(key=lambda r: r["calibration"]["selection_score"], reverse=True)
        winners = {}
        for candidate in ranking:
            if candidate["family"] not in winners:
                candidate["test"] = candidate_metrics(test[candidate["heads"]].mean(0), sample_groups["test"])
                winners[candidate["family"]] = candidate
        report["rankings"][method] = ranking
        report["winners"][method] = winners
        for family, candidate in winners.items():
            if family not in global_winners or candidate["calibration"]["selection_score"] > global_winners[family]["calibration"]["selection_score"]:
                global_winners[family] = candidate
        log(stage="ranked", readout=method, candidates=len(ranking), winners={f: {"heads": r["name"], "calibration": r["calibration"]["macro_pairwin"], "test": r["test"]["macro_pairwin"], "test_top1": r["test"]["macro_top1"]} for f, r in winners.items()})
    report["global_winners"] = global_winners
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    overview = defaultdict(list)
    for row in cases:
        directory = args.output / f"episode_{row['episode']:05d}_{row['camera']}"
        with np.load(directory / "fields.npz") as cache:
            panels = render_case(row, cache, global_winners, directory, args.video_root)
        if row["split"] == "test" and row["camera"] == "agentview":
            overview[row["suite"]].append(panels[0])
    for suite, panels in overview.items():
        sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), "white")
        offset = 0
        for panel in panels:
            sheet.paste(panel, (0, offset))
            offset += panel.height
        sheet.save(args.output / f"{suite}_test_overview.jpg", quality=93)
    report["elapsed_seconds"] = time.monotonic() - start
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    log(stage="complete", output=str(args.output.resolve()), winners={f: r["name"] for f, r in global_winners.items()})


if __name__ == "__main__":
    main()
