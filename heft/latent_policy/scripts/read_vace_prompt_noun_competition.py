#!/usr/bin/env python3
"""Compare noun-relative attention with native token mass using saved fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_vace_prompt_k_guidance import (
    FONT, ROOT, calibrate, evaluation_masks, field, figure, guide_heatmaps,
    read_video, score_heatmap, write_video,
)


def noun_relative(raw):
    # Per-head spatial mean removes noun length/global token-mass priors.
    normalized = raw / np.maximum(raw.mean(axis=(-3, -2, -1), keepdims=True), 1e-12)
    return normalized / np.maximum(normalized.sum(axis=2, keepdims=True), 1e-12)


def mean_heads(raw, heads):
    return np.stack([raw[layer, head] for layer, head in heads]).mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = json.loads((args.input / "report.json").read_text())
    display = dict(np.load(ROOT / "runs/vace_k_time_normalization_visual_20261006/fixed_display_projection.npz"))
    calibration = next(case for case in report["cases"] if case["calibration_only"])
    directory = args.input / "episode_00039_agentview"
    with np.load(directory / "model_fields.npz") as z:
        raw = noun_relative(z["attn2"].astype(np.float32))
        phrases = z["phrases"].astype(str).tolist()
        times = z["times"]
    labels = {"black bowl": "akita_black_bowl_1", "plate": "plate_1", "wooden cabinet": "wooden_cabinet_1"}
    masks = evaluation_masks(ROOT / "runs/vace_tracking_all_suites_20261004/tracks/episode_00039_agentview_visible_surface.npz", times, 32, 256)
    # The calibration objective prioritizes objects; the gripper remains a noun competitor.
    object_indices = [phrases.index(phrase) for phrase in labels]
    heads, ranking = calibrate(raw[:, :, object_indices], labels, masks, 3)
    (args.output / "calibration_ranking.json").write_text(json.dumps(ranking, indent=2))
    output = {"input": str(args.input.resolve()), "readout": "per-head spatial-mean-normalized noun mass / sum of listed noun masses",
              "head_selection": "episode39 agentview only; top3 mean object AUC, gripper excluded from selection objective",
              "selected_heads": heads, "guidance_strength": report["strength"],
              "bandwidth_scale": report["bandwidth_scale"], "training": False,
              "no_gt_in_guidance": True, "background_class": False,
              "interpretation": "relative competition among prompt nouns, not calibrated object-presence probability",
              "cases": []}
    print(json.dumps({"selected_heads": heads, "calibration_object_auc": ranking[0]["mean_auc"]}), flush=True)
    head_names = "+".join(f"T{layer}H{head}" for layer, head in heads)
    panels = []
    for case in report["cases"]:
        if case["calibration_only"]:
            continue
        ep, camera = case["episode"], case["camera"]
        name = f"episode_{ep:05d}_{camera}"
        directory = args.output / name
        directory.mkdir()
        with np.load(args.input / name / "model_fields.npz") as z:
            k = z["k"].astype(np.float32)
            raw = z["attn2"].astype(np.float32)
            phrases = z["phrases"].astype(str).tolist()
            times = z["times"]
        raw = mean_heads(noun_relative(raw), heads)
        guided, isotropic, _ = guide_heatmaps(raw, k, report["strength"], report["bandwidth_scale"])
        np.savez_compressed(directory / "heatmaps.npz", raw=raw, k_guided=guided,
                            isotropic=isotropic, times=times, phrases=np.array(phrases))
        video = read_video(ROOT / f"runs/vace_tracking_all_suites_20261004/videos/episode_{ep:05d}/{camera}.mp4")
        masks = evaluation_masks(ROOT / f"runs/vace_tracking_all_suites_20261004/tracks/episode_{ep:05d}_{camera}_visible_surface.npz", times, 32, 256)
        from evaluate_vace_prompt_k_guidance import CASE_PHRASES
        labels = {**CASE_PHRASES[ep], "robot gripper": "robot gripper"}
        metrics = {}
        for p, phrase in enumerate(phrases):
            metrics[phrase] = {method: score_heatmap(value[p], masks, labels[phrase])
                              for method, value in [("raw", raw), ("k_guided", guided), ("isotropic", isotropic)]}
            frames = []
            for ti, frame_index in enumerate(times):
                image = figure(video[frame_index], field(k[ti], display), raw[p, ti], guided[p, ti],
                               phrase, int(frame_index), (1 / len(phrases), 1),
                               f"{name} | noun-relative attention", head_names)
                draw = ImageDraw.Draw(image)
                draw.rectangle((548, 57, 805, 80), fill="#f5f6f8")
                draw.text((548, 60), "attn2: noun competition", fill="black", font=ImageFont.truetype(FONT, 13))
                draw.rectangle((12, 346, image.width, 374), fill="#f5f6f8")
                draw.text((12, 350), f"Fixed scale: {1 / len(phrases):.2f} to 1. Relative noun scores; not background segmentation. No GT masks in guidance.",
                          fill="black", font=ImageFont.truetype(FONT, 13))
                frames.append(np.asarray(image))
            strip = Image.new("RGB", (frames[0].shape[1], 3 * frames[0].shape[0]), "white")
            for row, ti in enumerate([0, len(times) // 2, len(times) - 1]):
                strip.paste(Image.fromarray(frames[ti]), (0, row * frames[ti].shape[0]))
            slug = phrase.replace(" ", "_")
            strip.save(directory / f"{slug}_comparison.jpg", quality=94)
            write_video(directory / f"{slug}_comparison.mp4", np.stack(frames), 5)
            if camera == "agentview" and p == 0:
                panels.append(Image.fromarray(frames[len(times) // 2]))
        output["cases"].append({"episode": ep, "camera": camera, "metrics": metrics})
        (directory / "metrics.json").write_text(json.dumps(output["cases"][-1], indent=2))
        print(json.dumps({"case": name, "auc": {p: {m: v["auc"] for m, v in ms.items()} for p, ms in metrics.items()}}), flush=True)
    for method in ("raw", "k_guided", "isotropic"):
        values = [ms[method]["auc"] for case in output["cases"] for ms in case["metrics"].values() if ms[method]["auc"] is not None]
        output.setdefault("held_out_mean_auc", {})[method] = float(np.mean(values))
    (args.output / "report.json").write_text(json.dumps(output, indent=2))
    if panels:
        overview = Image.new("RGB", (panels[0].width, sum(image.height for image in panels)), "white")
        for row, panel in enumerate(panels):
            overview.paste(panel, (0, row * panel.height))
        overview.save(args.output / "held_out_overview.jpg", quality=94)
    print(json.dumps({"complete": True, "held_out_mean_auc": output["held_out_mean_auc"]}), flush=True)


if __name__ == "__main__":
    main()
