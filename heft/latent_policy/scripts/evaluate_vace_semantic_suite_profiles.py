#!/usr/bin/env python3
"""Choose a frozen semantic candidate per suite using calibration tasks only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_vace_prompt_k_guidance import ROOT, evaluation_masks, score_heatmap
from scan_vace_semantics_all_suites import group_maps, render_case, sample_metrics
from search_vace_tracking_heads import CAMERAS


def select_profile(ranking, suite):
    rows = []
    for candidate in ranking:
        groups = [v for k, v in candidate["calibration"]["by_suite_view"].items() if k.startswith(suite + "/")]
        if len(groups) != 2:
            raise RuntimeError("Each suite needs calibration data in both cameras")
        values = [v["pairwin"] for v in groups]
        rows.append((.75 * float(np.mean(values)) + .25 * min(values), candidate))
    return max(rows, key=lambda item: item[0])


def evaluate_maps(row, maps, masks):
    phrase_scores, samples = [], []
    targets = [np.logical_or.reduce([masks[label] for label in phrase["labels"]]) for phrase in row["phrases"]]
    for pi, phrase in enumerate(row["phrases"]):
        evaluation = {label: mask for label, mask in masks.items() if label not in phrase["labels"]}
        evaluation["__target__"] = targets[pi]
        metric = score_heatmap(maps[pi], evaluation, "__target__")
        phrase_scores.append({"phrase": phrase["phrase"], "auc": metric["auc"]})
        negatives = [mask for label, mask in evaluation.items() if label != "__target__"]
        for ti, target in enumerate(targets[pi]):
            other = [mask[ti] for mask in negatives if mask[ti].any()]
            if not target.any() or not other:
                continue
            values = [float(maps[pi, ti, mask].mean()) for mask in [target, *other]]
            samples.append(np.pad(values, (0, 16 - len(values)), constant_values=np.nan))
    pairwin, top1 = sample_metrics(np.asarray(samples))
    specificity = []
    for pi, target in enumerate(targets):
        for pj, other_target in enumerate(targets):
            if pi == pj or set(row["phrases"][pi]["labels"]) & set(row["phrases"][pj]["labels"]):
                continue
            for ti in range(len(target)):
                if target[ti].any() and other_target[ti].any():
                    a, b = maps[pi, ti, target[ti]].mean(), maps[pj, ti, target[ti]].mean()
                    specificity.append(float(a > b) + .5 * float(a == b))
    return {"group_macro_auc": float(np.mean([p["auc"] for p in phrase_scores if p["auc"] is not None])),
            "pairwin": float(pairwin.mean()), "top1": float(top1.mean()),
            "own_phrase_win": float(np.mean(specificity)) if specificity else None, "phrases": phrase_scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.scan / "report.json").read_text())
    manifest = json.loads((args.scan / "manifest.json").read_text())
    ranking = [candidate for rows in report["rankings"].values() for candidate in rows]
    profiles = {}
    for suite in sorted({row["suite"] for row in manifest}):
        score, candidate = select_profile(ranking, suite)
        profiles[suite] = {"calibration_selection_score": score, "family": candidate["family"],
                           "heads": candidate["heads"], "name": candidate["name"], "readout": candidate["readout"]}
    results, grouped, overview = [], defaultdict(list), defaultdict(list)
    for task in manifest:
        if task["split"] != "test":
            continue
        profile = profiles[task["suite"]]
        for camera in CAMERAS:
            row = {**task, "camera": camera}
            source = args.scan / f"episode_{task['episode']:05d}_{camera}"
            with np.load(source / "fields.npz") as cache:
                if int(cache["protocol_version"]) != 2:
                    raise RuntimeError("Expected stable log-space readout")
                maps = cache[profile["readout"]].astype(np.float32)[profile["heads"]].mean(0)
                maps = group_maps(maps[None], cache["regions"])[0]
                masks = evaluation_masks(ROOT / f"runs/vace_tracking_all_suites_20261004/tracks/episode_{task['episode']:05d}_{camera}_visible_surface.npz", cache["times"], 32, 256)
                metric = evaluate_maps(row, maps, masks)
            results.append({"episode": task["episode"], "suite": task["suite"], "task_id": task["task_id"], "camera": camera, **metric})
            grouped[task["suite"], camera].append(metric)
            print(json.dumps({"stage": "suite_profile", "cases": len(results), "total": 40}), flush=True)
    summary = {}
    for (suite, camera), rows in grouped.items():
        summary[f"{suite}/{camera}"] = {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
                                       for key in ("group_macro_auc", "pairwin", "top1", "own_phrase_win")}
    output = {"training": False, "new_model_inference": False, "test_used_for_selection": False,
              "requires_suite_identity": True, "profiles": profiles, "results": results,
              "by_suite_view": summary,
              "macro": {key: float(np.mean([r[key] for r in summary.values()])) for key in ("group_macro_auc", "pairwin", "top1", "own_phrase_win")}}
    (args.scan / "suite_profiles.json").write_text(json.dumps(output, indent=2))
    lines = ["# Suite-specific frozen semantic profiles", "",
             "This diagnostic assumes the suite identity is known at inference. It is not a single universal semantic head.",
             "Each profile is chosen from the same 1,710 candidates using that suite's five calibration tasks in both cameras. All results use its other five tasks.",
             "No new model inference or training; reuse protocol2 fields. Selection criterion is .75 mean plus .25 worst camera pairwise object-mean win rate.", "",
             "| Suite | Profile | Readout | Agentview macro AUC | Wrist macro AUC |", "| --- | --- | --- | ---: | ---: |"]
    for suite, profile in profiles.items():
        lines.append(f"| {suite} | {profile['name']} | {profile['readout']} | {summary[suite + '/agentview']['group_macro_auc']:.4f} | {summary[suite + '/robot0_eye_in_hand']['group_macro_auc']:.4f} |")
    lines += ["", "Across eight suite/view groups: " + json.dumps(output["macro"]), "",
              "AUROC compares visible surface cells of each phrase's annotated object against other objects. It does not penalize background area or measure full segmentation.",
              "The main current_report.md retains the universal-candidate results; these profiles never overwrite them.", ""]
    (args.scan / "suite_profiles.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "profiles": profiles, "by_suite_view": summary, "macro": output["macro"]}), flush=True)


if __name__ == "__main__":
    main()
