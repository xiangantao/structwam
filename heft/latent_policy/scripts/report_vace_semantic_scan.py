#!/usr/bin/env python3
"""Report held-out semantic localization for calibration-selected candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_vace_prompt_k_guidance import ROOT, evaluation_masks, score_heatmap
from scan_vace_semantics_all_suites import group_maps
from search_vace_tracking_heads import CAMERAS


def macro(values):
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.scan / "report.json").read_text())
    if report["protocol_version"] != 2 or "elapsed_seconds" not in report:
        raise RuntimeError("The stable all-suite scan must finish before reporting")
    manifest = json.loads((args.scan / "manifest.json").read_text())
    families = ("previous", "single", "layer", "head_combination", "layer_combination", "all_heads")
    winners = report["global_winners"]
    cases = []
    for task in manifest:
        if task["split"] != "test":
            continue
        for camera in CAMERAS:
            directory = args.scan / f"episode_{task['episode']:05d}_{camera}"
            with np.load(directory / "fields.npz") as cache:
                readouts = {method: cache[method].astype(np.float32) for method in {winners[f]["readout"] for f in families}}
                masks = evaluation_masks(ROOT / f"runs/vace_tracking_all_suites_20261004/tracks/episode_{task['episode']:05d}_{camera}_visible_surface.npz", cache["times"], 32, 256)
                row = {"suite": task["suite"], "task_id": task["task_id"], "episode": task["episode"], "camera": camera, "candidates": {}}
                for family in families:
                    winner = winners[family]
                    pixel = readouts[winner["readout"]][winner["heads"]].mean(0)
                    grouped = group_maps(pixel[None], cache["regions"])[0]
                    per_phrase = []
                    targets = [np.logical_or.reduce([masks[label] for label in phrase["labels"]]) for phrase in task["phrases"]]
                    specificity = []
                    for pi, target in enumerate(targets):
                        for pj, other_target in enumerate(targets):
                            if pi == pj or set(task["phrases"][pi]["labels"]) & set(task["phrases"][pj]["labels"]):
                                continue
                            for ti in range(len(cache["times"])):
                                if not target[ti].any() or not other_target[ti].any():
                                    continue
                                own = float(grouped[pi, ti, target[ti]].mean())
                                other = float(grouped[pj, ti, target[ti]].mean())
                                specificity.append(float(own > other) + .5 * float(own == other))
                    for pi, phrase in enumerate(task["phrases"]):
                        evaluation = {label: mask for label, mask in masks.items() if label not in phrase["labels"]}
                        evaluation["__target__"] = np.logical_or.reduce([masks[label] for label in phrase["labels"]])
                        values = {method: score_heatmap(heatmap[pi], evaluation, "__target__")
                                  for method, heatmap in (("pixel", pixel), ("group", grouped))}
                        per_phrase.append({"phrase": phrase["phrase"], "target_labels": phrase["labels"], **values})
                    row["candidates"][family] = {"phrases": per_phrase, "own_phrase_win": macro(specificity),
                                                  "phrase_comparison_count": len(specificity),
                        **{method + "_macro_auc": macro([p[method]["auc"] for p in per_phrase if p[method]["auc"] is not None])
                           for method in ("pixel", "group")}}
            cases.append(row)
            print(json.dumps({"stage": "held_out_auc", "cases": len(cases), "total": 40, "episode": row["episode"], "camera": camera}), flush=True)
    summary = {}
    for family in families:
        by_group = defaultdict(lambda: {"pixel": [], "group": [], "specificity": []})
        for row in cases:
            specificity = row["candidates"][family]["own_phrase_win"]
            if specificity is not None:
                by_group[f"{row['suite']}/{row['camera']}"]["specificity"].append(specificity)
            for method in ("pixel", "group"):
                value = row["candidates"][family][method + "_macro_auc"]
                if value is not None:
                    by_group[f"{row['suite']}/{row['camera']}"][method].append(value)
        rows = {group: {**{method + "_macro_auc": macro(values[method]) for method in ("pixel", "group")},
                        "own_phrase_win": macro(values["specificity"])}
                for group, values in by_group.items()}
        summary[family] = {"name": winners[family]["name"], "heads": winners[family]["heads"], "readout": winners[family]["readout"],
                           "by_suite_view": rows,
                           **{method + "_macro_auc": macro([v[method + "_macro_auc"] for v in rows.values() if v[method + "_macro_auc"] is not None])
                              for method in ("pixel", "group")},
                           **{camera + "_group_macro_auc": macro([v["group_macro_auc"] for g, v in rows.items() if g.endswith("/" + camera) and v["group_macro_auc"] is not None])
                              for camera in CAMERAS},
                           "own_phrase_win": macro([v["own_phrase_win"] for v in rows.values() if v["own_phrase_win"] is not None]),
                           "test_pairwin": winners[family]["test"]["macro_pairwin"],
                           "test_top1": winners[family]["test"]["macro_top1"]}
    output = {"selection_uses_test": False, "metric": "target-vs-each-other-object surface-cell AUROC, macro over objects, task phrases, tasks, then eight suite/view groups",
              "training": False, "cases": cases, "summary": summary}
    (args.scan / "held_out_localization.json").write_text(json.dumps(output, indent=2))
    lines = ["# Frozen VACE attn2 scan across four LIBERO suites", "",
             "40 tasks, one observed clip per task, both cameras (80 windows). Model weights remain frozen.",
             "20 tasks select candidates; 20 different tasks evaluate them. Each suite contributes five calibration and five test tasks.",
             "All selected episodes are also held out from the existing spatial K LoRA training split. Scene layouts can overlap across tasks.", "",
             "## Search", "",
             "Two noun-specific readouts use the same complete instruction and the same observed video/control/noise:", "",
             "- Native video-to-text attention over all 512 text keys, summing the object's phrase columns and normalizing spatial mean per frame/head/phrase. Log-space computation avoids underflow.",
             "- Spatial softmax for each object's phrase token, then averaging those tokens. This is a derived Q/K readout rather than the model's native attention probabilities.",
             "- Each readout scans all 180 heads, all 15 full layers, all 105 pairs of full layers, pairs among the top 24 calibration heads, triples among the top 12 calibration heads, within-layer subsets, and global top-k subsets.",
             "- Combinations average maps with equal weights. No learned semantic adapter or per-test-task head selection.",
             "- The original three heads and all-head averaging are included as controls. The old semantic calibration task (Spatial task9) remains on the calibration side.", "",
             "Heads are selected by 0.75 times mean suite/view object-mean pairwise win rate plus 0.25 times the worst suite/view rate, using calibration tasks only.",
             "A win compares the target's mean response with each other visible object's mean within the same frame. Ties count as half; top1 shares credit on ties.", "",
             "## Held-out results", "",
             "The AUROCs below are diagnostic reports for already-selected candidates. They do not choose the candidates. Chance AUROC and pairwise win rate are 0.5.", "",
             "| Candidate family | Readout | Pixel macro AUC | K-group macro AUC | Agentview group AUC | Wrist group AUC | Object-mean pairwise win | Top1 | Own-phrase win |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for family, value in summary.items():
        lines.append(f"| {family} | {value['readout']} | {value['pixel_macro_auc']:.4f} | {value['group_macro_auc']:.4f} | {value['agentview_group_macro_auc']:.4f} | {value['robot0_eye_in_hand_group_macro_auc']:.4f} | {value['test_pairwin']:.4f} | {value['test_top1']:.4f} | {value['own_phrase_win']:.4f} |")
    lines += ["", "Own-phrase win asks whether an object region responds more to its own phrase than to another simultaneously visible object phrase, using spatially normalized maps in the same video. Chance/ties are 0.5. It is a diagnostic only and never reselects heads on the test set.", "", "## Fixed Candidates", ""]
    for family, value in summary.items():
        name = value["name"] if len(value["heads"]) <= 24 else f"{len(value['heads'])} heads; full indices in report.json"
        lines.append(f"- {family}: {name} ({value['readout']}).")
    lines += ["", "## Per Suite And View", "",
              "| Suite/view | Previous | Single | Layer | Head combination | Layer combination | All heads |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in sorted(summary["single"]["by_suite_view"]):
        values = [summary[family]["by_suite_view"][group]["group_macro_auc"] for family in families]
        lines.append("| " + group + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines += ["", "## Interpretation Limits", "",
              "- Official BDDL objects of interest associate prompt noun spans with evaluation targets. The complete instruction remains the model condition. This scan does not supply a general live instruction parser.",
              "- These scores compare annotated visible surface cells, not full segmentation masks or background false-positive area. A good score alone does not establish clean localization.",
              "- Drawer spans are evaluated with whole-cabinet point labels. The Goal task5 table-region target has no tracked object root and is explicitly unscored; its plate target is still evaluated.",
              "- Empty-prompt K retains all 128 channels and defines the same region averaging for every candidate. Ground-truth pixels do not generate, trim, or smooth heatmaps.",
              "- Visual overlays show response above uniform density, with one scale shared across candidates and frames for each phrase/window. GT bounds appear only in the RGB reference panel.",
              "- This is observed-video semantic localization. It is not a future prediction, tracking accuracy, or robot rollout success evaluation.", "",
              "## Artifacts", "",
              "- manifest.json: all 40 tasks, splits, object phrase annotations and unscored targets.",
              "- report.json: every calibration candidate and the frozen family winners.",
              "- held_out_localization.json: all 40 held-out camera-window localization results.",
              "- Per-episode/view folders: full head/layer maps and first/middle/last-frame comparisons.",
              "- Four *_test_overview.jpg files: all five held-out tasks per suite, agentview.", ""]
    (args.scan / "current_report.md").write_text("\n".join(lines))
    print(json.dumps({"complete": True, "report": str((args.scan / "current_report.md").resolve()), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
