#!/usr/bin/env python3
"""Report fixed-head results and suite-held-out head selection without new forwards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark_vace_motion_heads_all_suites import PHASES, dump


def fixed(rows, branch, layer, head, state, experiment="gt_assisted", decoder="anchor_qk"):
    return next(row for row in rows if (row["branch"], row["layer"], row["head"], row["state_after_steps"],
                                       row["experiment"], row["decoder"]) == (branch, layer, head, state, experiment, decoder))


def value(number, percentage=False):
    if number is None:
        return "n/a"
    return f"{100 * number:.2f}%" if percentage else f"{number:.2f}"


def suite_holdout(ranking):
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    results = []
    for experiment, state in (("gt_assisted", 13), ("observed_all_noise", 0)):
        for branch in ("main", "control"):
            candidates = [row for row in ranking if row["experiment"] == experiment
                          and row["state_after_steps"] == state and row["branch"] == branch]
            for phase in ("moving_object_displaced", "gripper_visible"):
                for held_out in suites:
                    scored = []
                    for row in candidates:
                        strata = row["metrics"][phase]["strata"]
                        train = [item["epe_px"] for name, item in strata.items()
                                 if not name.startswith(held_out + "/") and item["epe_px"] is not None]
                        if train:
                            scored.append((float(np.mean(train)), row))
                    if not scored:
                        continue
                    score, selected = min(scored, key=lambda item: (item[0], item[1]["layer"], item[1]["head"], item[1]["decoder"]))
                    test_strata = {name: item for name, item in selected["metrics"][phase]["strata"].items()
                                   if name.startswith(held_out + "/")}
                    test = {}
                    for metric in ("epe_px", "within8", "trajectory80_at8"):
                        values = [item[metric] for item in test_strata.values() if item[metric] is not None]
                        test[metric] = float(np.mean(values)) if values else None
                    test["eligible_cases"] = sum(item["cases"] for item in test_strata.values())
                    results.append({"experiment": experiment, "state": state, "branch": branch, "phase": phase,
                                    "held_out_suite": held_out, "selection_epe_other_suites": score,
                                    "layer": selected["layer"], "head": selected["head"], "decoder": selected["decoder"],
                                    "test": test, "test_strata": test_strata})
    return results


def plots(output, rows, baselines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    states = [0, 10, 13, 18, 22]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, phase, title, control_head in zip(axes, ("moving_object_displaced", "gripper_visible"),
                                              ("Displaced object points", "Visible gripper points"), ((11, 1), (7, 11))):
        for branch, layer, head, color in (("main", 15, 4, "#a43187"), ("control", *control_head, "#19866b")):
            y = [fixed(rows, branch, layer, head, state)["metrics"][phase]["epe_px"] for state in states]
            ax.plot(states, y, "o-", color=color, label=f"{branch} L{layer}H{head}")
        for name, label, color in (("hold_position", "Hold position", "#777777"), ("recent_velocity", "Recent GT-point velocity", "#3377aa")):
            y = baselines[name][phase]["epe_px"]
            if y is not None:
                ax.axhline(y, linestyle="--", color=color, linewidth=1.2, label=label)
        ax.set(title=title, xlabel="Completed denoising updates", ylabel="Macro EPE (256-image pixels)")
        ax.set_xticks(states)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(output / "fixed_head_stages.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    rows = json.loads((output / "fixed_head_summary.json").read_text())
    ranking = json.loads((output / "global_ranking_exploratory.json").read_text())
    baselines = json.loads((output / "baseline_summary.json").read_text())
    completion = json.loads((output / "completion.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    folds = suite_holdout(ranking)
    dump(output / "leave_one_suite_out.json", {
        "selection": "Choose one head and decoder on the other three suites, evaluate on the excluded suite; stage13 fixed in advance",
        "calibration": "Episode39 excluded everywhere; original fixed L15H4 results reported separately",
        "results": folds,
    })
    report = ["# Multi-suite native QK trajectory comparison", "",
              f"Completed {completion['finished_cases']}/{completion['planned_cases']} camera windows; primary aggregation uses {completion['independent_cases']} independent windows.", "",
              "Original frozen VACE, no LoRA or training. Episode39, used to choose L15H4 previously, is excluded from all primary aggregation and suite-holdout selection. Both views of the same episode share the same simulator-selected window.", "",
              "## Fixed main head across denoising stages", "",
              "Errors use original256x256 coordinates. Objects and suite/camera strata receive equal weight; moving and static points are scored separately.", "",
              "| Updates | Displaced object EPE | Within8 | Visible gripper EPE | Within8 |", "| --- | ---: | ---: | ---: | ---: |"]
    for state in [0, 10, 13, 18, 22]:
        metric = fixed(rows, "main", 15, 4, state)["metrics"]
        obj, grip = metric["moving_object_displaced"], metric["gripper_visible"]
        report.append(f"| {state} | {value(obj['epe_px'])} | {value(obj['within8'], True)} | {value(grip['epe_px'])} | {value(grip['within8'], True)} |")
    report += ["", "## Matched fixed-head comparison at state13", "",
               "All heads below are fixed globally, not selected per test case. Both branches read the same model forward, prompt, noisy state, grid and query seeds.", "",
               "| Candidate | Displaced object EPE | Within8 | Object trajectory80 | Visible gripper EPE | Within8 |",
               "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for branch, layer, head in (("main", 15, 4), ("control", 11, 1), ("control", 7, 11)):
        metric = fixed(rows, branch, layer, head, 13)["metrics"]
        obj, grip = metric["moving_object_displaced"], metric["gripper_visible"]
        report.append(f"| {branch} L{layer}H{head} | {value(obj['epe_px'])} | {value(obj['within8'], True)} | {value(obj['trajectory80_at8'], True)} | {value(grip['epe_px'])} | {value(grip['within8'], True)} |")
    for name, metric in baselines.items():
        obj, grip = metric["moving_object_displaced"], metric["gripper_visible"]
        report.append(f"| {name} | {value(obj['epe_px'])} | {value(obj['within8'], True)} | {value(obj['trajectory80_at8'], True)} | {value(grip['epe_px'])} | {value(grip['within8'], True)} |")
    report += ["", "## Suite and camera breakdown", "",
               "Moving-object rows require actual world displacement above1cm and image displacement above8px, and object excursion above2cm. Missing rows are unavailable supervision, not tracking successes. Gripper-visible includes rigid near-static wrist-camera points and must be compared with hold-position.", "",
               "| Suite / camera | Eligible object cases | Main object EPE / within8 | Control11H1 object EPE | Main gripper EPE / within8 | Control7H11 gripper EPE |",
               "| --- | ---: | --- | ---: | --- | ---: |"]
    main_metrics = fixed(rows, "main", 15, 4, 13)["metrics"]
    obj_control = fixed(rows, "control", 11, 1, 13)["metrics"]
    grip_control = fixed(rows, "control", 7, 11, 13)["metrics"]
    for name, obj in main_metrics["moving_object_displaced"]["strata"].items():
        grip = main_metrics["gripper_visible"]["strata"][name]
        other_obj = obj_control["moving_object_displaced"]["strata"][name]
        other_grip = grip_control["gripper_visible"]["strata"][name]
        report.append(f"| {name} | {obj['cases']} | {value(obj['epe_px'])} / {value(obj['within8'], True)} | {value(other_obj['epe_px'])} | {value(grip['epe_px'])} / {value(grip['within8'], True)} | {value(other_grip['epe_px'])} |")
    report += ["", "## All-observed, pure-noise-main control", "",
               "The identical49-frame window is fully supplied to the condition branch, mask0, empty prompt, main pure noise at timestep1000, one forward. This resembles the earlier observation-tracking input route, with a matched short query horizon. These later frames are known observations in this protocol, not predictions.", "",
               "| Candidate | Displaced object EPE | Within8 | Visible gripper EPE | Within8 |", "| --- | ---: | ---: | ---: | ---: |"]
    for branch, layer, head in (("main", 15, 4), ("control", 11, 1), ("control", 7, 11)):
        metric = fixed(rows, branch, layer, head, 0, "observed_all_noise")["metrics"]
        obj, grip = metric["moving_object_displaced"], metric["gripper_visible"]
        report.append(f"| {branch} L{layer}H{head} | {value(obj['epe_px'])} | {value(obj['within8'], True)} | {value(grip['epe_px'])} | {value(grip['within8'], True)} |")
    report += ["", "## Head selected without looking at the held-out suite", "",
               "For each row, choose one head/decoder using the other three suites and evaluate the excluded suite. Selection is per target type; a gripper head and an object head may differ. There are360 main versus180 control heads, with two decoders each, so search budgets differ.", "",
               "| Protocol | Excluded suite | Target | Branch | Selected head / decoder | Test EPE | Test within8 |", "| --- | --- | --- | --- | --- | ---: | ---: |"]
    for fold in folds:
        report.append(f"| {fold['experiment']} | {fold['held_out_suite']} | {fold['phase']} | {fold['branch']} | L{fold['layer']}H{fold['head']} / {fold['decoder']} | {value(fold['test']['epe_px'])} | {value(fold['test']['within8'], True)} |")
    report += ["", "## Interpretation and limits", "",
               "- The short horizon remains16 later RGB frames,0.8 seconds, represented by4 native future temporal positions. This does not establish multi-second, full-episode tracking.",
               "- GT-assisted initialization uses true later video latents. This experiment measures privileged trajectory extraction, not autonomous future prediction.",
               "- Error against simulator GT combines content drift during denoising and correspondence-decoding error. It does not isolate either cause.",
               "- The main query token is located in the last observed temporal slice, but native attention is global and can incorporate later inputs.",
               "- Control receives a main-input patch embedding at its first block. Its input is not completely independent of main noisy video content.",
               "- All comparisons use original frozen heads. The previous trained tracking/object LoRA checkpoints and beam processing are not included here.",
               "- Windows are selected for simulator motion independently of head predictions. This is a motion-enriched diagnostic subset, not all frames or a LIBERO task-success benchmark.",
               "- Seeds are chosen using only anchor visibility and spatial spread per rigid body. GT later visibility is used for scoring; no future reanchoring, smoothing or spatial matching masks are applied.",
               "- Object-trajectory80 requires at least3 eligible later samples and80% within8px. It is a short-trajectory metric and can have less coverage than EPE.",
               "- Per-case GT-selected best heads and global test-set rankings are stored as exploratory diagnostics; they are not the fixed-head result.", "",
               "## Artifacts", "",
               "`videos.md` links every case; `fixed_head_stages.png` plots stage comparisons; `fixed_head_summary.json` includes all suite/view strata; `leave_one_suite_out.json` contains independent-suite selections; each case stores exact fixed trajectories, query protocol and all native-head scores.", ""]
    dump(output / "report_status.json", completion)
    (output / "current_report.md").write_text("\n".join(report))
    index = ["# All-suite tracking videos", "", "State13; GT green and QK magenta. Left is fixed main L15H4, right is frozen control11H1 for objects or7H11 for gripper. Original RGB backgrounds; no generated frames or interpolated tracks.", ""]
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        index += ["## " + suite, "", "| Task | Camera | Moving object | Gripper | Endpoint image |", "| --- | --- | --- | --- | --- |"]
        for case in manifest:
            if case["suite"] != suite:
                continue
            directory = output / "cases" / case["name"]
            links = []
            for filename, label in (("moving_object_displaced.mp4", "video"), ("gripper_visible.mp4", "video"), ("endpoint_comparison.jpg", "image")):
                path = directory / filename
                links.append(f"[{label}]({path.resolve()})" if path.exists() else "n/a")
            suffix = " (calibration, excluded)" if case["calibration_episode"] else ""
            index.append(f"| {case['task_id']}: {case['task']}{suffix} | {case['camera']} | " + " | ".join(links) + " |")
        index.append("")
    (output / "videos.md").write_text("\n".join(index))
    with (output / "fixed_head_per_suite.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "updates", "branch", "layer", "head", "decoder", "phase", "suite_camera", "eligible_cases", "epe_px", "within8", "trajectory80_at8"])
        for row in rows:
            for phase in PHASES:
                for name, metrics in row["metrics"][phase]["strata"].items():
                    writer.writerow([row["experiment"], row["state_after_steps"], row["branch"], row["layer"], row["head"], row["decoder"], phase, name,
                                     metrics["cases"], metrics["epe_px"], metrics["within8"], metrics["trajectory80_at8"]])
    plots(output, rows, baselines)
    print(json.dumps(completion), flush=True)


if __name__ == "__main__":
    main()
