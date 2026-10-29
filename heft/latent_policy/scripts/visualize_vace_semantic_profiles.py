#!/usr/bin/env python3
"""Render predetermined held-out examples for suite-specific semantic profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_vace_prompt_k_guidance import ROOT
from scan_vace_semantics_all_suites import render_case
from search_vace_tracking_heads import CAMERAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.scan / "report.json").read_text())
    profiles = json.loads((args.scan / "suite_profiles.json").read_text())["profiles"]
    manifest = json.loads((args.scan / "manifest.json").read_text())
    examples = {}
    for row in manifest:
        if row["split"] == "test":
            examples.setdefault(row["suite"], row)
    output = args.scan / "suite_profile_visuals"
    output.mkdir(exist_ok=True)
    for suite, task in examples.items():
        winners = {**report["global_winners"], "head_combination": profiles[suite]}
        for camera in CAMERAS:
            row = {**task, "camera": camera}
            name = f"episode_{task['episode']:05d}_{camera}"
            directory = output / name
            directory.mkdir(exist_ok=True)
            with np.load(args.scan / name / "fields.npz") as cache:
                panels = render_case(row, cache, winners, directory,
                                     ROOT / "runs/vace_tracking_all_suites_20261004/videos")
            sheet = Image.new("RGB", (panels[0].width, sum(panel.height for panel in panels)), "white")
            for index, panel in enumerate(panels):
                sheet.paste(panel, (0, index * panel.height))
            sheet.save(output / f"{suite}_{camera}.jpg", quality=94)
            print(json.dumps({"suite": suite, "camera": camera, "episode": task["episode"], "profile": profiles[suite]["name"]}), flush=True)
    (output / "protocol.json").write_text(json.dumps({"new_inference": False, "GT_generates_heatmaps": False,
        "example_selection": "first held-out task per suite, deterministic; no visual/metric-based choice",
        "columns": ["RGB with GT reference bounds", "shared T14H9", "shared full layer6", "suite-specific frozen combination"],
        "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
