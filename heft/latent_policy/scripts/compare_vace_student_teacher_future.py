#!/usr/bin/env python3
"""Put saved teacher future fields and student future fields through one readout.

Teacher: the saved 1-update (``sparse``) and 13-update (``reference``) future fields
of ``render_vace_sparse_future_demo.py`` for tracking episode 39.  Student: the
future branch exported by ``render_vace_student_region_bundles.py`` for the same
window of the LeRobot recording (source episode 1653), which is an exact
horizontal mirror; student grids are mirrored back (x -> 31 - x) so all modes
share the tracking-render coordinates and simulator GT.

Every mode uses the same prompt -> K region -> 64 seeds rule, the same beam
decoder and the same drawing code.  GT scoring reuses the saved GT query cells
and the evaluator's ``measure``.  Display background is actual recorded RGB.
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

from vace_latent_distill.teacher import local_scripts

local_scripts()
import demo_main_prompt_k_tracking as demo
import render_vace_sparse_future_demo as future_demo
from evaluate_vace_sparse_future_tracks import summarize
from render_vace_student_region_bundles import write_h264
from scan_vace_main_prompt_attention import read_window
from search_vace_tracking_heads import xy_to_cell
from vace_latent_distill.bundles import phrase_positions
from vace_latent_distill.config import PipelineConfig

GRID = 32
ANCHORS = (8, 10)
LABELS = {"sparse": "TEACHER 1-update", "reference": "TEACHER 13-update", "student": "STUDENT"}


def mirror_cells(array):
    """[T, 1024, ...] -> horizontally mirrored grid cells."""
    shape = array.shape
    return array.reshape(shape[0], GRID, GRID, *shape[2:])[:, :, ::-1].reshape(shape).copy()


def teacher_values(saved, mode):
    return {"heat": saved[f"{mode}_prompt_heat"], "q": saved[f"{mode}_q"], "k": saved[f"{mode}_k"],
            "fields": {22: saved[f"{mode}_k22_fields"], 15: saved[f"{mode}_k15_fields"]}}


def student_values(fields, phrase_mask):
    future = {key: value.float().numpy() for key, value in fields["future"].items()}
    heat = mirror_cells(future["text_attention"][..., phrase_mask].sum(-1))
    return {"heat": heat.reshape(-1, GRID, GRID),
            "q": mirror_cells(future["track_q"]).reshape(-1, future["track_q"].shape[-1]),
            "k": mirror_cells(future["track_k"]).reshape(-1, future["track_k"].shape[-1]),
            "fields": {22: mirror_cells(future["seg_k_l22h9"]), 15: mirror_cells(future["seg_k_l15h7"])}}


def caption(image, lines, height=34):
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    for row, text in enumerate(lines):
        cv2.putText(image, text, (6, 13 + 15 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return image


def pca_colors(modes, key):
    """One label-free PCA3 over all modes; unit vectors, per-frame mean removed (display only)."""
    def centered(field):
        unit = field / np.maximum(np.linalg.norm(field, axis=-1, keepdims=True), 1e-8)
        return unit - unit.mean(1, keepdims=True)

    sample = np.concatenate([centered(v["fields"][key]).reshape(-1, 128)[::3] for v in modes.values()])
    mean = sample.mean(0)
    _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
    projection = vt[:3]
    low, high = np.percentile((sample - mean) @ projection.T, (1, 99), axis=0)
    result = {}
    for name, values in modes.items():
        colors = (centered(values["fields"][key]).reshape(-1, 128) - mean) @ projection.T
        colors = np.clip((colors - low) / np.maximum(high - low, 1e-8), 0, 1)
        result[name] = np.uint8(colors.reshape(-1, GRID, GRID, 3) * 255)
    return result


def heat_overlay(rgb, heat, bounds):
    return future_demo.overlay_heat(rgb, heat, bounds)


def keyframe_sheet(background, modes, decoded, videos, output):
    frames = (0, 4, 8, 12, 15)
    k22, k15 = pca_colors(modes, 22), pca_colors(modes, 15)
    rows = []
    for name, values in modes.items():
        bounds = np.percentile(values["heat"][ANCHORS[0]:], (1, 99))
        for kind in ("K L22H9", "K L15H7", "black bowl heat", "QK + beam tracks"):
            tiles = []
            for frame in frames:
                native = 8 + frame / 4
                rgb = cv2.resize(background[frame], (512, 512))
                if kind.startswith("K"):
                    colors = (k22 if "22" in kind else k15)[name][int(native)]
                    colors = cv2.resize(colors, (512, 512), interpolation=cv2.INTER_NEAREST)
                    image = np.uint8(rgb * 0.25 + colors * 0.75)
                elif kind.endswith("heat"):
                    left, right = int(native), min(int(native) + 1, len(values["heat"]) - 1)
                    weight = native - left
                    heat = values["heat"][left] * (1 - weight) + values["heat"][right] * weight
                    image = heat_overlay(rgb, heat, bounds)
                else:
                    image = videos[name][frame].copy()
                image = caption(cv2.resize(image, (320, 320)), [f"{LABELS[name]} | {kind}", f"FUT {frame:02d}"], 32)
                tiles.append(image)
            rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(output / "keyframes_teacher_vs_student.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", type=Path, required=True,
                        help="Directory with future_track_comparison.npz and future_gt_accuracy_paths.npz")
    parser.add_argument("--student-fields", type=Path, required=True)
    parser.add_argument("--tracking-video", type=Path, required=True, help="Tracking-render agentview.mp4 of episode 39")
    parser.add_argument("--config", type=Path, required=True, help="pipeline.json of the student run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)
    config = PipelineConfig.load(args.config)
    protocol = json.loads((args.teacher_run / "protocol.json").read_text())
    start = protocol["source_frames"][0]
    with np.load(args.teacher_run / "future_track_comparison.npz") as archive:
        saved = {key: archive[key] for key in archive.files}
    if tuple(saved["anchor_latents"].tolist()) != ANCHORS:
        raise ValueError("Unexpected teacher future anchors")
    student = torch.load(str(args.student_fields), map_location="cpu", weights_only=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(config.teacher.text_root) / "tokenizer", local_files_only=True)
    phrase = phrase_positions(tokenizer, protocol["prompt"], [protocol["phrase"]])[protocol["phrase"]]

    modes = {"sparse": teacher_values(saved, "sparse"), "reference": teacher_values(saved, "reference"),
             "student": student_values(student["target"], phrase)}
    decoded = {name: future_demo.decode_tracks(values, ANCHORS, config.teacher) for name, values in modes.items()}
    for name in ("sparse", "reference"):  # The shared decoder must reproduce the saved teacher demo.
        np.testing.assert_array_equal(decoded[name]["seeds"], saved[f"{name}_seeds"])
        np.testing.assert_allclose(decoded[name]["paths"], saved[f"{name}_paths"], atol=1e-4)

    background = read_window(args.tracking_video, start + 33, 16)
    videos = {}
    for name in modes:
        frames = future_demo.render_future(background, decoded[name]["paths"], (0, 8), name)
        videos[name] = np.stack([caption(frame, [f"{LABELS[name]} | L15H4-style QK + beam | same decoder",
                                                 f"future frame {index:02d} | anchors 0,8 | actual RGB"])
                                 for index, frame in enumerate(frames)])
        selections = future_demo.render_selections(background, modes[name], decoded[name], ANCHORS, name)
        for row in range(2):
            for column, title in enumerate(("Actual RGB", f"{protocol['phrase']} heat", "Prompt/K score + 64 seeds")):
                y, x = row * 512, column * 512
                selections[y:y + 512, x:x + 512] = caption(selections[y:y + 512, x:x + 512],
                                                           [f"{LABELS[name]} | future anchor {row * 8} | {title}"])
        cv2.imwrite(str(args.output / f"selections_{name}.jpg"), cv2.cvtColor(selections, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    write_h264(args.output / "future_tracks_teacher1_vs_student.mp4",
               np.concatenate((videos["sparse"], videos["student"]), axis=2))
    write_h264(args.output / "future_tracks_teacher1_teacher13_student.mp4",
               np.concatenate((videos["sparse"], videos["reference"], videos["student"]), axis=2))
    keyframe_sheet(background, {k: modes[k] for k in ("sparse", "student")}, decoded, videos, args.output)

    # GT: identical saved query cells for every mode, each mode's own prompt/K gate.
    with np.load(args.teacher_run / "future_gt_accuracy_paths.npz") as archive:
        gt = {key: archive[key] for key in archive.files}
    records = []
    for index, anchor in enumerate(ANCHORS):
        xy = gt[f"anchor{index}_gt"]
        row = {"groups": gt[f"anchor{index}_groups"], "xy": xy, "valid": gt[f"anchor{index}_valid"],
               "displaced": gt[f"anchor{index}_displaced"], "hold": np.repeat(xy[:, :1], xy.shape[1], axis=1),
               "sparse": gt[f"anchor{index}_sparse_prediction"],
               "reference": gt[f"anchor{index}_reference_prediction"]}
        chosen = xy_to_cell(xy[:, 0], GRID, 256)
        end = min(anchor + 3, 13)
        for name in modes:
            values = modes[name]
            raw = demo.track_dense(values["q"][:end * GRID**2], values["k"][:end * GRID**2], chosen, anchor, GRID,
                                   decoded[name]["region_scores"][index, :end], topk=config.teacher.beam_topk,
                                   beam_size=config.teacher.beam_size, max_step=config.teacher.beam_max_step,
                                   motion_weight=config.teacher.motion_weight, local_weight=config.teacher.local_weight,
                                   region_weight=config.teacher.region_weight,
                                   acceleration_weight=config.teacher.acceleration_weight,
                                   descriptor_ema=config.teacher.descriptor_ema, direction="q_to_k")
            paths = demo.regularize_paths(raw, config.teacher.regularize_max_step,
                                          config.teacher.regularize_max_acceleration, passes=2)
            prediction = (paths + 0.5) * (256 / GRID) - 0.5
            if name != "student":
                np.testing.assert_allclose(prediction, row[name], atol=1e-3)
            row[name] = prediction
        records.append(row)
    seed_overlap = [{"anchor": a, "student_vs_teacher1": len(np.intersect1d(decoded["student"]["seeds"][i],
                                                                            decoded["sparse"]["seeds"][i]))}
                    for i, a in enumerate(ANCHORS)]
    result = {"teacher_run": str(args.teacher_run), "student_fields": str(args.student_fields),
              "student_step": student.get("step"), "phrase": protocol["phrase"],
              "orientation": "Student LeRobot grid mirrored x->31-x into tracking-render coordinates",
              "scoring": "Saved GT query cells, each mode's own prompt/K gate, frozen beam decoder; 256px",
              "seed_overlap_of_64": seed_overlap,
              "fixed_gt_queries": summarize(records, ("sparse", "reference", "student", "hold"))}
    (args.output / "gt_track_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
