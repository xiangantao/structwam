#!/usr/bin/env python3
"""Display public K/QK fields and phrase-selected region/trajectory bundles."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from render_vace_sparse_future_demo import write_h264
from scan_vace_main_prompt_attention import read_window
from vace_latent_distill.bundles import phrase_positions, read_region_bundles
from vace_latent_distill.config import NEGATIVE_PROMPT, PipelineConfig
from vace_latent_distill.data import stable_seed
from vace_latent_distill.teacher import VaceTeacher, content_token_masks

COLORS = np.array([(244, 225, 95), (255, 96, 78), (53, 210, 255),
                   (187, 123, 255), (85, 239, 140), (255, 154, 57)], np.uint8)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PLANS = [
    {"episode": 39, "start": 48, "phrases": ["black bowl", "plate", "wooden cabinet"]},
    {"episode": 40, "start": 32, "phrases": ["alphabet soup", "basket"]},
]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def progress(**value):
    print(json.dumps(value, allow_nan=False), flush=True)


def timeline_index(frame):
    if frame < 33:
        return "observed", min(4 + frame / 4, 11)
    return "future", min(8 + (frame - 33) / 4, 12)


def at_time(field, time):
    left, right = int(time), min(int(time) + 1, len(field) - 1)
    return field[left] * (1 - (time - left)) + field[right] * (time - left)


def tile(image, caption, size=512):
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    result = np.full((size + 40, size, 3), 249, np.uint8)
    result[40:] = image
    cv2.putText(result, caption, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.55 if size == 512 else 0.43, (35, 35, 35), 1, cv2.LINE_AA)
    return result


def overlay(rgb, heat, color, bounds=(0, 1), alpha=0.6):
    value = np.clip((heat - bounds[0]) / max(float(bounds[1] - bounds[0]), 1e-8), 0, 1)
    value = cv2.resize(value.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR)
    opacity = (alpha * value)[..., None]
    return np.uint8(rgb * (1 - opacity) + np.asarray(color) * opacity)


def fit_display(cases):
    result = {}
    for key in ("seg_k_l22h9", "seg_k_l15h7"):
        fields = [case["target"][branch][key].float().numpy()
                  for case in cases for branch in ("observed", "future")]
        centered = [field - field.mean(1, keepdims=True) for field in fields]
        sample = np.concatenate([field.reshape(-1, 128)[::4] for field in centered])
        sample_tensor = torch.from_numpy(sample)
        mean = sample.mean(0)
        torch.manual_seed(42)
        _, singular, rotation = torch.pca_lowrank(sample_tensor, q=3, center=True, niter=5)
        projection = rotation.T.numpy()
        low, high = np.percentile((sample - mean) @ projection.T, (1, 99), axis=0)
        variance = singular.square() / (len(sample) - 1) / sample_tensor.var(0).sum()
        result[key] = {"projection": projection, "mean": mean,
                       "low": low, "high": high,
                       "variance": variance.numpy()}
        for case in cases:
            case.setdefault("k_colors", {})[key] = {}
            for branch in ("observed", "future"):
                field = case["target"][branch][key].float().numpy()
                colors = ((field - field.mean(1, keepdims=True)).reshape(-1, 128) - mean) @ projection.T
                colors = np.clip((colors - low) / np.maximum(high - low, 1e-8), 0, 1)
                case["k_colors"][key][branch] = np.uint8(colors.reshape(len(field), 32, 32, 3) * 255)
    return result


def draw_paths(rgb, decoded, branch, frame, bundle=None):
    image = rgb.copy()
    if bundle is None:
        paths, masks, seeds = (decoded[key] for key in
                               ("public_paths", "public_path_valid", "public_seeds"))
    else:
        paths, masks, seeds = (decoded[key][bundle] for key in ("paths", "path_valid", "seeds"))
    anchors_rgb = (0, 8, 16, 24) if branch == "observed" else (33, 41)
    for ai, (anchor, rgb_anchor) in enumerate(zip(decoded["anchor_native"], anchors_rgb, strict=True)):
        if frame < rgb_anchor:
            continue
        ids = np.flatnonzero(masks[ai, 0])
        coordinates = paths[ai, :, ids].transpose(1, 0, 2)
        elapsed = min((frame - rgb_anchor) / 4, len(ids) - 1)
        left, right = int(elapsed), min(int(elapsed) + 1, len(ids) - 1)
        current = coordinates[:, left] * (1 - (elapsed - left)) + coordinates[:, right] * (elapsed - left)
        for point, cell in enumerate(seeds[ai]):
            if bundle is None and (cell % 32 % 4 or cell // 32 % 4):
                continue
            history = np.concatenate((coordinates[point, :left + 1], current[point:point + 1]))
            pixels = np.rint((history + 0.5) * 16).astype(np.int32)
            color = tuple(map(int, COLORS[bundle % len(COLORS)])) if bundle is not None else (
                tuple(map(int, cv2.cvtColor(np.uint8([[[cell % 32 * 5, 210, 255]]]), cv2.COLOR_HSV2RGB)[0, 0])))
            cv2.polylines(image, [pixels], False, color, 1, cv2.LINE_AA)
            if frame <= rgb_anchor + 8:
                cv2.circle(image, tuple(pixels[-1]), 2, color, -1, cv2.LINE_AA)
    return image


def labeled_sheet(rows, captions, frames, title):
    panels = [np.concatenate([tile(image, caption, 384) for image, caption in
                             zip(images, captions, strict=True)], axis=1) for images in rows]
    width = panels[0].shape[1] + 100
    result = Image.new("RGB", (width, 62 + sum(x.shape[0] for x in panels)), "#f9fafb")
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype(FONT, 19)
    draw.text((12, 14), title, font=font, fill="#222222")
    offset = 62
    for frame, panel in zip(frames, panels, strict=True):
        label = f"OBS {frame}" if frame < 33 else f"FUT {frame - 33}"
        draw.text((8, offset + 50), label, font=ImageFont.truetype(FONT, 15), fill="#333333")
        result.paste(Image.fromarray(panel), (100, offset))
        offset += panel.shape[0]
    return result


def diagnostics(decoded):
    output = {}
    names = decoded["names"]
    for branch in ("observed", "future"):
        value = decoded[branch]
        overlaps = []
        for left in range(1, len(names)):
            for right in range(left + 1, len(names)):
                a, b = value["seeds"][left], value["seeds"][right]
                overlaps.append({"left": names[left], "right": names[right],
                                 "seed_jaccard_by_anchor": [len(set(x) & set(y)) / len(set(x) | set(y))
                                                            for x, y in zip(a, b, strict=True)]})
        output[branch] = {"phrase_seed_overlap": overlaps,
                          "public_points_per_anchor": int(value["public_paths"].shape[1]),
                          "region_heat_shape": list(value["region_heat"].shape),
                          "trajectory_heat_shape": list(value["trajectory_heat"].shape)}
    return output


def render_case(case, output, source="teacher"):
    names, values = case["decoded"]["names"], case["decoded"]
    keys = ("seg_k_l22h9", "seg_k_l15h7")
    color_bounds = []
    for bundle in range(len(names)):
        heat = np.concatenate([values[b]["prompt_heat"][bundle, start:].ravel()
                               for b, start in (("observed", 4), ("future", 8))])
        color_bounds.append(np.percentile(heat, (1, 99)))
    public_video, public_tracks, selected_video = [], [], []
    phrase_videos = [[] for _ in names]
    keyframes = (0, 16, 32, 41, 48)
    public_rows, raw_rows, region_rows, trajectory_rows = [], [], [], []
    for frame, pixels in enumerate(case["rgb"]):
        rgb = cv2.resize(pixels, (512, 512))
        branch, native = timeline_index(frame)
        decoded = values[branch]
        k_maps = [cv2.resize(case["k_colors"][key][branch][int(native)], (512, 512),
                            interpolation=cv2.INTER_NEAREST) for key in keys]
        public_images = [rgb, *(np.uint8(rgb * 0.25 + colored * 0.75) for colored in k_maps)]
        raw_images, regions, motion = [rgb], [rgb], [rgb]
        combined_region, combined_tracks = rgb.copy(), rgb.copy()
        for bundle, name in enumerate(names):
            heat = at_time(decoded["prompt_heat"][bundle], native)
            region = at_time(decoded["region_heat"][bundle], native)
            raw = overlay(rgb, heat, COLORS[bundle % len(COLORS)], color_bounds[bundle])
            selection = overlay(rgb, region, COLORS[bundle % len(COLORS)])
            tracks = draw_paths(selection, decoded, branch, frame, bundle)
            density = at_time(decoded["trajectory_heat"][bundle], native)
            maximum = max(float(decoded["trajectory_heat"][bundle].max()), 1e-8)
            trajectory = overlay(rgb, density, COLORS[bundle % len(COLORS)], (0, maximum), 0.8)
            raw_images.append(raw)
            regions.append(selection)
            motion.append(trajectory)
            phrase_videos[bundle].append(np.concatenate(
                (tile(raw, f"{name}: word response"), tile(selection, "Prompt + K region"),
                 tile(tracks, "Region + dense QK tracks")), axis=1))
            if bundle:
                combined_region = overlay(combined_region, region, COLORS[bundle % len(COLORS)], alpha=0.45)
                combined_tracks = draw_paths(combined_tracks, decoded, branch, frame, bundle)
        future_tag = "1 update" if source == "teacher" else "predicted"
        phase = f"OBS {frame:02d}" if branch == "observed" else f"FUT {frame - 33:02d} | {future_tag}"
        public_video.append(np.concatenate([tile(image, caption) for image, caption in zip(
            public_images, (f"Actual RGB | {phase}", "Full-scene K: L22H9", "Full-scene K: L15H7"),
            strict=True)], axis=1))
        public_tracks.append(tile(draw_paths(rgb, decoded, branch, frame), f"Public QK tracks | {phase}"))
        selected_video.append(np.concatenate((tile(rgb, f"Actual RGB | {phase}"),
                                               tile(combined_region, "Named regions (overlap retained)"),
                                               tile(combined_tracks, "Named dense trajectory bundles")), axis=1))
        if frame in keyframes:
            public_rows.append(public_images)
            raw_rows.append(raw_images)
            region_rows.append(regions)
            trajectory_rows.append(motion)
    stem = f"ep{case['episode']:02d} {case['camera']} | actual RGB | OBS33 + FUT16 {source}"
    labeled_sheet(public_rows, ("RGB", "Public K L22H9", "Public K L15H7"), keyframes,
                  stem + " | shared fixed K color projection").save(output / "public_regions.jpg", quality=95)
    for filename, rows, caption in (("phrase_response.jpg", raw_rows, "Raw word response"),
                                     ("phrase_regions.jpg", region_rows, "Prompt + K regions"),
                                     ("trajectory_fields.jpg", trajectory_rows, "Dense bundle occupancy")):
        labeled_sheet(rows, ("RGB", *names), keyframes, stem + " | " + caption).save(output / filename, quality=95)
    write_h264(output / "public_regions.mp4", np.stack(public_video))
    write_h264(output / "public_qk_tracks.mp4", np.stack(public_tracks))
    write_h264(output / "named_regions_and_tracks.mp4", np.stack(selected_video))
    for name, frames in zip(names, phrase_videos, strict=True):
        write_h264(output / f"{name.lower().replace(' ', '_')}_bundle.mp4", np.stack(frames))
    arrays = {f"{branch}_{key}": value for branch in ("observed", "future")
              for key, value in values[branch].items()}
    np.savez_compressed(output / "region_bundles.npz", names=np.asarray(names), **arrays)
    result = diagnostics(values)
    dump(output / "diagnostics.json", result)
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vace_structured.json")
    parser.add_argument("--data", type=Path, default=WORKSPACE / "runs/vace_tracking_all_suites_20261004")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if args.output.exists() and not args.render_only:
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True, exist_ok=args.render_only)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    config = PipelineConfig.load(args.config)
    if (config.teacher.denoise_updates, config.teacher.denoise_reference_updates) != (1, 13):
        parser.error("This demo follows the one-update mainline")
    manifest = json.loads((args.data / "videos/manifest.json").read_text())
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(config.teacher.text_root) / "tokenizer", local_files_only=True)
    cases = []
    for plan in PLANS:
        metadata = next(row for row in manifest if row["episode"] == plan["episode"])
        for camera in ("agentview", "robot0_eye_in_hand"):
            video = args.data / "videos" / f"episode_{plan['episode']:05d}" / f"{camera}.mp4"
            rgb = read_window(video, plan["start"], 49)
            case = {**plan, "camera": camera, "prompt": metadata["task"],
                    "suite": metadata["suite"], "video": str(video), "rgb": rgb,
                    "seed": stable_seed(42, plan["episode"], camera, plan["start"])}
            case["directory"] = args.output / f"episode_{plan['episode']:05d}_{camera}"
            case["phrase_masks"] = phrase_positions(tokenizer, case["prompt"], plan["phrases"])
            case["directory"].mkdir(exist_ok=args.render_only)
            cases.append(case)
    content = content_token_masks(config.teacher.text_root, [case["prompt"] for case in cases])
    started = time.monotonic()
    if not args.render_only:
        progress(stage="load_teacher", cases=len(cases))
        teacher = VaceTeacher(config.teacher, "shared", "cuda:0", text_cache[NEGATIVE_PROMPT])
        try:
            for begin in range(0, len(cases), args.batch_size):
                batch = cases[begin:begin + args.batch_size]
                embeddings = [text_cache[case["prompt"]] for case in batch]
                targets = teacher.extract_batch(
                    [torch.from_numpy(case["rgb"]).permute(0, 3, 1, 2).contiguous() for case in batch],
                    embeddings, [embedding.abs().sum(-1) != 0 for embedding in embeddings],
                    [case["seed"] for case in batch], [content[case["prompt"]] for case in batch])
                for case, target in zip(batch, targets, strict=True):
                    case["target"] = {branch: {key: value for key, value in target[branch].items()
                                              if key not in {"depth", "seeds", "paths", "path_valid"}}
                                      for branch in ("observed", "future")}
                    case["teacher_protocol"] = dict(teacher.protocol)
                    torch.save({"target": case["target"], "teacher_protocol": case["teacher_protocol"]},
                               case["directory"] / "teacher_fields.pt")
                    progress(stage="teacher_captured", episode=case["episode"], camera=case["camera"],
                             capture_sigma=teacher.protocol["capture_sigma"])
        finally:
            teacher.close()
            del teacher
            gc.collect()
            torch.cuda.empty_cache()
        progress(stage="gpu_released")
    else:
        for case in cases:
            case.update(torch.load(case["directory"] / "teacher_fields.pt", map_location="cpu", weights_only=True))
    projection = fit_display(cases)
    np.savez(args.output / "display_projection_only.npz",
             **{f"{key}_{name}": value for key, settings in projection.items() for name, value in settings.items()})
    summary = []
    for case in cases:
        progress(stage="decode_regions", episode=case["episode"], camera=case["camera"])
        case["decoded"] = {"names": ["Full instruction", *case["phrase_masks"]]}
        for branch in ("observed", "future"):
            case["decoded"][branch] = read_region_bundles(
                case["target"][branch], branch, case["phrase_masks"], content[case["prompt"]],
                config.teacher, lambda **event: progress(stage="bundle_decoded", **event))
        metrics = render_case(case, case["directory"])
        record = {key: case[key] for key in ("episode", "camera", "suite", "prompt", "phrases", "start", "seed", "video")}
        record.update(directory=str(case["directory"]), teacher_protocol=case["teacher_protocol"], diagnostics=metrics)
        summary.append(record)
        dump(args.output / "summary.json", summary)
        progress(stage="case_rendered", episode=case["episode"], camera=case["camera"])
    dump(args.output / "protocol.json", {
        "config": config.to_dict(), "student_loaded": False, "training": False, "depth_displayed": False,
        "teacher_input": "Observed33 +15 copies of frame0; future full49 GT+sigma0.8, control past33, one update to original step13 endpoint",
        "phrases": "Explicit noun phrases from the full instruction; all occurrences read in the same contextual text embedding",
        "full_instruction_heat": "Maximum content-token response; never the constant sum over all valid text tokens",
        "public_regions": "Direct full-scene normalized K, both heads retained; no cluster centers/count or GT labels",
        "public_tracking": "Every one of the 1024 cells seeded at every anchor; neutral region gate, same QK/beam decoder",
        "public_track_display": "Draw every fourth row/column to keep RGB visible; all 1024 tracks are saved",
        "phrase_regions": "Existing prompt seed5% and K keep15% readout, independent per phrase; overlaps retained, no forced disjoint assignment",
        "colors": "Phrase colors name readout slots, not GT identities; fixed K PCA3 only for display, fitted without labels over all cases/branches",
        "heat_scale": "One p1-p99 scale per phrase across both branches, no per-frame display stretching",
        "region_vs_trajectory": "Region heat is read independently from prompt/K; trajectory heat is bilinear occupancy with point/anchor paths retained",
        "temporal_support": "Short 8-frame multi-anchor paths; validity is time support, not visibility confidence",
        "background": "Actual recorded RGB including the16 future GT frames; not a generated video or sparse latent decode",
        "fps": 10, "elapsed_seconds": time.monotonic() - started,
    })
    progress(stage="completed", output=str(args.output), seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
