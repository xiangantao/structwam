#!/usr/bin/env python3
"""Frozen minimal-pair instruction contrasts on the same observed K regions."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score

from evaluate_vace_prompt_k_guidance import FONT, ROOT, encode_prompts, evaluation_masks
from evaluate_vace_instruction_regions import capture_instruction, region_image, relevance_metrics
from search_vace_tracking_heads import CAMERAS, ControlTower, load_vae, read_video, write_video
from train_vace_k_vector_lora import install_k_lora
from vace_vae_warmup import observation_control, observation_window_indices

PAIRS = [
    ("pick up the alphabet soup and place it in the basket", "pick up the butter and place it in the basket"),
    ("grasp the alphabet soup and put it into the basket", "grasp the butter and put it into the basket"),
    ("take the alphabet soup and move it into the basket", "take the butter and move it into the basket"),
]
SOUP, BUTTER = "alphabet_soup_1", "butter_1"


def load_pair(cache, camera):
    directory = cache / f"episode_00040_{camera}"
    with np.load(directory / "K_regions.npz") as z:
        regions, times = z["regions"], z["times"]
    result = {}
    for source in ("task", "swap"):
        with np.load(directory / f"{source}_scores.npz") as z:
            if not np.array_equal(z["times"], times):
                raise RuntimeError("Instruction and K times differ")
            result[source] = {method: z[method].astype(np.float32) for method in ("group", "pixel")}
    return regions, times, {method: result["swap"][method] - result["task"][method]
                           for method in ("group", "pixel")}


def contrast_metrics(score, masks):
    a, b = score[masks[SOUP]], score[masks[BUTTER]]
    pair_auc = float(roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])) if len(a) and len(b) else None
    per_frame = []
    for ti in range(len(score)):
        av, bv = score[ti][masks[SOUP][ti]], score[ti][masks[BUTTER][ti]]
        if len(av) and len(bv):
            per_frame.append(float(roc_auc_score(np.r_[np.zeros(len(av)), np.ones(len(bv))], np.r_[av, bv])))
    means = {label: float(score[mask].mean()) if mask.any() else None for label, mask in masks.items()}
    positive_order = sorted([label for label, value in means.items() if value is not None], key=lambda label: -means[label])
    negative_order = list(reversed(positive_order))
    return {"butter_vs_soup_auc": pair_auc, "per_frame_pair_auc": float(np.mean(per_frame)) if per_frame else None,
            "butter_positive_fraction": float((b > 0).mean()) if len(b) else None,
            "soup_negative_fraction": float((a < 0).mean()) if len(a) else None,
            "butter_vs_all_macro_auc": relevance_metrics(score, masks, [BUTTER])["mean_auc"],
            "soup_vs_all_macro_auc": relevance_metrics(-score, masks, [SOUP])["mean_auc"],
            "object_mean_contrast": means,
            "butter_rank_of_object_means": positive_order.index(BUTTER) + 1 if BUTTER in positive_order else None,
            "soup_rank_of_object_means": negative_order.index(SOUP) + 1 if SOUP in negative_order else None,
            "visible_objects": len(positive_order)}


def collect_paraphrases(data, cache_report, device_name):
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(4)
    device = torch.device(device_name)
    tokenizer, texts = encode_prompts(ROOT / "models/Wan2.1-T2V-1.3B-Diffusers",
                                     [prompt for pair in PAIRS[1:] for prompt in pair], device)
    payload = torch.load(cache_report["checkpoint"], map_location="cpu", weights_only=False)
    model_root = ROOT / "models/Wan2.1-VACE-1.3B"
    tower = ControlTower.load(model_root / "diffusion_pytorch_model.safetensors", device)
    install_k_lora(tower, int(payload["layer"]), int(payload["head"]), int(payload["rank"]), float(payload["alpha"]))
    tower.load_state_dict(payload["lora"], strict=False)
    tower.eval().requires_grad_(False)
    vae = load_vae(model_root / "Wan2.1_VAE.pth", device)
    special_ids, zeros = set(tokenizer.all_special_ids), {}
    heads = cache_report["selected_heads"]
    for camera, row in data.items():
        times, regions = row["times"], row["regions"]
        video = read_video(ROOT / f"runs/vace_tracking_all_suites_20261004/videos/episode_00040/{camera}.mp4")
        indices, _ = observation_window_indices(len(video), 32, int(times[-1] - times[0] + 1), 16)
        with torch.inference_mode():
            pixels = torch.from_numpy(video[indices].copy()).to(device).permute(0, 3, 1, 2).float()
            pixels = F.interpolate(pixels, size=(512, 512), mode="bilinear", align_corners=False).permute(1, 0, 2, 3) / 127.5 - 1
            control = observation_control(vae, pixels, 16, zeros)
            generator = torch.Generator(device=device).manual_seed(42 + 40 * 37 + CAMERAS.index(camera))
            noise = torch.randn((1, 16, len(times), 64, 64), device=device, generator=generator).bfloat16()
            for pi, pair in enumerate(PAIRS[1:], 1):
                scores = []
                for prompt in pair:
                    embedding, ids = texts[prompt]
                    positions = [i for i, token in enumerate(ids) if token not in special_ids]
                    print(json.dumps({"stage": "frozen_paraphrase", "camera": camera, "pair": pi, "prompt": prompt}), flush=True)
                    group, pixel = capture_instruction(tower, regions, positions, noise, control, embedding.to(device))
                    scores.append({method: np.stack([values[l, h] for l, h in heads]).mean(0).reshape(len(times), 32, 32)
                                   for method, values in [("group", group), ("pixel", pixel)]})
                row["contrasts"].append({method: scores[1][method] - scores[0][method] for method in ("group", "pixel")})
            del pixels, control, noise
            gc.collect()
            torch.cuda.empty_cache()
    del tower, vae
    gc.collect()
    torch.cuda.empty_cache()


def signed_overlay(rgb, score, scale):
    values = cv2.resize(score, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    colors = np.where((values >= 0)[..., None], np.array([238, 65, 53]), np.array([37, 99, 235]))
    alpha = (.75 * np.clip(np.abs(values) / scale, 0, 1))[..., None]
    return np.round(rgb * (1 - alpha) + colors * alpha).astype(np.uint8)


def references(rgb, masks, ti):
    image = Image.fromarray(rgb)
    draw, font = ImageDraw.Draw(image), ImageFont.truetype(FONT, 12)
    for label, color in [(SOUP, (37, 99, 235)), (BUTTER, (238, 65, 53))]:
        y, x = np.where(masks[label][ti])
        if not len(x):
            continue
        bounds = (int(x.min() * 8), int(y.min() * 8), int((x.max() + 1) * 8 - 1), int((y.max() + 1) * 8 - 1))
        draw.rectangle(bounds, outline=color, width=2)
    draw.rectangle((0, 237, 255, 255), fill="white")
    draw.text((5, 239), "Soup GT", fill=(37, 99, 235), font=font)
    draw.text((132, 239), "Butter GT", fill=(238, 65, 53), font=font)
    return np.asarray(image)


def render(rgb, masks, ti, regions, pixel, group, scale, title, method):
    size, gap = 256, 12
    canvas = Image.new("RGB", (1084, 406), "#f5f6f8")
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 13)
    draw.text((gap, 8), title, fill="black", font=font)
    draw.text((gap, 35), method + " | score = butter instruction - soup instruction", fill="black", font=small)
    draw.text((gap, 57), "RED: butter instruction scores higher. BLUE: soup instruction scores higher. Color is not an object mask.", fill="black", font=small)
    panels = [references(rgb, masks, ti), cv2.resize(region_image(regions), (256, 256), interpolation=cv2.INTER_NEAREST),
              signed_overlay(rgb, pixel, scale), signed_overlay(rgb, group, scale)]
    captions = ["RGB + GT sample bounds", "Same empty-prompt K groups", "Pixel contrast", "Group contrast"]
    for col, (panel, caption) in enumerate(zip(panels, captions)):
        x = gap + col * (size + gap)
        draw.text((x, 91), caption, fill="black", font=small)
        canvas.paste(Image.fromarray(panel), (x, 117))
    draw.text((gap, 384), f"One fixed signed scale across views/frames/readouts: +/-{scale:.4f}. GT only draws reference boxes and evaluates scores.", fill="black", font=small)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/vace_instruction_K_regions_20261006")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-paraphrases", action="store_true")
    parser.add_argument("--paraphrase-cache", type=Path,
                        help="Reuse saved paraphrase contrasts without semantic inference")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.paraphrase_cache and not args.with_paraphrases:
        parser.error("--paraphrase-cache requires --with-paraphrases")
    args.output.mkdir(parents=True, exist_ok=False)
    parent = json.loads((args.cache / "report.json").read_text())
    data = {}
    for camera in CAMERAS:
        regions, times, contrast = load_pair(args.cache, camera)
        data[camera] = {"regions": regions, "times": times, "contrasts": [contrast]}
    if args.with_paraphrases:
        if args.paraphrase_cache:
            for camera, row in data.items():
                for pi in range(1, len(PAIRS)):
                    with np.load(args.paraphrase_cache / camera / f"pair_{pi:02d}_contrast.npz") as z:
                        if not np.array_equal(z["times"], row["times"]) or not np.array_equal(z["regions"], row["regions"]):
                            raise RuntimeError("Paraphrase cache must use identical K groups and observation times")
                        row["contrasts"].append({method: z[method].astype(np.float32) for method in ("group", "pixel")})
        else:
            collect_paraphrases(data, parent, args.device)
    scale = max(float(np.percentile(np.abs(np.concatenate([value[method].ravel() for row in data.values()
                                                           for value in row["contrasts"] for method in ("group", "pixel")])), 98)), 1e-5)
    report = {"cache": str(args.cache.resolve()), "training": False, "new_K_extraction": False,
              "new_semantic_inference": args.with_paraphrases and args.paraphrase_cache is None,
              "paraphrase_cache": str(args.paraphrase_cache.resolve()) if args.paraphrase_cache else None,
              "fixed_heads": parent["selected_heads"],
              "pairs": PAIRS if args.with_paraphrases else PAIRS[:1],
              "readout": "butter instruction minus soup instruction; cached neutral cancels exactly",
              "fixed_display_scale": scale, "GT_used_to_generate_heatmaps": False,
              "GT_reference_boxes": "bounds of visible annotated surface cells, not full object boxes",
              "scope": "one held-out episode, two cameras, same K groups and diffusion noise", "cases": {}}
    overview = []
    for camera, row in data.items():
        directory = args.output / camera
        directory.mkdir()
        times, regions = row["times"], row["regions"]
        video = read_video(ROOT / f"runs/vace_tracking_all_suites_20261004/videos/episode_00040/{camera}.mp4")
        masks = evaluation_masks(ROOT / f"runs/vace_tracking_all_suites_20261004/tracks/episode_00040_{camera}_visible_surface.npz", times, 32, 256)
        outputs = {f"pair_{i:02d}": value for i, value in enumerate(row["contrasts"])}
        if args.with_paraphrases:
            outputs["ensemble"] = {method: np.stack([value[method] for value in row["contrasts"]]).mean(0)
                                   for method in ("group", "pixel")}
        report["cases"][camera] = {}
        for name, value in outputs.items():
            if not np.isfinite(value["group"]).all():
                raise RuntimeError("Non-finite instruction contrast")
            for ti, partition in enumerate(regions):
                for label in np.unique(partition):
                    if np.ptp(value["group"][ti].ravel()[partition == label]) > 1e-6:
                        raise RuntimeError("Group readout must remain constant within each K region")
            metric = {method: contrast_metrics(value[method], masks) for method in ("group", "pixel")}
            report["cases"][camera][name] = metric
            np.savez_compressed(directory / f"{name}_contrast.npz", **value, times=times, regions=regions)
            frames = [render(video[frame], masks, ti, regions[ti], value["pixel"][ti], value["group"][ti], scale,
                             f"episode00040 {camera} | observed frame {frame}", name)
                      for ti, frame in enumerate(times)]
            for ti in [0, int(np.argmin(np.abs(times - 44))), len(times) - 1]:
                frames[ti].save(directory / f"{name}_frame_{times[ti]:03d}.jpg", quality=95)
            write_video(directory / f"{name}_contrast.mp4", np.stack([np.asarray(frame) for frame in frames]), 5)
            if name in ("pair_00", "ensemble"):
                overview.append(frames[int(np.argmin(np.abs(times - 44)))])
            print(json.dumps({"camera": camera, "readout": name, "group": metric["group"], "pixel_pair_auc": metric["pixel"]["butter_vs_soup_auc"]}), flush=True)
        if args.with_paraphrases:
            group_values = np.stack([value["group"] for value in row["contrasts"]])
            sign_consistency = np.maximum((group_values > 0).mean(0), (group_values < 0).mean(0))
            np.savez_compressed(directory / "paraphrase_consistency.npz", agreement=sign_consistency, times=times)
            report["cases"][camera]["paraphrase_consistency"] = {
                "soup_all_same_sign": float((sign_consistency[masks[SOUP]] == 1).mean()),
                "butter_all_same_sign": float((sign_consistency[masks[BUTTER]] == 1).mean())}
    sheet = Image.new("RGB", (overview[0].width, sum(frame.height for frame in overview)), "white")
    for row, frame in enumerate(overview):
        sheet.paste(frame, (0, row * frame.height))
    sheet.save(args.output / "same_frame_contrasts.jpg", quality=95)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"complete": True, "output": str(args.output.resolve()), "paraphrases": args.with_paraphrases}), flush=True)


if __name__ == "__main__":
    main()
