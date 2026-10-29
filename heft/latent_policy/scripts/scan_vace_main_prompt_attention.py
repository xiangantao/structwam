#!/usr/bin/env python3
"""Scan main-branch prompt cross-attention on an observed-only VACE window."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(ROOT / "heft/latent_policy/scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from benchmark_vace_observed33 import observed_inputs  # noqa: E402
from evaluate_vace_pure_noise_future import load_transformer  # noqa: E402
from search_vace_tracking_heads import read_video, load_vae  # noqa: E402


def load_text(model_root: Path, prompt: str, device: torch.device):
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(
        model_root / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()
    tokens = tokenizer([prompt], padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        embedding = encoder(**tokens).last_hidden_state
        embedding[:, int(tokens.attention_mask[0].sum()):] = 0
    ids = tokens.input_ids[0].detach().cpu().tolist()
    mask = tokens.attention_mask[0].detach().cpu().tolist()
    del encoder, tokenizer, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return embedding.detach(), ids, mask


def phrase_token_ids(model_root: Path, phrase: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    return tokenizer(phrase, add_special_tokens=False)["input_ids"]


def find_phrase_positions(input_ids, phrase_ids):
    for start in range(len(input_ids) - len(phrase_ids) + 1):
        if input_ids[start:start + len(phrase_ids)] == phrase_ids:
            return list(range(start, start + len(phrase_ids)))
    raise ValueError(f"Phrase token sequence not found: {phrase_ids}")


def read_window(path: Path, start: int, count: int):
    with av.open(str(path)) as container:
        frames = []
        for index, frame in enumerate(container.decode(video=0)):
            if index >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == count:
                break
    if len(frames) != count:
        raise ValueError(f"Expected {count} frames from {path}, got {len(frames)}")
    return np.stack(frames)


def dense_masks(tracks_path: Path, times: np.ndarray, labels, grid: int):
    with np.load(tracks_path, allow_pickle=True) as source:
        xy = source["xy"][times]
        visible = source["visible"][times]
        point_labels = source["point_label"].astype(str)
    masks = {}
    for label in labels:
        mask = np.zeros((len(times), grid, grid), dtype=bool)
        ids = np.flatnonzero(point_labels == label)
        if len(ids):
            cells = np.floor((np.nan_to_num(xy[:, ids]) + 0.5) * grid / 256.0).astype(np.int64)
            cells = np.clip(cells, 0, grid - 1)
            for ti in range(len(times)):
                valid = visible[ti, ids] & np.isfinite(xy[ti, ids]).all(-1)
                mask[ti, cells[ti, valid, 1], cells[ti, valid, 0]] = True
        masks[label] = mask
    return masks


def capture_main_attn2(model, phrase_positions, noise, control, text, timestep, scales):
    captures = {}

    def make_callback(layer):
        def callback(query, key, **_):
            q, k = query[0].float(), key[0].float()
            selected = []
            for start in range(0, q.shape[1], 2048):
                logits = torch.einsum(
                    "hnd,htd->hnt", q[:, start:start + 2048], k
                ) / math.sqrt(q.shape[-1])
                probability = torch.softmax(logits, dim=-1)
                selected.append(torch.stack(
                    [probability[..., positions].sum(-1) for positions in phrase_positions.values()], dim=1
                ).cpu())
            captures[layer] = torch.cat(selected, dim=-1)

        return callback

    for layer, block in enumerate(model.blocks):
        block.attn2._heft_capture = make_callback(layer)
    try:
        with torch.inference_mode():
            model(
                hidden_states=noise,
                timestep=timestep,
                encoder_hidden_states=text,
                control_hidden_states=control,
                control_hidden_states_scale=scales,
                return_dict=False,
            )
    finally:
        for block in model.blocks:
            if hasattr(block.attn2, "_heft_capture"):
                del block.attn2._heft_capture
    if set(captures) != set(range(len(model.blocks))):
        raise RuntimeError(f"Missing main attn2 captures: {sorted(set(range(len(model.blocks))) - set(captures))}")
    return torch.stack([captures[layer] for layer in range(len(model.blocks))])


def auc_pairwise(target, distractor):
    if not len(target) or not len(distractor):
        return None
    comparison = target[:, None] - distractor[None, :]
    return float((comparison > 0).mean() + 0.5 * (comparison == 0).mean())


def score_heatmaps(raw, masks, phrases):
    # raw: [layers, heads, phrases, time, grid, grid]
    scores = []
    for layer in range(raw.shape[0]):
        for head in range(raw.shape[1]):
            for phrase_index, (phrase, target_label) in enumerate(phrases.items()):
                target = raw[layer, head, phrase_index][masks[target_label]]
                distractors = [raw[layer, head, phrase_index][mask]
                               for label, mask in masks.items() if label != target_label and mask.any()]
                distractor = np.concatenate(distractors) if distractors else np.empty(0, dtype=np.float32)
                auc = auc_pairwise(target, distractor)
                scores.append({
                    "layer": layer,
                    "head": head,
                    "phrase": phrase,
                    "target_label": target_label,
                    "target_mean": float(target.mean()) if len(target) else 0.0,
                    "distractor_mean": float(distractor.mean()) if len(distractor) else 0.0,
                    "target_margin": float(target.mean() - distractor.mean()) if len(target) and len(distractor) else None,
                    "auc": auc,
                    "target_cells": int(len(target)),
                    "distractor_cells": int(len(distractor)),
                })
    return scores


def render_top(video, raw, rows, masks, phrases, output, grid):
    selected = []
    seen = set()
    for row in sorted(rows, key=lambda x: (-1 if x["auc"] is None else -x["auc"])):
        key = (row["phrase"], row["layer"], row["head"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len([x for x in selected if x["phrase"] == row["phrase"]]) >= 3 and len(seen) >= 9:
            break
    frames = [0, raw.shape[3] // 2, raw.shape[3] - 1]
    tiles = []
    for row in selected:
        phrase_index = list(phrases).index(row["phrase"])
        heat = raw[row["layer"], row["head"], phrase_index]
        for time_index in frames:
            image = cv2.resize(video[min(time_index * 4, len(video) - 1)], (512, 512), interpolation=cv2.INTER_AREA)
            values = heat[time_index]
            values = (values - values.min()) / max(float(values.max() - values.min()), 1e-8)
            values = cv2.resize(values.astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)
            colored = cv2.applyColorMap(np.uint8(values * 255), cv2.COLORMAP_JET)[:, :, ::-1]
            blend = np.uint8(0.58 * image + 0.42 * colored)
            cv2.putText(blend, f"{row['phrase']} L{row['layer']}H{row['head']} AUC={row['auc']:.3f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(blend)
    if tiles:
        rows_out = []
        for start in range(0, len(tiles), 3):
            chunk = tiles[start:start + 3]
            while len(chunk) < 3:
                chunk.append(np.zeros_like(tiles[0]))
            rows_out.append(np.concatenate(chunk, axis=1))
        cv2.imwrite(str(output), cv2.cvtColor(np.concatenate(rows_out, axis=0), cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=39)
    parser.add_argument("--camera", choices=("agentview", "robot0_eye_in_hand"), default="agentview")
    parser.add_argument("--start", type=int, default=48)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", default="pick up the black bowl on the wooden cabinet and place it on the plate")
    parser.add_argument("--phrases", nargs="+", default=("black bowl", "plate", "wooden cabinet"))
    parser.add_argument("--labels", nargs="+", default=("akita_black_bowl_1", "plate_1", "wooden_cabinet_1"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--timestep", type=int, default=1000)
    args = parser.parse_args()
    if len(args.phrases) != len(args.labels):
        parser.error("--phrases and --labels must have equal length")
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    video_path = args.data / "videos" / f"episode_{args.episode:05d}" / f"{args.camera}.mp4"
    video = read_window(video_path, args.start, args.frames)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    control, encoded = observed_inputs(vae, video, argparse.Namespace(size=args.size), device)
    noise = torch.randn((1, 16, encoded.shape[1], args.size // 8, args.size // 8),
                        device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(42))
    del vae, encoded
    torch.cuda.empty_cache()
    text, input_ids, attention_mask = load_text(args.text_model, args.prompt, device)
    tokenizer_ids = {phrase: phrase_token_ids(args.text_model, phrase) for phrase in args.phrases}
    positions = {phrase: find_phrase_positions(input_ids, ids) for phrase, ids in tokenizer_ids.items()}
    model = load_transformer(args.model, device)
    scales = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
    raw = capture_main_attn2(model, positions, noise, control, text,
                             torch.tensor([args.timestep], device=device), scales).numpy()
    times = np.arange(args.start, args.start + args.frames, 4)
    tracks = args.data / "tracks" / f"episode_{args.episode:05d}_{args.camera}_visible_surface.npz"
    masks = dense_masks(tracks, times, args.labels, args.size // 16)
    # [layers, heads, phrases, time, grid, grid]
    raw = raw.reshape(raw.shape[0], raw.shape[1], len(args.phrases), len(times), args.size // 16, args.size // 16)
    phrase_map = dict(zip(args.phrases, args.labels))
    rows = score_heatmaps(raw, masks, phrase_map)
    payload = {
        "protocol": {
            "branch": "main.attn2",
            "input": "33 recorded RGB frames to VACE control; main hidden_states are pure Gaussian noise",
            "future_gt": False,
            "episode": args.episode,
            "camera": args.camera,
            "start": args.start,
            "frames": args.frames,
            "timestep": args.timestep,
            "prompt": args.prompt,
            "phrases": positions,
        },
        "top_by_phrase": {
            phrase: sorted([row for row in rows if row["phrase"] == phrase],
                           key=lambda row: -1 if row["auc"] is None else -row["auc"])[:12]
            for phrase in args.phrases
        },
        "all_scores": rows,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    np.savez_compressed(args.output / "main_attn2_heatmaps.npz", heatmaps=raw, times=times)
    render_top(video, raw, rows, masks, phrase_map, args.output / "top_heatmaps.jpg", args.size // 16)
    print(json.dumps({"output": str(args.output), "layers": raw.shape[0], "heads": raw.shape[1],
                      "phrases": positions, "top": payload["top_by_phrase"]}, indent=2))


if __name__ == "__main__":
    main()
