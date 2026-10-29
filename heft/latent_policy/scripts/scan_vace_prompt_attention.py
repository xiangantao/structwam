#!/usr/bin/env python3
"""Scan prompt-conditioned VACE cross-attention against LIBERO object masks."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(ROOT / "heft/src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from search_vace_tracking_heads import (  # noqa: E402
    CAMERAS,
    ControlTower,
    load_vae,
    read_video,
)
from heft_postprocess_vace_tracking import prepare_case  # noqa: E402



def load_prompt(model_root: Path, prompt: str, device: torch.device):
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(
        model_root / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()
    tokens = tokenizer([prompt], padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = encoder(**tokens).last_hidden_state
    embedding[:, int(tokens.attention_mask.sum()):] = 0
    return embedding.detach(), tokenizer, tokens.input_ids[0].detach().cpu().tolist(), tokens.attention_mask[0].detach().cpu().tolist()


def phrase_token_ids(tokenizer, phrase: str):
    result = tokenizer(phrase, add_special_tokens=False)["input_ids"]
    return result


def find_phrase_positions(input_ids, phrase_ids):
    for start in range(len(input_ids) - len(phrase_ids) + 1):
        if input_ids[start : start + len(phrase_ids)] == phrase_ids:
            return list(range(start, start + len(phrase_ids)))
    raise ValueError(f"phrase token sequence not found: {phrase_ids}")


def attention_capture(tower, selected_positions, device):
    captures = {}
    selected_positions = torch.as_tensor(selected_positions, device=device, dtype=torch.long)

    def make_callback(layer):
        def callback(query, key, **_):
            # query: [1, heads, video_tokens, dim], key: [1, heads, text_tokens, dim]
            q = query[0].float()
            k = key[0].float()
            values = []
            for start in range(0, q.shape[1], 2048):
                logits = torch.einsum("hnd,htd->hnt", q[:, start : start + 2048], k) / math.sqrt(q.shape[-1])
                values.append(torch.softmax(logits, dim=-1)[..., selected_positions].cpu())
            captures[layer] = torch.cat(values, dim=1)

        return callback

    for layer, block in enumerate(tower.vace_blocks):
        block.attn2._heft_capture = make_callback(layer)
    return captures


def dense_object_masks(tracks_root: Path, episode: int, camera: str, times: np.ndarray, grid: int, labels):
    path = tracks_root / f"episode_{episode:05d}_{camera}_visible_surface.npz"
    with np.load(path, allow_pickle=True) as z:
        xy = z["xy"][times]
        visible = z["visible"][times]
        point_labels = z["point_label"].astype(str)
    masks = {}
    for label in labels:
        mask = np.zeros((len(times), grid, grid), dtype=bool)
        ids = np.flatnonzero(point_labels == label)
        if len(ids):
            points = np.round((xy[:, ids] + 0.5) * grid / 256.0).astype(int)
            points = np.clip(points, 0, grid - 1)
            for ti in range(len(times)):
                valid = visible[ti, ids]
                mask[ti, points[ti, valid, 1], points[ti, valid, 0]] = True
        masks[label] = mask
    return masks


def score_maps(attention, masks, target_label):
    rows = []
    # attention: [heads, video_tokens], flattened time-major spatial grid.
    heads, tokens = attention.shape
    times, grid = next(iter(masks.values())).shape[0], next(iter(masks.values())).shape[-1]
    if tokens != times * grid * grid:
        raise ValueError(f"attention token shape {tokens} does not match {times}x{grid}x{grid}")
    maps = attention.reshape(heads, times, grid, grid)
    labels = list(masks)
    for head in range(heads):
        per_label = {}
        for label, mask in masks.items():
            values = maps[head][mask]
            all_values = maps[head].reshape(times, -1)
            per_label[label] = {
                "mass": float(values.mean()) if len(values) else 0.0,
                "peak": float(values.max()) if len(values) else 0.0,
                "support": float((values > 0.01).float().mean()) if len(values) else 0.0,
            }
        target = target_label
        distractors = [x for x in labels[1:] if x != target]
        target_mass = per_label[target]["mass"]
        distractor_mass = float(np.mean([per_label[x]["mass"] for x in distractors])) if distractors else 0.0
        rows.append({
            "head": head,
            "target": target,
            "target_mass": target_mass,
            "distractor_mass": distractor_mass,
            "target_margin": target_mass - distractor_mass,
            "labels": per_label,
        })
    return rows, maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=39)
    parser.add_argument("--camera", choices=CAMERAS, default="agentview")
    parser.add_argument("--video-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/videos")
    parser.add_argument("--tracks-root", type=Path, default=ROOT / "runs/vace_tracking_all_suites_20261004/tracks")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--t2v-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", default="pick up the black bowl on the wooden cabinet and place it on the plate")
    parser.add_argument("--phrases", nargs="+", default=("black bowl", "plate", "wooden cabinet"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--skip-frames", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=129)
    parser.add_argument("--points-per-subject", type=int, default=24)
    parser.add_argument("--min-motion", type=float, default=0.02)
    parser.add_argument("--timestep", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    text, tokenizer, input_ids, attention_mask = load_prompt(args.t2v_model, args.prompt, device)
    phrase_positions = {phrase: find_phrase_positions(input_ids, phrase_token_ids(tokenizer, phrase)) for phrase in args.phrases}
    # Run the model once per prompt; capture every attn2 layer. The callback is
    # installed directly on attn2 processors, so existing self-attention code is unchanged.
    name, video, times, points, control, noise = prepare_case(
        args.video_root, args.tracks_root, args.model, args.episode, args.camera,
        args.size, args.skip_frames, args.max_frames, args.points_per_subject,
        args.min_motion, device, 42,
    )
    grid = args.size // 16
    masks = dense_object_masks(args.tracks_root, args.episode, args.camera, times, grid, [
        "akita_black_bowl_1", "plate_1", "wooden_cabinet_1"
    ])
    selected_positions = sorted({position for positions in phrase_positions.values() for position in positions})
    tower = ControlTower.load(args.model / "diffusion_pytorch_model.safetensors", device)
    callback_capture = attention_capture(tower, selected_positions, device)
    with torch.inference_mode():
        tower(noise, control, text, torch.tensor([args.timestep], device=device), lambda *_: None)
    records = {}
    for layer, raw in sorted(callback_capture.items()):
        for phrase, positions in phrase_positions.items():
            local_positions = [selected_positions.index(position) for position in positions]
            maps = raw[..., local_positions].sum(-1)
            target_label = {"black bowl": "akita_black_bowl_1", "plate": "plate_1", "wooden cabinet": "wooden_cabinet_1"}[phrase]
            rows, heatmaps = score_maps(maps, masks, target_label)
            records[f"layer_{layer:02d}/{phrase}"] = {"token_positions": positions, "scores": rows}
            np.save(args.output / f"layer_{layer:02d}_{phrase.replace(' ', '_')}.npy", heatmaps.numpy())
    (args.output / "results.json").write_text(json.dumps({"episode": args.episode, "camera": args.camera, "prompt": args.prompt, "phrases": phrase_positions, "results": records}, indent=2))
    print(json.dumps({"output": str(args.output), "episode": args.episode, "layers": len(callback_capture), "phrases": phrase_positions}, indent=2))


if __name__ == "__main__":
    main()
