#!/usr/bin/env python3
"""Export observed/future matrices from a student checkpoint and past33 RGB only."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from vace_latent_distill.config import ModelConfig
from vace_latent_distill.data import decode_indices, resize_rgb
from vace_latent_distill.model import StructuredStudent


def encode_instruction(instruction, text_root, device):
    from transformers import AutoTokenizer, UMT5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(text_root / "tokenizer", local_files_only=True)
    encoder = (
        UMT5EncoderModel.from_pretrained(
            text_root / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    tokens = tokenizer(
        instruction, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text = encoder(**tokens).last_hidden_state.float().cpu()
    text[:, int(tokens.attention_mask.sum()) :] = 0
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return text[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-cache", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("schema") != "vace-structured-student-v1":
        raise ValueError("Expected a structured student checkpoint")
    config = ModelConfig(**saved["config"]["model"])
    device = torch.device(args.device)
    cache_path = args.text_cache or Path(saved["config"]["train"]["text_cache"])
    cache = (
        torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache_path.is_file()
        else {}
    )
    text = cache.get(args.instruction)
    if text is None:
        text = encode_instruction(
            args.instruction, Path(saved["config"]["teacher"]["text_root"]), device
        )
    indices = list(range(args.start, args.start + 33))
    rgb = resize_rgb(
        torch.from_numpy(decode_indices(args.video, indices, args.fps)), config.image_size
    )
    model = StructuredStudent(config, pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    del saved, cache
    model.to(device).eval()
    with (
        torch.no_grad(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        output = model(
            rgb[None].to(device), text[None].to(device), (text.abs().sum(-1) != 0)[None].to(device)
        )
    arrays = {
        f"{branch}_{key}": value[0].float().cpu().numpy()
        for branch in ("observed", "future")
        for key, value in output[branch].items()
    }
    arrays.update(
        {
            f"{branch}_{key}": value.cpu().numpy()
            for branch, values in output["timeline"].items()
            for key, value in values.items()
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "model": dataclasses.asdict(config),
                "video": str(args.video.resolve()),
                "past_frame_indices": indices,
                "instruction": args.instruction,
                "fps": args.fps,
                "student_future_rgb": False,
                "future_source": "predicted_from_past_and_instruction",
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "shapes": {key: list(value.shape) for key, value in arrays.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
