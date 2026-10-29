#!/usr/bin/env python3
"""Cache frozen VACE-compatible UMT5-XXL instruction tokens locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vace_latent_distill.config import NEGATIVE_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.max_length != 512:
        parser.error("The VACE teacher protocol uses exactly512 padded text positions")
    from transformers import AutoTokenizer, UMT5EncoderModel

    prompts = sorted(
        {
            json.loads(line)["task"]
            for line in args.manifest.read_text().splitlines()
            if line.strip()
        }
        | {NEGATIVE_PROMPT}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.text_model / "tokenizer", local_files_only=True)
    model = (
        UMT5EncoderModel.from_pretrained(
            args.text_model / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    result = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), 4):
            group = prompts[start : start + 4]
            tokens = tokenizer(
                group,
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(model.device)
            values = model(**tokens).last_hidden_state.cpu()
            for index, prompt in enumerate(group):
                length = int(tokens.attention_mask[index].sum())
                values[index, length:] = 0
                result[prompt] = values[index].float().contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(
        json.dumps(
            {
                "prompts": len(result),
                "tokens": args.max_length,
                "width": 4096,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
