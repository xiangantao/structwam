#!/usr/bin/env python3
"""Precompute the Wan2.2 UMT5 text context for every LIBERO prompt on one NPU, so training can
leave the frozen 5.7B text encoder out (model.video_backbone.text_cache_path=<out>).

    python scripts/structwam/npu_entry.py scripts/structwam/build_wan_text_cache.py \
        --model-path <Wan2.2-TI2V-5B> --dataset-dir <libero> --out <cache.pt>

Prompts are the meta/tasks.parquet strings, stripped, exactly as the LIBERO reader hands them to
training (LeRobotV3Reader._load_prompts / _resolve_prompt).  Each batch goes through the training
path's own ``wan.encode.encode_text``: the same tokenizer and 512-token padding, the UMT5 weights
loaded by the same loader in bf16, in eval mode under no_grad, as in ``preprocess_input_for_train``.
Only the first seq_len rows of each context are stored; ``cached_text_context`` zero-fills the
rest, as ``encode_text`` does.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch

from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.loader import load_wan_components
from openwam.model.video_backbone.wan.pipeline_builder import discover_model_files, is_text_encoder_config


def libero_prompts(dataset_dir: Path) -> list:
    tasks = pd.read_parquet(dataset_dir / "meta" / "tasks.parquet")
    prompts = sorted({str(t).strip() for t in tasks.index})
    if not prompts or any(not p for p in prompts):
        raise ValueError(f"empty prompt in {dataset_dir}/meta/tasks.parquet")
    return prompts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=8, help="the training micro-batch size")
    args = p.parse_args()

    prompts = libero_prompts(args.dataset_dir)
    model_configs, tokenizer_config = discover_model_files(str(args.model_path))
    te_configs = [c for c in model_configs if is_text_encoder_config(c)]
    if len(te_configs) != 1:
        raise FileNotFoundError(f"expected one UMT5 weight file in {args.model_path}, found {len(te_configs)}")
    device = torch.device("cuda")  # npu via npu_entry.py's transfer_to_npu
    holder = load_wan_components(te_configs, tokenizer_config, device="cuda", torch_dtype=torch.bfloat16)
    text_encoder = holder.text_encoder
    if text_encoder is None or holder.tokenizer is None:
        raise RuntimeError("UMT5 text encoder or tokenizer did not load")
    text_encoder.to(dtype=torch.bfloat16, device=device)
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    embeds = {}
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i : i + args.batch_size]
            context, seq_lens = wan_encode.encode_text(
                chunk, tokenizer=holder.tokenizer, text_encoder=text_encoder, device=device
            )
            for prompt, ctx, n in zip(chunk, context, seq_lens.tolist()):
                if not 0 < n < wan_encode.TEXT_MAX_LENGTH:
                    raise ValueError(f"{n} tokens for {prompt!r}")
                embeds[prompt] = ctx[:n].to("cpu").clone()

    dims = {tuple(e.shape[1:]) for e in embeds.values()}
    dtypes = {e.dtype for e in embeds.values()}
    payload = {
        "embeds": embeds,
        "max_length": wan_encode.TEXT_MAX_LENGTH,
        "model_file": str(te_configs[0].path),
        "tokenizer": str(tokenizer_config.path),
        "dataset_dir": str(args.dataset_dir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, args.out)
    lens = [e.shape[0] for e in embeds.values()]
    print(
        f"[build_wan_text_cache] {len(embeds)} prompts, tokens {min(lens)}-{max(lens)}, dim {dims}, "
        f"dtype {dtypes} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
