#!/usr/bin/env python3
"""Policy features for every frame of every LIBERO episode (one process per NPU).

For frame ``t`` of an episode the frozen student sees the 33 frames ``t-32 .. t`` (frame 0
repeated before the episode start) and the instruction; ``policy_features.window_features``
turns its fields into K blocks and point tracks.  One ``.npz`` per episode and camera
holds the per-frame arrays; ``finalize_policy_features.py`` assembles them.  Reruns skip
finished files.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WINDOW = 33
KEYS = ("seg_k_l22h9", "seg_k_l15h7", "track_k", "text_attention", "depth")
FIELDS = ("valid", "centre", "size", "k", "depth", "relevance", "trajectory", "confidence", "tracks", "owner")


def log(**values):
    print(json.dumps({"time": round(time.time(), 1), **values}), flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="episodes.jsonl (one row per episode and camera)")
    parser.add_argument("--words", type=Path, required=True, help="select_policy_words.py output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--episodes", default="", help="comma list to restrict (testing)")
    parser.add_argument("--max-frames", type=int, default=0, help="first N frames per episode (testing)")
    parser.add_argument("--block-workers", type=int, default=16, help="CPU processes for K blocks")
    args = parser.parse_args()
    from vace_latent_distill.config import ModelConfig
    from vace_latent_distill.data import decode_indices, resize_rgb
    from vace_latent_distill.model import StructuredStudent
    from multiprocessing import get_context

    from vace_latent_distill import fast_rope
    from vace_latent_distill.policy_features import block_inputs, blocks_job, window_features

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=True)
    words = json.loads(args.words.read_text())
    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    if args.episodes:
        wanted = {int(e) for e in args.episodes.split(",")}
        rows = [r for r in rows if r["episode"] in wanted]
    rows = [r for r in rows if r["episode"] % args.world == args.rank]
    todo = [r for r in rows if not (args.output / f"{r['id']}.npz").is_file()]
    log(stage="start", rank=args.rank, rows=len(rows), todo=len(todo))
    saved = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False, mmap=True)
    model = StructuredStudent(ModelConfig(**saved["config"]["model"]), pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    text_cache = torch.load(saved["config"]["train"]["text_cache"], map_location="cpu", weights_only=True)
    del saved
    model = model.eval().to(device)
    fast_rope.install(model.backbone)  # bit-identical cached RoPE (the original rebuilds it every call)
    pool = get_context("spawn").Pool(args.block_workers)
    for row in todo:
        tick = time.monotonic()
        length = row["length"] if not args.max_frames else min(row["length"], args.max_frames)
        frames = decode_indices(row["video"], range(row["episode_start_frame"], row["episode_start_frame"] + length),
                                row["fps"])
        frames = resize_rgb(torch.from_numpy(frames), 256)                           # [L, 3, 256, 256] uint8
        text = text_cache[row["task"]].float()[None].to(device)
        mask = text.abs().sum(-1) != 0
        tokens = words[row["task"]]["words"]
        wrist = row["camera"] == "observation.images.image2"
        outputs = {key: [] for key in FIELDS}
        for begin in range(0, length, args.batch_size):
            ends = range(begin, min(begin + args.batch_size, length))
            index = torch.stack([torch.arange(t - WINDOW + 1, t + 1).clamp(min=0) for t in ends])
            rgb = frames[index].to(device)                                            # [B, 33, 3, H, W]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # The observed branch is only read by the point tracks (track K).
                output = model(rgb, text.expand(len(index), -1, -1), mask.expand(len(index), -1),
                               observed_heads=("tracking",))
            windows = [{branch: {k: output[branch][k][b] for k in KEYS if k in output[branch]}
                        for branch in ("observed", "future")} for b in range(len(index))]
            jobs = [(*block_inputs(w["future"], tokens), wrist) for w in windows]
            for fields, blocks in zip(windows, pool.map(blocks_job, jobs)):
                result = window_features(fields["observed"], fields["future"], tokens, wrist, blocks)
                for key in FIELDS:
                    outputs[key].append(result[key])
        arrays = {key: np.stack(values) for key, values in outputs.items()}
        np.savez(args.output / f"{row['id']}.tmp.npz", **arrays, episode=row["episode"], camera=row["camera"],
                 task=row["task"], words=json.dumps(tokens))
        os.replace(args.output / f"{row['id']}.tmp.npz", args.output / f"{row['id']}.npz")
        seconds = time.monotonic() - tick
        log(stage="row", rank=args.rank, id=row["id"], frames=length, seconds=round(seconds, 1),
            fps=round(length / seconds, 1), blocks=float(arrays["valid"].sum(1).mean()))
    pool.close()
    log(stage="done", rank=args.rank)


if __name__ == "__main__":
    main()
