#!/usr/bin/env python3
"""Extract VACE teacher targets for a fixed window list into a local cache.

Small-scale debugging cache only (full training keeps the online teacher).  Run one
process per NPU with ``--rank r --world n``; existing targets are skipped, so a rerun
resumes.  ``--merge`` writes ``manifest.jsonl`` with local ``target_path`` plus the
sha256/bytes/provenance fields that ``OfflineStructuredDataset`` verifies.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from vace_latent_distill.config import NEGATIVE_PROMPT, PipelineConfig
from vace_latent_distill.data import decode_indices
from vace_latent_distill.offline import pack_target, provenance
from vace_latent_distill.offline_store import file_sha256

METADATA = ("sample_id", "teacher_seed", "frame_indices", "task", "camera", "split", "start",
            "episode", "id")


def log(**values):
    print(json.dumps({"time": round(time.time(), 1), **values}), flush=True)


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def merge(args, rows):
    identity = json.loads((args.output / "provenance.json").read_text())["id"]
    records = []
    for row in rows:
        path = args.output / "targets" / f"{row['sample_id']}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({**row, "target_path": str(path.resolve()), "target_sha256": file_sha256(path),
                        "target_bytes": path.stat().st_size, "provenance_id": identity})
    for split in ("train", "heldout"):
        chosen = [r for r in records if r["split"] == split]
        (args.output / f"manifest_{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chosen))
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    log(stage="merged", windows=len(records), gib=round(sum(r["target_bytes"] for r in records) / 2**30, 2))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    rows = load_rows(args.windows)
    config = PipelineConfig.load(args.config)
    (args.output / "targets").mkdir(parents=True, exist_ok=True)
    if args.merge:
        return merge(args, rows)
    identity = provenance(config, args.windows, config.train.text_cache)
    if args.rank == 0:
        (args.output / "provenance.json").write_text(json.dumps(identity, indent=2) + "\n")
        (args.output / "pipeline.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    todo = [row for row in rows[args.rank::args.world]
            if not (args.output / "targets" / f"{row['sample_id']}.pt").is_file()]
    log(stage="start", rank=args.rank, windows=len(rows[args.rank::args.world]), todo=len(todo))
    if not todo:
        return
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)  # one process per NPU; default device must match
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    from vace_latent_distill.teacher import VaceTeacher, content_token_masks

    content = content_token_masks(config.teacher.text_root, sorted({row["task"] for row in rows}))
    teacher = VaceTeacher(config.teacher, "shared", args.device, text_cache[NEGATIVE_PROMPT])
    try:
        for begin in range(0, len(todo), args.batch_size):
            batch = todo[begin:begin + args.batch_size]
            tick = time.monotonic()
            videos = [torch.from_numpy(decode_indices(r["video"], r["frame_indices"], r["fps"])) for r in batch]
            texts = [text_cache[r["task"]] for r in batch]
            targets = teacher.extract_batch(videos, texts, [t.abs().sum(-1) != 0 for t in texts],
                                            [r["teacher_seed"] for r in batch],
                                            [content[r["task"]] for r in batch])
            for row, target in zip(batch, targets, strict=True):
                target = {b: {k: v.cpu() for k, v in target[b].items()} for b in ("observed", "future")}
                payload = pack_target(target, {k: row[k] for k in METADATA}, teacher.protocol, identity["id"])
                path = args.output / "targets" / f"{row['sample_id']}.pt"
                temporary = path.with_suffix(".tmp")
                torch.save(payload, temporary)
                temporary.replace(path)
            log(stage="batch", rank=args.rank, done=begin + len(batch), todo=len(todo),
                seconds=round(time.monotonic() - tick, 1))
    finally:
        teacher.close()


if __name__ == "__main__":
    main()
