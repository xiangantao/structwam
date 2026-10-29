#!/usr/bin/env python3
"""Single-NPU breakdown of one teacher extract_batch call (needs VACE_NPU=1)."""
import collections
import os
import sys
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from vace_latent_distill import teacher as T
from vace_latent_distill.config import PipelineConfig
from vace_latent_distill.data import StructuredDataset, collate_structured

COUNT = int(os.environ.get("COUNT", "4"))
TIMES = collections.defaultdict(float)
CALLS = collections.defaultdict(int)


DETAIL = []


def wrap(obj, name, label):
    fn = getattr(obj, name)

    def inner(*a, **k):
        torch.npu.synchronize()
        t = time.monotonic()
        out = fn(*a, **k)
        torch.npu.synchronize()
        dt = time.monotonic() - t
        TIMES[label] += dt
        CALLS[label] += 1
        if label == "vae_encode":
            DETAIL.append((dt, [tuple(v.shape) for v in a[0]],
                           [float(v.abs().max()) == 0.0 for v in a[0]]))
        return out

    setattr(obj, name, inner)


def main():
    config = PipelineConfig.load(ROOT / "configs/vace_structured_npu.json")
    ds = StructuredDataset(config.train.manifest, config.train.text_cache,
                           image_size=config.model.image_size, seed=0,
                           windows_per_episode_camera=1)
    batch = collate_structured([ds[i] for i in range(COUNT)])
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    from vace_latent_distill.config import NEGATIVE_PROMPT
    neg = ds.text[NEGATIVE_PROMPT]
    teacher = T.VaceTeacher(config.teacher, config.model.task, device, neg)
    content = T.content_token_masks(config.teacher.text_root, sorted({m["task"] for m in batch["metadata"]}))
    wrap(teacher.vae, "encode", "vae_encode")
    wrap(teacher, "_forward", "transformer_forward")
    wrap(T, "multi_anchor_paths", "beam_tracking_cpu")
    wrap(teacher, "_readout", "readout_total")
    args = (batch["teacher_rgb"], batch["text"], batch["text_mask"],
            [r["teacher_seed"] for r in batch["metadata"]],
            [content[r["task"]] for r in batch["metadata"]])
    for run in range(3):
        TIMES.clear(); CALLS.clear(); DETAIL.clear()
        torch.npu.synchronize(); t = time.monotonic()
        teacher.extract_batch(*args)
        torch.npu.synchronize(); total = time.monotonic() - t
        print(f"run {run}: total {total:.1f}s for {COUNT} windows")
        for k in sorted(TIMES):
            print(f"   {k:20s} {TIMES[k]:7.1f}s  calls={CALLS[k]}")
        for dt, shapes, zeros in DETAIL:
            print(f"      vae call {dt:5.2f}s  n={len(shapes)} shape={shapes[0]} all_zero={zeros[0]}")


if __name__ == "__main__":
    main()
