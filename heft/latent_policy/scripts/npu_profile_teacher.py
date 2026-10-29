#!/usr/bin/env python3
"""Capture one steady-state teacher extract_batch with the Ascend profiler.

Writes a trace to $OUT (default /tmp/npu_prof_teacher) and prints the top device
kernels, AICPU kernels, and per-op host launch counts so slow operators can be
found without opening the trace UI.
"""
import os
import sys
from pathlib import Path

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from vace_latent_distill import teacher as T
from vace_latent_distill.config import NEGATIVE_PROMPT, PipelineConfig
from vace_latent_distill.data import StructuredDataset, collate_structured

COUNT = int(os.environ.get("COUNT", "4"))
OUT = os.environ.get("OUT", "/tmp/npu_prof_teacher")


def main():
    config = PipelineConfig.load(ROOT / "configs/vace_structured_npu.json")
    ds = StructuredDataset(config.train.manifest, config.train.text_cache,
                           image_size=config.model.image_size, seed=0,
                           windows_per_episode_camera=1)
    batch = collate_structured([ds[i] for i in range(COUNT)])
    device = torch.device("npu", 0)
    torch.npu.set_device(device)
    teacher = T.VaceTeacher(config.teacher, config.model.task, device, ds.text[NEGATIVE_PROMPT])
    content = T.content_token_masks(config.teacher.text_root,
                                    sorted({m["task"] for m in batch["metadata"]}))
    args = (batch["teacher_rgb"], batch["text"], batch["text_mask"],
            [r["teacher_seed"] for r in batch["metadata"]],
            [content[r["task"]] for r in batch["metadata"]])
    teacher.extract_batch(*args)  # warm-up: op compilation and caches
    torch.npu.synchronize()

    experimental = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        with_stack=False,
        experimental_config=experimental,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(OUT),
    ) as prof:
        teacher.extract_batch(*args)
        torch.npu.synchronize()
    print("trace written to", OUT, flush=True)
    table = prof.key_averages().table(sort_by="self_npu_time_total", row_limit=30)
    print(table, flush=True)


if __name__ == "__main__":
    main()
