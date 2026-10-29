#!/usr/bin/env python3
"""Eight-GPU DDP smoke test for the selected-head VACE tracking LoRA."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from search_vace_tracking_heads import ControlTower
from train_vace_tracking_lora import install_lora, parse_candidates, trajectory_loss


def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.manual_seed(42 + rank)
    candidates = parse_candidates("6:1,10:6,7:11,14:8")
    model = ControlTower.load(ROOT / "models/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors", device)
    params = install_lora(model, candidates, rank=8, alpha=16)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    t, grid, points = 20, 32, 8
    noise = torch.randn(1, 16, t, grid * 2, grid * 2, device=device, dtype=torch.bfloat16)
    control = torch.randn(1, 96, t, grid * 2, grid * 2, device=device, dtype=torch.bfloat16)
    text = torch.zeros(1, 512, 4096, device=device, dtype=torch.bfloat16)
    cells = np.stack([np.arange(points) + i for i in range(t)], axis=1) % (grid * grid)
    fake = {
        "anchor": np.zeros(points, dtype=np.int64),
        "cells": cells.astype(np.int64),
        "valid": np.ones((points, t), dtype=bool),
        "displaced": np.ones((points, t), dtype=bool),
        "groups": np.asarray(["moving_object"] * (points - 1) + ["gripper"]),
        "labels": np.asarray(["object_a"] * (points - 1) + ["gripper base"]),
    }
    captures = {}
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(
            noise, control, text, torch.tensor([1000], device=device),
            lambda layer, q, k: captures.__setitem__(layer, (q, k)),
        )
        loss = torch.stack([
            trajectory_loss(captures[layer][0], captures[layer][1], fake, head, grid,
                            rollout_weight=1.0, separation_weight=0.25, margin=1.0)
            for layer, head in candidates
        ]).mean()
    loss.backward()
    grad_norm = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in params if p.grad is not None))
    optimizer.step()
    values = torch.tensor([loss.detach(), grad_norm.detach()], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.AVG)
    if rank == 0:
        print(json.dumps({"world_size": world, "loss_avg": float(values[0]),
                          "grad_norm_avg": float(values[1]),
                          "trainable_parameters": sum(p.numel() for p in params),
                          "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                          "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 2)}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
