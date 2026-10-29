"""Student-only FSDP2 and resumable optimizer-step checkpoints."""

import contextlib
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import PipelineConfig


def wrap_student(model, mode, device):
    if not dist.is_initialized():
        return model
    if mode == "ddp":
        return DDP(model, device_ids=[device.index] if device.type in ("cuda", "npu") else None)
    if device.type not in ("cuda", "npu"):
        raise ValueError("FSDP2 recipe requires CUDA/NPU; use DDP for CPU checks")
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    options = {
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        "reshard_after_forward": True,
    }
    for block in model.backbone.blocks:
        fully_shard(block, **options)
    # The student explicitly runs patch_embed/blocks/norm to interleave text
    # fusion.  backbone.forward is never called, so its remaining parameters
    # must belong to the root forward's shard group rather than an idle wrapper.
    for group in (model.language, model.observed_blocks, model.future_blocks):
        for block in group:
            fully_shard(block, **options)
    if model.config.task == "shared":
        for head in model.heads.values():
            fully_shard(head, **options)
    else:
        fully_shard(model.head, **options)
    fully_shard(model, **options)
    return model


def optimizer_for(model, config):
    base = model.module if isinstance(model, DDP) else model
    backbone, other = [], []
    for name, parameter in base.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith("backbone.") else other).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": config.backbone_lr, "base_lr": config.backbone_lr},
            {"params": other, "lr": config.head_lr, "base_lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )


def learning_rate(optimizer, step, config):
    factor = (
        (step + 1) / max(1, config.warmup_steps)
        if step < config.warmup_steps
        else (
            0.1
            + 0.9
            * 0.5
            * (
                1
                + math.cos(
                    math.pi
                    * (step - config.warmup_steps)
                    / max(1, config.steps - config.warmup_steps)
                )
            )
        )
    )
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def save_checkpoint(path, model, optimizer, *, config, step, epoch, cursor, provenance):
    base = model.module if isinstance(model, DDP) else model
    model_state, optimizer_state = get_state_dict(
        base, optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    states = [None] * dist.get_world_size() if dist.is_initialized() else [rng_state()]
    if dist.is_initialized():
        dist.all_gather_object(states, rng_state())
        if dist.get_rank() != 0:
            return
    payload = {
        "schema": "vace-structured-student-v1",
        "config": config.to_dict(),
        "model": model_state,
        "optimizer": optimizer_state,
        "step": step,
        "epoch": epoch,
        "cursor": cursor,
        "rng": states,
        "provenance": provenance,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def saved_options(saved_config):
    """Options added after a checkpoint was written take their (old-behavior) defaults."""
    defaults = PipelineConfig().to_dict()
    return {
        name: {**{k: v for k, v in defaults[name].items() if k not in saved_config[name]},
               **saved_config[name]}
        for name in ("model", "teacher", "train")
    }


def load_checkpoint(path, model, optimizer, config, provenance):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    options = saved_options(saved["config"])
    if (
        saved["schema"] != "vace-structured-student-v1"
        or options["model"] != config.to_dict()["model"]
    ):
        raise ValueError("Checkpoint architecture/task mismatch")
    if (
        options["teacher"] != config.to_dict()["teacher"]
        or saved["provenance"] != provenance
    ):
        raise ValueError("Checkpoint teacher/data protocol mismatch")
    mutable = {
        "steps", "output", "save_every", "log_every", "workers",
        "batch_size", "teacher_batch_size", "accumulation_steps", "tracking_diagnostics",
    }
    for key, value in options["train"].items():
        if key not in mutable and getattr(config.train, key) != value:
            raise ValueError(f"Resume changes training option {key}")
    base = model.module if isinstance(model, DDP) else model
    set_state_dict(
        base,
        optimizer,
        model_state_dict=saved["model"],
        optim_state_dict=saved["optimizer"],
        options=StateDictOptions(full_state_dict=True, strict=True),
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    if len(saved["rng"]) != (dist.get_world_size() if dist.is_initialized() else 1):
        raise ValueError("Resume requires the same number of ranks")
    restore_rng(saved["rng"][rank])
    return saved["step"], saved["epoch"], saved["cursor"]


def accumulation_context(model, boundary):
    return model.no_sync() if isinstance(model, DDP) and not boundary else contextlib.nullcontext()


def finite_gradient_norm(model, limit):
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), limit)
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite student gradients")
    return float(norm)
