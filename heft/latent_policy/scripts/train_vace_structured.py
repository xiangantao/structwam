#!/usr/bin/env python3
"""Full-LIBERO structured distillation with online or cached teacher targets."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

if os.environ.get("VACE_NPU", "0") == "1":
    # Ascend: remap torch.cuda / "cuda" devices / nccl onto torch_npu / hccl.
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))

from vace_latent_distill.config import HEAD_TASKS, NEGATIVE_PROMPT, TASKS, PipelineConfig
from vace_latent_distill.data import StructuredDataset, collate_structured, color_augmentation
from vace_latent_distill.losses import augmentation_loss, structured_loss, tracking_diagnostics
from vace_latent_distill.model import StructuredStudent
from vace_latent_distill.seg_invariance import MODES, appearance_augmentation, seg_invariance_loss
from vace_latent_distill.targets import stack_targets
from vace_latent_distill.teacher import VaceTeacher, content_token_masks
from vace_latent_distill.training import (
    accumulation_context,
    finite_gradient_norm,
    learning_rate,
    load_checkpoint,
    optimizer_for,
    save_checkpoint,
    wrap_student,
)


_PROFILE = os.environ.get("VACE_PROFILE") == "1"


def _tick(device, store, key, t0):
    """Accumulate synchronized wall time into store[key] when profiling is on."""
    if not _PROFILE:
        return t0
    torch.npu.synchronize()
    now = time.monotonic()
    store[key] = store.get(key, 0.0) + now - t0
    return now


def log(rank, **values):
    if rank == 0:
        print(json.dumps(values, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vace_structured.json")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--teacher-batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--teacher-device", default=None)
    parser.add_argument("--offline-manifest", type=Path)
    parser.add_argument("--teacher-cache-dir", type=Path)
    parser.add_argument("--seg-invariance", choices=MODES, default="none")
    parser.add_argument("--stop-step", type=int,
                        help="Stop (and save) after this step; the schedule still uses train.steps")
    args = parser.parse_args()
    config = PipelineConfig.load(args.config)
    for name in ("steps", "batch_size", "teacher_batch_size", "workers"):
        if getattr(args, name) is not None:
            setattr(config.train, name, getattr(args, name))
    if args.task:
        config.model.task = args.task
    if args.offline_manifest:
        config.train.offline_manifest = str(args.offline_manifest.resolve())
    if args.teacher_cache_dir:
        config.train.teacher_cache_dir = str(args.teacher_cache_dir.resolve())
    output = args.output.resolve() if args.output else Path(config.train.output) / config.model.task
    config.train.output = str(output)
    config.validate()
    seg_invariance = args.seg_invariance != "none"
    if seg_invariance and config.model.task != "shared" and not config.model.task.startswith("seg_"):
        parser.error("--seg-invariance requires shared or seg training")
    if seg_invariance and not config.train.consistency_weight:
        parser.error("--seg-invariance requires a positive consistency_weight")
    # The teacher loss below always uses config.train.head_weights.  When the
    # dedicated segmentation-invariance branch is enabled, only its generic
    # appearance-consistency copy skips seg heads to avoid counting that second
    # consistency objective twice; the dedicated branch still uses their weights.
    weak_weights = dict(config.train.head_weights)
    if seg_invariance:
        weak_weights.update({name: 0.0 for name in HEAD_TASKS if name.startswith("seg_")})
    run_weak_consistency = config.train.consistency_weight and (
        not seg_invariance or (config.model.task == "shared" and any(weak_weights.values()))
    )
    run_seg_consistency = seg_invariance and (
        config.model.task != "shared" or any(
            weight for name, weight in config.train.head_weights.items() if name.startswith("seg_")
        )
    )
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("hccl" if os.environ.get("VACE_NPU") == "1" else "nccl")
    random.seed(config.train.seed + rank)
    np.random.seed(config.train.seed + rank)
    torch.manual_seed(config.train.seed + rank)
    torch.set_num_threads(4)
    offline = config.train.offline_manifest is not None
    if offline:
        from vace_latent_distill.offline import OfflineStructuredDataset

        if config.model.task != "shared":
            raise ValueError("The offline five-head recipe currently requires task=shared")
        dataset = OfflineStructuredDataset(
            config.train.offline_manifest, config.train.text_cache,
            cache_dir=config.train.teacher_cache_dir,
            cache_limit_bytes=config.train.teacher_cache_limit_bytes,
            tos_config=config.train.tos_config, tos_endpoint=config.train.tos_endpoint,
            image_size=config.model.image_size,
            teacher_config=config.teacher,
        )
    else:
        dataset = StructuredDataset(
            config.train.manifest,
            config.train.text_cache,
            image_size=config.model.image_size,
            seed=config.train.seed,
            windows_per_episode_camera=config.train.windows_per_episode_camera,
        )
    sampler = DistributedSampler(
        dataset, num_replicas=world, rank=rank, shuffle=True, seed=config.train.seed
    )
    provenance = {
        "manifest_sha256": hashlib.sha256(Path(config.train.manifest).read_bytes()).hexdigest(),
        "text_sha256": hashlib.sha256(Path(config.train.text_cache).read_bytes()).hexdigest(),
        "world_size": world,
        "online_teacher": not offline,
        "student_future_rgb": False,
    }
    if offline:
        provenance["offline_manifest_sha256"] = hashlib.sha256(
            Path(config.train.offline_manifest).read_bytes()
        ).hexdigest()
    if seg_invariance:
        from vace_latent_distill import seg_invariance as seg_module

        provenance["seg_invariance"] = {
            "mode": args.seg_invariance,
            "source_sha256": hashlib.sha256(Path(seg_module.__file__).read_bytes()).hexdigest(),
            "recipe": "seg-appearance-consistency-v1",
        }
    if NEGATIVE_PROMPT not in dataset.text:
        raise ValueError("Text cache must include the frozen negative prompt")
    log(
        rank,
        event="loading_cached_targets" if offline else "loading_teacher",
        task=config.model.task,
        dataset_rows=len(dataset.rows),
        windows_per_epoch=len(dataset),
        world_size=world,
        global_batch=config.train.batch_size * config.train.accumulation_steps * world,
        teacher_batch_size=config.train.teacher_batch_size,
    )
    teacher = None
    if not offline:
        content = content_token_masks(config.teacher.text_root, {row["task"] for row in dataset.rows})
        teacher = VaceTeacher(
            config.teacher,
            config.model.task,
            args.teacher_device or str(device),
            dataset.text[NEGATIVE_PROMPT],
        )
        log(rank, event="teacher_ready", device=str(teacher.device),
            denoise_updates=config.teacher.denoise_updates,
            denoise_reference_updates=config.teacher.denoise_reference_updates)
    model = wrap_student(
        StructuredStudent(config.model).to(device), config.train.parallelism, device
    )
    optimizer = optimizer_for(model, config.train)
    start, epoch, cursor = (
        load_checkpoint(args.resume, model, optimizer, config, provenance)
        if args.resume
        else (0, 0, 0)
    )
    stop = min(args.stop_step or config.train.steps, config.train.steps)
    if stop <= start:
        parser.error("--stop-step must be after the resumed step")
    relation_sampling = config.train.relation_sampling_options()
    content_masks = None
    if config.train.semantic_spatial_weight:
        content_masks = content if not offline else content_token_masks(
            config.teacher.text_root, {row["task"] for row in dataset.rows}
        )
    depth_weights = config.train.depth_camera_weights
    if depth_weights is not None:
        missing = {row["camera"] for row in dataset.rows} - set(depth_weights)
        if missing:
            parser.error(f"depth_camera_weights lacks cameras {sorted(missing)}")
    diagnose = config.train.tracking_diagnostics and config.model.task in {"tracking", "shared"}
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / "pipeline.json").write_text(json.dumps(config.to_dict(), indent=2))
        if seg_invariance:
            (output / "seg_invariance.json").write_text(json.dumps({
                **provenance["seg_invariance"],
                "weight": config.train.consistency_weight,
                "clean_target_stopgrad": True,
                "future_weight": config.train.future_weight,
            }, indent=2))
    log(rank, event="student_ready", output=str(output), resume_step=start)

    def new_loader(epoch):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        return DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            sampler=sampler,
            num_workers=config.train.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
            collate_fn=collate_structured,
            generator=torch.Generator().manual_seed(config.train.seed + epoch),
        )

    loader = new_loader(epoch)
    iterator = iter(loader)
    for _ in range(cursor):
        next(iterator)
    model.train()
    started = time.monotonic()
    try:
        for step in range(start, stop):
            optimizer.zero_grad(set_to_none=True)
            learning_rate(optimizer, step, config.train)
            metrics = {}
            prof = {}
            _t = time.monotonic()
            for micro in range(config.train.accumulation_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch, cursor = epoch + 1, 0
                    loader = new_loader(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                cursor += 1
                _t = _tick(device, prof, "data_wait", _t)
                if offline:
                    target = {branch: {key: value.to(device, non_blocking=True)
                                       for key, value in values.items()}
                              for branch, values in batch["target"].items()}
                else:
                    targets = []
                    for begin in range(0, len(batch["metadata"]), config.train.teacher_batch_size):
                        end = min(begin + config.train.teacher_batch_size, len(batch["metadata"]))
                        rows = batch["metadata"][begin:end]
                        targets.extend(
                            teacher.extract_batch(
                                batch["teacher_rgb"][begin:end],
                                batch["text"][begin:end],
                                batch["text_mask"][begin:end],
                                [row["teacher_seed"] for row in rows],
                                [content[row["task"]] for row in rows],
                            )
                        )
                    target = stack_targets(targets, device)
                    del targets
                    _t = _tick(device, prof, "teacher", _t)
                rgb = batch["rgb"].to(device, non_blocking=True)
                text = batch["text"].to(device, non_blocking=True)
                text_mask = batch["text_mask"].to(device, non_blocking=True)
                metadata = batch["metadata"]
                optional = {}
                if content_masks is not None:
                    optional["content_mask"] = torch.stack(
                        [torch.as_tensor(content_masks[row["task"]]) for row in metadata]
                    ).to(device)
                if depth_weights is not None:
                    optional["depth_sample_weights"] = torch.tensor(
                        [float(depth_weights[row["camera"]]) for row in metadata], device=device
                    )
                boundary = micro == config.train.accumulation_steps - 1
                with accumulation_context(model, boundary):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        prediction = model(rgb, text, text_mask)
                        loss, details = structured_loss(
                            prediction,
                            target,
                            task=config.model.task,
                            samples=config.train.relation_queries,
                            temperature=config.train.relation_temperature,
                            future_weight=config.train.future_weight,
                            trajectory_weight=config.train.trajectory_weight,
                            cycle_weight=config.train.cycle_weight,
                            depth_spatial_weight=config.train.depth_spatial_weight,
                            head_weights=config.train.head_weights,
                            relation_sampling=relation_sampling,
                            semantic_spatial_weight=config.train.semantic_spatial_weight,
                            relation_weight=config.train.relation_weight,
                            trajectory_sigma=config.train.trajectory_sigma,
                            tracking_target=config.train.tracking_target,
                            **optional,
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite distillation loss")
                    _t = _tick(device, prof, "student_fwd_loss", _t)
                    (loss / config.train.accumulation_steps).backward()
                    _t = _tick(device, prof, "student_bwd", _t)
                details["distillation_total"] = details["total"]
                logging_step = (step + 1) % config.train.log_every == 0 or step == start
                if diagnose and logging_step:
                    details.update(tracking_diagnostics(
                        prediction, target, config.train.relation_temperature
                    ))
                detached = {
                    b: {k: v.detach() for k, v in prediction[b].items()}
                    for b in ("observed", "future")
                }
                detached["timeline"] = prediction["timeline"]
                del prediction, target
                consistency_total = 0.0
                if run_weak_consistency:
                    with accumulation_context(model, boundary):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            augmented = model(color_augmentation(rgb), text, text_mask)
                            consistency = augmentation_loss(
                                augmented,
                                detached,
                                config.model.task,
                                weak_weights,
                            )
                            scaled = (
                                config.train.consistency_weight
                                * consistency
                                / config.train.accumulation_steps
                            )
                        if not torch.isfinite(consistency):
                            raise RuntimeError("Nonfinite augmentation consistency loss")
                        scaled.backward()
                    consistency_total = consistency.detach()
                    del augmented
                if run_seg_consistency:
                    kind = args.seg_invariance
                    if kind == "alternating":
                        kind = ("color", "texture")[(step * config.train.accumulation_steps + micro) % 2]
                    with accumulation_context(model, boundary):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            augmented = model(appearance_augmentation(rgb, kind), text, text_mask)
                            seg_consistency, seg_details = seg_invariance_loss(
                                augmented, detached, config.model.task,
                                config.train.head_weights,
                                future_weight=config.train.future_weight,
                            )
                            scaled = (config.train.consistency_weight * seg_consistency
                                      / config.train.accumulation_steps)
                        if not torch.isfinite(seg_consistency):
                            raise RuntimeError("Nonfinite segmentation invariance loss")
                        scaled.backward()
                    consistency_total = consistency_total + seg_consistency.detach()
                    details.update({f"seg_invariance/{kind}/{key}": value
                                    for key, value in seg_details.items()})
                    del augmented
                if run_weak_consistency or run_seg_consistency:
                    details["augmentation"] = consistency_total
                    details["augmentation_weighted"] = config.train.consistency_weight * consistency_total
                    details["total"] = details["distillation_total"] + details["augmentation_weighted"]
                for key, value in details.items():
                    metrics[key] = (
                        metrics.get(key, 0.0) + float(value) / config.train.accumulation_steps
                    )
                del detached
            norm = finite_gradient_norm(model, config.train.grad_clip)
            optimizer.step()
            if (step + 1) % config.train.log_every == 0 or step == start:
                if world > 1:
                    tensor = torch.tensor(list(metrics.values()), device=device)
                    dist.all_reduce(tensor)
                    metrics = dict(zip(metrics, (tensor / world).tolist(), strict=True))
                log(
                    rank,
                    step=step + 1,
                    epoch=epoch,
                    cursor=cursor,
                    **metrics,
                    grad_norm=norm,
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                    **({"prof_" + k: round(v, 2) for k, v in prof.items()} if _PROFILE else {}),
                    elapsed_seconds=time.monotonic() - started,
                )
                if rank == 0:
                    with (output / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps({"step": step + 1, **metrics}) + "\n")
                    (output / "teacher_protocol.json").write_text(
                        json.dumps(teacher.protocol if teacher else {
                            "source": "offline verified targets",
                            "manifest": config.train.offline_manifest,
                            "provenance_id": dataset.rows[0]["provenance_id"],
                        }, indent=2)
                    )
            if (step + 1) % config.train.save_every == 0 or step + 1 in {config.train.steps, stop}:
                save_checkpoint(
                    output / f"step_{step + 1:07d}.pt",
                    model,
                    optimizer,
                    config=config,
                    step=step + 1,
                    epoch=epoch,
                    cursor=cursor,
                    provenance=provenance,
                )
    finally:
        del iterator, loader
        if teacher:
            teacher.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
