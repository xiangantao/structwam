"""Train one V-JEPA2 probe, or evaluate a retained checkpoint separately."""

import argparse
import importlib.util
import json
import logging
import os
import random
import time
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler

from ...checkpoint import load_checkpoint, publish_checkpoint_alias, save_checkpoint
from ...data import EvaluationSampler, manifest_identity, write_manifest
from ...engine import ProbeOptimization, run_epoch, unwrap
from ...monitoring import WandbMonitor
from ...upstream import load_upstream
from .config import RunConfig
from .data import build_dataset, read_manifest
from .encoder import FrozenEncoder, file_sha256

logger = logging.getLogger(__name__)


def preflight(config, *, evaluation=False):
    issues = []
    required = {
        "model_path": config.model_path,
        "data.val": config.data.val,
    }
    if not evaluation:
        required["data.train"] = config.data.train
    for name, value in required.items():
        if not config.path(value).is_file():
            issues.append(f"{name} is not a file: {config.path(value)}")
    for name in ("videos", "labels"):
        value = getattr(config.data, name)
        if value is not None and not config.path(value).exists():
            issues.append(f"data.{name} does not exist: {config.path(value)}")
    root = config.path(config.vjepa_root)
    for relative in (
        "src/models/vision_transformer.py",
        "evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py",
    ):
        if not (root / relative).is_file():
            issues.append(f"Missing official source: {root / relative}")
    packages = ["torchvision", "timm", "decord", "pandas", "einops"]
    if config.wandb.enabled:
        packages.append("wandb")
    missing = [p for p in packages if importlib.util.find_spec(p) is None]
    if missing:
        issues.append(
            f"Missing dependencies: {', '.join(missing)}; no automatic installation"
        )
    return issues


def distributed_device():
    if not torch.cuda.is_available():
        raise RuntimeError("Training/evaluation requires CUDA; --check supports CPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    return (
        torch.device("cuda", local_rank),
        dist.get_rank() if dist.is_initialized() else 0,
        dist.get_world_size() if dist.is_initialized() else 1,
    )


def rank_zero_action(action, rank):
    # Propagate filesystem failures before other ranks wait for the next collective.
    error = [None]
    if rank == 0:
        try:
            action()
        except Exception as exc:
            logger.exception("Rank-zero filesystem action failed")
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])


def make_loader(dataset, config, *, rank, world_size, training, steps=None):
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed
        )
        if training
        else EvaluationSampler(len(dataset), rank, world_size)
    )
    batching = {
        "batch_size": config.optimization.batch_size,
        "sampler": sampler,
        "drop_last": False,
    }
    if steps is not None:
        # Stop scheduling work at the sampler, so decoders finish before workers exit.
        batching = {
            "batch_sampler": LimitedLoader(
                BatchSampler(sampler, config.optimization.batch_size, False), steps
            )
        }
    return DataLoader(
        dataset,
        **batching,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(config.seed + rank),
        multiprocessing_context="spawn" if config.num_workers else None,
    ), sampler


def protocol_record(
    config, encoder, classifier, runtime, train_records, val_records, world_size
):
    directory = Path(__file__).resolve().parent
    shared = directory.parents[1]
    sources = list(directory.glob("*.py")) + [
        shared / name
        for name in (
            "config.py",
            "data.py",
            "engine.py",
            "checkpoint.py",
            "monitoring.py",
            "upstream.py",
        )
    ]
    return {
        "representation": {
            "baseline": "vjepa2_vitl_singlehead",
            "encoder": encoder.metadata(),
            "probe": {
                "embed_dim": encoder.embed_dim,
                "heads": config.num_heads,
                "depth": config.num_probe_blocks,
                "classes": config.data.num_classes,
                "parameters": sum(p.numel() for p in unwrap(classifier).parameters()),
            },
            "preprocessing": {
                "frames": config.clip.frames,
                "crop_size": config.data.crop_size,
                "frame_step": config.data.frame_step,
                "segments": config.data.num_segments,
                "validation_views": config.data.num_views_per_segment,
                "rgb_adapter": "official ImageNet-normalized CTHW; no uint8 roundtrip",
                "sampling": "per-video seeded original sampler; validation epoch fixed to zero",
                "seed": config.seed,
            },
            "vjepa2": runtime.provenance,
            "runner_sources": {
                str(p.relative_to(shared)): file_sha256(p)
                for p in sorted(sources)
                if not p.name.startswith("test_")
            },
        },
        "training": {
            "optimization": asdict(config.optimization),
            "world_size": world_size,
            "extract_batch_size": config.extract_batch_size,
            "extract_lanes": config.extract_lanes,
            "validate_every": 0,
            "manifest": manifest_identity(train_records)
            if train_records is not None
            else None,
        },
        "validation_manifest": manifest_identity(val_records),
    }


class LimitedLoader:
    def __init__(self, loader, steps):
        self.loader, self.steps = loader, steps

    def __len__(self):
        return min(len(self.loader), self.steps)

    def __iter__(self):
        yield from islice(self.loader, self.steps)


class TimedLoader:
    def __init__(self, loader):
        self.loader = loader
        self.last_wait_seconds = 0.0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        start = time.perf_counter()
        iterator = iter(self.loader)
        for _ in range(len(self)):
            batch = next(iterator)
            self.last_wait_seconds = time.perf_counter() - start
            yield batch
            start = time.perf_counter()
        # Exhaust the finite sampler so DataLoader joins its workers normally.
        try:
            next(iterator)
        except StopIteration:
            return
        raise RuntimeError("DataLoader produced more batches than its declared length")


def run(config, *, resume=None, evaluate=None, smoke_steps=None):
    if smoke_steps is not None and (
        smoke_steps < 1 or resume is not None or evaluate is not None
    ):
        raise ValueError(
            "Smoke steps must be positive and cannot be combined with resume/evaluate"
        )
    issues = preflight(config, evaluation=evaluate is not None)
    if issues:
        raise ValueError("\n".join(issues))
    logger.info("Reading training/validation manifests")
    train_records = (
        None
        if evaluate is not None
        else read_manifest(config.path(config.data.train), config)
    )
    val_records = read_manifest(config.path(config.data.val), config)
    logger.info(
        "Manifests ready: train=%s val=%d",
        len(train_records) if train_records is not None else None,
        len(val_records),
    )
    if train_records is not None and {r.id for r in train_records} & {
        r.id for r in val_records
    }:
        raise ValueError("Training and validation video IDs overlap")
    output = config.path(config.output_dir)
    if evaluate is not None:
        output = output / "evaluations" / evaluate.stem
    device, rank, world_size = distributed_device()
    monitor = None
    encoder = None
    succeeded = False
    try:

        def prepare():
            if (output / "latest.pt").exists() and resume is None and evaluate is None:
                raise FileExistsError("Use --resume or a new output_dir")
            output.mkdir(parents=True, exist_ok=True)
            if train_records is not None:
                write_manifest(output / "train_paths.csv", train_records)
            write_manifest(output / "val_paths.csv", val_records)

        rank_zero_action(prepare, rank)
        monitor = WandbMonitor(
            config.wandb,
            output,
            rank=rank,
            metadata=config.metadata(),
            job_type="evaluate" if evaluate is not None else "train",
        )
        runtime = load_upstream(config.path(config.vjepa_root))
        training = evaluate is None
        records = train_records if training else val_records
        dataset = build_dataset(
            runtime,
            output / ("train_paths.csv" if training else "val_paths.csv"),
            records,
            config,
            training=training,
        )
        loader, sampler = make_loader(
            dataset,
            config,
            rank=rank,
            world_size=world_size,
            training=training,
            steps=smoke_steps,
        )
        full_epoch_steps = (
            len(sampler) + config.optimization.batch_size - 1
        ) // config.optimization.batch_size
        if smoke_steps is not None:
            loader = TimedLoader(loader)
        logger.info(
            "rank=%d videos=%d batches=%d loading frozen ViT-L",
            rank,
            len(dataset),
            len(loader),
        )
        encoder = FrozenEncoder.from_local(config, device)
        encoder.verify_parallel = smoke_steps is not None and config.extract_lanes > 1
        encoder.measure = smoke_steps is not None
        random.seed(config.seed)
        np.random.seed(config.seed % 2**32)
        torch.manual_seed(config.seed)
        classifier = runtime.classifier(
            embed_dim=encoder.embed_dim,
            num_heads=config.num_heads,
            depth=config.num_probe_blocks,
            num_classes=config.data.num_classes,
            use_activation_checkpointing=True,
        ).to(device)
        if dist.is_initialized():
            classifier = DistributedDataParallel(
                classifier, device_ids=[device.index], static_graph=True
            )
        protocol = protocol_record(
            config, encoder, classifier, runtime, train_records, val_records, world_size
        )
        if smoke_steps is not None:
            protocol["training"]["smoke_steps"] = smoke_steps
        monitor.update_config({"protocol": protocol})
        if evaluate is not None:
            epoch, _ = load_checkpoint(
                evaluate, classifier, protocol=protocol, device=device
            )
            metrics = run_epoch(
                classifier,
                encoder,
                loader,
                training=False,
                amp_dtype=config.optimization.amp_dtype,
                predictions=output / f"predictions_r{rank}.jsonl",
            )
            if metrics["samples"] != len(val_records):
                raise RuntimeError("Validation did not cover the exact split")
            monitor.log(
                {"epoch": epoch, **{f"evaluation/{k}": v for k, v in metrics.items()}}
            )
            rank_zero_action(
                lambda: (output / "evaluation.json").write_text(
                    json.dumps(
                        {
                            "checkpoint": str(evaluate.resolve()),
                            "epoch": epoch,
                            "metrics": metrics,
                            "protocol": protocol,
                        },
                        indent=2,
                    )
                ),
                rank,
            )
            succeeded = True
            return metrics
        optimization = ProbeOptimization.create(
            runtime, classifier, config.optimization, full_epoch_steps
        )
        start_epoch, best = 0, float("-inf")
        if resume is not None:
            start_epoch, best = load_checkpoint(
                resume,
                classifier,
                protocol=protocol,
                device=device,
                optimization=optimization,
            )
        monitor.update_config(
            {
                "start_epoch": start_epoch,
                "resume_checkpoint": str(resume) if resume else None,
            }
        )

        def record_protocol():
            (output / "config.json").write_text(json.dumps(config.metadata(), indent=2))
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2))

        rank_zero_action(record_protocol, rank)
        smoke_metrics = []

        def log_smoke(values):
            telemetry = torch.tensor(
                [loader.last_wait_seconds, encoder.last_extract_seconds], device=device
            )
            if dist.is_initialized():
                dist.all_reduce(telemetry, op=dist.ReduceOp.MAX)
            data_wait, extraction = telemetry.tolist()
            values = {
                **values,
                "timing/data_wait_seconds_max": data_wait,
                "timing/extraction_seconds_max": extraction,
            }
            smoke_metrics.append(values)
            monitor.log(values)
            if rank == 0:
                logger.info("smoke_step=%s", json.dumps(values))

        for epoch in range(start_epoch, config.optimization.num_epochs):
            sampler.set_epoch(epoch)
            dataset.set_epoch(epoch)
            before = (
                next(unwrap(classifier).parameters()).detach().clone()
                if smoke_steps
                else None
            )
            metrics = run_epoch(
                classifier,
                encoder,
                loader,
                training=True,
                amp_dtype=config.optimization.amp_dtype,
                optimization=optimization,
                on_step=log_smoke if smoke_steps else monitor.log,
                log_every=1 if smoke_steps else config.wandb.log_every,
                epoch=epoch,
            )
            if smoke_steps is not None:
                after = next(unwrap(classifier).parameters()).detach()
                if torch.equal(before, after):
                    raise RuntimeError("Classifier parameters did not update")
                if any(
                    p.requires_grad or p.grad is not None
                    for wrapper in encoder.wrappers
                    for p in wrapper.parameters()
                ):
                    raise RuntimeError("Frozen encoder accumulated gradients")
                copies = [torch.empty_like(after) for _ in range(world_size)]
                if dist.is_initialized():
                    dist.all_gather(copies, after.contiguous())
                    if any(not torch.equal(after, value) for value in copies):
                        raise RuntimeError("DDP classifier parameters diverged")
                checkpoint = output / "smoke.pt"
                save_checkpoint(
                    checkpoint,
                    classifier,
                    optimization,
                    epoch=0,
                    best_top1=best,
                    protocol=protocol,
                    device=device,
                )
                if dist.is_initialized():
                    dist.barrier()
                restored_epoch, _ = load_checkpoint(
                    checkpoint,
                    classifier,
                    protocol=protocol,
                    device=device,
                    optimization=optimization,
                )
                if restored_epoch != 0:
                    raise RuntimeError("Smoke checkpoint restore failed")
                report = {
                    "complete": True,
                    "world_size": world_size,
                    "batch_size_per_rank": config.optimization.batch_size,
                    "extract_batch_size": config.extract_batch_size,
                    "extract_lanes": config.extract_lanes,
                    "parallel_comparison": encoder.comparison,
                    "full_epoch_steps": full_epoch_steps,
                    "steps": smoke_metrics,
                    "metrics": metrics,
                    "classifier_updated": True,
                    "encoder_frozen": True,
                    "ddp_parameter_agreement": True,
                    "checkpoint_restored": True,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                    / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                    "measurement": "Real full-split DataLoader; distinct sampled videos; includes loader wait",
                }
                (output / f"smoke_result_r{rank}.json").write_text(
                    json.dumps(report, indent=2)
                )
                logger.info("Smoke complete: rank=%d metrics=%s", rank, metrics)
                succeeded = True
                return report
            epoch_path = output / f"epoch_{epoch + 1:04d}.pt"

            def check_destination(epoch_path=epoch_path):
                if epoch_path.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite retained checkpoint: {epoch_path}"
                    )

            rank_zero_action(check_destination, rank)
            save_checkpoint(
                epoch_path,
                classifier,
                optimization,
                epoch=epoch + 1,
                best_top1=best,
                protocol=protocol,
                device=device,
            )

            def publish(epoch_path=epoch_path, epoch=epoch, metrics=metrics):
                publish_checkpoint_alias(epoch_path, output / "latest.pt")
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps({"epoch": epoch + 1, "train": metrics, "val": None})
                        + "\n"
                    )

            rank_zero_action(publish, rank)
            monitor.log(
                {
                    "step": (epoch + 1) * len(loader),
                    "epoch": epoch + 1,
                    **{f"train/{k}": v for k, v in metrics.items()},
                }
            )
            logger.info(
                "epoch=%d train=%s checkpoint=%s", epoch + 1, metrics, epoch_path
            )
        succeeded = True
    finally:
        try:
            if encoder is not None:
                encoder.close()
        finally:
            try:
                if monitor is not None:
                    monitor.finish(exit_code=0 if succeeded else 1)
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--extract-lanes", type=int, choices=(1, 2))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", type=Path)
    modes.add_argument("--evaluate", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help="Run bounded real training steps and checkpoint verification",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = RunConfig.load(args.config)
    for name in ("model_path", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            config = replace(config, **{name: str(value.resolve())})
    if args.extract_lanes is not None:
        config = replace(config, extract_lanes=args.extract_lanes)
    if args.check:
        issues = preflight(config, evaluation=args.evaluate is not None)
        checkpoint = args.evaluate or args.resume
        if checkpoint is not None and not checkpoint.is_file():
            issues.append(f"Probe checkpoint is not a file: {checkpoint}")
        print(
            json.dumps(
                {"ready": not issues, "issues": issues, "config": config.metadata()},
                indent=2,
            )
        )
        return 1 if issues else 0
    try:
        run(
            config,
            resume=args.resume,
            evaluate=args.evaluate,
            smoke_steps=args.smoke_steps,
        )
    except Exception:
        logger.exception("V-JEPA2 baseline failed")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
