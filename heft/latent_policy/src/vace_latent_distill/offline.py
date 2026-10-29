"""Loss-only teacher serialization and past-only cached distillation samples."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import timeline
from .data import decode_indices, resize_rgb, stable_seed
from .offline_store import ArtifactCache, TosStore, file_sha256
from .targets import validate_target

SCHEMA = "vace-structured-teacher-lossfp16-v1"
WINDOW_SCHEMA = "vace-eight-offset-windows-v1"
DENSE_KEYS = ("seg_k_l22h9", "seg_k_l15h7", "track_q", "track_k", "text_attention", "depth")


def offset_starts(length, count=8, seed=42):
    last = int(length) - 49
    if last + 1 < count:
        raise ValueError("Episode has too few distinct 49-frame windows")
    # Cover the episode while varying all eight start-frame phases.  Nearby
    # windows therefore place the same physical frames at different token times.
    phases = np.random.default_rng(seed).permutation(count)
    chosen = []
    for index, phase in enumerate(phases):
        candidates = np.arange(int(phase), last + 1, count)
        if len(candidates) == 0:
            raise ValueError("No window exists for this offset phase")
        desired = last * (index + 0.5) / count
        chosen.append(int(candidates[np.abs(candidates - desired).argmin()]))
    if len(set(chosen)) != count:
        raise RuntimeError("Offset windows repeated a start position")
    return sorted(chosen)


def build_windows(rows, *, seed=42, count=8):
    windows = []
    for row in rows:
        starts = offset_starts(row["length"], count, stable_seed(seed, row["episode"]))
        for slot, start in enumerate(starts):
            item = dict(row)
            item.update(
                ordinal=len(windows), window_slot=slot, start=start,
                frame_indices=list(range(row["episode_start_frame"] + start,
                                         row["episode_start_frame"] + start + 49)),
                sample_id=f"{row['id']}-w{slot:02d}-s{start:04d}",
                teacher_seed=stable_seed(seed, row["id"], slot, start, "offline-v1"),
            )
            item["target_key"] = f"targets/{item['ordinal'] // 256:05d}/{item['sample_id']}.pt"
            windows.append(item)
    return windows


def provenance(config, manifest, text_cache):
    root = Path(__file__).resolve().parents[3]
    source_files = [
        Path(__file__), Path(__file__).with_name("teacher.py"),
        Path(__file__).with_name("config.py"),
        Path(__file__).with_name("schedules.py"),
        root / "latent_policy/scripts/demo_main_prompt_k_tracking.py",
        root / "latent_policy/scripts/evaluate_vace_pure_noise_future.py",
    ]
    source_hashes = {str(path.relative_to(root)): file_sha256(path) for path in source_files}
    weights = []
    for path in sorted(Path(config.teacher.model_root).glob("*")):
        if path.is_file():
            stat = path.stat()
            weights.append({"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    value = {
        "schema": SCHEMA, "window_schema": WINDOW_SCHEMA,
        "teacher": dataclasses.asdict(config.teacher),
        "manifest_sha256": file_sha256(manifest), "text_sha256": file_sha256(text_cache),
        "source_sha256": source_hashes, "weight_file_signatures": weights,
        "observed_native_indices": list(range(4, 12)),
        "future_native_indices": list(range(8, 13)),
        "student_future_rgb": False,
    }
    value["id"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value


def pack_target(target, metadata, protocol, identity, *, grid=32, text_length=512):
    validate_target(target, "shared", grid=grid, text_length=text_length)
    branches = {}
    for branch, times in timeline().items():
        indices = times["valid"].nonzero().flatten()
        source = target[branch]
        values = {key: source[key][indices].to(torch.float16).contiguous() for key in DENSE_KEYS}
        values.update(
            seeds=source["seeds"].to(torch.int64).contiguous(),
            paths=source["paths"][:, :, indices].to(torch.float32).contiguous(),
            path_valid=source["path_valid"][:, :, indices].bool().contiguous(),
            native_indices=indices,
        )
        branches[branch] = values
    return {
        "schema": SCHEMA, "provenance_id": identity,
        "metadata": metadata, "teacher_protocol": dict(protocol), "target": branches,
        "grid": grid, "text_length": text_length,
    }


def unpack_target(payload, *, identity=None, grid=None, text_length=None):
    if payload.get("schema") != SCHEMA or (identity and payload["provenance_id"] != identity):
        raise ValueError("Cached teacher schema/protocol differs from this run")
    grid = payload["grid"] if grid is None else grid
    text_length = payload["text_length"] if text_length is None else text_length
    target = {}
    for branch, times in timeline().items():
        source = payload["target"][branch]
        indices = source["native_indices"]
        expected = times["valid"].nonzero().flatten()
        if not torch.equal(indices, expected):
            raise ValueError("Cached teacher time indices differ from the loss timeline")
        temporal = len(times["valid"])
        values = {}
        for key in DENSE_KEYS:
            value = source[key].float()
            restored = torch.zeros((temporal, *value.shape[1:]), dtype=torch.float32)
            restored[indices] = value
            values[key] = restored
        paths = source["paths"]
        values["paths"] = torch.zeros((*paths.shape[:2], temporal, 2), dtype=torch.float32)
        values["paths"][:, :, indices] = paths.float()
        values["path_valid"] = torch.zeros((*paths.shape[:2], temporal), dtype=torch.bool)
        values["path_valid"][:, :, indices] = source["path_valid"]
        values["seeds"] = source["seeds"].long()
        target[branch] = values
    validate_target(target, "shared", grid=grid, text_length=text_length)
    return target


class OfflineStructuredDataset(Dataset):
    def __init__(self, manifest, text_cache, *, cache_dir, cache_limit_bytes=50_000_000_000,
                 tos_config="/root/.tosutilconfig", tos_endpoint=None, image_size=256,
                 teacher_config=None):
        manifest = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not self.rows or any(not row.get("target_sha256") for row in self.rows):
            raise ValueError("Use the verified uploaded.jsonl manifest, not planned windows.jsonl")
        if len({row["provenance_id"] for row in self.rows}) != 1:
            raise ValueError("Offline manifest mixes different teacher protocols")
        self.text = torch.load(text_cache, map_location="cpu", weights_only=True)
        self.image_size = image_size
        self.cache = ArtifactCache(cache_dir, cache_limit_bytes)
        self.store_options = {"config": tos_config, "endpoint": tos_endpoint}
        self.manifest = manifest
        self.epoch = 0
        self.denoising = None if teacher_config is None else (
            teacher_config.denoise_updates, teacher_config.denoise_reference_updates
        )

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        if "target_path" in row:
            # Small local debugging cache (cache_vace_teacher_local.py); no TOS round trip.
            path = Path(row["target_path"])
            if path.stat().st_size != row["target_bytes"]:
                raise ValueError(f"Local cached target changed size: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
        else:
            with self.cache.local(row["target_uri"], TosStore(**self.store_options),
                                  sha256=row["target_sha256"], expected_bytes=row["target_bytes"]) as path:
                payload = torch.load(path, map_location="cpu", weights_only=True)
        fields = ("sample_id", "teacher_seed", "frame_indices", "task", "camera")
        if any(payload["metadata"][field] != row[field] for field in fields):
            raise ValueError("Cached teacher window differs from sampled RGB")
        protocol = payload["teacher_protocol"]
        denoising = (
            protocol.get("denoise_updates"),
            protocol.get("denoise_reference_updates", protocol.get("denoise_updates")),
        )
        if self.denoising is not None and denoising != self.denoising:
            raise ValueError("Cached teacher denoising protocol differs from this run")
        target = unpack_target(payload, identity=row["provenance_id"])
        # Only past pixels are decoded; privileged future RGB is absent at training.
        video = decode_indices(row["video"], row["frame_indices"][:33], row["fps"])
        text = self.text[row["task"]].float()
        return {
            "rgb": resize_rgb(video, self.image_size), "text": text,
            "text_mask": text.abs().sum(-1) != 0, "metadata": row, "target": target,
        }
