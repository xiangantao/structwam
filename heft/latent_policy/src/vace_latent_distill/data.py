"""Deterministic random windows; privileged video stays outside the student API."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def stable_seed(*values):
    return int.from_bytes(
        hashlib.sha256(":".join(map(str, values)).encode()).digest()[:8], "little"
    ) % (2**31)


def decode_indices(video, indices, fps):
    import av

    indices = list(map(int, indices))
    if not indices or min(indices) < 0 or fps <= 0:
        raise ValueError("Invalid video frame indices / fps")
    wanted, found = set(indices), {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        if stream.average_rate is None or abs(float(stream.average_rate) - fps) > 1e-6:
            raise ValueError("Video FPS differs from metadata")
        origin = stream.start_time or 0
        container.seek(
            origin + round(min(indices) / (fps * float(stream.time_base))),
            stream=stream,
            backward=True,
        )
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame lacks PTS")
            position = (frame.pts - origin) * float(stream.time_base) * fps
            index = round(position)
            if abs(position - index) > 1e-3:
                raise ValueError("Frame timestamp is not on stored FPS grid")
            if index in wanted:
                found[index] = frame.to_ndarray(format="rgb24").transpose(2, 0, 1).copy()
            if index >= max(indices):
                break
    if set(found) != wanted:
        raise RuntimeError(f"Missing frames from {video}: {sorted(wanted - set(found))[:4]}")
    return np.stack([found[index] for index in indices])


def resize_rgb(rgb, size):
    value = torch.as_tensor(rgb)
    if value.dtype != torch.uint8 or value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("Expected uint8 [T,3,H,W]")
    if value.shape[-2:] != (size, size):
        # LIBERO and the teacher both use square cameras.  No crop is applied
        # without also transforming the teacher's spatial coordinate field.
        if value.shape[-2] != value.shape[-1]:
            raise ValueError("Only square LIBERO cameras are supported")
        value = (
            F.interpolate(
                value.float(), (size, size), mode="bilinear", align_corners=False, antialias=True
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
    return value.contiguous()


class StructuredDataset(Dataset):
    def __init__(
        self, manifest, text_cache, *, image_size=256, seed=42, windows_per_episode_camera=4
    ):
        self.rows = [
            json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()
        ]
        if not self.rows or len({row["id"] for row in self.rows}) != len(self.rows):
            raise ValueError("Manifest is empty or repeats episode/camera ids")
        self.text = torch.load(text_cache, map_location="cpu", weights_only=True)
        self.image_size, self.seed = image_size, seed
        self.repetitions, self.epoch = windows_per_episode_camera, 0
        if self.repetitions < 1:
            raise ValueError("Window repetitions must be positive")
        for row in self.rows:
            if row["length"] < 49 or row["task"] not in self.text:
                raise ValueError("Episode is too short or task text is missing")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows) * self.repetitions

    def window(self, index):
        row = dict(self.rows[index // self.repetitions])
        seed = stable_seed(self.seed, self.epoch, index, row["id"])
        start = int(np.random.default_rng(seed).integers(0, row["length"] - 49 + 1))
        absolute = row["episode_start_frame"] + start
        row.update(
            start=start,
            frame_indices=list(range(absolute, absolute + 49)),
            sample_id=f"{row['id']}-s{start:04d}",
            teacher_seed=seed,
        )
        return row

    def __getitem__(self, index):
        row = self.window(index)
        video = torch.from_numpy(decode_indices(row["video"], row["frame_indices"], row["fps"]))
        text = self.text[row["task"]].float()
        if text.ndim != 2 or text.shape[-1] != 4096 or not torch.isfinite(text).all():
            raise ValueError("Invalid UMT5 cache entry")
        mask = text.abs().sum(-1) != 0
        return {
            "rgb": resize_rgb(video[:33], self.image_size),
            "teacher_rgb": video,
            "text": text,
            "text_mask": mask,
            "metadata": row,
        }


def collate_structured(batch):
    result = {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "text": torch.stack([item["text"] for item in batch]),
        "text_mask": torch.stack([item["text_mask"] for item in batch]),
        "metadata": [item["metadata"] for item in batch],
    }
    if "target" in batch[0]:
        from .targets import stack_targets

        result["target"] = stack_targets([item["target"] for item in batch], "cpu")
    else:
        result["teacher_rgb"] = torch.stack([item["teacher_rgb"] for item in batch])
    return result


def color_augmentation(rgb):
    batch = len(rgb)
    brightness = 0.95 + 0.1 * torch.rand(batch, 1, 1, 1, 1, device=rgb.device)
    color = 0.95 + 0.1 * torch.rand(batch, 1, 3, 1, 1, device=rgb.device)
    return (rgb.float() * brightness * color).round().clamp(0, 255).to(torch.uint8)
