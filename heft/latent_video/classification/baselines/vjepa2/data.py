"""Official normalized inputs with the same per-video sampling as the Wan run."""

import csv
import os
from pathlib import Path

from ...data import IMAGENET_NORMALIZATION, OnlineVideoDataset, Sample
from ...data import read_manifest as read_original_manifest


def read_manifest(path, config):
    if path.suffix != ".csv":
        return read_original_manifest(path, config)
    directories = {}
    records = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream, delimiter=" ", skipinitialspace=True):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError("Expected a space-delimited video-path/label CSV")
            video = Path(row[0]).expanduser()
            if not video.is_absolute():
                video = path.parent / video
            parent = video.parent
            if parent not in directories:
                resolved = parent.resolve()
                # Directory entries expose symlinks without a per-video stat on GPFS.
                try:
                    with os.scandir(resolved) as entries:
                        links = {entry.name for entry in entries if entry.is_symlink()}
                except FileNotFoundError:
                    links = set()
                directories[parent] = resolved, links
            resolved, links = directories[parent]
            canonical = (
                video.resolve() if video.name in links else resolved / video.name
            )
            records.append(Sample(video.stem, str(canonical), int(row[1])))
    if not records:
        raise ValueError(f"Empty data split: {path}")
    if len({r.id for r in records}) != len(records) or len(
        {r.path for r in records}
    ) != len(records):
        raise ValueError(f"Duplicate video IDs or paths in {path}")
    if any(not 0 <= r.label < config.data.num_classes for r in records):
        raise ValueError("Out-of-range SSv2 label")
    return records


def build_dataset(runtime, manifest, records, config, *, training):
    transform = runtime.make_transforms(
        training=training,
        num_views_per_clip=1 if training else config.data.num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=config.data.crop_size,
        normalize=IMAGENET_NORMALIZATION,
    )
    dataset = runtime.video_dataset(
        data_paths=[str(manifest)],
        frames_per_clip=config.clip.frames,
        frame_step=config.data.frame_step,
        num_clips=config.data.num_segments,
        random_clip_sampling=True,
        allow_clip_overlap=True,
        transform=transform,
    )
    if list(dataset.samples) != [r.path for r in records] or list(dataset.labels) != [
        r.label for r in records
    ]:
        raise ValueError("V-JEPA2 read a different video/label manifest")
    return OnlineVideoDataset(dataset, records, seed=config.seed, training=training)
