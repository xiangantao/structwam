#!/usr/bin/env python3
"""Verify real cached targets, optional cold TOS reads and a small CPU backward."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn

from vace_latent_distill.config import ModelConfig
from vace_latent_distill.data import collate_structured
from vace_latent_distill.losses import structured_loss
from vace_latent_distill.model import StructuredStudent
from vace_latent_distill.offline import OfflineStructuredDataset


class SmallPatch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(3, dim, (2, 16, 16), (2, 16, 16))

    def forward(self, video):
        return self.conv(video).flatten(2).transpose(1, 2)


class SmallBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, tokens, **_):
        return tokens + self.linear(self.norm(tokens))


class SmallBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_embed = SmallPatch(dim)
        self.blocks = nn.ModuleList([SmallBlock(dim), SmallBlock(dim)])
        self.norm = nn.LayerNorm(dim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    job = json.loads((args.run / "job.json").read_text())
    dataset = OfflineStructuredDataset(
        args.run / "uploaded.jsonl", job["config"]["train"]["text_cache"],
        cache_dir=job["cache_dir"], cache_limit_bytes=job["cache_limit_bytes"],
        tos_config=job["tos_config"], tos_endpoint=job["tos_endpoint"], image_size=32,
    )
    results = []
    for index in range(min(args.samples, len(dataset))):
        row = dataset.rows[index]
        if args.cold:
            dataset.cache.evict(row["target_uri"])
        started = time.monotonic()
        item = dataset[index]
        assert "teacher_rgb" not in item and item["rgb"].shape == (33, 3, 32, 32)
        result = {"sample_id": row["sample_id"], "seconds": time.monotonic() - started,
                  "cold_read": args.cold, "bytes": row["target_bytes"], "student_future_rgb": False,
                  "shapes": {branch: {key: list(value.shape) for key, value in values.items()}
                             for branch, values in item["target"].items()}}
        if args.backward and index == 0:
            config = ModelConfig(
                task="shared", image_size=32, grid_size=32, dim=24, text_dim=4096,
                channels=128, attention_heads=4, language_layers=1, decoder_layers=1,
                head_layers=1, window_size=8, mlp_ratio=1, activation_checkpointing=False,
            )
            model = StructuredStudent(config, SmallBackbone(config.dim))
            batch = collate_structured([item])
            prediction = model(batch["rgb"], batch["text"], batch["text_mask"])
            loss, _ = structured_loss(prediction, batch["target"], task="shared", samples=4)
            assert torch.isfinite(loss)
            loss.backward()
            assert model.backbone.patch_embed.conv.weight.grad.abs().sum() > 0
            result.update(small_cpu_backward=True, loss=float(loss.detach()))
        results.append(result)
    payload = {"verified": results, "cache": dataset.cache.summary(job["run_id"])}
    (args.run / "readback_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
