"""Validate and batch the online teacher's native readout matrices."""

import torch


def validate_target(target, task, grid=32, channels=128, text_length=512):
    for branch, temporal in (("observed", 12), ("future", 13)):
        values = target[branch]
        if task.startswith("seg_"):
            expected = {"seg_k": (temporal, grid * grid, channels)}
        elif task == "tracking":
            expected = {
                "track_q": (temporal, grid * grid, channels),
                "track_k": (temporal, grid * grid, channels),
            }
        elif task == "semantic":
            expected = {"text_attention": (temporal, grid * grid, text_length)}
        elif task == "shared":
            expected = {
                "seg_k_l22h9": (temporal, grid * grid, channels),
                "seg_k_l15h7": (temporal, grid * grid, channels),
                "track_q": (temporal, grid * grid, channels),
                "track_k": (temporal, grid * grid, channels),
                "text_attention": (temporal, grid * grid, text_length),
                "depth": (temporal, grid * grid),
            }
        else:
            expected = {
                "depth": (temporal, grid * grid),
                "seg_k": (temporal, grid * grid, channels),
            }
        for name, shape in expected.items():
            value = values[name]
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {branch}.{name}: expected {shape}, got {value.shape}")


def stack_targets(targets, device):
    return {
        branch: {
            key: torch.stack([item[branch][key] for item in targets]).to(device)
            for key in targets[0][branch]
        }
        for branch in ("observed", "future")
    }
