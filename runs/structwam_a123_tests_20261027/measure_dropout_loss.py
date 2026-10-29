"""Action / video loss of one checkpoint with video dropout forced off and forced on.

Same samples, timesteps and noise for both settings (paired), each with and without the run's own
augmentation.  Usage: measure_dropout_loss.py CKPT_DIR CKPT_NAME N_BATCHES OUT_JSON
"""

import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import torch
from omegaconf import OmegaConf

from openwam.dataloader import build_dataset
from openwam.deploy.model_loader import load_from_checkpoint_dir

BS = 8  # the training per-device batch


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def summary(xs):
    return {"mean": round(st.mean(xs), 5), "sem": round(st.stdev(xs) / len(xs) ** 0.5, 5), "n_batches": len(xs)}


def main():
    ck, name, n_batches, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    cfg = OmegaConf.load(ck / "config.yaml")
    _, arch = load_from_checkpoint_dir(str(ck), device="cuda:0", ckpt_name=name)
    arch.normalizer = None  # the trainer never attaches one; the dataset already normalizes actions
    arch.init_training_schedulers(1000)
    arch.set_training_runtime(use_gradient_checkpointing=False,
                              max_timestep_boundary=float(cfg.training.max_timestep_boundary),
                              min_timestep_boundary=float(cfg.training.min_timestep_boundary))
    arch.train()
    arch.requires_grad_(False)

    datasets = {"noaug": build_dataset(OmegaConf.merge(cfg.dataloader, {"structure_augment": {"enabled": False}}),
                                       split="train"),
                "aug": build_dataset(cfg.dataloader, split="train")}
    idx = random.Random(0).sample(range(len(datasets["noaug"])), n_batches * BS)
    drops = {"off": lambda b, d: None, "on": lambda b, d: torch.ones(b, dtype=torch.bool, device=d)}
    res = {f"{a}/{m}/{k}": [] for a in datasets for m in drops for k in ("action", "video")}

    for i in range(n_batches):
        for a, ds in datasets.items():
            seed_all(1000 + i)
            samples = [ds[j] for j in idx[i * BS:(i + 1) * BS]]
            with torch.no_grad():
                seed_all(7 + i)
                inputs = arch.prepare_inputs(samples)
                inputs["use_gradient_checkpointing"] = False
                for m, fn in drops.items():
                    arch._video_drop = fn
                    seed_all(7 + i)
                    r = arch.compute_loss(**dict(inputs), lambda_video=1.0, lambda_action=1.0)
                    res[f"{a}/{m}/action"].append(float(r["loss_action"]))
                    res[f"{a}/{m}/video"].append(float(r["loss_video"]))
        print(f"batch {i + 1}/{n_batches}: " + " ".join(f"{k}={v[-1]:.4f}" for k, v in res.items() if "action" in k),
              flush=True)

    report = {k: summary(v) for k, v in res.items()}
    for a in datasets:
        on, off = res[f"{a}/on/action"], res[f"{a}/off/action"]
        report[f"{a}/action on-off (paired)"] = summary([x - y for x, y in zip(on, off)])
        report[f"{a}/action 0.7*off+0.3*on"] = round(0.7 * st.mean(off) + 0.3 * st.mean(on), 5)
    out.write_text(json.dumps(report, indent=1))
    print("REPORT", json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
