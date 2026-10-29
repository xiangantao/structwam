"""Correctness checks for the opt-in A2 / A3 switches on the s40000 checkpoint (one NPU).

1. action_video_dropout=1: perturbing the video latents leaves the action prediction unchanged;
   with 0 it changes (the perturbation is effective).
2. Mixed batch [drop, keep]: the dropped sample's action is unchanged, the kept one's changes.
3. structure_expert.groups without trunk: 202 structure tokens instead of 234, finite losses.
4. inference.structure_only_action marks every sample at eval.
"""

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import numpy as np
import torch
from omegaconf import OmegaConf

from openwam.dataloader import build_dataset
from openwam.deploy.model_loader import load_from_checkpoint_dir
from openwam.model.architectures.tri_system import struct_system

CKPT = Path("/home/ma-user/work/2337888765/workspace/runs/structwam_step40000_eval_ckpt")
NAME = "checkpoint_step_40000.safetensors"
CAPTURE = {}
_action_loss = struct_system.TriSystemStructArchitecture._compute_action_loss


def _capture(self, noise_pred, *args, **kwargs):
    CAPTURE["action"] = noise_pred.detach().float().clone()
    return _action_loss(self, noise_pred, *args, **kwargs)


struct_system.TriSystemStructArchitecture._compute_action_loss = _capture


def load(ckpt_dir):
    _, arch = load_from_checkpoint_dir(str(ckpt_dir), device="cuda:0", ckpt_name=NAME)
    arch.normalizer = None
    arch.init_training_schedulers(1000)
    arch.train()
    arch.requires_grad_(False)
    return arch


def action_pred(arch, inputs, seed, perturb):
    inputs = copy.copy(inputs)
    if perturb:
        g = torch.Generator(device="cpu").manual_seed(7)
        for key in ("input_latents", "first_frame_latents"):
            if inputs.get(key) is not None:
                noise = torch.randn(inputs[key].shape, generator=g).to(inputs[key].device, inputs[key].dtype)
                inputs[key] = inputs[key] + noise
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        result = arch.compute_loss(**inputs, lambda_video=1.0, lambda_action=1.0)
    return CAPTURE["action"], {k: float(v) for k, v in result.items() if k.startswith("loss")}


def main():
    report = {}
    cfg = OmegaConf.load(CKPT / "config.yaml")
    ds = build_dataset(cfg.dataloader, split="train")
    samples = [ds[i] for i in (1234, 98765)]
    arch = load(CKPT)
    with torch.no_grad():
        inputs = arch.prepare_inputs(samples)
    inputs["use_gradient_checkpointing"] = False

    def diff(a, b):
        return [float((a[i] - b[i]).abs().max()) for i in range(a.shape[0])]

    for p in (0.0, 1.0):
        arch.structure_cfg["action_video_dropout"] = p
        a0, l0 = action_pred(arch, inputs, 11, perturb=False)
        a1, l1 = action_pred(arch, inputs, 11, perturb=True)
        report[f"p={p}"] = {"max_abs_action_change_per_sample": diff(a0, a1), "losses": l0}
    arch.structure_cfg["action_video_dropout"] = 0.0

    original = arch._video_drop
    arch._video_drop = lambda batch, device: torch.tensor([True, False], device=device)
    a0, _ = action_pred(arch, inputs, 11, perturb=False)
    a1, _ = action_pred(arch, inputs, 11, perturb=True)
    report["mixed [drop, keep]"] = {"max_abs_action_change_per_sample": diff(a0, a1)}
    arch._video_drop = original

    arch.eval()
    arch.inference_cfg["structure_only_action"] = True
    report["eval structure_only_action"] = arch._video_drop(3, "cpu").tolist()
    arch.inference_cfg["structure_only_action"] = False
    report["eval default"] = arch._video_drop(3, "cpu")
    arch.train()
    report["tokens_per_camera_default"] = arch.structure_expert.tokenizer.tokens_per_camera
    del arch
    torch.cuda.empty_cache()

    tmp = Path(tempfile.mkdtemp(prefix="a3_ckpt_", dir="/cache" if os.path.isdir("/cache") else None))
    for entry in CKPT.iterdir():
        if entry.name != "config.yaml":
            os.symlink(entry, tmp / entry.name)
    cfg3 = OmegaConf.load(CKPT / "config.yaml")
    cfg3.model.architecture.structure_expert.groups = ["map", "keypoint", "word"]
    OmegaConf.save(cfg3, tmp / "config.yaml")
    try:
        arch3 = load(tmp)
        tok = arch3.structure_expert.tokenizer
        with torch.no_grad():
            features = arch3.student.encode({c: torch.from_numpy(np.stack([s["structure_frames"][c] for s in samples]))
                                             for c in ("head", "wrist")}, [s["prompt"] for s in samples])
            out = tok(features, 2)
        _, l3 = action_pred(arch3, arch3.prepare_inputs(samples), 11, perturb=False)
        report["A3 groups"] = {"groups": list(tok.groups), "tokens_per_camera": tok.tokens_per_camera,
                               "token_tensor": list(out["tokens"].shape), "losses": l3}
    finally:
        shutil.rmtree(tmp)
    report["eval default"] = None if report["eval default"] is None else "unexpected"
    print("REPORT", json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
