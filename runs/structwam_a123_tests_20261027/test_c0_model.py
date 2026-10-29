"""c0 model check on one NPU: s40000 weights with the side-by-side canvas (256x512) and the c0 aug.
compute_loss runs and is finite; a canvas/model layout mismatch raises; wrist tokens sit in the right half."""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import torch
from omegaconf import OmegaConf

from openwam.dataloader import build_dataset
from openwam.deploy.model_loader import load_from_checkpoint_dir

CK = Path("/home/ma-user/work/2337888765/workspace/runs/structwam_step40000_eval_ckpt")
NAME = "checkpoint_step_40000.safetensors"


def ckpt_with(cfg_overrides):
    tmp = Path(tempfile.mkdtemp(prefix="c0_ckpt_"))
    for entry in CK.iterdir():
        if entry.name != "config.yaml":
            os.symlink(entry, tmp / entry.name)
    cfg = OmegaConf.merge(OmegaConf.load(CK / "config.yaml"), cfg_overrides)
    OmegaConf.save(cfg, tmp / "config.yaml")
    return tmp, cfg


def main():
    report = {}
    sbs = {"dataloader": {"canvas_layout": "side_by_side", "height": 256, "width": 512,
                          "structure_augment": {"enabled": True, "rotate_deg": 5.0, "scale_max": 1.1, "brightness": 0.3,
                                                "contrast": 0.4, "saturation": 0.5, "hue": 0.05, "blur_prob": 0.0,
                                                "independent_color": True}}}
    for tag, model_layout in (("matched", "side_by_side"), ("mismatched", "lshape")):
        tmp, cfg = ckpt_with({**sbs, "model": {"architecture": {"structure": {"canvas_layout": model_layout}}}})
        try:
            _, arch = load_from_checkpoint_dir(str(tmp), device="cuda:0", ckpt_name=NAME)
            arch.normalizer = None
            arch.init_training_schedulers(1000)
            arch.train()
            arch.requires_grad_(False)
            ds = build_dataset(cfg.dataloader, split="train")
            samples = [ds[i] for i in (1234, 98765)]
            with torch.no_grad():
                inputs = arch.prepare_inputs(samples)
                inputs["use_gradient_checkpointing"] = False
                try:
                    result = arch.compute_loss(**inputs, lambda_video=1.0, lambda_action=1.0)
                    report[tag] = {k: round(float(v), 5) for k, v in result.items() if k.startswith("loss")}
                    report[tag]["latent shape"] = list(inputs["input_latents"].shape)
                    report[tag]["layout tiles"] = [list(t) for t in arch.layout.tiles]
                except ValueError as e:
                    report[tag] = "ValueError: " + str(e)[:200]
            del arch
            torch.cuda.empty_cache()
        finally:
            shutil.rmtree(tmp)
    print("REPORT", json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
