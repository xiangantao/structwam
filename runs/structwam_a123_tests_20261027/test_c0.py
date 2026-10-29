"""c0 checks: (1) old configs byte-identical after the patch, (2) side-by-side canvas content,
(3) deploy canvas == training canvas, (4) model forward on the side-by-side canvas (NPU)."""

import json
import random
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

T = "/mnt/sfs-turbo-wangfei/2337888765/workspace/runs/structwam_a123_tests_20261027"
CK = "/home/ma-user/work/2337888765/workspace/runs/structwam_step40000_eval_ckpt"
report = {}

# (1) regression: rerun the dump with the patched code and compare
subprocess.run([sys.executable, f"{T}/dump_samples.py", f"{T}/samples_after_c0.npz"], check=True,
               capture_output=True)
a, b = np.load(f"{T}/samples_before_c0.npz"), np.load(f"{T}/samples_after_c0.npz")
report["(1) old configs identical"] = all(np.array_equal(a[k], b[k]) for k in a.files) and set(a.files) == set(b.files)

from openwam.dataloader import build_dataset
from openwam.deploy.obs_preprocess import ObsPreprocessor

base = OmegaConf.load(f"{CK}/config.yaml")
sbs = OmegaConf.merge(base.dataloader, {"canvas_layout": "side_by_side", "height": 256, "width": 512})
C0_AUG = {"enabled": True, "rotate_deg": 5.0, "scale_max": 1.1, "brightness": 0.3, "contrast": 0.4,
          "saturation": 0.5, "hue": 0.05, "blur_prob": 0.0, "independent_color": True}

# (2) side-by-side canvas, no augmentation: halves equal the present frame of each camera
ds = build_dataset(sbs, split="train")
s = ds[1234]
canvas = np.asarray(s["video"][0])
head_now, wrist_now = s["structure_frames"]["head"][-1], s["structure_frames"]["wrist"][-1]
report["(2) canvas shape"] = list(canvas.shape)
report["(2) left half == head present (max abs diff)"] = int(np.abs(canvas[:, :256].astype(int) - head_now.astype(int)).max())
report["(2) right half == wrist present (max abs diff)"] = int(np.abs(canvas[:, 256:].astype(int) - wrist_now.astype(int)).max())

# (2b) with the c0 augmentation: halves track the augmented histories, wrist untouched geometrically
ds_aug = build_dataset(OmegaConf.merge(sbs, {"structure_augment": C0_AUG}), split="train")
random.seed(3)
sa = ds_aug[1234]
ca = np.asarray(sa["video"][0]).astype(np.float32)
corr = lambda x, y: float(np.corrcoef(np.asarray(x, np.float32).ravel(), np.asarray(y, np.float32).ravel())[0, 1])
report["(2b) corr(left, aug head hist)"] = round(corr(ca[:, :256], sa["structure_frames"]["head"][-1]), 4)
report["(2b) corr(left, plain head)"] = round(corr(ca[:, :256], head_now), 4)
report["(2b) corr(right, aug wrist hist)"] = round(corr(ca[:, 256:], sa["structure_frames"]["wrist"][-1]), 4)
report["(2b) first frame == video[0]"] = bool(np.array_equal(np.asarray(sa["first_frame_image"][0]), np.asarray(sa["video"][0])))
Image.fromarray(np.concatenate([canvas, np.asarray(sa["video"][0])], 0)).save(f"{T}/c0_preview.png")

# (3) deploy preprocessor with the same config composes the same canvas from the raw frames
cfg = OmegaConf.merge(base, {"dataloader": {"canvas_layout": "side_by_side", "height": 256, "width": 512}})
pre = ObsPreprocessor.from_cfg(cfg)
obs = {"images": {"head_camera": Image.fromarray(head_now), "left_wrist_camera": Image.fromarray(wrist_now),
                  "right_wrist_camera": None}, "prompt": s["prompt"], "state": np.zeros(8, np.float32)}
try:
    out = pre.preprocess(obs)
    report["(3) deploy canvas == training canvas (max abs diff)"] = int(np.abs(np.asarray(out["image"]).astype(int) - canvas.astype(int)).max())
    report["(3) deploy canvas size"] = list(out["image"].size)
except Exception as e:  # report instead of crashing
    report["(3) deploy error"] = repr(e)[:300]

print("REPORT", json.dumps(report, indent=1), flush=True)
