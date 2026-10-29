"""B1 augmentation checks on CPU: off = byte-identical to the backed-up reader; on = one draw shared by
the canvas video (head tile) and the student's history, no exposed border; writes a preview image."""

import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

W = Path("/home/ma-user/work/2337888765/workspace")
OUT = W / "runs/structwam_a123_tests_20261027"
cfg = OmegaConf.load(W / "runs/structwam_step40000_eval_ckpt/config.yaml").dataloader

from openwam.dataloader.libero_struct import LiberoStructDataset as New

spec = importlib.util.spec_from_file_location(
    "old_libero_struct", W / "backups/structwam_before_b1_20261027/openwam/dataloader/libero_struct.py")
old_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_mod)
Old = old_mod.LiberoStructDataset


def arrays(sample):
    out = {"video": np.stack([np.asarray(f) for f in sample["video"]]),
           "first": np.asarray(sample["first_frame_image"][0]),
           "action": np.asarray(sample["action"]), "proprio": np.asarray(sample["proprio"])}
    for cam, clip in sample["structure_frames"].items():
        out[f"hist_{cam}"] = clip
    return out


report = {}
idx = [1234, 98765, 40000]
old_ds = Old.from_config(cfg, split="train")
new_ds = New.from_config(cfg, split="train")
for i in idx:
    random.seed(0); a = arrays(old_ds[i])
    random.seed(0); b = arrays(new_ds[i])
    report[f"off idx{i} identical"] = all(np.array_equal(a[k], b[k]) for k in a)

cfg_on = OmegaConf.merge(cfg, {"structure_augment": {"enabled": True}})
aug_ds = New.from_config(cfg_on, split="train")
rows = []
for i in idx:
    random.seed(i)
    t0 = time.time(); s = aug_ds[i]; dt = time.time() - t0
    random.seed(0); base = new_ds[i]
    canvas = np.asarray(s["video"][0]).astype(np.float32)
    head_tile = canvas[:256, :320]
    hist_now = s["structure_frames"]["head"][-1]
    hist_plain = base["structure_frames"]["head"][-1]
    tile_from_hist = np.asarray(Image.fromarray(hist_now).resize((320, 256), Image.BILINEAR)).astype(np.float32)
    tile_from_plain = np.asarray(Image.fromarray(hist_plain).resize((320, 256), Image.BILINEAR)).astype(np.float32)
    corr = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    border = np.concatenate([hist_now[0], hist_now[-1], hist_now[:, 0], hist_now[:, -1]])
    rows.append({"idx": i, "seconds": round(dt, 2),
                 "corr(head tile, augmented history)": round(corr(head_tile, tile_from_hist), 4),
                 "corr(head tile, plain history)": round(corr(head_tile, tile_from_plain), 4),
                 "first frame == video[0]": bool(np.array_equal(np.asarray(s["first_frame_image"][0]), np.asarray(s["video"][0]))),
                 "history border near-black px": int((border.max(-1) < 8).sum()),
                 "plain border near-black px": int((np.concatenate([hist_plain[0], hist_plain[-1], hist_plain[:, 0], hist_plain[:, -1]]).max(-1) < 8).sum())})
    if i == idx[0]:
        tiles = [np.asarray(base["video"][0]), np.asarray(s["video"][0])]
        hist = np.concatenate([np.asarray(Image.fromarray(x).resize((320, 320))) for x in
                               (hist_plain, hist_now, base["structure_frames"]["wrist"][-1], s["structure_frames"]["wrist"][-1])], 1)
        Image.fromarray(np.concatenate([np.concatenate(tiles, 1), np.zeros((384, 0, 3), np.uint8)], 1)).save(OUT / "b1_preview_canvas.png")
        Image.fromarray(hist).save(OUT / "b1_preview_history.png")
report["on"] = rows
print("REPORT", json.dumps(report, indent=1), flush=True)
