"""Offline check of the 'online calibration' idea on LIBERO demos (no training).

Per window ending at present t (student run on the 33 frames ending at t and at t+16):
  image signal  v = [head hand-keypoint centroid velocity (2), wrist object-keypoint centroid velocity (2)]
                over 4-frame intervals, in 32-grid cells
  action signal A = sum of the 4 raw delta-xyz actions of the interval
Fit J (4x3, ridge) on the 8 past intervals (-32..0, tracked) and test on the 4 future intervals (0..+16):
  forward : v_fut ~ J A_fut                       (does the local linear map hold?)
  inverse : A_hat = J^+ v_fut, v_fut from the real future track (t+16 window) or the student's imagination
Baselines for A_fut: zero, and repeat the last past interval's action.
"""

import json
import random
import sys
from pathlib import Path

import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import numpy as np
import torch
from omegaconf import OmegaConf

from openwam.dataloader import build_dataset
from openwam.structure.layout import KeypointLayout
from openwam.structure.student import StructureStudent, StudentConfig

W = Path("/home/ma-user/work/2337888765/workspace")
CK = OmegaConf.load(W / "runs/structwam_step40000_eval_ckpt/config.yaml")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OUT = W / "runs/structwam_a123_tests_20261027/calib_check.json"

if __name__ == "__main__":
    ds = build_dataset(CK.dataloader, split="train")
    student = StructureStudent(StudentConfig(**OmegaConf.to_container(CK.model.architecture.structure.student)), "cuda:0")


    def centroid(kp, role):
        """kp [32,57] -> per-step centroid [13,2] of valid points of a role (nan where none)."""
        p = KeypointLayout.split(torch.as_tensor(kp)[None])
        xy, valid, r = p["xy"][0].numpy(), p["valid"][0].numpy(), p["role"][0].numpy()
        out = np.full((13, 2), np.nan)
        for s in range(13):
            m = (r == role) & valid[:, s]
            if m.sum() >= 2:
                out[s] = xy[m, s].mean(0)
        return out


    def encode(frames, prompt):
        f = student.encode({c: torch.from_numpy(frames[c][None]) for c in ("head", "wrist")}, [prompt])
        return {c: f[c]["keypoints"][0].cpu().numpy() for c in ("head", "wrist")}


    random.seed(0)
    rows, tried = [], 0
    eps_actions = {}
    while len(rows) < N and tried < N * 6:
        tried += 1
        idx = random.randrange(len(ds))
        ep = int(np.searchsorted(ds._cum_n_starts, idx, side="right") - 1)
        t = int(ds._ep_valid_start[ep]) + (idx - int(ds._cum_n_starts[ep])) * ds._window_stride
        if t - 32 < int(ds._ep_valid_start[ep]) or t + 16 >= int(ds._ep_valid_end[ep]):
            continue
        row = ds._eps_df.iloc[ep]
        if ep not in eps_actions:
            table = ds._load_data_table(int(row["data/chunk_index"]), int(row["data/file_index"]))
            win = table.slice(int(ds._ep_data_row_offset[ep]), int(ds._ep_valid_end[ep])).to_pandas()
            eps_actions[ep] = ds._raw_action_eef10(win)[:, :3]
        acts = eps_actions[ep]
        sample = ds[idx]
        now = encode(sample["structure_frames"], sample["prompt"])
        fut_frames = {c: ds._decode_history(cam, row, ep, t + 16, kind=c)
                      for c, cam in (("head", ds._head_camera), ("wrist", ds._left_wrist_camera))}
        fut = encode(fut_frames, sample["prompt"])
        # centroid tracks: steps 0..8 = frames -32..0 (tracked), 9..12 = +4..+16 (imagined)
        h_now, w_now = centroid(now["head"], 0), centroid(now["wrist"], 1)
        h_fut, w_fut = centroid(fut["head"], 0), centroid(fut["wrist"], 1)   # fut steps 4..8 = frames t..t+16 tracked
        sig_now = np.concatenate([h_now, w_now], 1)          # [13,4]
        sig_fut = np.concatenate([h_fut, w_fut], 1)
        v_past = np.diff(sig_now[0:9], axis=0)               # 8 intervals
        v_true = np.diff(sig_fut[4:9], axis=0)               # 4 future intervals (tracked)
        v_imag = np.diff(sig_now[8:13], axis=0)              # 4 future intervals (student imagination)
        A = np.stack([acts[t - 32 + 4 * k: t - 28 + 4 * k].sum(0) for k in range(12)])   # 8 past + 4 future
        A_past, A_fut = A[:8], A[8:]
        if np.isnan(v_past).any() or np.isnan(v_true).any() or np.isnan(v_imag).any():
            continue
        lam = 1e-3 * np.trace(A_past.T @ A_past) / 3 + 1e-9
        J = v_past.T @ A_past @ np.linalg.inv(A_past.T @ A_past + lam * np.eye(3))     # 4x3
        Jp = np.linalg.pinv(J)
        rows.append({"idx": idx, "task": sample["prompt"],
                     "A_fut": A_fut.tolist(), "A_last": A_past[-1].tolist(),
                     "v_true": v_true.tolist(), "v_imag": v_imag.tolist(), "v_last": v_past[-1].tolist(),
                     "v_fwd": (A_fut @ J.T).tolist(),
                     "A_inv_true": (v_true @ Jp.T).tolist(), "A_inv_imag": (v_imag @ Jp.T).tolist(),
                     "J_cond": float(np.linalg.cond(J))})
        if len(rows) % 25 == 0:
            print("windows", len(rows), flush=True)


    def stats(pred, true, d=3):
        pred, true = np.asarray(pred).reshape(-1, d), np.asarray(true).reshape(-1, d)
        err = ((pred - true) ** 2).sum(1).mean()
        var = ((true - true.mean(0)) ** 2).sum(1).mean()
        cos = (pred * true).sum(1) / (np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1) + 1e-9)
        return {"mse": float(err), "R2": float(1 - err / var), "cos_median": float(np.median(cos))}


    A_fut = [r["A_fut"] for r in rows]
    rep = {"windows": len(rows),
           "forward: v_fut from J*A_fut": stats([r["v_fwd"] for r in rows], [r["v_true"] for r in rows], 4),
           "forward baseline: v_fut = last past velocity": stats([[r["v_last"]] * 4 for r in rows], [r["v_true"] for r in rows], 4),
           "inverse (true future track)": stats([r["A_inv_true"] for r in rows], A_fut),
           "inverse (imagined future track)": stats([r["A_inv_imag"] for r in rows], A_fut),
           "baseline: repeat last past action": stats([[r["A_last"]] * 4 for r in rows], A_fut),
           "baseline: zero": stats(np.zeros((len(rows), 4, 3)), A_fut),
           "J condition number median": float(np.median([r["J_cond"] for r in rows]))}
    v_true = np.asarray([r["v_true"] for r in rows]).reshape(-1, 4)
    v_fwd = np.asarray([r["v_fwd"] for r in rows]).reshape(-1, 4)
    rep["forward R2 per channel (head x,y, wrist x,y)"] = [float(1 - ((v_fwd[:, i] - v_true[:, i]) ** 2).mean() / v_true[:, i].var()) for i in range(4)]
    OUT.write_text(json.dumps({"report": rep, "rows": rows}))
    print("REPORT", json.dumps(rep, indent=1), flush=True)
