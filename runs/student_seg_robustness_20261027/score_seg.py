"""Emergent-segmentation robustness of the StructWAM student vs raw V-JEPA2 (Burns et al. style).

For each rendered LIBERO-plus scene (``render_seg.py``) and camera, the static frame is repeated
to a 33-frame history (as at the first deploy step) and these features are read at the present:

    student_k15   student partition K, head L15H7   [32x32, 128]   (what StructWAM's maps use)
    student_k22   student partition K, head L22H9   [32x32, 128]
    student_trunk student fine-tuned V-JEPA2 trunk  [16x16, 1024]  (StructWAM trunk tokens)
    vjepa_raw     official V-JEPA2 ViT-L, no text   [16x16, 1024]  (the student's init)

Scores against the simulator instance segmentation (majority label per grid cell; robot parts
merged; cells under 50 % purity ignored).  Ground truth is used for scoring only.

    knn_purity    for object/robot cells, share of the 8 cosine-nearest cells with the same label
    cluster_miou  k-means (k = labels present, background included), Hungarian-matched IoU,
                  mean over object/robot labels  (the Jaccard-style score of Burns et al.)
    separation    mean within-label cosine - mean between-label cosine (object/robot cells)

    python score_seg.py --scenes DIR --output DIR
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

W = Path("/home/ma-user/work/2337888765/workspace")
STUDENT = W / "runs/mainline_20261016/g2_full/step_0002000.pt"
CLEAN = W / "runs/libero_plus_structwam_s24000_v10"
PRESENT = 8
FEATURES = ("student_k15", "student_k22", "student_k15_16", "student_k22_16", "student_trunk", "vjepa_raw")
CAMERAS = ("head", "wrist")


def grid_labels(seg: np.ndarray, instances: list, grid: int) -> np.ndarray:
    """[256, 256] instance ids (0 = background, i + 1 = instances[i]) -> [grid*grid] labels, -1 = ignore.
    Robot parts merge into one label; labels: 0 background, 1 robot, 2.. objects."""
    remap = np.zeros(len(instances) + 2, np.int64)
    for i, name in enumerate(instances):
        remap[i + 1] = 1 if ("Panda" in name or "Mount" in name) else i + 2
    lab = remap[np.clip(seg, 0, len(instances) + 1)]
    block = seg.shape[0] // grid
    cells = lab.reshape(grid, block, grid, block).transpose(0, 2, 1, 3).reshape(grid * grid, -1)
    out = np.full(grid * grid, -1, np.int64)
    for c, values in enumerate(cells):
        counts = np.bincount(values)
        if counts.max() >= 0.5 * len(values):
            out[c] = counts.argmax()
    return out


def scores(feat: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict:
    valid = labels >= 0
    x = feat[valid].astype(np.float64)
    y = labels[valid]
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    fg = y >= 1
    present = np.unique(y)
    if fg.sum() < 4 or len(present) < 3:
        return {}
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)
    sizes = {l: int((y == l).sum()) for l in present}
    per_label = []
    for l in present:
        k = min(8, sizes[l] - 1)            # an object of n cells has at most n - 1 same-label neighbours
        if l < 1 or k < 1:
            continue
        rows = np.flatnonzero(y == l)
        per_label.append((y[order[rows, :k]] == l).mean())
    if not per_label:
        return {}
    np.fill_diagonal(sim, 1.0)
    same = y[:, None] == y[None, :]
    fgm = fg[:, None] & fg[None, :]
    off = ~np.eye(len(y), dtype=bool)
    within = sim[same & fgm & off].mean() if (same & fgm & off).any() else np.nan
    between = sim[~same & fgm].mean() if (~same & fgm).any() else np.nan
    k = len(present)
    pred = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(x)
    iou = np.zeros((k, k))
    for i, l in enumerate(present):
        for j in range(k):
            a, b = y == l, pred == j
            iou[i, j] = (a & b).sum() / max((a | b).sum(), 1)
    rows, cols = linear_sum_assignment(-iou)
    matched = {present[r]: iou[r, c] for r, c in zip(rows, cols)}
    return {"knn_purity": float(np.mean(per_label)),
            "cluster_miou": float(np.mean([matched[l] for l in present if l >= 1])),
            "separation": float(within - between)}


class Models:
    def __init__(self, device):
        sys.path[:0] = [str(W / "heft/latent_policy/src"), str(W / "heft")]
        from latent_dp.online import OnlineMaps
        from vace_latent_distill.config import ModelConfig
        from vace_latent_distill.model import load_backbone

        self.online = OnlineMaps(STUDENT, None, device, workers=2)
        self.student = self.online.student
        saved = torch.load(str(STUDENT), map_location="cpu", weights_only=False, mmap=True)
        self.raw = load_backbone(ModelConfig(**saved["config"]["model"]), pretrained=True).eval().requires_grad_(False).to(device)
        del saved
        self.device = device
        sys.path.insert(0, str(CLEAN))
        from clean_instructions import clean

        base = json.loads((CLEAN / "libero_base_instructions.json").read_text())
        self.clean = lambda s: clean(s, base)

    @torch.no_grad()
    def raw_trunk(self, rgb):
        """Official V-JEPA2 on the same 34-frame tubelet input the student builds -> present [B,16,16,1024]."""
        pixels = torch.cat((rgb, rgb[:, -1:]), 1).permute(0, 2, 1, 3, 4).float() / 255
        pixels = (pixels - self.student.rgb_mean) / self.student.rgb_std
        x = self.raw.patch_embed(pixels)
        t, h, w = pixels.shape[2] // 2, pixels.shape[3] // 16, pixels.shape[4] // 16
        for block in self.raw.blocks:
            x = block(x, mask=None, attn_mask=None, T=t, H_patches=h, W_patches=w)
        return self.raw.norm(x).reshape(len(pixels), t, h, w, -1)[:, -1]

    @torch.no_grad()
    def features(self, frames, instruction):
        """frames [N, 256, 256, 3] uint8 (one per camera) -> {feature: [N, cells, dim] numpy}."""
        rgb = torch.from_numpy(frames).permute(0, 3, 1, 2)[:, None].repeat(1, 33, 1, 1, 1).to(self.device)
        text, mask = self.online.text_for([instruction] * len(frames))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.student(rgb, text, mask, observed_heads=("tracking", "semantic"))
        fut = out["future"]
        res = {"student_k15": fut["seg_k_l15h7"][:, PRESENT], "student_k22": fut["seg_k_l22h9"][:, PRESENT],
               "student_trunk": out["trunk"][:, -1].reshape(len(frames), -1, out["trunk"].shape[-1])}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            res["vjepa_raw"] = self.raw_trunk(rgb).reshape(len(frames), -1, 1024)
        for name in ("student_k15", "student_k22"):   # same 16x16 grid as the trunks: 2x2 mean pool
            k = res[name].float().reshape(len(frames), 32, 32, -1).permute(0, 3, 1, 2)
            res[f"{name}_16"] = F.avg_pool2d(k, 2).permute(0, 2, 3, 1).reshape(len(frames), 256, -1)
        return {k: v.float().reshape(len(frames), -1, v.shape[-1]).cpu().numpy() for k, v in res.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    models = Models("cuda:0")
    files = sorted(Path(args.scenes).glob("*.npz"))[: args.limit or None]
    rows = []
    for n, f in enumerate(files):
        meta = json.loads(f.with_suffix(".json").read_text())
        data = np.load(f)
        instruction = models.clean(meta["language"])
        frames = np.stack([data[f"{c}_rgb"] for c in CAMERAS])
        feats = models.features(frames, instruction)
        for ci, camera in enumerate(CAMERAS):
            seg = data[f"{camera}_seg"]
            for name in FEATURES:
                feat = feats[name][ci]
                grid = int(round(np.sqrt(feat.shape[0])))
                s = scores(feat, grid_labels(seg, meta["instances"], grid))
                if s:
                    rows.append({"scene": f.stem, "category": meta["category"], "suite": meta["suite"],
                                 "camera": camera, "feature": name, "grid": grid, **s})
        print(f"[{n + 1}/{len(files)}] {f.stem} {meta['category']} | {instruction}", flush=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    table = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ("knn_purity", "cluster_miou", "separation"):
            table[(r["camera"], r["feature"], m)][r["category"]].append(r[m])
    summary = {f"{c}|{feat}|{m}": {cat: [float(np.mean(v)), len(v)] for cat, v in sorted(d.items())}
               for (c, feat, m), d in sorted(table.items())}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("COMPLETE", flush=True)
    models.online.close()


if __name__ == "__main__":
    main()
