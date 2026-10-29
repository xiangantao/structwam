import sys, json, numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import check_policy_maps as c
from torch.nn import functional as F
from pathlib import Path
from latent_dp.policy_maps import PRESENT, segments, distal_part
W = Path("/home/ma-user/work/2337888765/workspace/runs"); root = W / "viz_inputs_20261015/runs/vace_tracking_all_suites_20261004"
FD = W / "eval_mainline_20261016/main_s2000/heldout/fields/main_s2000"
rows = [json.loads(l) for l in open(W / "vace_teacher_cache_small_20261015/manifest_heldout.jsonl")]
data = []
for row in rows:
    if "image2" in row["camera"]: continue
    gt = c.ground_truth(row, root)
    if gt is None or not gt["hand"]: continue
    f = c.load_fields(FD / f"{row['sample_id']}.pt")
    k = F.normalize(f["track_k"], dim=-1)
    sim = lambda a, b: (k[a] * k[b]).sum(-1).numpy()
    ch1 = 1 - 0.5 * (sim(PRESENT, PRESENT - 1) + sim(PRESENT, PRESENT + 1))
    chp = 1 - sim(PRESENT, PRESENT - 1)
    ch2 = 1 - 0.5 * (sim(PRESENT, PRESENT - 2) + sim(PRESENT, PRESENT + 2))
    parts = segments(f["seg_k_l22h9"], f["seg_k_l15h7"], False)
    data.append((gt, {"ch1": ch1, "past1": chp, "ch2": ch2}, parts))
def score(sets):
    p, r, t, n = [], [], [], []
    for (gt, _, _), s in zip(data, sets):
        hand = c.dilate(gt["hand"]); things = c.dilate(gt["things"])
        if s.any():
            p.append(hand[s].mean()); t.append(things[s].mean())
        r.append(c.dilate(np.flatnonzero(s))[sorted(gt["hand"])].mean() if s.any() else 0.0)
        n.append(s.sum())
    return "prec %.2f recall %.2f things %.2f cells %.0f nonempty %d/%d" % (np.mean(p) if p else 0, np.mean(r), np.mean(t) if t else 0, np.mean(n), len(p), len(data))
for name in ("ch1", "past1", "ch2"):
    for tau in (0.15, 0.25, 0.35, 0.45):
        cells = [m[name] > tau for _, m, _ in data]
        print(name, tau, "cells:", score(cells))
        seg = []
        for (gt, m, parts) in data:
            s = np.zeros(1024, bool)
            for part in parts:
                if not part["background"] and m[name][part["mask"]].mean() > tau:
                    s |= part["mask"]
            seg.append(s)
        print(name, tau, "segments:", score(seg))
        print(name, tau, "cells+distal:", score([distal_part(s) for s in cells]))
    # relative threshold
    for z in (2.0, 3.0, 4.0):
        cells = []
        for _, m, _ in data:
            v = m[name]; med = np.median(v); mad = np.median(np.abs(v - med)) + 1e-6
            cells.append(v > med + z * 1.4826 * mad)
        print(name, "z", z, "cells:", score(cells))
