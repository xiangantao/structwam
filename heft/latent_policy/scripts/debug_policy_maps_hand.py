import sys, json, numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import check_policy_maps as c
from torch.nn import functional as F
from pathlib import Path
from latent_dp.policy_maps import present_k, PRESENT, STEPS, STILL_MIN, STILL_MARGIN, segments
W = Path("/home/ma-user/work/2337888765/workspace/runs"); root = W / "viz_inputs_20261015/runs/vace_tracking_all_suites_20261004"
FD = W / "eval_mainline_20261016/main_s2000/heldout/fields/main_s2000"
rows = [json.loads(l) for l in open(W / "vace_teacher_cache_small_20261015/manifest_heldout.jsonl")]
def wrist_hand(f):
    k = F.normalize(f["track_k"], dim=-1)
    still = (k * k[PRESENT:PRESENT + 1]).sum(-1)[[i for i in range(STEPS) if i != PRESENT]].mean(0).numpy()
    level = float(np.median(still)); hand = np.zeros(1024, bool)
    for part in segments(f["seg_k_l22h9"], f["seg_k_l15h7"], True):
        if part["mask"].sum() >= 3 and still[part["mask"]].mean() > max(STILL_MIN, level + STILL_MARGIN):
            hand |= part["mask"]
    return hand
protos, pair = [], {}
for row in rows:
    if "image2" not in row["camera"]: continue
    f = c.load_fields(FD / f"{row['sample_id']}.pt")
    h = wrist_hand(f)
    if h.sum() < 3: continue
    for name in ("seg", "track"):
        pass
    kseg = present_k(f["seg_k_l22h9"], f["seg_k_l15h7"]).numpy()
    ktr = F.normalize(f["track_k"][PRESENT], dim=-1).numpy()
    pair[row["sample_id"].replace("image2", "image")] = (kseg[h].mean(0), ktr[h].mean(0))
    protos.append((kseg[h].mean(0), ktr[h].mean(0)))
gseg = np.mean([p[0] for p in protos], 0); gtr = np.mean([p[1] for p in protos], 0)
res = {k: [] for k in ("seg_global", "seg_pair", "track_global", "change")}
for row in rows:
    if "image2" in row["camera"]: continue
    gt = c.ground_truth(row, root)
    if gt is None or not gt["hand"]: continue
    f = c.load_fields(FD / f"{row['sample_id']}.pt")
    kseg = present_k(f["seg_k_l22h9"], f["seg_k_l15h7"]).numpy()
    ktr = F.normalize(f["track_k"], dim=-1)
    change = 1 - 0.5 * ((ktr[PRESENT] * ktr[PRESENT - 1]).sum(-1) + (ktr[PRESENT] * ktr[PRESENT + 1]).sum(-1)).numpy()
    maps = {"seg_global": kseg @ gseg, "track_global": ktr[PRESENT].numpy() @ gtr, "change": change}
    if row["sample_id"] in pair: maps["seg_pair"] = kseg @ pair[row["sample_id"]][0]
    hand_gt = c.dilate(gt["hand"]); n = max(len(gt["hand"]), 10)
    for name, m in maps.items():
        top = np.argsort(m)[-n:]
        res[name].append((float(hand_gt[int(np.argmax(m))]), float(hand_gt[top].mean()), float(c.dilate(gt["things"])[top].mean())))
for name, v in res.items():
    v = np.array(v)
    print(name, "windows", len(v), "argmax on gripper %.2f" % v[:, 0].mean(), "top-N on gripper %.2f" % v[:, 1].mean(), "top-N on things %.2f" % v[:, 2].mean())
print("chance gripper cells fraction")
