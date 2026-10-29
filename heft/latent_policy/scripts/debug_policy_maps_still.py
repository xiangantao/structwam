import sys, json, numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import check_policy_maps as c
from torch.nn import functional as F
from pathlib import Path
W = Path("/home/ma-user/work/2337888765/workspace/runs"); root = W / "viz_inputs_20261015/runs/vace_tracking_all_suites_20261004"
rows = [json.loads(l) for l in open(W / "vace_teacher_cache_small_20261015/manifest_heldout.jsonl")]
for row in [r for r in rows if "image2" in r["camera"]][::4]:
    sid = row["sample_id"]
    f = c.load_fields(W / "eval_mainline_20261016/main_s2000/heldout/fields/main_s2000" / f"{sid}.pt")
    for key in ("track_k",):
        k = F.normalize(f[key], dim=-1)
        sim = (k * k[8:9]).sum(-1)
        score = sim[[0, 2, 4, 6, 7, 9, 10, 11, 12]].mean(0).numpy()
    gt = c.ground_truth(row, root)
    if gt is None:
        continue
    hand = sorted(gt["hand"]); other = sorted(set(range(1024)) - gt["hand"])
    hs = score[hand] if hand else np.zeros(1)
    print(sid, "med all %.3f" % np.median(score), "gt hand cells", len(hand), "hand med %.3f" % np.median(hs),
          "other med %.3f" % np.median(score[other]), "hand>0.85 %.2f" % (hs > 0.85).mean(), "other>0.85 %.2f" % (score[other] > 0.85).mean(),
          "hand>0.9 %.2f" % (hs > 0.9).mean(), "other>0.9 %.2f" % (score[other] > 0.9).mean())
