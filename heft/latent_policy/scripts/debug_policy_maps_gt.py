import json, sys
from pathlib import Path
import numpy as np, torch, cv2
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import check_policy_maps as c
from latent_dp.policy_maps import window_maps, role_tokens
from transformers import AutoTokenizer
from vace_latent_distill.config import PipelineConfig
W = Path("/home/ma-user/work/2337888765/workspace/runs")
root = W / "viz_inputs_20261015/runs/vace_tracking_all_suites_20261004"
rows = [json.loads(l) for l in open(W / "policy_maps_check_20261018/manifest_test.jsonl")]
tok = AutoTokenizer.from_pretrained(Path(PipelineConfig.load(W / "vace_teacher_cache_small_20261015/pipeline.json").teacher.text_root) / "tokenizer", local_files_only=True)
words = json.load(open(W / "policy_features_20261017/words.json"))
out = W / "policy_maps_check_20261018/debug"; out.mkdir(exist_ok=True)
for row in rows:
    gt = c.ground_truth(row, root)
    frame = cv2.cvtColor(c.present_frame(row), cv2.COLOR_RGB2BGR)
    img = cv2.resize(frame, None, fx=2, fy=2)
    tr = gt["tracks"]; f = gt["present"]; vis = tr["visible"][f]
    for i in np.flatnonzero(vis)[::3]:
        lab = tr["point_label"][i]; body = gt["point_body"][i]
        col = (0, 0, 255) if lab in c.HAND else (0, 255, 0) if body == gt["target_name"] else (255, 120, 0) if body == gt["destination_name"] else (200, 200, 200)
        cv2.circle(img, tuple(np.round(tr["xy"][f, i] * 2).astype(int)), 1, col, -1)
    cv2.imwrite(str(out / f"{row['sample_id']}_gt.png"), img)
    fields = c.load_fields(W / "eval_mainline_20261016/main_s2000/heldout/fields/main_s2000" / f"{row['sample_id']}.pt")
    tokens, readable = role_tokens(row["task"], tok, words.get(row["task"]))
    wrist = "image2" in row["camera"]
    maps, kp, d = window_maps(fields, tokens, wrist)
    cam = np.linalg.norm(d["camera"], axis=-1).max(0)
    disp = np.linalg.norm(np.where(d["dense"]["valid"][..., None], d["displacement"], 0), axis=-1).max(0)
    print(row["sample_id"], "wrist" if wrist else "third", "target", gt["target_name"], "dest", gt["destination_name"],
          "reliable", int(d["reliable"].sum()), "valid+1", int(d["dense"]["valid"][9].sum()), "valid-1", int(d["dense"]["valid"][7].sum()),
          "thr", round(d["threshold"], 2), "moving", int(d["moving"].sum()), "cam max", round(float(cam.max()), 2),
          "still(<0.4)", int((d["reliable"] & (disp < 0.4)).sum()), "fb median", np.round(np.median(d["dense"]["fb"][[7, 9]], 1), 2),
          "distinct med", np.round(np.median(d["dense"]["distinct"][9]), 3))
