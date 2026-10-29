import sys, json, numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import check_policy_maps as c
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from vace_latent_distill.config import PipelineConfig
from vace_latent_distill.offline import unpack_target
from vace_latent_distill.teacher import content_token_masks
from latent_dp.policy_maps import role_tokens
W = Path("/home/ma-user/work/2337888765/workspace/runs"); root = W / "viz_inputs_20261015/runs/vace_tracking_all_suites_20261004"
FD = W / "eval_mainline_20261016/main_s2000/heldout/fields/main_s2000"
cache = W / "vace_teacher_cache_small_20261015"
rows = [json.loads(l) for l in open(cache / "manifest_heldout.jsonl")]
text_root = PipelineConfig.load(cache / "pipeline.json").teacher.text_root
tok = AutoTokenizer.from_pretrained(Path(text_root) / "tokenizer", local_files_only=True)
content = content_token_masks(text_root, sorted({r["task"] for r in rows}))
words = json.load(open(W / "policy_features_20261017/words.json"))
def heat(att, tokens, mode, mask):
    att = np.asarray(att, np.float32)
    if mode == "contrast":
        att = att / np.maximum(att[:, mask].sum(1, keepdims=True), 1e-8)
    h = att[:, tokens].sum(1)
    return h / max(h.mean(), 1e-8)
stats = defaultdict(list)
for row in rows:
    tokens, readable = role_tokens(row["task"], tok, words.get(row["task"]))
    gt = c.ground_truth(row, root, readable)
    if gt is None:
        continue
    cam = "wrist" if "image2" in row["camera"] else "third"
    s = torch.load(FD / f"{row['sample_id']}.pt", map_location="cpu", weights_only=True)
    t = unpack_target(torch.load(row["target_path"], map_location="cpu", weights_only=True), identity=row["provenance_id"])
    sources = {"student_obs11": s["observed"]["text_attention"][11].float().numpy(),
               "student_mean": torch.cat((s["observed"]["text_attention"][8:12], s["future"]["text_attention"][8:9])).float().mean(0).numpy(),
               "student_fut12": s["future"]["text_attention"][12].float().numpy(),
               "teacher_obs11": t["observed"]["text_attention"][11].float().numpy()}
    mask = np.flatnonzero(content[row["task"]])
    for name, att in sources.items():
        for mode in ("plain", "contrast"):
            for role, truth in (("target_1", gt["target"]), ("destination_1", gt["destination"])):
                if not tokens[role] or not truth:
                    continue
                h = heat(att, tokens[role], mode, mask)
                d = c.dilate(truth)
                stats[(cam, role, name, mode)].append(float(d[int(np.argmax(h))]))
                stats[(cam, role, "chance", "-")].append(float(d.mean()))
            if gt["distractor"] and gt["target"] and tokens["target_1"]:
                right, wrong = sorted(gt["target"]), sorted(gt["distractor"] - gt["target"])
                if wrong:
                    h = heat(att, tokens["target_1"], mode, mask)
                    stats[(cam, "instance", name, mode)].append(float(h[right].mean() > h[wrong].mean()))
for key in sorted(stats):
    if key[2] == "chance" and key[1] == "instance":
        continue
    print(key, "n=%d" % len(stats[key]), "%.3f" % np.mean(stats[key]))
