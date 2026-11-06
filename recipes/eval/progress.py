"""LIBERO-plus 测评进度和结果（任意 python3 即可，只用标准库）。

    python3 recipes/eval/progress.py OUT_DIR [--compare OTHER_OUT_DIR] [--diff]

OUT_DIR 是 eval_libero_plus.sh 的输出目录（下面有 spatial/ object/ goal/ long/ 子目录）。
输出每个 suite 的完成数 / 成功率 / 预计剩余时间，以及每类扰动的成功率；
--compare 给另一次测评（例如 10% 抽样）做同 suite 同类别对比，--diff 按难度 1–5 展开。
任务分类表读自 $LIBERO_PLUS_REPO/libero/libero/benchmark/task_classification.json。
"""
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict

argv = sys.argv[1:]
SHOW_DIFF = "--diff" in argv
COMPARE = argv[argv.index("--compare") + 1] if "--compare" in argv else None
positional = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--compare")]
if not positional:
    sys.exit(__doc__)
OUT_DIR = positional[0]
LIBERO_PLUS_REPO = os.environ.get("LIBERO_PLUS_REPO") or os.path.join(os.environ["STRUCTWAM_ASSETS"], "repos", "LIBERO-plus")
SUITES = [("long", "libero_10"), ("goal", "libero_goal"), ("object", "libero_object"), ("spatial", "libero_spatial")]
CAT_ZH = {
    "Background Textures": "背景纹理",
    "Camera Viewpoints": "相机视角",
    "Language Instructions": "语言指令",
    "Light Conditions": "光照条件",
    "Objects Layout": "物体布局",
    "Robot Initial States": "机器人初始状态",
    "Sensor Noise": "传感器噪声",
}
TC = json.load(open(os.path.join(LIBERO_PLUS_REPO, "libero", "libero", "benchmark", "task_classification.json")))


def load(pattern):
    """{task_id: (successes, mtime)} — task_id is the 0-based LIBERO-plus index (classification id - 1)."""
    out = {}
    for f in glob.glob(pattern):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        out[int(r["task_id"])] = (int(r.get("successes", 0)), os.path.getmtime(f))
    return out


def pct(o, n):
    return f"{100.0 * o / n:5.1f}%" if n else "   -  "


def width(s):
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in s)


def pad(s, w):
    return s + " " * max(0, w - width(s))


now = time.time()
grand = Counter()
for short, suite in SUITES:
    meta = {r["id"] - 1: r for r in TC[suite]}
    full = load(f"{OUT_DIR}/{short}/videos/{suite}/task_*/*/attempt_*/results.json")
    samp = {}
    if COMPARE:  # per-suite layout (this launcher) or one dir for all suites (run_all_suites default)
        base = f"{COMPARE}/{short}/videos" if os.path.isdir(f"{COMPARE}/{short}/videos") else f"{COMPARE}/videos"
        samp = load(f"{base}/{suite}/task_*/*/attempt_*/results.json")
    if not full and not samp:
        continue
    total, done = len(meta), len(full)
    ok = sum(s for s, _ in full.values())
    recent = sum(1 for _, t in full.values() if now - t < 1800)
    rate = recent * 2  # 每小时
    eta = f"{(total - done) / rate:.1f} 小时" if rate and done < total else ("已完成" if done >= total else "未开始/暂停")
    s_ok = sum(s for s, _ in samp.values())
    print(f"\n=== {suite}: 完成 {done}/{total}  成功率 {pct(ok, done)}  | 对比 {pct(s_ok, len(samp))} (n={len(samp)})"
          f"  | 近30分钟 {rate}/小时, 剩余约 {eta}")
    grand.update(full_n=done, full_ok=ok, total=total)
    if done:  # 合计行的对比只算已开测的 suite
        grand.update(samp_n=len(samp), samp_ok=s_ok)

    rows = defaultdict(lambda: Counter())
    for tid, m in meta.items():
        keys = [(m["category"], None)] + ([(m["category"], m["difficulty_level"])] if SHOW_DIFF else [])
        for k in keys:
            rows[k]["total"] += 1
            if tid in full:
                rows[k]["n"] += 1
                rows[k]["ok"] += full[tid][0]
            if tid in samp:
                rows[k]["sn"] += 1
                rows[k]["sok"] += samp[tid][0]
            if tid in full and tid in samp:
                rows[k]["pair"] += 1
                rows[k]["agree"] += int(full[tid][0] == samp[tid][0])
    print(f"  {pad('扰动类别', 16)} {'完成/总数':>11}  {'本次SR':>7}  {'对比SR':>7} {'对比n':>5}  {'差值':>6}  同任务一致")
    for (cat, diff) in sorted(rows, key=lambda k: (k[0], -1 if k[1] is None else k[1])):
        c = rows[(cat, diff)]
        label = CAT_ZH.get(cat, cat) if diff is None else f"  难度{diff}"
        delta = (f"{100.0 * c['ok'] / c['n'] - 100.0 * c['sok'] / c['sn']:+5.1f}" if c["n"] and c["sn"] else "   - ")
        agree = f"{c['agree']}/{c['pair']}" if c["pair"] else "-"
        print(f"  {pad(label, 16)} {c['n']:>5}/{c['total']:<5}  {pct(c['ok'], c['n']):>7}  {pct(c['sok'], c['sn']):>7} "
              f"{c['sn']:>5}  {delta:>6}  {agree}")

if grand["full_n"]:
    print(f"\n=== 全部 suite 合计: 完成 {grand['full_n']}/{grand['total']}  成功率 {pct(grand['full_ok'], grand['full_n'])}"
          f"  | 对比 {pct(grand['samp_ok'], grand['samp_n'])} (n={grand['samp_n']})")
print("\n说明: 差值 = 本次SR - 对比SR（同 suite 同类别）；中途的SR只覆盖已测的任务，类别按任务编号顺序依次测。"
      "\n      同任务一致 = 两次都测过的任务里成败相同的个数（测评是确定性的，同一检查点应全部一致）。")
