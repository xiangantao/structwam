"""Every 10 min: if any LIBERO-plus result changed, recompute the paired comparisons and write
latest.md (overwritten) and append the same snapshot to history.md. Also lists training progress.
    nohup python3 monitor.py > monitor.out 2>&1 &
"""
import collections, csv, glob, json, os, re, time
from math import comb

W = "/mnt/sfs-turbo-wangfei/2337888765/workspace"
OUT = W + "/runs/eval_monitor"
EVALS = {  # name -> results dir (sample10_clean protocol unless noted)
    "OpenWAM 40k": W + "/runs/stage0_libero_plus_20261027/openwam_cosmos_s40000",
    "StructWAM 40k": W + "/runs/libero_plus_sample10_clean/structwam_s40000",
}
EXTRA_S40 = W + "/runs/libero_plus_structwam_s40000_v10/libero10_weak4_sample10_clean"
S24 = W + "/runs/libero_plus_structwam_s24000_v10/sample10_clean/summary.csv"
cls = json.load(open(W + "/repos/LIBERO-plus/libero/libero/benchmark/task_classification.json"))
CAT = {(s, int(i["id"]) - 1): i["category"] for s, v in cls.items() for i in v}


def load(path):
    out = {}
    for f in glob.glob(path + "/videos/*/*/*/*/results.json"):
        try:
            d = json.load(open(f)); out[(d["suite"], d["task_id"])] = int(d["successes"])
        except Exception:
            pass
    return out


def mcnemar(a, b):
    m = a + b
    return min(1.0, 2 * sum(comb(m, i) for i in range(min(a, b) + 1)) / 2 ** m) if m else 1.0


def table(title, ref, cols, keys):
    """ref: dict used for wins; cols: [(name, dict)], keys: tasks present in all."""
    lines = [f"### {title}  (n={len(keys)})", "",
             "| group | n | " + " | ".join(n for n, _ in cols) + f" | {cols[1][0]} only / {cols[0][0]} only | p |",
             "|---|---|" + "---|" * len(cols) + "---|---|"]
    groups = collections.defaultdict(list)
    for k in keys:
        for g in ("ALL", k[0], CAT[k]):
            groups[g].append(k)
    order = ["ALL"] + sorted(g for g in groups if g.startswith("libero")) + sorted(g for g in groups if g != "ALL" and not g.startswith("libero"))
    a_name, a = cols[0]; b_name, b = cols[1]
    for g in order:
        ks = groups[g]; n = len(ks)
        rates = " | ".join(f"{100 * sum(d[k] for k in ks) / n:.1f}%" for _, d in cols)
        bw = sum(b[k] > a[k] for k in ks); aw = sum(a[k] > b[k] for k in ks)
        lines.append(f"| {g} | {n} | {rates} | {bw} / {aw} | {mcnemar(bw, aw):.3f} |")
    return "\n".join(lines) + "\n"


def training():
    rows = []
    for d in sorted(glob.glob(W + "/runs/structwam_a123_20261027/*/train.log")):
        arm = d.split("/")[-2]
        try:
            with open(d, "rb") as f:
                f.seek(max(0, os.path.getsize(d) - 4000)); tail = f.read().decode(errors="ignore")
            m = re.findall(r"(\d+)/40000 \[[^<]*<([^,]*), *([\d.]+)s/step, loss=([\d.]+)", tail)
            rows.append(f"| {arm} | {m[-1][0]}/40000 | {m[-1][2]} s/step | ETA {m[-1][1]} | loss {m[-1][3]} |" if m else f"| {arm} | starting | | | |")
        except Exception as e:
            rows.append(f"| {arm} | ? ({e}) | | | |")
    return "### Training\n\n| arm | step | speed | remaining | loss |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n"


def snapshot():
    data = {n: load(p) for n, p in EVALS.items()}
    o, s = data["OpenWAM 40k"], data["StructWAM 40k"]
    s24 = {(r["suite"], int(r["task_id"])): int(r["successes"]) for r in csv.DictReader(open(S24))}
    s_comb = dict(s)
    for k, v in load(EXTRA_S40).items():
        s_comb.setdefault(k, v)
    parts = [f"## {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
             "Progress: " + ", ".join(f"{n} {len(d)}/1005" for n, d in data.items()) + "\n"]
    k1 = [k for k in s if k in o and k in s24]
    parts.append(table("Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)", o,
                       [("OpenWAM 40k", o), ("StructWAM 40k", s), ("StructWAM 24k", s24)], k1))
    k2 = [k for k in s_comb if k in o and k in s24]
    parts.append(table("Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k", o,
                       [("OpenWAM 40k", o), ("StructWAM 40k", s_comb), ("StructWAM 24k", s24)], k2))
    k3 = [k for k in s if k in s24]
    parts.append(table("StructWAM 40k vs 24k (full eval tasks)", s24,
                       [("StructWAM 24k", s24), ("StructWAM 40k", s)], k3))
    parts.append(training())
    return "\n".join(parts), sum(len(d) for d in data.values())


last = -1
while True:
    try:
        text, count = snapshot()
        if count != last:
            open(OUT + "/latest.md", "w").write(text)
            with open(OUT + "/history.md", "a") as h:
                h.write(text + "\n---\n")
            last = count
    except Exception as e:
        open(OUT + "/monitor_error.txt", "a").write(f"{time.ctime()} {e!r}\n")
    time.sleep(600)
