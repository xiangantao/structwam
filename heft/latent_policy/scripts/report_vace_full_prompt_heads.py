#!/usr/bin/env python3
"""Score the full-instruction head scan against simulator GT (scoring only).

GT target, without reading the instruction: the object that physically moves most from the
present frame over the next 6 s (bodies of that object that move at least half as much,
so a drawer counts, not its cabinet).  Windows where nothing moves >= 3 cm ahead (task done)
or where the target is not visible in that camera are skipped and counted.

The GT comes from re-rendering each demo by replaying its actions from the init state, which
sometimes diverges (a missed grasp).  A window is kept only if the replay still looks like
the dataset video at the present frame: the third-person grey-level difference (64x64,
x-mirrored replay) may exceed its frame-0 difference by at most ``ALIGN_MAX``.  Subtracting
frame 0 removes the constant floor-texture / lighting offset (large in libero_object, where
object positions nevertheless match).  Both cameras of a window use the third-person check.

Metrics per candidate readout: ``hit`` = the argmax cell lies on the (3x3-dilated) target;
``moved`` = it lies on any object the episode moves; ``right`` = where another instance
of the same category is visible (two black bowls), the mean heat is higher on the target.

Selection protocol: within each suite, half the tasks are calibration and half test (fixed
seed).  Every choice (head, readout, combination) is made on calibration windows by the
camera-macro hit rate; test numbers are reported for the chosen candidates only.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_policy_maps import CAMERAS, HAND, dilate, is_object, prefix  # noqa: E402

PASSES = ("observed", "future", "both")
CAMERA_NAMES = {"agentview": "第三人称", "robot0_eye_in_hand": "腕部"}
SPLIT_SEED = 20261019
ALIGN_MAX = 6.0
STATE = {}


HEAD_ALIASES = {"drawer": "cabinet", "drawers": "cabinet", "compartment": "caddy", "pots": "pot", "bowls": "bowl",
                "plates": "plate", "mugs": "mug"}


def head_noun(group):
    return re.sub(r"_\d+$", "", group).split("_")[-1]


def mentioned_groups(prompt, names):
    """Object groups whose head noun (``akita_black_bowl_1`` -> bowl) the instruction contains."""
    words = {HEAD_ALIASES.get(w, w) for w in re.findall(r"[a-z]+", prompt.lower())}
    groups = defaultdict(list)
    for index, name in enumerate(names):
        if is_object(name):
            groups[prefix(name)].append(index)
    return {g: members for g, members in groups.items() if head_noun(g) in words}


def grey64(image):
    return image.astype(np.float32).mean(-1).reshape(64, image.shape[0] // 64, 64, image.shape[1] // 64).mean((1, 3))


def alignment_job(args):
    """Third-person replay-vs-dataset drift per start of one episode: {start: diff - frame-0 diff}."""
    import av
    from vace_latent_distill.data import decode_indices

    rows, root = args
    episode = rows[0]["tracking_episode"]
    origin = rows[0]["frame_indices"][0] - rows[0]["start"]
    wanted = sorted({0} | {r["start"] + 32 for r in rows})
    replay = {}
    with av.open(str(root / "videos" / f"episode_{episode:05d}" / "agentview.mp4")) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                replay[index] = grey64(frame.to_ndarray(format="rgb24")[:, ::-1])
            if index >= wanted[-1]:
                break
    dataset = {}
    for index in wanted:
        frame = decode_indices(rows[0]["video"], [origin + index], rows[0]["fps"])[0]
        dataset[index] = grey64(np.transpose(frame, (1, 2, 0)))
    base = float(np.abs(dataset[0] - replay[0]).mean())
    return episode, {r["start"]: (float(np.abs(dataset[r["start"] + 32] - replay[r["start"] + 32]).mean()) - base
                                  if r["start"] + 32 in replay else None) for r in rows}


def replay_alignment(rows, root, workers):
    groups = defaultdict(list)
    for row in rows:
        if CAMERAS[row["camera"]] == "agentview":
            groups[row["tracking_episode"]].append(row)
    with ProcessPoolExecutor(workers) as pool:
        return dict(pool.map(alignment_job, [(group, root) for group in groups.values()]))


def ground_truth(row, root, swap_prompt=None):
    camera = CAMERAS[row["camera"]]
    episode = row["tracking_episode"]
    with np.load(root / "tracks" / f"episode_{episode:05d}_{camera}_visible_surface.npz") as archive:
        xy = archive["xy"].copy()
        visible_all, label = archive["visible"], archive["point_label"]
        body_id, body_names = archive["body_id"], archive["body_names"]
        annotated = set(archive["annotated_frame_indices"].tolist())
    xy[..., 0] = 255 - xy[..., 0]
    xpos = np.load(root / "videos" / f"episode_{episode:05d}" / "sim_state.npz")["body_xpos"]
    names = json.loads((root / "videos" / f"episode_{episode:05d}" / "metadata.json").read_text())["body_names"]
    present = row["start"] + 32
    if present not in annotated or present >= len(xpos):
        return None, "present frame not annotated"
    point_body = np.asarray([str(body_names[b]) for b in body_id])
    visible = visible_all[present]
    safe = np.nan_to_num(xy[present], nan=-1.0)
    cell = (np.clip(np.floor(safe[:, 1] / 8), 0, 31) * 32 + np.clip(np.floor(safe[:, 0] / 8), 0, 31)).astype(int)

    def cells_of(mask):
        return sorted(set(cell[visible & mask].tolist()))

    groups = defaultdict(list)
    for index, name in enumerate(names):
        if is_object(name):
            groups[prefix(name)].append(index)
    ahead = np.linalg.norm(xpos[present:present + 61] - xpos[present], axis=-1).max(0)
    episode_travel = np.linalg.norm(xpos[10:] - xpos[10], axis=-1).max(0)
    best = max(groups, key=lambda g: ahead[groups[g]].max())
    peak = float(ahead[groups[best]].max())
    if peak < 0.03:
        return None, "nothing moves ahead"
    bodies = [names[i] for i in groups[best] if ahead[i] >= 0.5 * peak]
    target = cells_of(np.isin(point_body, bodies))
    if not target:
        return None, "target not visible"
    moved = [names[i] for g, members in groups.items() if episode_travel[members].max() >= 0.03 for i in members]
    stem = re.sub(r"_\d+$", "", best)
    others = [names[i] for g, members in groups.items()
              if g != best and re.sub(r"_\d+$", "", g) == stem for i in members]
    distractor = sorted(set(cells_of(np.isin(point_body, others))) - set(target))
    result = {"target": target, "moved": cells_of(np.isin(point_body, moved)), "distractor": distractor,
              "hand": cells_of(np.isin(label, HAND)), "target_name": best, "travel": peak}
    if swap_prompt is not None:
        named = mentioned_groups(swap_prompt, names)
        named.pop(best, None)
        bodies = [names[i] for members in named.values() for i in members]
        result["swap_named"] = sorted(set(cells_of(np.isin(point_body, bodies))) - set(target))
    return result, None


def load_maps(path):
    with np.load(path) as archive:
        observed, future = archive["observed"].astype(np.float32), archive["future"].astype(np.float32)
    return np.stack((observed, future, (observed + future) / 2))      # [P, blocks, R, T, H, C]


def metrics(maps, gt):
    """``maps [..., C]`` -> dict of arrays over the leading axes."""
    peaks = maps.argmax(-1)
    out = {"hit": dilate(gt["target"])[peaks], "moved": dilate(gt["moved"])[peaks]}
    if gt["distractor"]:
        out["right"] = maps[..., gt["target"]].mean(-1) > maps[..., gt["distractor"]].mean(-1)
    return out


def single_job(row):
    if row["drift"] is None:
        return row["sample_id"], None, "replay shorter than dataset"
    if row["drift"] > ALIGN_MAX:
        return row["sample_id"], None, "replay diverged from dataset video"
    gt, reason = ground_truth(row, STATE["tracking"])
    if gt is None:
        return row["sample_id"], None, reason
    maps = load_maps(STATE["maps"] / f"{row['sample_id']}.npz")
    blocks = maps.shape[1]
    main = [i for i, name in enumerate(STATE["blocks"]) if name.startswith("m")]
    vace = [i for i, name in enumerate(STATE["blocks"]) if name.startswith("v")]
    result = {
        "single": metrics(maps, gt),                                     # [P, blocks, R, T, H]
        "layer": metrics(maps.mean(4), gt),                              # [P, blocks, R, T]
        "all_main": metrics(maps[:, main].mean((1, 4)), gt),             # [P, R, T]
        **({"all_vace": metrics(maps[:, vace].mean((1, 4)), gt)} if vace else {}),
        "chance": float(dilate(gt["target"]).mean()),
        "target_name": gt["target_name"],
        "has_distractor": bool(gt["distractor"]),
    }
    if blocks != len(STATE["blocks"]):
        raise RuntimeError("Block count differs from the scan")
    return row["sample_id"], result, None


def combo_job(args):
    """Combined-candidate metrics; with a swapped-instruction window, also the control:
    ``stay`` = the swapped map still peaks on the original target, ``named`` = it peaks on an
    object the swapped instruction names, ``corr`` = map correlation across the two prompts."""
    row, combos = args
    swap = STATE["swaps"].get(row["sample_id"])
    gt, _ = ground_truth(row, STATE["tracking"], swap["task"] if swap else None)
    maps = load_maps(STATE["maps"] / f"{row['sample_id']}.npz")
    swapped = load_maps(STATE["maps"] / f"{swap['sample_id']}.npz") if swap else None
    out = {}
    for name, members in combos.items():
        combined = np.mean([maps[p, b, r, t, h] for p, b, r, t, h in members], 0)
        out[name] = {key: bool(value) for key, value in metrics(combined, gt).items()}
        if swap:
            other = np.mean([swapped[p, b, r, t, h] for p, b, r, t, h in members], 0)
            peak = int(other.argmax())
            out[name]["stay"] = bool(dilate(gt["target"])[peak])
            out[name]["named"] = bool(dilate(gt["swap_named"])[peak]) if gt["swap_named"] else None
            out[name]["corr"] = float(np.corrcoef(combined, other)[0, 1])
    return row["sample_id"], out


def init_worker(tracking, maps, blocks, swaps=None):
    STATE.update(tracking=tracking, maps=maps, blocks=blocks, swaps=swaps or {})


def split_tasks(rows):
    rng = np.random.default_rng(SPLIT_SEED)
    calibration = set()
    for suite in sorted({r["suite"] for r in rows}):
        tasks = sorted({r["task_index"] for r in rows if r["suite"] == suite})
        rng.shuffle(tasks)
        calibration.update(tasks[:len(tasks) // 2])
    return {r["sample_id"]: "calibration" if r["task_index"] in calibration else "test" for r in rows}


def rate(values):
    return float(np.mean(values)) if len(values) else float("nan")


def camera_macro(per_camera):
    values = [v for v in per_camera.values() if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def bootstrap_difference(a, b, episodes, seed=0, rounds=2000):
    """95% interval of mean(a - b), resampling episodes (both cameras and starts together)."""
    a, b, episodes = np.asarray(a, float), np.asarray(b, float), np.asarray(episodes)
    unique = np.unique(episodes)
    index = {e: np.flatnonzero(episodes == e) for e in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(rounds):
        chosen = np.concatenate([index[e] for e in rng.choice(unique, len(unique))])
        draws.append((a[chosen] - b[chosen]).mean())
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.scan / "windows.jsonl").read_text().splitlines()]
    swaps = {r["swap_of"]: r for r in rows if "swap_of" in r}
    rows = [r for r in rows if "swap_of" not in r]
    with np.load(args.scan / "maps" / f"{rows[0]['sample_id']}.npz") as archive:
        blocks = [str(b) for b in archive["blocks"]]
        readouts = [str(r) for r in archive["readouts"]]
        times = [str(t) for t in archive["times"]]
    drift = replay_alignment(rows, args.tracking_root, args.workers)
    for row in rows:
        row["drift"] = drift[row["tracking_episode"]].get(row["start"])
    split = split_tasks(rows)
    by_id = {r["sample_id"]: r for r in rows}

    with ProcessPoolExecutor(args.workers, initializer=init_worker,
                             initargs=(args.tracking_root, args.scan / "maps", blocks)) as pool:
        scored = list(pool.map(single_job, rows, chunksize=2))
    skipped = defaultdict(int)
    windows = {}
    for sample_id, result, reason in scored:
        if result is None:
            skipped[f"{by_id[sample_id]['suite']} {reason}"] += 1
        else:
            windows[sample_id] = result
    groups = defaultdict(list)                       # (split, camera) -> sample ids
    for sample_id in windows:
        groups[split[sample_id], CAMERAS[by_id[sample_id]["camera"]]].append(sample_id)
    cameras = sorted({camera for _, camera in groups})

    def stack(kind, metric, ids):
        return np.stack([windows[i][kind][metric] for i in ids]).astype(float)

    def kind_rates(kind, split_name, metric="hit"):
        return {camera: stack(kind, metric, groups[split_name, camera]).mean(0) for camera in cameras}

    calibration = kind_rates("single", "calibration")
    score = np.mean([calibration[c] for c in cameras], 0)            # [P, blocks, R, T, H]
    order = np.argsort(-score, axis=None)

    def describe(index):
        p, b, r, t, h = (int(v) for v in index)
        return {"pass": PASSES[p], "block": blocks[b], "head": h, "readout": readouts[r], "time": times[t]}

    def label(index):
        d = describe(index)
        return f"{d['pass']} {d['block']}h{d['head']} {d['readout']} {d['time']}"

    baseline = (PASSES.index("observed"), blocks.index("m22"), readouts.index("content_snmax"),
                times.index("recent"), 4)
    student_like = (PASSES.index("both"), blocks.index("m22"), readouts.index("content_snmax"),
                    times.index("recent"), 4)
    top = [tuple(int(v) for v in np.unravel_index(i, score.shape)) for i in order[:max(args.top, 10)]]
    per_camera_best = {camera: tuple(int(v) for v in np.unravel_index(np.argmax(calibration[camera]), score.shape))
                       for camera in cameras}
    family_best = {}
    for p in range(len(PASSES)):
        for r in range(len(readouts)):
            sub = score[p, :, r]
            b, t, h = np.unravel_index(np.argmax(sub), sub.shape)
            family_best[PASSES[p], readouts[r]] = (p, int(b), r, int(t), int(h))
    layer_score = np.mean([kind_rates("layer", "calibration")[c] for c in cameras], 0)
    layer_best = tuple(int(v) for v in np.unravel_index(np.argmax(layer_score), layer_score.shape))
    combos = {f"top{k}": [top[i] for i in range(k)] for k in (3, 5, 10)}
    combos["layer"] = [(layer_best[0], layer_best[1], layer_best[2], layer_best[3], h) for h in range(12)]
    jobs = [(by_id[i], {**combos, **{f"single:{label(c)}": [c] for c in
                                      {baseline, student_like, *top, *per_camera_best.values(),
                                       *family_best.values()}}})
            for i in windows]
    with ProcessPoolExecutor(args.workers, initializer=init_worker,
                             initargs=(args.tracking_root, args.scan / "maps", blocks, swaps)) as pool:
        combined = dict(pool.map(combo_job, jobs, chunksize=2))

    def summary(name):
        out = {}
        for split_name in ("calibration", "test"):
            for camera in cameras:
                ids = groups[split_name, camera]
                out[f"{split_name}_{camera}_hit"] = rate([combined[i][name]["hit"] for i in ids])
                out[f"{split_name}_{camera}_moved"] = rate([combined[i][name]["moved"] for i in ids])
                right = [combined[i][name]["right"] for i in ids if "right" in combined[i][name]]
                out[f"{split_name}_{camera}_right"] = rate(right)
                out[f"{split_name}_{camera}_right_n"] = len(right)
            out[f"{split_name}_macro_hit"] = camera_macro({c: out[f"{split_name}_{c}_hit"] for c in cameras})
        return out

    candidates = {
        "当前读法（观测段 m22h4 content_snmax recent）": f"single:{label(baseline)}",
        "当前读法，观测+未来平均（接近学生热图）": f"single:{label(student_like)}",
        f"最佳单头（{label(top[0])}）": f"single:{label(top[0])}",
        **{f"前 {k} 名单头平均": f"top{k}" for k in (3, 5, 10)},
        f"最佳整层 12 头平均（{PASSES[layer_best[0]]} {blocks[layer_best[1]]} {readouts[layer_best[2]]} "
        f"{times[layer_best[3]]}）": "layer",
    }
    for camera, index in per_camera_best.items():
        candidates[f"按{CAMERA_NAMES.get(camera, camera)}单独选的最佳单头（{label(index)}）"] = f"single:{label(index)}"
    main_table = {name: summary(key) for name, key in candidates.items()}
    all_heads = {}
    for kind in [k for k in ("all_main", "all_vace") if k in next(iter(windows.values()))]:
        rates_by = {split_name: kind_rates(kind, split_name) for split_name in ("calibration", "test")}
        macro = np.mean([rates_by["calibration"][c] for c in cameras], 0)
        p, r, t = np.unravel_index(np.argmax(macro), macro.shape)
        all_heads[kind] = {"choice": f"{PASSES[p]} {readouts[r]} {times[t]}",
                           **{f"{s}_{c}_hit": float(rates_by[s][c][p, r, t]) for s in rates_by for c in cameras}}
    top_table = [{"candidate": label(index), "calibration_macro": float(score[index]), **summary(f"single:{label(index)}")}
                 for index in top[:args.top]]
    family_table = [{"pass": p, "readout": r, "candidate": label(index), **summary(f"single:{label(index)}")}
                    for (p, r), index in family_best.items()]

    chance = {f"{s}_{c}": rate([windows[i]["chance"] for i in groups[s, c]]) for s in ("calibration", "test")
              for c in cameras}
    counts = {f"{s}_{c}": len(groups[s, c]) for s in ("calibration", "test") for c in cameras}
    suites = sorted({r["suite"] for r in rows})
    best_combo = max(("top3", "top5", "top10", "layer", f"single:{label(top[0])}"),
                     key=lambda key: summary(key)["calibration_macro_hit"])
    per_suite = {}
    for key, name in ((f"single:{label(baseline)}", "当前读法"), (f"single:{label(top[0])}", "最佳单头"),
                      (best_combo, f"校准集最好的组合（{best_combo}）")):
        per_suite[name] = {f"{suite}_{camera}": rate([combined[i][key]["hit"] for i in groups["test", camera]
                                                     if by_id[i]["suite"] == suite])
                           for suite in suites for camera in cameras}
    test_ids = [i for c in cameras for i in groups["test", c]]
    difference = {
        "best_combo_minus_current": rate([combined[i][best_combo]["hit"] - combined[i][f"single:{label(baseline)}"]["hit"]
                                          for i in test_ids]),
        "interval_95": bootstrap_difference(
            [combined[i][best_combo]["hit"] for i in test_ids],
            [combined[i][f"single:{label(baseline)}"]["hit"] for i in test_ids],
            [by_id[i]["episode"] for i in test_ids]),
    }
    control = {}
    if swaps:
        for name, key in candidates.items():
            row = {}
            for camera in cameras:
                for kind, clean in (("clean", True), ("unclean", False)):
                    ids = [i for s in ("calibration", "test") for i in groups[s, camera]
                           if i in swaps and swaps[i]["swap_clean"] == clean]
                    hit = [i for i in ids if combined[i][key]["hit"]]
                    named = [combined[i][key]["named"] for i in ids if combined[i][key]["named"] is not None]
                    row[f"{camera}_{kind}"] = {
                        "n": len(ids), "hit": rate([combined[i][key]["hit"] for i in ids]),
                        "stay_given_hit": rate([combined[i][key]["stay"] for i in hit]),
                        "swap_hit": rate([combined[i][key]["stay"] for i in ids]),
                        "named": rate(named), "named_n": len(named),
                        "corr": rate([combined[i][key]["corr"] for i in ids])}
            control[name] = row
    report = {"windows": len(rows), "control": control, "scored": len(windows), "skipped": dict(skipped), "counts": counts,
              "chance": chance, "main": main_table, "all_heads": all_heads, "top": top_table,
              "families": family_table, "per_suite_test": per_suite, "best_combo": best_combo,
              "test_difference": difference,
              "split": {s: sorted({by_id[i]["task_index"] for i in windows if split[i] == s}) for s in ("calibration", "test")}}
    (args.scan / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.scan / "report.md").write_text(markdown(report, cameras, suites))
    print(markdown(report, cameras, suites))


def markdown(report, cameras, suites):
    def pct(value):
        return "—" if value is None or np.isnan(value) else f"{100 * value:.0f}%"

    names = [CAMERA_NAMES.get(c, c) for c in cameras]
    lines = [f"窗口 {report['windows']} 个，参与打分 {report['scored']} 个；跳过：{report['skipped']}",
             f"每组窗口数：{report['counts']}", f"随机落点命中率（目标面积占比）：" +
             "，".join(f"{k} {pct(v)}" for k, v in report["chance"].items()), "",
             "| 读法 | " + " | ".join(f"{n} 校准 | {n} 测试" for n in names) +
             " | 测试：同类干扰选对 | 测试：落在任一被移动物体 |",
             "| --- |" + " --- |" * (2 * len(cameras) + 2)]
    for name, row in report["main"].items():
        right = [row[f"test_{c}_right"] for c in cameras if not np.isnan(row[f"test_{c}_right"])]
        moved = [row[f"test_{c}_moved"] for c in cameras]
        lines.append(f"| {name} | " + " | ".join(f"{pct(row[f'calibration_{c}_hit'])} | {pct(row[f'test_{c}_hit'])}"
                                                 for c in cameras) +
                     f" | {pct(np.mean(right)) if right else '—'} | {pct(np.mean(moved))} |")
    for kind, row in report["all_heads"].items():
        label = "主干全部 30 层×12 头平均" if kind == "all_main" else "VACE 分支全部 15 层×12 头平均"
        lines.append(f"| {label}（{row['choice']}） | " +
                     " | ".join(f"{pct(row[f'calibration_{c}_hit'])} | {pct(row[f'test_{c}_hit'])}" for c in cameras) +
                     " | — | — |")
    lines += ["", f"测试集上 最好组合 − 当前读法 = {pct(report['test_difference']['best_combo_minus_current'])}，"
              f"95% 区间 [{pct(report['test_difference']['interval_95'][0])}, "
              f"{pct(report['test_difference']['interval_95'][1])}]（按 episode 重采样）", "",
              "按 suite 的测试集命中率：", "",
              "| 读法 | " + " | ".join(f"{s} {n}" for s in suites for n in names) + " |",
              "| --- |" + " --- |" * (len(suites) * len(cameras))]
    for name, row in report["per_suite_test"].items():
        lines.append(f"| {name} | " + " | ".join(pct(row[f"{s}_{c}"]) for s in suites for c in cameras) + " |")
    lines += ["", "校准集排名前列的单头：", "", "| 候选 | 校准（两相机平均） | " +
              " | ".join(f"{n} 测试" for n in names) + " |", "| --- | --- |" + " --- |" * len(cameras)]
    for row in report["top"]:
        lines.append(f"| {row['candidate']} | {pct(row['calibration_macro'])} | " +
                     " | ".join(pct(row[f"test_{c}_hit"]) for c in cameras) + " |")
    lines += ["", "每种读法、每个前向段里校准集最好的头：", "", "| 段 | 读法 | 候选 | " +
              " | ".join(f"{n} 校准 | {n} 测试" for n in names) + " |",
              "| --- | --- | --- |" + " --- |" * (2 * len(cameras))]
    for row in report["families"]:
        lines.append(f"| {row['pass']} | {row['readout']} | {row['candidate']} | " +
                     " | ".join(f"{pct(row[f'calibration_{c}_hit'])} | {pct(row[f'test_{c}_hit'])}" for c in cameras) + " |")
    if report["control"]:
        lines += ["", "换指令对照（同一窗口、同一噪声，只把指令换成同 suite 另一条；校准+测试全部窗口）。"
                  "干净对照 = 新指令点名了场景里别的物体、没点名原目标；"
                  "指令定位的读法应当离开原目标、落到新指令点名的物体上，只看运动的读法会留在原目标。", ""]
        for camera, name in zip(cameras, names):
            lines += [f"{name}：", "", "| 读法 | 干净对照窗口数 | 原指令命中 | 换指令后仍在原目标 | 原本命中的里面仍留下 | "
                      "换后落到新指令点名的物体 | 两张图相关系数 | 非干净对照（如 spatial）仍在原目标 |",
                      "| --- |" + " --- |" * 7]
            for label, row in report["control"].items():
                clean, other = row[f"{camera}_clean"], row[f"{camera}_unclean"]
                corr = "—" if np.isnan(clean["corr"]) else f"{clean['corr']:.2f}"
                lines.append(f"| {label} | {clean['n']} | {pct(clean['hit'])} | {pct(clean['swap_hit'])} | "
                             f"{pct(clean['stay_given_hit'])} | {pct(clean['named'])}（{clean['named_n']}） | "
                             f"{corr} | {pct(other['swap_hit'])}（{other['n']}） |")
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
