#!/usr/bin/env python3
"""Grammar-free word maps: does attention concentration + the student's own K segments find objects?

Read-only analysis on saved student fields of the held-out windows; simulator GT only scores.

* Segments: ``policy_maps.segments`` (K-means + connected components of the present K); the
  large ones are flagged background by a fixed area threshold.  Precision / recall of the
  non-background cells against GT object surfaces (robot excluded).
* Words: every content word of the instruction (generic stop-word list, no grammar), its
  spatially normalized heat (``phrase_heat``) on the present map; concentration = peak log
  heat.  Words are ranked by concentration.  For rank r: does the raw peak / the peak inside
  the best non-background segment (segment-mean pooling) land on an object, on the target
  (the object the episode moves next), on the destination (the object the target ends on).
* Union over the top-K words, and on-object rate per concentration bin (for a threshold).
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_policy_maps import CAMERAS, dilate, is_object, load_fields, prefix  # noqa: E402

STATE = {}
RANKS = 8


def init_worker(text_root):
    from transformers import AutoTokenizer

    STATE["tokenizer"] = AutoTokenizer.from_pretrained(Path(text_root) / "tokenizer", local_files_only=True)


def ground_truth(row, root):
    """Present-frame cells: objects (robot excluded), target (moves most in the next 6 s, its moving
    bodies) and destination (other object nearest to where the target ends)."""
    camera = CAMERAS[row["camera"]]
    episode = row["tracking_episode"]
    with np.load(root / "tracks" / f"episode_{episode:05d}_{camera}_visible_surface.npz") as archive:
        xy = archive["xy"].copy()
        visible_all, body_id, body_names = archive["visible"], archive["body_id"], archive["body_names"]
        annotated = set(archive["annotated_frame_indices"].tolist())
    xy[..., 0] = 255 - xy[..., 0]
    xpos = np.load(root / "videos" / f"episode_{episode:05d}" / "sim_state.npz")["body_xpos"]
    names = json.loads((root / "videos" / f"episode_{episode:05d}" / "metadata.json").read_text())["body_names"]
    present = row["start"] + 32
    if present not in annotated or present >= len(xpos):
        return None
    point_body = np.asarray([str(body_names[b]) for b in body_id])
    visible = visible_all[present]
    safe = np.nan_to_num(xy[present], nan=-1.0)
    cell = (np.clip(np.floor(safe[:, 1] / 8), 0, 31) * 32 + np.clip(np.floor(safe[:, 0] / 8), 0, 31)).astype(int)

    def cells_of(bodies):
        return sorted(set(cell[visible & np.isin(point_body, bodies)].tolist()))

    groups = defaultdict(list)
    for index, name in enumerate(names):
        if is_object(name) and not any(s in name for s in ("floor", "table", "wall")):
            groups[prefix(name)].append(index)
    objects = cells_of([names[i] for members in groups.values() for i in members])
    ahead = np.linalg.norm(xpos[present:present + 61] - xpos[present], axis=-1).max(0)
    best = max(groups, key=lambda g: ahead[groups[g]].max())
    peak = float(ahead[groups[best]].max())
    robot = sorted(set(cell[visible & ~np.isin(point_body, [n for n in names if is_object(n)])].tolist()))
    out = {"objects": objects, "robot": robot, "target": [], "destination": []}
    if peak >= 0.03:
        out["target"] = cells_of([names[i] for i in groups[best] if ahead[i] >= 0.5 * peak])
        final = xpos[-1, groups[best]].mean(0)
        others = {g: m for g, m in groups.items() if g != best}
        if others:
            near = min(others, key=lambda g: np.linalg.norm(xpos[-1, others[g]] - final, axis=-1).min())
            out["destination"] = cells_of([names[i] for i in others[near]])
    return out


def job(args):
    row, fields_dir, root = args
    from latent_dp.policy_maps import STOP, instruction_words, phrase_heat, segments

    path = fields_dir / f"{row['sample_id']}.pt"
    if not path.is_file():
        return None
    gt = ground_truth(row, root)
    if gt is None or not gt["objects"]:
        return None
    fields = load_fields(path)
    wrist = CAMERAS[row["camera"]] != "agentview"
    words, tokens = instruction_words(row["task"], STATE["tokenizer"])
    content_words = [(w, t) for w, t in zip(words, tokens) if w not in STOP]
    content = [i for _, t in content_words for i in t]
    attention = np.asarray(fields["text_attention"], np.float32)               # [2, cells, L]
    objects = dilate(gt["objects"], 0)
    robot = dilate(gt["robot"], 0) & ~objects
    target, destination = dilate(gt["target"]), dilate(gt["destination"])
    parts = segments(fields["seg_k_l22h9"], fields["seg_k_l15h7"], wrist)
    foreground = np.zeros(1024, bool)
    candidates = [p for p in parts if not p["background"] and p["mask"].sum() >= 2]
    for p in candidates:
        foreground |= p["mask"]
    result = {"camera": "wrist" if wrist else "third", "suite": row["suite"],
              "fg_precision": float(objects[foreground].mean()) if foreground.any() else None,
              "fg_recall": float(foreground[objects].mean()) if objects.any() else None,
              "fg_area": float(foreground.mean()), "object_area": float(objects.mean()),
              "robot_area": float(robot.mean()), "segments": len(parts), "background_segments":
              sum(p["background"] for p in parts),
              "has_target": bool(gt["target"]), "has_destination": bool(gt["destination"]), "words": []}
    for word, token_ids in content_words:
        heat = phrase_heat(attention[0], token_ids, content)
        later = phrase_heat(attention[1], token_ids, content)
        peak = int(np.argmax(heat))
        entry = {"word": word, "concentration": float(heat.max()), "raw_object": bool(objects[peak]),
                 "raw_robot": bool(robot[peak]),
                 "raw_target": bool(target[peak]) if gt["target"] else None,
                 "raw_destination": bool(destination[int(np.argmax(later))]) if gt["destination"] else None}
        if candidates:
            scores = [float(heat[p["mask"]].mean()) for p in candidates]
            chosen = candidates[int(np.argmax(scores))]["mask"]
            inside = np.flatnonzero(chosen)[int(np.argmax(heat[chosen]))]
            later_scores = [float(later[p["mask"]].mean()) for p in candidates]
            later_chosen = candidates[int(np.argmax(later_scores))]["mask"]
            later_inside = np.flatnonzero(later_chosen)[int(np.argmax(later[later_chosen]))]
            entry.update(seg_score=float(max(scores)), seg_object=bool(objects[inside]),
                         seg_on_objects=float(objects[chosen].mean()),
                         seg_target=bool(target[inside]) if gt["target"] else None,
                         seg_destination=bool(destination[later_inside]) if gt["destination"] else None)
        result["words"].append(entry)
    result["words"].sort(key=lambda e: -e["concentration"])
    return result


def rate(values):
    values = [v for v in values if v is not None]
    return (float(np.mean(values)), len(values)) if values else (float("nan"), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--text-root", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(args.text_root,)) as pool:
        results = [r for r in pool.map(job, [(row, args.fields, args.tracking_root) for row in rows]) if r]
    report, lines = {"windows": len(results)}, []
    for camera in ("third", "wrist"):
        group = [r for r in results if r["camera"] == camera]
        block = {"windows": len(group)}
        for key in ("fg_precision", "fg_recall", "fg_area", "object_area", "robot_area", "segments",
                    "background_segments"):
            block[key] = rate([r[key] for r in group])
        by_rank = []
        for k in range(RANKS):
            entries = [r["words"][k] for r in group if len(r["words"]) > k]
            by_rank.append({key: rate([e.get(key) for e in entries]) for key in
                            ("raw_object", "raw_robot", "seg_object", "seg_on_objects", "raw_target", "seg_target",
                             "raw_destination", "seg_destination")} | {"n": len(entries)})
        block["by_rank"] = by_rank
        union = []
        for k in range(1, RANKS + 1):
            union.append({
                "raw_target": rate([any(e["raw_target"] for e in r["words"][:k]) for r in group if r["has_target"]]),
                "seg_target": rate([any(e.get("seg_target") for e in r["words"][:k]) for r in group if r["has_target"]]),
                "seg_destination": rate([any(e.get("seg_destination") for e in r["words"][:k])
                                         for r in group if r["has_destination"]]),
            })
        block["union"] = union
        concentration = np.array([e["concentration"] for r in group for e in r["words"]])
        on_object = np.array([e["raw_object"] for r in group for e in r["words"]], float)
        edges = np.quantile(concentration, np.linspace(0, 1, 6)) if len(concentration) else []
        block["bins"] = [{"low": float(lo), "high": float(hi),
                          "raw_object": float(on_object[(concentration >= lo) & (concentration <= hi)].mean()),
                          "n": int(((concentration >= lo) & (concentration <= hi)).sum())}
                         for lo, hi in zip(edges[:-1], edges[1:])]
        report[camera] = block
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "windows.json").write_text(json.dumps(results, indent=1) + "\n")

    def pct(pair):
        value, count = pair
        return "—" if count == 0 else f"{100 * value:.0f}%"

    for camera in ("third", "wrist"):
        block = report[camera]
        lines += [f"## {camera} ({block['windows']} windows)",
                  f"non-background segments: precision on objects {pct(block['fg_precision'])}, recall of object "
                  f"cells {pct(block['fg_recall'])}, area {pct(block['fg_area'])} (objects cover "
                  f"{pct(block['object_area'])})", "",
                  "| rank | n | raw peak on object | seg peak on object | seg cells on objects | raw target | "
                  "seg target | raw destination(+16) | seg destination(+16) |", "|" + " --- |" * 9]
        for k, e in enumerate(block["by_rank"]):
            lines.append(f"| {k + 1} | {e['n']} | " + " | ".join(pct(e[key]) for key in (
                "raw_object", "seg_object", "seg_on_objects", "raw_target", "seg_target", "raw_destination",
                "seg_destination")) + " |")
        lines += ["", "| top-K | any raw target | any seg target | any seg destination |", "|" + " --- |" * 4]
        for k, e in enumerate(block["union"]):
            lines.append(f"| {k + 1} | {pct(e['raw_target'])} | {pct(e['seg_target'])} | {pct(e['seg_destination'])} |")
        lines += ["", "| concentration bin | n | raw peak on object |", "| --- | --- | --- |"]
        for b in block["bins"]:
            lines.append(f"| {b['low']:.2f}–{b['high']:.2f} | {b['n']} | {100 * b['raw_object']:.0f}% |")
        lines.append("")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
