#!/usr/bin/env python3
"""Check the policy maps (A) and keypoint tracks (C) on held-out windows against simulator GT.

Inputs are the saved student fields of the held-out windows (no future frames enter the
maps); simulator GT is used only to score them:

* heat: does the argmax of the target / destination heat land on the object that the episode
  moves / on the object it ends up on; where a same-category distractor is visible (two black
  bowls), is the heat higher on the right one.  Also for the automatically selected words.
* motion filter: of the cells kept, how many lie on GT objects or the gripper (before vs after
  filtering); hand cells against the GT gripper; how many GT moving cells survive.
* C keypoints: end-point error against the GT point under each keypoint, past and future, vs
  holding the present position.

Writes ``summary.json`` / ``summary.md`` and per-window renders (``--render``).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CAMERAS = {"observation.images.image": "agentview", "observation.images.image2": "robot0_eye_in_hand"}
HAND = ("gripper base", "left finger", "right finger", "wrist")
ROLE_COLOURS = {0: (40, 150, 255), 1: (60, 220, 60), 2: (255, 200, 40)}        # BGR: hand orange, target green, destination cyan
STATE = {}


def init_worker(text_root, words_path, k_basis, heat_source):
    from transformers import AutoTokenizer

    HEAT_SOURCE["name"] = heat_source
    torch.set_num_threads(2)
    STATE["tokenizer"] = AutoTokenizer.from_pretrained(Path(text_root) / "tokenizer", local_files_only=True)
    STATE["words"] = json.loads(Path(words_path).read_text())
    STATE["k_basis"] = k_basis


HEAT_SOURCE = {"name": "mean"}


def load_fields(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    observed, future = saved["observed"], saved["future"]
    attention = {"future": future["text_attention"][8], "observed": observed["text_attention"][11],
                 "mean": torch.cat((observed["text_attention"][8:12], future["text_attention"][8:9])).float().mean(0)}
    now = attention[HEAT_SOURCE["name"]]
    return {"track_k": torch.cat((observed["track_k"][4:12], future["track_k"][8:13]), 0).float(),
            "seg_k_l22h9": future["seg_k_l22h9"][8].float(), "seg_k_l15h7": future["seg_k_l15h7"][8].float(),
            "text_attention": torch.stack((now.float(), future["text_attention"][12].float())).numpy(),
            "depth": future["depth"][8].float().numpy()}


def present_k_job(path):
    from latent_dp.policy_maps import present_k

    torch.set_num_threads(2)
    fields = load_fields(path)
    k = present_k(fields["seg_k_l22h9"], fields["seg_k_l15h7"]).numpy()
    return k[np.random.default_rng(0).choice(len(k), 256, replace=False)]


def heft_tracker(track_k):
    from vace_latent_distill.policy_features import track_many

    def run(cells):
        starts = (np.stack((cells % 32, cells // 32), -1) * 8 + 3.5).astype(np.float32)[None]
        sequence = track_k[None]
        forward, forward_sim = track_many(sequence, starts, (8, 9, 10, 11, 12))
        backward, backward_sim = track_many(sequence, starts, (8, 7, 6, 5, 4, 3, 2, 1, 0))
        pixels = torch.cat((backward.flip(2)[:, :, :8], forward), 2)[0]
        similarity = torch.cat((backward_sim.flip(2)[:, :, :8], forward_sim), 2)[0]
        return pixels.numpy(), similarity.numpy()

    return run


def dilate(cells, size=1):
    from scipy import ndimage

    grid = np.zeros((32, 32), bool)
    cells = np.asarray(sorted(cells), int)
    if len(cells):
        grid.flat[cells] = True
    return ndimage.binary_dilation(grid, np.ones((2 * size + 1,) * 2)).ravel() if size else grid.ravel()


def prefix(name):
    match = re.match(r"(.+?_\d+)", name)
    return match.group(1) if match else name


def is_object(name):
    return not name.startswith(("robot0_", "gripper0_", "mount0_")) and name not in ("world", "table")


ADJECTIVES = {"black", "white", "yellow", "red", "left", "right", "top", "middle", "bottom", "back", "front",
              "wooden", "center", "table", "both"}
ALIASES = {"pots": "pot", "compartment": "caddy"}


def named_groups(phrase, names):
    """Object groups (by instance prefix) whose sim name contains a noun of ``phrase`` (scoring only)."""
    words = [ALIASES.get(w, w) for w in phrase.split()]
    groups = defaultdict(list)
    for i, name in enumerate(names):
        if not is_object(name):
            continue
        tokens = name.split("_")
        if "drawer" in words:
            level = [w for w in words if w in ("top", "middle", "bottom")]
            if "cabinet" in tokens and tokens[-1] in ("top", "middle", "bottom") and (not level or tokens[-1] == level[0]):
                groups[name].append(i)
            continue
        if any(w in tokens for w in words if w not in ADJECTIVES):
            groups[prefix(name)].append(i)
    return groups


def ground_truth(row, root, roles):
    """GT at the present frame: hand / object cells, moving cells, target / destination / distractor cells.

    The target is the instance of the instruction's target noun that the episode moves most;
    the destination the instance of the destination noun nearest to the target's final position.
    """
    camera = CAMERAS[row["camera"]]
    episode = row["tracking_episode"]
    with np.load(root / "tracks" / f"episode_{episode:05d}_{camera}_visible_surface.npz") as archive:
        tracks = {k: archive[k] for k in ("xy", "visible", "point_label", "body_id", "body_names",
                                          "annotated_frame_indices")}
    tracks["xy"] = tracks["xy"].copy()
    tracks["xy"][..., 0] = 255 - tracks["xy"][..., 0]
    xpos = np.load(root / "videos" / f"episode_{episode:05d}" / "sim_state.npz")["body_xpos"]
    names = json.loads((root / "videos" / f"episode_{episode:05d}" / "metadata.json").read_text())["body_names"]
    present = row["start"] + 32
    annotated = set(tracks["annotated_frame_indices"].tolist())
    if present >= len(tracks["xy"]) or present not in annotated:
        return None
    point_body = np.asarray([str(tracks["body_names"][b]) for b in tracks["body_id"]])
    visible = tracks["visible"][present]
    xy = tracks["xy"][present]
    safe = np.nan_to_num(xy, nan=-1.0)
    cell = (np.clip(np.floor(safe[:, 1] / 8), 0, 31) * 32 + np.clip(np.floor(safe[:, 0] / 8), 0, 31)).astype(int)
    hand_point = np.isin(tracks["point_label"], HAND)
    travel = np.linalg.norm(xpos[10:] - xpos[10:11], axis=-1).max(0)          # after the objects settle

    def cells_of(mask):
        return set(cell[visible & mask].tolist())

    def group_cells(members):
        return cells_of(np.isin(point_body, [names[i] for i in members]))

    result = {"tracks": tracks, "present": present, "annotated": annotated, "point_body": point_body,
              "hand": cells_of(hand_point), "object": cells_of(~hand_point), "things": cells_of(np.ones(len(xy), bool)),
              "target": set(), "target_name": None, "destination": set(), "destination_name": None, "distractor": set()}
    targets = named_groups(roles.get("target_1", ""), names)
    if targets:
        chosen = max(targets, key=lambda g: travel[targets[g]].max())
        result["target"], result["target_name"] = group_cells(targets[chosen]), chosen
        result["distractor"] = set().union(*[group_cells(m) for g, m in targets.items() if g != chosen])
        final = xpos[-1, targets[chosen]].mean(0)
        destinations = named_groups(roles.get("destination_1", ""), names)
        destinations.pop(chosen, None)
        if destinations:
            near = min(destinations, key=lambda g: np.linalg.norm(xpos[-1, destinations[g]] - final, axis=-1).min())
            result["destination"], result["destination_name"] = group_cells(destinations[near]), near
    moving = np.zeros(len(xy), bool)
    for frame in (present - 16, present + 16):
        if 0 <= frame < len(tracks["xy"]) and frame in annotated:
            both = visible & tracks["visible"][frame]
            moving |= both & (np.linalg.norm(tracks["xy"][frame] - xy, axis=-1) > 8)
    result["moving"], result["moving_points"] = cells_of(moving), moving
    return result


def heat_metrics(heat, tokens, readable, goals, gt):
    from latent_dp.policy_maps import HEAT_CHANNELS, WORD_HEAT

    out = {}
    things = dilate(gt["things"])
    targets = dilate(gt["target"])
    destination = dilate(gt["destination"])
    for h, name in enumerate(HEAT_CHANNELS):
        if not tokens[name]:
            continue
        top = int(np.argmax(heat[h]))
        out[f"{name}_on_things"] = float(things[top])
        if name == "target_1" and gt["target"]:
            out[f"{name}_hit"] = float(targets[top])
            out[f"{name}_chance"] = float(targets.mean())
        if name == "destination_1" and gt["destination"]:
            out["destination_1_hit"] = float(destination[top])
            out["destination_1_chance"] = float(destination.mean())
    words = [h for h, n in enumerate(HEAT_CHANNELS) if n in WORD_HEAT and tokens[n]]
    if words and gt["target"]:
        out["words_best_target_hit"] = float(max(targets[int(np.argmax(heat[h]))] for h in words))
        if gt["destination"]:
            out["words_any_destination_hit"] = float(max(destination[int(np.argmax(heat[h]))] for h in words))
    if gt["distractor"] and gt["target"]:
        right, wrong = sorted(gt["target"]), sorted(gt["distractor"] - gt["target"])
        if wrong:
            t1 = HEAT_CHANNELS.index("target_1")
            out["target_1_right_instance"] = float(heat[t1][right].mean() > heat[t1][wrong].mean())
            if words:
                combined = np.mean([np.exp(heat[h]) for h in words], 0)
                out["words_right_instance"] = float(combined[right].mean() > combined[wrong].mean())
    return out


def filter_metrics(diag, gt):
    things, hand_gt, moving_gt = dilate(gt["things"]), dilate(gt["hand"]), sorted(gt["moving"])
    raw = diag["raw_moving"]
    kept = diag["kept"]
    hand = diag["roles"]["hand"] & kept
    out = {"raw_cells": int(raw.sum()), "kept_cells": int(kept.sum()), "hand_cells": int(hand.sum())}
    if raw.any():
        out["raw_on_things"] = float(things[raw].mean())
    if kept.any():
        out["kept_on_things"] = float(things[kept].mean())
    if hand.any():
        out["hand_on_gripper"] = float(hand_gt[hand].mean())
    if gt["hand"]:
        out["gripper_found"] = float(dilate(np.flatnonzero(hand))[sorted(gt["hand"])].mean()) if hand.any() else 0.0
    for role, truth in (("target", gt["target"]), ("destination", gt["destination"])):
        mask = diag["roles"][role]
        if mask.any() and truth:
            on = dilate(truth)[mask].mean()
            out[f"{role}_segment_on_gt"] = float(on)
            out[f"{role}_segment_hit"] = float(on >= 0.5)
    if moving_gt:
        out["moving_recall_raw"] = float(dilate(np.flatnonzero(raw))[moving_gt].mean()) if raw.any() else 0.0
        out["moving_recall_kept"] = float(dilate(np.flatnonzero(kept))[moving_gt].mean()) if kept.any() else 0.0
    return out


def keypoint_metrics(keypoints, gt):
    from latent_dp.policy_maps import PRESENT, ROLES, STEPS

    tracks, present = gt["tracks"], gt["present"]
    visible = tracks["visible"][present]
    candidates = np.flatnonzero(visible)
    errors = defaultdict(list)
    matched = defaultdict(list)
    valid_steps = defaultdict(list)
    for i in np.flatnonzero(keypoints["role"] >= 0):
        role = ROLES[keypoints["role"][i]]
        pixels = keypoints["xy"][i] * 8 + 3.5
        distance = np.linalg.norm(tracks["xy"][present, candidates] - pixels[PRESENT], axis=-1)
        matched[role].append(float(distance.min() <= 6) if len(candidates) else 0.0)
        if not len(candidates) or distance.min() > 6:
            continue
        g = candidates[int(np.argmin(distance))]
        moving = bool(gt["moving_points"][g])
        for k in range(STEPS):
            frame = present + 4 * (k - PRESENT)
            if k == PRESENT or not (0 <= frame < len(tracks["xy"])) or frame not in gt["annotated"] \
                    or not tracks["visible"][frame, g]:
                continue
            valid_steps[role].append(float(keypoints["valid"][i, k]))
            if not keypoints["valid"][i, k]:
                continue
            span = "past" if k < PRESENT else "future"
            truth = tracks["xy"][frame, g]
            for subset in ("all", "moving") if moving else ("all",):
                errors[f"{role}_{span}_{subset}_epe"].append(float(np.linalg.norm(pixels[k] - truth)))
                errors[f"{role}_{span}_{subset}_hold"].append(float(np.linalg.norm(pixels[PRESENT] - truth)))
    out = {f"{role}_on_things": float(np.mean(v)) for role, v in matched.items() if v}
    out.update({f"{role}_valid_steps": float(np.mean(v)) for role, v in valid_steps.items() if v})
    out.update({key: float(np.mean(v)) for key, v in errors.items()})
    out.update({f"{key}_n": len(v) for key, v in errors.items()})
    return out


def present_frame(row):
    from vace_latent_distill.data import decode_indices, resize_rgb

    video = decode_indices(row["video"], [row["frame_indices"][32]], row["fps"])
    frame = resize_rgb(torch.from_numpy(video), 256).numpy().transpose(0, 2, 3, 1)[0]
    return np.ascontiguousarray(frame.clip(0, 255).astype(np.uint8))


def render(row, frame, maps, keypoints, diag, readable, gt, metrics, path):
    import cv2

    from latent_dp.policy_maps import HEAT_CHANNELS, MAP_CHANNELS, PRESENT, STEPS

    scale = 2
    base = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
    size = base.shape[0]

    def px(cells_xy):
        return np.round((cells_xy + 0.5) * 8 * scale).astype(int)

    def title(image, lines):
        bar = np.full((22 * len(lines) + 8, image.shape[1], 3), 24, np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(bar, line, (8, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return np.vstack((bar, image))

    positions, valid = diag["dense"]["positions"], diag["dense"]["valid"]
    displacement = np.linalg.norm(diag["displacement"], axis=-1).max(0)
    # 1: before filtering -- every cell whose raw chained track moves more than half a cell.
    before = (base * 0.55).astype(np.uint8)
    for c in np.flatnonzero(displacement > 0.5):
        line = px(positions[:, c])
        cv2.polylines(before, [line[: PRESENT + 1].reshape(-1, 1, 2)], False, (200, 160, 120), 1, cv2.LINE_AA)
        cv2.polylines(before, [line[PRESENT:].reshape(-1, 1, 2)], False, (90, 90, 255), 1, cv2.LINE_AA)
    # 2: after filtering -- kept cells, coloured by role; past thin, future thick.
    after = (base * 0.55).astype(np.uint8)
    overlay = after.copy()
    for r, role in enumerate(("hand", "target", "destination")):
        mask = (diag["roles"][role] & diag["kept"]).reshape(32, 32)
        big = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay[big] = (0.5 * overlay[big] + 0.5 * np.array(ROLE_COLOURS[r])).astype(np.uint8)
    after = cv2.addWeighted(overlay, 0.6, after, 0.4, 0)
    for r, role in enumerate(("hand", "target", "destination")):
        for c in np.flatnonzero(diag["roles"][role] & diag["kept"]):
            ok = valid[:, c]
            line = px(positions[:, c])
            past = [p for k, p in enumerate(line[: PRESENT + 1]) if ok[k]]
            future = [p for k, p in enumerate(line[PRESENT:], PRESENT) if ok[k]]
            if len(past) > 1:
                cv2.polylines(after, [np.asarray(past).reshape(-1, 1, 2)], False, ROLE_COLOURS[r], 1, cv2.LINE_AA)
            if len(future) > 1:
                cv2.polylines(after, [np.asarray(future).reshape(-1, 1, 2)], False, ROLE_COLOURS[r], 2, cv2.LINE_AA)
    # 3: role heat (target green, destination blue, locator magenta).
    def heat_overlay(channels):
        image = (base * 0.45).astype(np.float32)
        for name, colour in channels:
            h = HEAT_CHANNELS.index(name)
            heat = diag["heat"][1 if name.startswith("destination") else 0]
            if not np.any(heat[h]):
                continue
            value = np.clip(heat[h], 0, None)
            value = cv2.resize((value / max(value.max(), 1e-6)).reshape(32, 32).astype(np.float32), (size, size),
                               interpolation=cv2.INTER_CUBIC)[..., None].clip(0, 1)
            image = image * (1 - 0.75 * value) + 0.75 * value * np.array(colour, np.float32)
            top = int(np.argmax(heat[h]))
            cv2.drawMarker(image, (int((top % 32 + 0.5) * 8 * scale), int((top // 32 + 0.5) * 8 * scale)), colour,
                           cv2.MARKER_CROSS, 18, 2)
        return image.clip(0, 255).astype(np.uint8)

    roles_image = heat_overlay((("target_1", (60, 230, 60)), ("destination_1", (255, 140, 40)),
                                ("locator", (230, 60, 230))))
    words_image = heat_overlay((("word_1", (60, 60, 255)), ("word_2", (60, 230, 60)), ("word_3", (255, 140, 40))))
    # 5: C keypoints, time-coloured (blue past -> white present -> red future), GT trails white.
    keys = (base * 0.55).astype(np.uint8)
    tracks = gt["tracks"]
    candidates = np.flatnonzero(tracks["visible"][gt["present"]])
    for i in np.flatnonzero(keypoints["role"] >= 0):
        pixels = keypoints["xy"][i] * 8 + 3.5
        if len(candidates):
            distance = np.linalg.norm(tracks["xy"][gt["present"], candidates] - pixels[PRESENT], axis=-1)
            if distance.min() <= 6:
                g = candidates[int(np.argmin(distance))]
                frames = [gt["present"] + 4 * (k - PRESENT) for k in range(STEPS)]
                truth = [tracks["xy"][f, g] * scale for f in frames
                         if 0 <= f < len(tracks["xy"]) and f in gt["annotated"] and tracks["visible"][f, g]]
                if len(truth) > 1:
                    cv2.polylines(keys, [np.round(truth).astype(int).reshape(-1, 1, 2)], False, (255, 255, 255), 1,
                                  cv2.LINE_AA)
        line = np.round(pixels * scale).astype(int)
        for k in range(STEPS - 1):
            if not (keypoints["valid"][i, k] and keypoints["valid"][i, k + 1]):
                continue
            t = (k + 0.5) / (STEPS - 1)
            colour = (255, int(255 * t / 0.62), int(255 * t / 0.62)) if k < PRESENT else \
                (int(255 * (1 - (k - PRESENT) / 4)), int(255 * (1 - (k - PRESENT) / 4)), 255)
            cv2.line(keys, tuple(line[k]), tuple(line[k + 1]), tuple(int(min(c, 255)) for c in colour), 2, cv2.LINE_AA)
        cv2.circle(keys, tuple(line[PRESENT]), 4, ROLE_COLOURS[int(keypoints["role"][i])], -1, cv2.LINE_AA)

    camera = CAMERAS[row["camera"]]
    tiles = [title(before, ["1 before: raw K-K tracks", "(past blue-grey, future red)"]),
             title(after, ["2 after: hand orange / target green", "destination cyan (future thick)"]),
             title(roles_image, [f"3 role heat T={readable.get('target_1', '')[:18]}",
                                 f"D(+16)={readable.get('destination_1', '')[:12]} L={readable.get('locator', '')[:12]}"]),
             title(words_image, [f"4 auto words: R={readable.get('word_1', '')[:10]}",
                                 f"G={readable.get('word_2', '')[:10]} B={readable.get('word_3', '')[:10]}"]),
             title(keys, ["5 C keypoints (blue past > white > red future)", "GT trails white"])]
    strip = np.hstack(tiles)
    head = (f"{row['sample_id']}  {camera}  |  {row['task'][:90]}")
    summary = "  ".join(f"{k}={v:.2f}" for k, v in metrics.items()
                        if isinstance(v, float) and k in ("target_1_hit", "destination_1_hit_+16", "target_1_right_instance",
                                                          "raw_on_things", "kept_on_things", "hand_on_gripper",
                                                          "gripper_found", "moving_recall_kept"))
    strip = title(strip, [head, summary])
    cv2.imwrite(str(path), strip)

    # Channel atlas: every A channel, 32x32 upscaled.
    cell = 112
    columns = 9
    rows = int(np.ceil(len(MAP_CHANNELS) / columns))
    atlas = np.full((rows * (cell + 18), columns * cell, 3), 24, np.uint8)
    for i, name in enumerate(MAP_CHANNELS):
        value = maps[i]
        signed = value.min() < 0
        limit = max(float(np.abs(value).max()), 1e-6)
        norm = (value / limit + 1) / 2 if signed else value / limit
        image = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT_SHIFTED if signed
                                  else cv2.COLORMAP_VIRIDIS)
        image = cv2.resize(image, (cell, cell), interpolation=cv2.INTER_NEAREST)
        y, x = (i // columns) * (cell + 18), (i % columns) * cell
        atlas[y + 18:y + 18 + cell, x:x + cell] = image
        cv2.putText(atlas, name[:16], (x + 3, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.imwrite(str(path.with_name(path.stem + "_channels.png")), atlas)


def window_job(job):
    from latent_dp.policy_maps import instruction_roles, instruction_words, role_tokens, window_maps

    row, fields_path, root, output, draw = job
    fields = load_fields(fields_path)
    selected = STATE["words"].get(row["task"])
    tokens, readable = role_tokens(row["task"], STATE["tokenizer"], selected)
    goals = len(instruction_roles(instruction_words(row["task"], STATE["tokenizer"])[0])[0])
    wrist = CAMERAS[row["camera"]] == "robot0_eye_in_hand"
    maps, keypoints, diag = window_maps(fields, tokens, wrist, heft_tracker(fields["track_k"]), STATE["k_basis"])
    diag["single_goal"] = goals == 1
    gt = ground_truth(row, root, readable)
    metrics = {"sample_id": row["sample_id"], "suite": row["suite"], "camera": "wrist" if wrist else "third",
               "task": row["task"], "roles": readable}
    if gt is not None:
        metrics.update(heat_metrics(diag["heat"][0], tokens, readable, goals, gt))
        metrics.update({f"{k}_+16": v for k, v in heat_metrics(diag["heat"][1], tokens, readable, goals, gt).items()
                        if k.startswith("destination_1") or k.startswith("target_1")})
        metrics.update(filter_metrics(diag, gt))
        metrics.update(keypoint_metrics(keypoints, gt))
        metrics["gt_target"], metrics["gt_destination"] = gt["target_name"], gt["destination_name"]
    if draw and gt is not None:
        render(row, present_frame(row), maps, keypoints, diag, readable, gt, metrics,
               output / "renders" / f"{row['sample_id']}.png")
    return metrics


def aggregate(results):
    groups = defaultdict(list)
    for r in results:
        groups["all"].append(r)
        groups[r["camera"]].append(r)
        groups[f"{r['suite']}/{r['camera']}"].append(r)
    keys = sorted({k for r in results for k, v in r.items() if isinstance(v, float)})
    table = {}
    for name, members in groups.items():
        table[name] = {k: round(float(np.mean([m[k] for m in members if k in m])), 3)
                       for k in keys if any(k in m for m in members)}
        table[name]["windows"] = len(members)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True, help="directory of saved student fields <sample_id>.pt")
    parser.add_argument("--words", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True, help="for pipeline.json (tokenizer path)")
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--render", type=int, default=80, help="render the first N windows")
    parser.add_argument("--heat-source", choices=("future", "observed", "mean"), default="mean",
                        help="present heat from future slot 8 (frame 32), observed time 11 (frame 28) or their mean over frames 16..32")
    args = parser.parse_args()
    from latent_dp.policy_maps import APPEARANCE
    from vace_latent_distill.config import PipelineConfig

    (args.output / "renders").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    text_root = PipelineConfig.load(args.cache_dir / "pipeline.json").teacher.text_root
    paths = [args.fields / f"{r['sample_id']}.pt" for r in rows]
    with ProcessPoolExecutor(args.workers) as pool:
        sample = np.concatenate(list(pool.map(present_k_job, paths)))
    mean = sample.mean(0)
    basis = np.linalg.svd(sample - mean, full_matrices=False)[2][:APPEARANCE]
    jobs = [(r, p, args.tracking_root, args.output, i < args.render) for i, (r, p) in enumerate(zip(rows, paths))]
    with ProcessPoolExecutor(args.workers, initializer=init_worker,
                             initargs=(text_root, args.words, (mean, basis), args.heat_source)) as pool:
        results = list(pool.map(window_job, jobs))
    (args.output / "per_window.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    table = aggregate(results)
    (args.output / "summary.json").write_text(json.dumps(table, indent=1) + "\n")
    for name in ("all", "third", "wrist"):
        print(name, json.dumps(table.get(name, {})))


if __name__ == "__main__":
    main()
