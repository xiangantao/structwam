#!/usr/bin/env python3
"""Policy-usefulness probe: do the latents help predict what the robot does next?

Train/held-out windows come from a ``cache_vace_teacher_local.py`` cache.  For every
source (cached teacher targets, each student checkpoint, and frozen pretrained
V-JEPA2 features) small pooled features are built from the past-only latents at the
last observed time and the future readout.  Ridge regression (alpha by
cross-validation, standardized features) predicts, from the LeRobot data itself:

* ``action_chunk``: the 8 actions after the observed segment (8 x 7);
* ``future_state``: end-effector translation over those 8 frames and the gripper
  state at their end (3 + 2).

Reported: held-out R^2 for proprio state alone and proprio state + each feature set.
Higher than the state-only / V-JEPA rows means the latent carries policy information.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

if os.environ.get("VACE_NPU", "0") == "1":
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))

from vace_latent_distill.config import ModelConfig, PipelineConfig
from vace_latent_distill.data import decode_indices, resize_rgb

GRID = 32
FIELDS = ("seg_k_l22h9", "seg_k_l15h7", "track_q", "track_k", "text_attention", "depth")
OBSERVED_LAST, FUTURE_LAST = 11, 12
LAST = {"observed": OBSERVED_LAST, "future": FUTURE_LAST}
BRANCHES = ("observed", "future")


def pool(field, size):
    """[cells, C] or [cells] -> [size*size*C] block means over the 32x32 grid."""
    value = np.asarray(field, np.float32).reshape(GRID, GRID, -1)
    step = GRID // size
    return value.reshape(size, step, size, step, -1).mean((1, 3)).ravel()


def motion(q, k, time):
    """Best-match displacement (dx, dy) and peak probability from t to t+1, pooled 8x8."""
    q = q[time] / np.maximum(np.linalg.norm(q[time], axis=-1, keepdims=True), 1e-8)
    k = k[time + 1] / np.maximum(np.linalg.norm(k[time + 1], axis=-1, keepdims=True), 1e-8)
    logits = q @ k.T / 0.07
    best = logits.argmax(1)
    peak = 1.0 / np.exp(logits - logits.max(1, keepdims=True)).sum(1)  # softmax maximum
    cells = np.arange(GRID * GRID)
    shift = np.stack(((best % GRID) - (cells % GRID), (best // GRID) - (cells // GRID), peak), -1)
    return pool(shift, 8)


def kk_motion(volume, before, after):
    """HEFT-style K-K displacement of an 8x8 lattice of cell centres, pixels."""
    from evaluate_kk_tracking import kk_track

    cells = np.arange(2, GRID, 4)
    start = np.stack(np.meshgrid(cells, cells, indexing="xy"), -1).reshape(-1, 2) * 8 + 3.5
    path = kk_track(volume, start, before, after)
    return (path[:, -1] - path[:, 0]).ravel() / 8


def slot_vector(values, heat, time, volume):
    """One prompt slot at one time: prompt+K region centre, extent, mean L15H7 K, K-K motion."""
    import demo_main_prompt_k_tracking as demo
    from evaluate_kk_tracking import kk_track

    k_fields = {22: values["seg_k_l22h9"], 15: values["seg_k_l15h7"]}
    score, selected = demo.region_from_prompt_and_k(heat, k_fields, time, GRID)
    weights = score.reshape(-1)[selected]
    weights = weights / max(float(weights.sum()), 1e-8)
    xy = np.stack((selected % GRID, selected // GRID), -1).astype(np.float32)
    centre = (weights[:, None] * xy).sum(0)
    spread = np.sqrt((weights[:, None] * (xy - centre) ** 2).sum(0))
    descriptor = (weights[:, None] * values["seg_k_l15h7"][time][selected]).sum(0)
    top = selected[np.argsort(weights)[-16:]]
    start = np.stack((top % GRID, top // GRID), -1) * 8 + 3.5
    path = kk_track(volume, start, time - 1, time)
    motion = (path[:, -1] - path[:, 0]).mean(0) / 8
    return np.concatenate([centre / GRID, spread / GRID, descriptor, motion]).astype(np.float32)


SLOT_SIZE = 134


def slots(fields, masks, limit, volume, branches=BRANCHES):
    """Fixed-size slot features (observed last + future last time), zero-padded to ``limit``."""
    features = []
    for branch in branches:
        values, time = fields[branch], LAST[branch]
        for mask in list(masks)[:limit] + [None] * max(0, limit - len(masks)):
            if mask is None:
                features.append(np.zeros(SLOT_SIZE, np.float32))
                continue
            heat = values["text_attention"][..., mask].sum(-1).reshape(-1, GRID, GRID)
            features.append(slot_vector(values, heat, time, volume[branch]))
    return np.concatenate(features)


def prompt_objects(fields, phrases, volume, branches=BRANCHES):
    """Prompt as a selector over the scoring-only phrase split (first two phrases)."""
    return slots(fields, list(phrases.values()), 2, volume, branches)


def token_slots(fields, content, mode, limit):
    """Parser-free slots from the teacher/student's own word maps at the last observed time.

    ``top``: the ``limit`` content tokens whose map is most concentrated (top-5% mass);
    ``groups``: neighbouring content tokens whose maps correlate > 0.6 are merged first;
    ``all``: every content token is a slot.  Slots keep instruction order.
    """
    attention = fields["observed"]["text_attention"][OBSERVED_LAST]
    indices = np.flatnonzero(content)
    maps = attention[:, indices]

    def concentration(column):
        column = column / max(float(column.sum()), 1e-8)
        return float(np.sort(column)[-51:].sum())

    if mode == "groups":
        groups = []
        for j in range(len(indices)):
            if groups and indices[j] == indices[groups[-1][-1]] + 1 \
                    and np.corrcoef(maps[:, j], maps[:, groups[-1][-1]])[0, 1] > 0.6:
                groups[-1].append(j)
            else:
                groups.append([j])
    else:
        groups = [[j] for j in range(len(indices))]
    if mode in ("top", "groups"):
        scored = sorted(groups, key=lambda g: -concentration(maps[:, g].sum(1)))[:limit]
        groups = sorted(scored, key=lambda g: g[0])
    masks = []
    for group in groups[:limit]:
        mask = np.zeros(len(content), bool)
        mask[indices[group]] = True
        masks.append(mask)
    return masks


def latent_features(fields, content, phrases=None, branches=BRANCHES):
    """Pooled features at the last time of each branch in ``branches``, concatenated."""
    from evaluate_kk_tracking import field_volume
    from vace_latent_distill.teacher import seed_heat

    chosen = [(fields[b], LAST[b]) for b in branches]
    features = {}
    for key, name in (("seg_k_l22h9", "seg_l22h9"), ("seg_k_l15h7", "seg_l15h7")):
        features[name] = np.concatenate([pool(f[key][t], 4) for f, t in chosen])
    features["depth"] = np.concatenate([pool(f["depth"][t], 8) for f, t in chosen])
    features["prompt"] = np.concatenate(
        [pool(seed_heat(f["text_attention"][[t]], content, "spatial_normalized_max")[0], 8) for f, t in chosen])
    features["motion"] = np.concatenate([motion(f["track_q"], f["track_k"], t - 1) for f, t in chosen])
    # K-K descriptor motion on the L15H7 K field (best student tracking descriptor).
    volume = {branch: field_volume(fields[branch], ("seg_k_l15h7", None)) for branch in branches}
    span = {"observed": 3, "future": 2}
    features["kk_motion"] = np.concatenate([kk_motion(volume[b], LAST[b] - span[b], LAST[b]) for b in branches])
    if phrases is not None:
        features["prompt_objects"] = prompt_objects(fields, phrases, volume, branches)
        for mode, limit in (("top", 3), ("groups", 3), ("all", 7)):
            features[f"slots_{mode}{limit}"] = slots(fields, token_slots(fields, content, mode, limit),
                                                     limit, volume, branches)
    return features


def source_features(fields, content, phrases, future_ablation):
    """Features from both branches; with ``future_ablation`` also observed-only copies (``name@obs``)."""
    features = latent_features(fields, content, phrases)
    if future_ablation:
        observed = latent_features(fields, content, phrases, ("observed",))
        features.update({f"{name}@obs": value for name, value in observed.items()})
    return features


def numpy_fields(fields):
    return {b: {k: v[k].float().numpy() for k in FIELDS} for b, v in fields.items() if b in ("observed", "future")}


def load_frames(dataset_root):
    import pyarrow.parquet as pq

    tables = [pq.read_table(p, columns=["episode_index", "frame_index", "observation.state", "action"])
              for p in sorted(Path(dataset_root).glob("data/chunk-*/file-*.parquet"))]
    lookup = {}
    for table in tables:
        data = table.to_pydict()
        for episode, frame, state, action in zip(data["episode_index"], data["frame_index"],
                                                 data["observation.state"], data["action"]):
            lookup[(episode, frame)] = (np.asarray(state, np.float32), np.asarray(action, np.float32))
    return lookup


def ridge_r2(train_x, train_y, test_x, test_y, groups, draws=1000):
    """Held-out R^2 and a 95% bootstrap interval that resamples held-out episodes."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    model = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(scaler.transform(train_x), train_y)
    prediction = model.predict(scaler.transform(test_x))
    r2 = float(r2_score(test_y, prediction, multioutput="uniform_average"))
    unique = np.unique(groups)
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(draws):
        chosen = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique))])
        samples.append(r2_score(test_y[chosen], prediction[chosen], multioutput="uniform_average"))
    low, high = np.percentile(samples, (2.5, 97.5))
    return r2, float(model.alpha_), float(low), float(high), prediction


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", default=[], help="name=path, repeatable")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-vjepa", action="store_true")
    parser.add_argument("--teacher-only", action="store_true", help="skip students and V-JEPA (CPU check)")
    parser.add_argument("--future-ablation", action="store_true",
                        help="also probe observed-only features and report the paired gain from the future branch")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig.load(args.cache_dir / "pipeline.json")
    rows = [json.loads(l) for l in (args.cache_dir / "manifest.jsonl").read_text().splitlines() if l.strip()]
    from vace_latent_distill.model import StructuredStudent
    from vace_latent_distill.offline import unpack_target
    from vace_latent_distill.teacher import content_token_masks

    content = content_token_masks(config.teacher.text_root, sorted({r["task"] for r in rows}))
    from transformers import AutoTokenizer

    from evaluate_vace_latents import phrases_of
    from vace_latent_distill.teacher import local_scripts

    local_scripts()
    tokenizer = AutoTokenizer.from_pretrained(Path(config.teacher.text_root) / "tokenizer", local_files_only=True)
    phrase_masks = {task: phrases_of(tokenizer, task, content[task]) for task in content}  # scoring-only split
    text_cache = torch.load(config.train.text_cache, map_location="cpu", weights_only=True)
    frames = load_frames(config.train.dataset_root)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    targets, base = {}, {}
    tasks = sorted({r["task"] for r in rows})
    instruction = {t: np.eye(len(tasks), dtype=np.float32)[i] for i, t in enumerate(tasks)}  # policy knows the task
    for row in rows:
        now = row["start"] + 32
        state_now = frames[(row["episode"], now)][0]
        actions = np.stack([frames[(row["episode"], now + i)][1] for i in range(1, 9)])
        state_end = frames[(row["episode"], now + 8)][0]
        targets[row["sample_id"]] = {"action_chunk": actions.ravel(),
                                     "future_state": np.concatenate([state_end[:3] - state_now[:3], state_end[6:8]])}
        base[row["sample_id"]] = {"state": state_now, "instr": instruction[row["task"]]}

    sources = {"teacher": {}}
    for row in rows:
        payload = torch.load(row["target_path"], map_location="cpu", weights_only=True)
        sources["teacher"][row["sample_id"]] = source_features(
            numpy_fields(unpack_target(payload, identity=row["provenance_id"])), content[row["task"]],
            phrase_masks[row["task"]], args.future_ablation)

    def frames_of(row):
        video = decode_indices(row["video"], row["frame_indices"][:33], row["fps"])
        return resize_rgb(torch.from_numpy(video), 256)

    models = []
    if args.teacher_only:
        args.checkpoint, args.no_vjepa = [], True
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        model = StructuredStudent(ModelConfig(**saved["config"]["model"]), pretrained=False)
        model.load_state_dict(saved["model"], strict=True)
        models.append((name, model.float().eval().to(device)))
        del saved
    vjepa = None
    if not args.no_vjepa:
        model_config = ModelConfig(**json.loads((args.cache_dir / "pipeline.json").read_text())["model"])
        vjepa = StructuredStudent(model_config, pretrained=True).float().eval()
        for fusion in vjepa.language:  # exact frozen V-JEPA2 features: no text fusion
            fusion.scale.zero_()
        vjepa = vjepa.to(device)
    for row in rows:
        rgb = frames_of(row)
        base[row["sample_id"]]["rgb"] = rgb[-1].float().numpy().transpose(1, 2, 0)[::32, ::32].ravel() / 255
        text = text_cache[row["task"]].float()[None].to(device)
        mask = text.abs().sum(-1) != 0
        batch = rgb[None].to(device)
        for name, model in models:
            output = model(batch, text, mask)
            fields = {b: {k: output[b][k][0].float().cpu() for k in FIELDS} for b in ("observed", "future")}
            sources.setdefault(name, {})[row["sample_id"]] = source_features(
                numpy_fields(fields), content[row["task"]], phrase_masks[row["task"]], args.future_ablation)
        if vjepa is not None:
            visual = vjepa.encode_past(batch, vjepa.text_projection(text), mask)[0, -1].float().cpu().numpy()
            base[row["sample_id"]]["vjepa"] = visual.reshape(16, 16, -1).reshape(4, 4, 4, 4, -1).mean((1, 3)).ravel()

    train = [r["sample_id"] for r in rows if r["split"] == "train"]
    test = [r["sample_id"] for r in rows if r["split"] == "heldout"]
    groups = np.asarray([r["episode"] for r in rows if r["split"] == "heldout"])  # both cameras together

    def stack(ids, parts):
        return np.stack([np.concatenate([parts[i][name] for name in sorted(parts[i])]) for i in ids])

    results, predictions = [], {}
    for goal in ("action_chunk", "future_state"):
        y_train = np.stack([targets[i][goal] for i in train])
        y_test = np.stack([targets[i][goal] for i in test])
        candidates = [("state", None, ("state",)), ("state+instr", None, ("state", "instr")),
                      ("state+rgb", None, ("state", "rgb"))]
        if vjepa is not None:
            candidates += [("state+vjepa", None, ("state", "vjepa")),
                           ("state+instr+vjepa", None, ("state", "instr", "vjepa"))]
        for source in sources:
            available = next(iter(sources[source].values()))
            names = sorted(n for n in available if not n.endswith("@obs"))
            for name in names:
                candidates.append((f"state+{source}:{name}", source, ("state", name)))
                candidates.append((f"state+instr+{source}:{name}", source, ("state", "instr", name)))
                if f"{name}@obs" in available:
                    candidates.append((f"state+instr+{source}:{name}@obs", source, ("state", "instr", f"{name}@obs")))
            candidates.append((f"state+instr+{source}:all", source, ("state", "instr", *names)))
            if args.future_ablation:
                candidates.append((f"state+instr+{source}:all@obs", source,
                                   ("state", "instr", *(f"{n}@obs" for n in names))))
            if vjepa is not None:
                candidates.append((f"state+instr+vjepa+{source}:all", source, ("state", "instr", "vjepa", *names)))
        for label, source, keys in candidates:
            def parts(ids):
                return {i: {k: (sources[source][i][k] if source and k in sources[source][i] else base[i][k])
                            for k in keys} for i in ids}
            r2, alpha, low, high, prediction = ridge_r2(stack(train, parts(train)), y_train,
                                                        stack(test, parts(test)), y_test, groups)
            predictions[(goal, label)] = prediction
            results.append({"target": goal, "inputs": label, "heldout_r2": r2, "ci95": [low, high],
                            "alpha": alpha})
            print(json.dumps(results[-1]), flush=True)
        # Paired bootstrap against reference baselines on identical episode resamples.
        from sklearn.metrics import r2_score

        rng = np.random.default_rng(1)
        unique = np.unique(groups)
        draws = [np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique))])
                 for _ in range(1000)]
        def paired(mine, reference):
            deltas = [r2_score(y_test[d], mine[d]) - r2_score(y_test[d], reference[d]) for d in draws]
            return [float(v) for v in np.percentile(deltas, (2.5, 97.5))]

        for reference_name, key in (("state+instr", "delta_vs_instr_ci95"), ("state+instr+vjepa", "delta_vs_vjepa_ci95")):
            if (goal, reference_name) not in predictions:
                continue
            for result in results:
                if result["target"] == goal:
                    result[key] = paired(predictions[(goal, result["inputs"])], predictions[(goal, reference_name)])
        for result in results:  # gain from adding the future branch to the same observed-only feature
            if result["target"] == goal and (goal, result["inputs"] + "@obs") in predictions:
                result["delta_vs_observed_ci95"] = paired(predictions[(goal, result["inputs"])],
                                                          predictions[(goal, result["inputs"] + "@obs")])
    (args.output / "probe_results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Policy-usefulness probe (ridge, held-out R^2; higher is better)", "",
             f"train windows {len(train)}, held-out windows {len(test)}; "
             "[95% bootstrap interval over held-out episodes]", "",
             "| inputs | action_chunk | future_state | Δ vs state+instr (action / state) "
             "| Δ vs state+instr+vjepa (action / state) | Δ future branch vs @obs (action / state) |",
             "|---|---|---|---|---|---|"]
    for label in dict.fromkeys(r["inputs"] for r in results):
        cells = {r["target"]: f"{r['heldout_r2']:.3f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"
                 for r in results if r["inputs"] == label}
        texts = []
        for key in ("delta_vs_instr_ci95", "delta_vs_vjepa_ci95", "delta_vs_observed_ci95"):
            delta = {r["target"]: r.get(key) for r in results if r["inputs"] == label}
            texts.append(" / ".join("–" if delta[g] is None else f"[{delta[g][0]:+.2f}, {delta[g][1]:+.2f}]"
                                    for g in ("action_chunk", "future_state")))
        lines.append(f"| {label} | {cells['action_chunk']} | {cells['future_state']} | {' | '.join(texts)} |")
    (args.output / "probe_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
