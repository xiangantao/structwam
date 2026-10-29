#!/usr/bin/env python3
"""Scan main-tower Q/K for linearly decodable LIBERO interaction outcomes.

The input contract is deliberately explicit and privileged:

* each sample is a native 49-frame, 20 Hz wrist-camera window;
* frames 0..32 are encoded into the VACE control tower, with no warmup copy;
* the full 49-frame recorded video is VAE-encoded and mixed with Gaussian
  noise at the scheduler sigma nearest to 0.8 in the 40-step flow schedule;
* the task instruction is supplied to the main tower as its text condition;
* only main-tower ``attn1`` Q/K are captured (the control tower is not ranked).

This is a teacher-side diagnostic: the future 16 frames are intentionally in
the main latent.  It tests whether a frozen main-tower head contains a linear
readout of the automatic grasp/place outcome labels; it is not a causal
interaction predictor.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "heft/diffusers/src"))
sys.path.insert(0, str(ROOT / "heft/latent_policy/scripts"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from diffusers import UniPCMultistepScheduler  # noqa: E402
from evaluate_vace_pure_noise_future import (  # noqa: E402
    load_transformer,
    load_vae,
    read_video,
)


EVENT_INDEX = 32
TOTAL_FRAMES = 49
OBSERVED_FRAMES = 33
DEFAULT_STATES = (0, 5, 10, 13, 18, 22)
NEGATIVE_PROMPT = "overexposed, blurry, subtitles, low quality, deformed, extra fingers, camera cut"


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def log(**values: object) -> None:
    print(json.dumps(values, allow_nan=False), flush=True)


def encode_prompts(model_root: Path, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    """Encode each task instruction once and keep only frozen CPU copies."""
    from transformers import AutoTokenizer, UMT5EncoderModel

    unique = sorted(set(prompts) | {NEGATIVE_PROMPT})
    tokenizer = AutoTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(
        model_root / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
    ).to(device).eval()
    result: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for start in range(0, len(unique), 4):
            batch = unique[start : start + 4]
            tokens = tokenizer(
                batch, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
            ).to(device)
            values = encoder(**tokens).last_hidden_state
            for index, text in enumerate(batch):
                length = int(tokens.attention_mask[index].sum())
                values[index, length:] = 0
                result[text] = values[index : index + 1].detach().cpu()
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def prepare_inputs(vae, truth: np.ndarray, size: int, observed: int, device: torch.device):
    """Build the exact no-warmup control and full-GT main latents."""
    pixels = torch.from_numpy(truth.copy()).to(device).permute(3, 0, 1, 2).float() / 127.5 - 1
    pixels = F.interpolate(
        pixels.permute(1, 0, 2, 3), size=(size, size), mode="bilinear", align_corners=False
    ).permute(1, 0, 2, 3)

    control_pixels = torch.zeros((3, len(truth), size, size), device=device)
    control_pixels[:, :observed] = pixels[:, :observed]
    inactive = vae.encode([control_pixels])[0]
    reactive = vae.encode([torch.zeros_like(control_pixels)])[0]
    expected_shape = (16, (len(truth) - 1) // 4 + 1, size // 8, size // 8)
    if tuple(inactive.shape) != expected_shape:
        raise RuntimeError(f"Unexpected VAE control shape {tuple(inactive.shape)} != {expected_shape}")

    mask = torch.ones((len(truth), size, size), device=device)
    mask[:observed] = 0
    mask = mask.view(len(truth), size // 8, 8, size // 8, 8)
    mask = mask.permute(2, 4, 0, 1, 3).flatten(0, 1)
    mask = F.interpolate(mask[None], size=inactive.shape[1:], mode="nearest-exact")[0]
    control = torch.cat((inactive, reactive, mask), dim=0)[None].bfloat16()
    clean = vae.encode([pixels])[0][None].float()
    if tuple(clean.shape) != (1, *expected_shape):
        raise RuntimeError(f"Unexpected VAE main shape {tuple(clean.shape)} != {(1, *expected_shape)}")
    return control, clean


def pool_qk(query: torch.Tensor, key: torch.Tensor, event_time: int, temporal: int):
    """Return compact per-head descriptors without materializing attention maps.

    The descriptor retains channel-wise spatial statistics at three temporal
    regions (pre-event, event, future) and two temporal deltas.  It is fixed
    before looking at labels, so head rankings are not produced by a learned
    clustering or a hand-selected object mask.
    """
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError(f"Unexpected Q/K shape: {tuple(query.shape)} / {tuple(key.shape)}")
    heads, tokens, dim = query.shape[1], query.shape[2], query.shape[3]
    if tokens % temporal:
        raise ValueError(f"Token count {tokens} is not divisible by temporal length {temporal}")
    spatial = tokens // temporal
    q = query[0].float().reshape(heads, temporal, spatial, dim)
    k = key[0].float().reshape(heads, temporal, spatial, dim)

    def describe(x: torch.Tensor) -> torch.Tensor:
        # [H,T,S,D] -> three spatial summaries [H,T,D].
        mean = x.mean(dim=2)
        std = x.std(dim=2, unbiased=False)
        regions = (
            mean[:, :event_time].mean(dim=1),
            mean[:, event_time],
            mean[:, event_time + 1 :].mean(dim=1),
            std[:, :event_time].mean(dim=1),
            std[:, event_time],
            std[:, event_time + 1 :].mean(dim=1),
        )
        pre, event, future = regions[0], regions[1], regions[2]
        return torch.cat(
            (*regions, event - pre, future - event), dim=-1
        )

    q_features = describe(q)
    k_features = describe(k)
    return q_features.cpu().numpy().astype(np.float16), k_features.cpu().numpy().astype(np.float16)


class MainQKCapture:
    def __init__(self, temporal: int, event_time: int):
        self.temporal = temporal
        self.event_time = event_time
        self.enabled = False
        self.state = -1
        self.q: dict[int, np.ndarray] = {}
        self.k: dict[int, np.ndarray] = {}

    def clear(self, state: int, enabled: bool) -> None:
        self.state = state
        self.enabled = enabled
        self.q.clear()
        self.k.clear()

    def __call__(self, layer: int, query: torch.Tensor, key: torch.Tensor, **_: object) -> None:
        if not self.enabled:
            return
        q, k = pool_qk(query, key, self.event_time, self.temporal)
        self.q[layer] = q
        self.k[layer] = k

    def arrays(self, layers: int, heads: int) -> tuple[np.ndarray, np.ndarray]:
        if sorted(self.q) != list(range(layers)) or sorted(self.k) != list(range(layers)):
            raise RuntimeError(f"Missing main Q/K captures at state {self.state}: {sorted(self.q)} / {sorted(self.k)}")
        q = np.stack([self.q[layer] for layer in range(layers)])
        k = np.stack([self.k[layer] for layer in range(layers)])
        if q.shape[0:2] != (layers, heads):
            raise RuntimeError(f"Unexpected pooled feature shape {q.shape}")
        return q, k


def install_hooks(model, capture: MainQKCapture) -> None:
    for layer, block in enumerate(model.blocks):
        block.attn1._heft_capture = lambda layer=layer, **kw: capture(layer, kw["query"], kw["key"])


def remove_hooks(model) -> None:
    for block in model.blocks:
        if hasattr(block.attn1, "_heft_capture"):
            del block.attn1._heft_capture


def stable_seed(base: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode()).digest()
    return int.from_bytes(digest[:4], "little") ^ base


def read_records(data: Path, event_type: str | None, limit: int | None) -> list[dict]:
    records = []
    for metadata_path in sorted(data.glob("sample_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("num_frames") != TOTAL_FRAMES:
            continue
        if metadata.get("event_index_in_window") != EVENT_INDEX:
            continue
        if event_type and metadata.get("event_type") != event_type:
            continue
        sample_dir = metadata_path.parent
        if not (sample_dir / "wrist.mp4").exists():
            continue
        records.append(
            {
                "sample_id": metadata["sample_id"],
                "episode": metadata["episode"],
                "event_type": metadata["event_type"],
                "label": int(metadata["label"]),
                "failure_type": metadata.get("failure_type"),
                "suite": metadata.get("suite"),
                "task_id": metadata.get("task_id"),
                "task": metadata["task"],
                "sample_dir": str(sample_dir),
                "label_source": metadata.get("label_source"),
                "margins": metadata.get("margins", {}),
            }
        )
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No valid interaction records in {data}")
    return records


def extract(args, records: list[dict], output: Path) -> dict:
    device = torch.device(args.device)
    torch.set_num_threads(4)
    prompts = [row["task"] for row in records]
    text_cpu = encode_prompts(args.text_model, prompts, device)
    model = load_transformer(args.model, device)
    vae = load_vae(args.model / "Wan2.1_VAE.pth", device)
    scheduler = UniPCMultistepScheduler.from_pretrained(
        args.text_model / "scheduler", local_files_only=True, flow_shift=args.flow_shift
    )
    scheduler.set_timesteps(args.steps, device=device)
    begin = int((scheduler.sigmas[:-1] - args.sigma).abs().argmin())
    scheduler.set_begin_index(begin)
    schedule_timesteps = scheduler.timesteps[begin:]
    actual_sigma = float(scheduler.sigmas[begin])
    denoise_steps = len(schedule_timesteps)
    temporal = (TOTAL_FRAMES - 1) // 4 + 1
    event_time = EVENT_INDEX // 4
    capture = MainQKCapture(temporal, event_time)
    install_hooks(model, capture)
    scales = torch.ones(len(model.vace_blocks), device=device, dtype=torch.bfloat16)
    states = tuple(sorted(set(args.states)))
    if states and max(states) >= denoise_steps:
        raise ValueError(f"Requested state {max(states)} but sigma={actual_sigma:.5f} has {denoise_steps} denoise steps")
    samples_root = output / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        for index, record in enumerate(records):
            feature_path = samples_root / record["sample_id"] / "features.npz"
            if args.resume and feature_path.exists():
                log(stage="skip_existing", sample=record["sample_id"], completed=index + 1, total=len(records))
                continue
            truth = read_video(Path(record["sample_dir"]) / "wrist.mp4")
            if truth.shape[0] != TOTAL_FRAMES:
                raise ValueError(f"{record['sample_id']} has {truth.shape[0]} frames")
            control, clean = prepare_inputs(vae, truth, args.size, OBSERVED_FRAMES, device)
            generator = torch.Generator(device=device).manual_seed(stable_seed(args.seed, record["sample_id"]))
            noise = torch.randn(clean.shape, dtype=torch.float32, device=device, generator=generator)
            # UniPC stores multistep history internally; reset it for every
            # independent video instead of carrying the previous sample's
            # derivative history into the next call to step().
            scheduler.set_timesteps(args.steps, device=device)
            scheduler.set_begin_index(begin)
            timesteps = scheduler.timesteps[begin:]
            latents = scheduler.add_noise(clean, noise, timesteps[:1])
            positive = text_cpu[record["task"]].to(device)
            negative = text_cpu[NEGATIVE_PROMPT].to(device)
            q_states, k_states = [], []
            for state, timestep in enumerate(timesteps):
                capture.clear(state, state in states)
                inputs = {
                    "hidden_states": latents.bfloat16(),
                    "timestep": timestep.reshape(1),
                    "control_hidden_states": control,
                    "control_hidden_states_scale": scales,
                    "return_dict": False,
                }
                conditional = model(**inputs, encoder_hidden_states=positive)[0]
                if state in states:
                    q, k = capture.arrays(len(model.blocks), model.config.num_attention_heads)
                    q_states.append(q)
                    k_states.append(k)
                if state == denoise_steps - 1:
                    break
                capture.clear(state, False)
                unconditional = model(**inputs, encoder_hidden_states=negative)[0]
                update = unconditional.float() + args.guidance * (conditional.float() - unconditional.float())
                latents = scheduler.step(update, timestep, latents, return_dict=False)[0]
                if not torch.isfinite(latents).all():
                    raise RuntimeError(f"Nonfinite latent at {record['sample_id']} step {state}")
            sample_out = samples_root / record["sample_id"]
            sample_out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                sample_out / "features.npz",
                states=np.asarray(states, dtype=np.int64),
                q=np.stack(q_states),
                k=np.stack(k_states),
            )
            dump(sample_out / "metadata.json", record | {
                "input_protocol": "33 observed wrist frames to VACE control, future control pixels zero, no warmup",
                "main_protocol": "full 49-frame GT VAE latent plus Gaussian noise at scheduler sigma near 0.8",
                "prompt": record["task"],
                "captured_branch": "main.blocks[*].attn1",
                "captured_values": "post-RoPE Q and K",
                "future_gt_used": True,
                "actual_noise_sigma": actual_sigma,
                "denoise_begin_index": begin,
                "denoise_steps": denoise_steps,
                "captured_states": list(states),
            })
            del control, clean, noise, latents, conditional, positive, negative
            torch.cuda.empty_cache()
            log(stage="sample_complete", sample=record["sample_id"], event_type=record["event_type"],
                label=record["label"], completed=index + 1, total=len(records),
                elapsed_seconds=time.monotonic() - started)
    finally:
        remove_hooks(model)
        del model, vae, text_cpu
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "actual_noise_sigma": actual_sigma,
        "denoise_begin_index": begin,
        "denoise_steps": denoise_steps,
        "states": list(states),
        "records": len(records),
    }


def metric(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def evaluate_candidate(x: np.ndarray, y: np.ndarray, groups: np.ndarray, c: float, splits,
                       probe_model: str) -> dict:
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scores = {"auroc": [], "average_precision": [], "balanced_accuracy": [], "f1": []}
    used = 0
    for train, test in splits:
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        if probe_model == "ridge":
            # RidgeClassifier is a closed-form linear readout.  It keeps the
            # scan practical for 30x12 heads while retaining the requested
            # episode-grouped, standardized linear probe.
            estimator = make_pipeline(
                StandardScaler(),
                RidgeClassifier(alpha=1.0 / max(c, 1e-8), class_weight="balanced"),
            )
        else:
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=2000),
            )
        estimator.fit(x[train], y[train])
        if probe_model == "ridge":
            score = estimator.decision_function(x[test])
            prediction = (score >= 0).astype(np.int64)
        else:
            score = estimator.predict_proba(x[test])[:, 1]
            prediction = (score >= 0.5).astype(np.int64)
        scores["auroc"].append(roc_auc_score(y[test], score))
        scores["average_precision"].append(average_precision_score(y[test], score))
        scores["balanced_accuracy"].append(balanced_accuracy_score(y[test], prediction))
        scores["f1"].append(f1_score(y[test], prediction, zero_division=0))
        used += 1
    return {name: metric(values) for name, values in scores.items()} | {"folds": used}


def probe(args, records: list[dict], output: Path) -> None:
    from sklearn.model_selection import GroupKFold

    rows = []
    for record in records:
        feature_path = output / "samples" / record["sample_id"] / "features.npz"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        with np.load(feature_path) as archive:
            states = archive["states"].astype(int)
            q = archive["q"].astype(np.float32)
            k = archive["k"].astype(np.float32)
        rows.append((record, states, q, k))
    if not rows:
        raise ValueError("No extracted features to probe")
    states = rows[0][1]
    if any(not np.array_equal(states, row[1]) for row in rows):
        raise ValueError("Feature files have inconsistent denoising states")
    q = np.stack([row[2] for row in rows])
    k = np.stack([row[3] for row in rows])
    labels = np.asarray([row[0]["label"] for row in rows], dtype=np.int64)
    groups = np.asarray([row[0]["episode"] for row in rows])
    event_types = np.asarray([row[0]["event_type"] for row in rows])
    suites = np.asarray([row[0]["suite"] for row in rows])
    layers, heads, feature_dim = q.shape[2], q.shape[3], q.shape[4]
    results = []
    for event_type in sorted(set(event_types)):
        selected = np.flatnonzero(event_types == event_type)
        if len(selected) < 10 or len(np.unique(labels[selected])) < 2:
            continue
        y = labels[selected]
        g = groups[selected]
        n_splits = min(args.folds, len(np.unique(g)))
        if n_splits < 2:
            continue
        splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(selected)), y, g))
        for state_index, state in enumerate(states):
            for layer in range(layers):
                for head in range(heads):
                    descriptors = {
                        "q": q[selected, state_index, layer, head],
                        "k": k[selected, state_index, layer, head],
                        "qk": np.concatenate((q[selected, state_index, layer, head],
                                               k[selected, state_index, layer, head]), axis=1),
                    }
                    for representation, x in descriptors.items():
                        best = None
                        for c in args.c_values:
                            score = evaluate_candidate(x, y, g, c, splits, args.probe_model)
                            key = (score.get(args.rank_metric) if score.get(args.rank_metric) is not None else -1.0)
                            if best is None or key > best[0]:
                                best = (key, c, score)
                        if best is None:
                            continue
                        results.append({
                            "event_type": event_type,
                            "state_after_steps": int(state),
                            "layer": layer,
                            "head": head,
                            "representation": representation,
                            "feature_dim": int(x.shape[1]),
                            "samples": int(len(selected)),
                            "positive": int(y.sum()),
                            "negative": int((y == 0).sum()),
                            "episodes": int(len(np.unique(g))),
                            "best_C": float(best[1]),
                            "metrics": best[2],
                        })
    results.sort(key=lambda row: (
        -(row["metrics"].get(args.rank_metric) if row["metrics"].get(args.rank_metric) is not None else -1),
        -(row["metrics"].get("auroc") if row["metrics"].get("auroc") is not None else -1),
    ))
    dump(output / "ranking.json", results)
    top = {}
    for event_type in sorted(set(event_types)):
        top[event_type] = [row for row in results if row["event_type"] == event_type][: args.top]
    dump(output / "top_heads.json", top)
    dump(output / "probe_protocol.json", {
        "probe": "episode-grouped linear classifier with per-fold StandardScaler",
        "probe_model": args.probe_model,
        "splits": f"GroupKFold({args.folds}) capped by unique episodes",
        "C_values": args.c_values,
        "rank_metric": args.rank_metric,
        "automatic_labels": True,
        "warning": "Results are against current replay labels; grasp labels are heuristic and place labels use simulator predicate.",
        "records": len(records),
        "event_counts": {name: int((event_types == name).sum()) for name in sorted(set(event_types))},
        "suite_counts": {name: int((suites == name).sum()) for name in sorted(set(suites))},
        "states": states.tolist(),
        "layers": layers,
        "heads": heads,
        "feature_dim_per_q_or_k": feature_dim,
    })
    log(stage="probe_complete", results=len(results), top={name: values[:3] for name, values in top.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/libero_interaction_probe_33plus16_20261008_all_native20_v4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Wan2.1-VACE-1.3B")
    parser.add_argument("--text-model", type=Path, default=ROOT / "models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--event-type", choices=("grasp", "place"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--states", type=int, nargs="+", default=list(DEFAULT_STATES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--c-values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--rank-metric", choices=("auroc", "balanced_accuracy", "average_precision", "f1"), default="auroc")
    parser.add_argument("--probe-model", choices=("ridge", "logistic"), default="ridge")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true",
                        help="Extract shard features without running the per-shard probe")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.size % 16 or OBSERVED_FRAMES >= TOTAL_FRAMES:
        parser.error("size must be divisible by 16 and the fixed 33+16 timeline must be valid")
    args.data = args.data.resolve(strict=True)
    args.output = args.output.resolve()
    args.model = args.model.resolve(strict=True)
    args.text_model = args.text_model.resolve(strict=True)
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error("--shards must be positive and --shard-index must be in [0, shards)")
    args.output.mkdir(parents=True, exist_ok=True)
    all_records = read_records(args.data, args.event_type, args.limit)
    records = [row for index, row in enumerate(all_records) if index % args.shards == args.shard_index]
    if not records:
        raise ValueError(f"Shard {args.shard_index}/{args.shards} has no records")
    dump(args.output / "manifest.json", {
        "data": str(args.data),
        "records": records,
        "all_records": len(all_records),
        "shard_index": args.shard_index,
        "shards": args.shards,
        "input_contract": "49 native wrist frames; frames 0..32 control; full GT main latent + sigma 0.8 noise; task prompt",
        "event_index": EVENT_INDEX,
        "observed_frames": OBSERVED_FRAMES,
        "total_frames": TOTAL_FRAMES,
        "event_counts": {name: sum(row["event_type"] == name for row in records) for name in ("grasp", "place")},
        "label_counts": {name: sum(row["label"] == 1 and row["event_type"] == name for row in records) for name in ("grasp", "place")},
    })
    protocol = {
        "branch": "main transformer blocks only",
        "attention": "main attn1 self-attention post-RoPE Q/K",
        "prompt": "per-sample full LIBERO task instruction supplied as encoder_hidden_states",
        "control": "33 recorded wrist frames, frames 33..48 zero, no warmup copies",
        "main": "full 49-frame GT VAE latent mixed at scheduler sigma nearest requested 0.8",
        "scheduler_steps": args.steps,
        "flow_shift": args.flow_shift,
        "guidance": args.guidance,
        "states": args.states,
        "teacher_diagnostic": True,
        "future_gt_used": True,
        "labels": "automatic v4 replay labels; not audited ground truth",
    }
    dump(args.output / "protocol.json", protocol)
    if not args.probe_only:
        extraction = extract(args, records, args.output)
        dump(args.output / "extraction.json", extraction)
    if not args.extract_only:
        probe(args, records, args.output)


if __name__ == "__main__":
    main()
