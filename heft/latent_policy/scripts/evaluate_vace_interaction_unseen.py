#!/usr/bin/env python3
"""Strict unseen-episode evaluation for the VACE interaction head scan.

This script keeps a deterministic episode holdout completely out of both head
selection and linear-probe fitting.  It is intentionally separate from the
exploratory ranking produced by ``scan_vace_interaction_heads.py``: selecting
the winner on all samples and then testing it on those same samples would
inflate the apparent interaction readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def read_records(data: Path, event_type: str) -> list[dict]:
    records = []
    for metadata_path in sorted(data.glob("sample_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("event_type") != event_type:
            continue
        records.append(metadata)
    if not records:
        raise ValueError(f"No {event_type} metadata in {data}")
    return records


def holdout_indices(records: list[dict], fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Select groups once, preserving label proportions at the group level."""
    groups = np.asarray([row["episode"] for row in records])
    labels = np.asarray([int(row["label"]) for row in records])
    unique, inverse = np.unique(groups, return_inverse=True)
    group_labels = np.zeros(len(unique), dtype=np.int64)
    for index in range(len(unique)):
        values = labels[inverse == index]
        if len(np.unique(values)) > 1:
            raise ValueError(f"Episode {unique[index]} has conflicting labels")
        group_labels[index] = values[0]
    if len(np.unique(group_labels)) < 2:
        raise ValueError("Cannot stratify holdout with one class")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train_groups, test_groups = next(splitter.split(np.zeros(len(unique)), group_labels))
    train_set, test_set = set(unique[train_groups]), set(unique[test_groups])
    train = np.asarray([index for index, group in enumerate(groups) if group in train_set], dtype=np.int64)
    test = np.asarray([index for index, group in enumerate(groups) if group in test_set], dtype=np.int64)
    return train, test


def load_features(feature_root: Path, records: list[dict]):
    states = None
    q_rows, k_rows = [], []
    for row in records:
        path = feature_root / "samples" / row["sample_id"] / "features.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            current_states = archive["states"].astype(np.int64)
            q_rows.append(archive["q"].astype(np.float32))
            k_rows.append(archive["k"].astype(np.float32))
        if states is None:
            states = current_states
        elif not np.array_equal(states, current_states):
            raise ValueError(f"Inconsistent states in {path}")
    q, k = np.stack(q_rows), np.stack(k_rows)
    return states, q, k


def score(estimator, x: np.ndarray, y: np.ndarray) -> dict:
    value = estimator.decision_function(x)
    prediction = (value >= 0).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y, value)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, value)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }


def fit_score(x_train, y_train, x_test, y_test, alpha):
    estimator = make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=alpha, class_weight="balanced"),
    )
    estimator.fit(x_train, y_train)
    return score(estimator, x_test, y_test)


def inner_cv_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float, folds: int) -> float:
    n_splits = min(folds, len(np.unique(groups)))
    if n_splits < 2:
        return -1.0
    values = []
    for train, test in GroupKFold(n_splits=n_splits).split(x, y, groups):
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        values.append(fit_score(x[train], y[train], x[test], y[test], alpha)["auroc"])
    return float(np.mean(values)) if values else -1.0


def candidate_descriptors(q, k, sample_indices, state_index, layer, head):
    q_value = q[sample_indices, state_index, layer, head]
    k_value = k[sample_indices, state_index, layer, head]
    return {"q": q_value, "k": k_value, "qk": np.concatenate((q_value, k_value), axis=1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-type", choices=("grasp", "place"), required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    args.data = args.data.resolve(strict=True)
    args.features = args.features.resolve(strict=True)
    args.output = args.output.resolve()
    records = read_records(args.data, args.event_type)
    states, q, k = load_features(args.features, records)
    train, test = holdout_indices(records, args.holdout_fraction, args.seed)
    y = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    groups = np.asarray([row["episode"] for row in records])
    if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
        raise ValueError("Holdout split must contain both labels")

    candidates = []
    for state_index, state in enumerate(states):
        for layer in range(q.shape[2]):
            for head in range(q.shape[3]):
                for representation, descriptor in candidate_descriptors(q, k, train, state_index, layer, head).items():
                    inner = inner_cv_score(descriptor, y[train], groups[train], args.alpha, args.inner_folds)
                    candidates.append({
                        "state_after_steps": int(state), "layer": layer, "head": head,
                        "representation": representation, "inner_auroc": inner,
                    })
    candidates.sort(key=lambda row: -row["inner_auroc"])
    selected = candidates[: args.top]
    evaluated = []
    for candidate in selected:
        state_index = int(np.flatnonzero(states == candidate["state_after_steps"])[0])
        descriptor_train = candidate_descriptors(
            q, k, train, state_index, candidate["layer"], candidate["head"]
        )[candidate["representation"]]
        descriptor_test = candidate_descriptors(
            q, k, test, state_index, candidate["layer"], candidate["head"]
        )[candidate["representation"]]
        evaluated.append(candidate | {
            "holdout_metrics": fit_score(descriptor_train, y[train], descriptor_test, y[test], args.alpha),
        })
    evaluated.sort(key=lambda row: -row["holdout_metrics"]["auroc"])

    # Task identity is a useful leakage check: it must be weaker than the
    # frozen feature result if the head is reading interaction state.
    task_values = np.asarray([f"{row.get('suite')}/{row.get('task_id')}" for row in records])
    names = sorted(set(task_values[train]))
    task_train = np.zeros((len(train), len(names)), dtype=np.float32)
    task_test = np.zeros((len(test), len(names)), dtype=np.float32)
    lookup = {name: index for index, name in enumerate(names)}
    for row_index, sample_index in enumerate(train):
        if task_values[sample_index] in lookup:
            task_train[row_index, lookup[task_values[sample_index]]] = 1
    for row_index, sample_index in enumerate(test):
        if task_values[sample_index] in lookup:
            task_test[row_index, lookup[task_values[sample_index]]] = 1
    task_metrics = fit_score(task_train, y[train], task_test, y[test], args.alpha)

    payload = {
        "protocol": {
            "event_type": args.event_type,
            "holdout_fraction": args.holdout_fraction,
            "seed": args.seed,
            "head_selection": "inner episode-grouped CV on train episodes only",
            "final_fit": "selected head refit on all train episodes only",
            "probe": "StandardScaler + balanced RidgeClassifier",
            "future_gt_in_features": True,
            "warning": "This is a privileged GT-assisted teacher diagnostic; it is not causal prediction.",
        },
        "counts": {
            "all_samples": len(records), "train_samples": len(train), "holdout_samples": len(test),
            "all_episodes": int(len(np.unique(groups))),
            "train_episodes": int(len(np.unique(groups[train]))),
            "holdout_episodes": int(len(np.unique(groups[test]))),
            "train_positive": int(y[train].sum()), "train_negative": int((y[train] == 0).sum()),
            "holdout_positive": int(y[test].sum()), "holdout_negative": int((y[test] == 0).sum()),
        },
        "holdout_episodes": sorted(set(groups[test])),
        "train_episodes": sorted(set(groups[train])),
        "task_id_baseline": task_metrics,
        "selected_head_holdout": evaluated,
        "inner_ranking_top": selected,
    }
    dump(args.output / "unseen_holdout.json", payload)
    print(json.dumps({
        "event_type": args.event_type,
        "train": payload["counts"]["train_samples"],
        "holdout": payload["counts"]["holdout_samples"],
        "holdout_positive": payload["counts"]["holdout_positive"],
        "task_id_baseline": task_metrics,
        "top": evaluated[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
