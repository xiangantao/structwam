"""Protect ready-first consumption from head-of-line blocking and cross-update mixing."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("lerobot")
spec = importlib.util.spec_from_file_location(
    "async_trial", Path(__file__).resolve().parents[1] / "scripts/trial_async_policy.py"
)
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def test_ready_batch_skips_slow_lane_without_consuming_future_update():
    queue = trial.ReadyBatchQueue(capacity=3)
    future = (1, 0, [100], "future labels", "future features")
    ready = (0, 4, [40], "current labels", "current features")
    queue.put(future, timeout=0.1)
    queue.put(ready, timeout=0.1)
    assert queue.get(0, lambda: None, timeout=0.1) == ready
    late = (0, 0, [0], "late labels", "late features")
    queue.put(late, timeout=0.1)
    assert queue.get(0, lambda: None, timeout=0.1) == late
    assert queue.get(1, lambda: None, timeout=0.1) == future


def test_duplicate_feature_batch_is_rejected():
    queue = trial.ReadyBatchQueue(capacity=2)
    payload = (0, 0, [0], "labels", "features")
    queue.put(payload, timeout=0.1)
    queue.get(0, lambda: None, timeout=0.1)
    queue.put(payload, timeout=0.1)
    with pytest.raises(ValueError, match="duplicate"):
        queue.get(0, lambda: None, timeout=0.1)
