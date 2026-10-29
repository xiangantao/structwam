"""Offline target parity, offset diversity, verified uploads and cache bounds."""

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vace_latent_distill.config import TeacherConfig, timeline
from vace_latent_distill.data import collate_structured
from vace_latent_distill.losses import structured_loss
from vace_latent_distill.offline import (
    OfflineStructuredDataset, build_windows, offset_starts, pack_target, unpack_target,
)
from vace_latent_distill.offline_store import ArtifactCache, file_sha256
from vace_latent_distill.model import StructuredStudent
from test_pipeline import TinyBackbone, inputs, teacher_target, tiny_config


def offline_model():
    config = tiny_config("shared")
    config.channels = 128
    return StructuredStudent(config, TinyBackbone(config.dim))


def single_target(prediction):
    return {branch: {key: value[0] for key, value in values.items()}
            for branch, values in teacher_target(prediction, "shared").items()}


def test_eight_offset_windows_are_reproducible_and_episode_bounded():
    row = {"id": "e0-image", "episode": 0, "length": 75,
           "episode_start_frame": 100, "task": "move cup", "camera": "image"}
    starts = offset_starts(75)
    assert len(starts) == len(set(starts)) == 8
    assert sorted(start % 8 for start in starts) == list(range(8))
    rows = build_windows([row])
    assert rows == build_windows([row])
    assert len({item["teacher_seed"] for item in rows}) == 8
    assert len({item["target_key"] for item in rows}) == 8
    assert all(100 <= min(item["frame_indices"]) <= max(item["frame_indices"]) < 175 for item in rows)
    assert any(len(set(rows[i]["frame_indices"]) & set(rows[i + 1]["frame_indices"])) > 0 for i in range(7))


def test_loss_only_roundtrip_keeps_paths_and_losses():
    torch.set_num_threads(2)
    prediction = offline_model()(*inputs())
    truth = single_target(prediction)
    payload = pack_target(truth, {}, {"sigma": 0.8}, "identity", grid=4, text_length=5)
    restored = unpack_target(payload, identity="identity")
    for branch, times in timeline().items():
        valid = times["valid"]
        assert payload["target"][branch]["track_q"].shape[0] == int(valid.sum())
        assert payload["target"][branch]["track_q"].dtype == torch.float16
        assert torch.equal(truth[branch]["paths"][:, :, valid], restored[branch]["paths"][:, :, valid])
        assert torch.equal(truth[branch]["seeds"], restored[branch]["seeds"])
        assert (restored[branch]["track_q"][~valid] == 0).all()
    left = {b: {k: v.unsqueeze(0).expand(2, *v.shape) for k, v in values.items()} for b, values in truth.items()}
    right = {b: {k: v.unsqueeze(0).expand(2, *v.shape) for k, v in values.items()} for b, values in restored.items()}
    torch.manual_seed(8)
    a, _ = structured_loss(prediction, left, task="shared", samples=4)
    torch.manual_seed(8)
    b, _ = structured_loss(prediction, right, task="shared", samples=4)
    assert torch.allclose(a, b, atol=0.002, rtol=0.002)
    b.backward()
    with pytest.raises(ValueError, match="protocol"):
        unpack_target(payload, identity="other")
    payload["target"]["observed"]["native_indices"] = torch.arange(8)
    with pytest.raises(ValueError, match="time indices"):
        unpack_target(payload)


class FakeStore:
    def __init__(self, remote):
        self.remote = remote
        self.downloads = 0

    def get(self, uri, path, sha256=None, expected_bytes=None):
        self.downloads += 1
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.remote[uri], path)
        if sha256 and file_sha256(path) != sha256:
            raise RuntimeError("SHA256 mismatch")


def register(cache, count):
    rows = [{"ordinal": index, "sample_id": str(index), "target_key": f"{index}.pt"} for index in range(count)]
    cache.register("run", rows, "tos://bucket/run")


def finish(cache, entry, owner, size=6000):
    path = Path(entry["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    sha = file_sha256(path)
    cache.ready(entry["uri"], path, sha, owner)
    upload = cache.claim_upload("run", owner)
    cache.uploaded(upload["uri"], owner, sha)
    return path, sha


def test_cache_never_evicts_pending_or_pinned_files(tmp_path):
    cache = ArtifactCache(tmp_path / "cache", 20_000)
    register(cache, 3)
    a = cache.claim_extract("run", "a", reservation=7000)
    path, sha = finish(cache, a, "a")
    b = cache.claim_extract("run", "b", reservation=7000)
    pending = Path(b["path"])
    pending.parent.mkdir(parents=True, exist_ok=True)
    pending.write_bytes(b"y" * 6000)
    cache.ready(b["uri"], pending, file_sha256(pending), "b")
    with cache.local(a["uri"], None, sha256=sha):
        assert cache.claim_extract("run", "c", reservation=7000) is None
        assert path.exists() and pending.exists()
    c = cache.claim_extract("run", "c", reservation=7000)
    assert c is not None and not path.exists() and pending.exists()
    assert cache.summary()["cache_accounted_bytes"] <= cache.data_limit
    exported = cache.export("run")
    assert exported[0]["target_bytes"] == 6000


def test_cache_reservations_are_atomic_and_dead_worker_recoverable(tmp_path):
    cache = ArtifactCache(tmp_path / "cache", 20_000)
    register(cache, 8)
    with ThreadPoolExecutor(8) as pool:
        entries = list(pool.map(lambda i: cache.claim_extract("run", str(i), reservation=6000), range(8)))
    claimed = [entry for entry in entries if entry]
    assert len(claimed) == 3
    assert len({entry["uri"] for entry in claimed}) == 3
    assert cache.summary()["cache_accounted_bytes"] <= cache.data_limit
    for owner in range(8):
        cache.recover_owner(str(owner))
    assert cache.summary()["cache_accounted_bytes"] == 0
    assert cache.summary()["states"] == {"planned": 8}


def test_cache_miss_downloads_once_and_checksum_failure_is_not_cached(tmp_path):
    cache = ArtifactCache(tmp_path / "cache", 20_000)
    remote = tmp_path / "remote.pt"
    remote.write_bytes(b"a" * 6000)
    sha = file_sha256(remote)
    store = FakeStore({"tos://bucket/test": remote, "tos://bucket/bad": remote})
    with cache.local("tos://bucket/test", store, sha256=sha, expected_bytes=6000) as path:
        assert path.read_bytes() == remote.read_bytes()
    with cache.local("tos://bucket/test", store, sha256=sha, expected_bytes=6000):
        pass
    assert store.downloads == 1
    with pytest.raises(RuntimeError, match="SHA256"):
        with cache.local("tos://bucket/bad", store, sha256="invalid", expected_bytes=6000):
            pass
    assert cache.summary()["states"].get("downloading", 0) == 0
    assert cache.summary()["cache_accounted_bytes"] == 6000


def test_offline_dataset_decodes_only_past_and_collates_cached_targets(tmp_path, monkeypatch):
    import vace_latent_distill.offline as module

    torch.set_num_threads(2)
    prediction = offline_model()(*inputs())
    row = {"sample_id": "sample", "teacher_seed": 3, "frame_indices": list(range(49)),
           "task": "move cup", "camera": "image", "video": "unused", "fps": 10,
           "provenance_id": "identity", "target_uri": "tos://bucket/sample"}
    payload = pack_target(single_target(prediction), row, {"denoise_updates": 13},
                          "identity", grid=4, text_length=5)
    remote = tmp_path / "remote.pt"
    torch.save(payload, remote)
    row.update(target_bytes=remote.stat().st_size, target_sha256=file_sha256(remote))
    manifest = tmp_path / "uploaded.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    text = tmp_path / "text.pt"
    torch.save({row["task"]: torch.randn(5, 12)}, text)
    monkeypatch.setattr(module, "TosStore", lambda **_: FakeStore({row["target_uri"]: remote}))
    seen = []

    def decode(_video, indices, _fps):
        seen.append(indices)
        return torch.zeros(33, 3, 32, 32, dtype=torch.uint8).numpy()

    monkeypatch.setattr(module, "decode_indices", decode)
    dataset = OfflineStructuredDataset(
        manifest, text, cache_dir=tmp_path / "cache", image_size=32,
        teacher_config=TeacherConfig(denoise_updates=13),
    )
    batch = collate_structured([dataset[0]])
    assert seen == [list(range(33))]
    assert batch["rgb"].shape == (1, 33, 3, 32, 32)
    assert "teacher_rgb" not in batch and "target" in batch
    assert batch["target"]["observed"]["track_q"].shape[1] == 12
    mainline = OfflineStructuredDataset(
        manifest, text, cache_dir=tmp_path / "cache", image_size=32,
        teacher_config=TeacherConfig(),
    )
    with pytest.raises(ValueError, match="denoising protocol"):
        mainline[0]
    assert seen == [list(range(33))]
