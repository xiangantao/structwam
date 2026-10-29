"""CPU contract checks; pretrained encoder and CUDA checks require local weights."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ...checkpoint import load_checkpoint, save_checkpoint
from ...config import DataConfig, OptimizationConfig
from ...config import RunConfig as WanConfig
from ...data import EvaluationSampler, Sample, WanRGBTransform, seeded_sample
from ...engine import ProbeOptimization, run_epoch
from ...upstream import load_upstream
from .config import RunConfig
from .data import build_dataset, read_manifest, read_original_manifest
from .encoder import FrozenEncoder, load_target_encoder
from .train import preflight, protocol_record

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def runtime():
    config = RunConfig.load(HERE / "ssv2.yaml")
    return load_upstream(config.path(config.vjepa_root))


def test_configuration_matches_wan_training_budget():
    config = RunConfig.load(HERE / "ssv2.yaml")
    wan = WanConfig.load(HERE.parents[1] / "configs/ssv2.yaml")
    assert config.data == wan.data
    assert config.optimization == wan.optimization
    assert config.seed == wan.seed == 42
    assert config.num_heads == wan.num_heads == 16
    assert config.num_probe_blocks == wan.num_probe_blocks == 4
    assert config.validate_every == wan.validate_every == 0
    assert len(config.optimization.upstream_kwargs()) == 1
    for updates in ({"num_heads": 8}, {"extract_batch_size": 1}, {"validate_every": 1}):
        with pytest.raises(ValueError):
            replace(config, **updates)


def test_strict_weight_loading(tmp_path):
    model = nn.Linear(3, 2)
    expected = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    path = tmp_path / "vitl.pt"
    torch.save(
        {"target_encoder": {"module.backbone." + k: v for k, v in expected.items()}},
        path,
    )
    load_target_encoder(model, path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])
    for state, message in (
        ({"weight": torch.zeros(2, 3)}, "missing="),
        ({**expected, "extra": torch.zeros(1)}, "unexpected="),
        ({**expected, "weight": torch.zeros(1, 3)}, "shape_mismatch="),
        ({**expected, "module.weight": expected["weight"]}, "Duplicate"),
    ):
        torch.save({"target_encoder": state}, path)
        with pytest.raises(ValueError, match=message):
            load_target_encoder(model, path)
    torch.save({"encoder": expected}, path)
    with pytest.raises(ValueError, match="target_encoder"):
        load_target_encoder(model, path)


class TinyWrapper(nn.Module):
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.calls = []

    def forward(self, clips):
        self.calls.append(sum(len(segment[0]) for segment in clips))
        return [
            torch.cat(
                [
                    (segment[0][:, 0, 0, 0, 0] * self.scale)[:, None, None].expand(
                        -1, 2048, self.embed_dim
                    )
                    for segment in clips
                ],
                dim=1,
            )
        ]


def fake_batch(size=5):
    values = torch.arange(size, dtype=torch.float32)
    return {
        "clips": [
            [
                (values + segment * 10 + view * 100)[:, None, None, None, None].expand(
                    -1, 3, 16, 256, 256
                )
                for view in range(3)
            ]
            for segment in range(2)
        ],
        "label": torch.arange(size) % 2,
        "id": [str(i) for i in range(size)],
    }


def test_chunking_preserves_video_segment_view_order_and_freezing():
    wrapper = TinyWrapper()
    encoder = FrozenEncoder(wrapper, device="cpu", extract_batch_size=4)
    tokens = encoder.encode_view(fake_batch(), 2)
    assert tokens.shape == (5, 4096, 8)
    assert wrapper.calls == [4, 4, 2]
    assert torch.equal(tokens[:, 0, 0], torch.arange(5) + 200)
    assert torch.equal(tokens[:, 2048, 0], torch.arange(5) + 210)
    assert not tokens.requires_grad and not wrapper.training
    head = nn.Linear(8, 2)
    head(tokens.mean(1)).sum().backward()
    assert head.weight.grad is not None
    assert wrapper.scale.grad is None and not wrapper.scale.requires_grad
    full = FrozenEncoder(TinyWrapper(), device="cpu", extract_batch_size=32)
    assert torch.equal(tokens, full.encode_view(fake_batch(), 2))
    batch = fake_batch()
    batch["clips"][0][0] = torch.zeros(5, 16, 3, 256, 256, dtype=torch.uint8)
    with pytest.raises(ValueError, match="normalized"):
        encoder.encode_view(batch, 0)


def test_official_augmentation_matches_wan_before_native_adapter(runtime):
    config = RunConfig.load(HERE / "ssv2.yaml")
    records = [Sample("a", "/a.webm", 1)]
    captured = {}

    def video_dataset(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(samples=["/a.webm"], labels=[1])

    adapted = SimpleNamespace(
        make_transforms=runtime.make_transforms, video_dataset=video_dataset
    )
    build_dataset(adapted, Path("/manifest.csv"), records, config, training=True)
    transform = captured["transform"]
    frames = np.random.default_rng(0).integers(
        0, 256, (16, 270, 280, 3), dtype=np.uint8
    )
    with seeded_sample(42):
        native = transform(frames)
    with seeded_sample(42):
        wan = WanRGBTransform(transform)(frames)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]
    assert native[0].shape == (3, 16, 256, 256)
    expected = (
        ((native[0] * std + mean) * 255)
        .round()
        .clamp(0, 255)
        .byte()
        .permute(1, 0, 2, 3)
    )
    assert torch.equal(expected, wan[0])
    assert native[0].dtype == torch.float32
    assert captured["num_clips"] == 2 and captured["frame_step"] == 4
    build_dataset(adapted, Path("/manifest.csv"), records, config, training=False)
    assert len(captured["transform"](frames)) == 3


def test_real_head_engine_checkpoint_and_evaluation(runtime, tmp_path):
    torch.manual_seed(42)
    device = torch.device("cpu")
    head = runtime.classifier(
        embed_dim=64,
        num_heads=4,
        depth=4,
        num_classes=174,
        use_activation_checkpointing=True,
    )
    config = OptimizationConfig(batch_size=2, num_epochs=2, amp_dtype="float32")
    opt = ProbeOptimization.create(runtime, head, config, 1)
    tokens = torch.randn(2, 12, 64)
    encoder = SimpleNamespace(
        device=device, encode_view=lambda batch, view: tokens + view
    )
    batch = {"clips": [[None]], "label": torch.tensor([0, 1]), "id": ["a", "b"]}
    before = {k: v.clone() for k, v in head.state_dict().items()}
    metrics = run_epoch(
        head, encoder, [batch], training=True, amp_dtype="float32", optimization=opt
    )
    assert metrics["samples"] == 2
    assert any(not torch.equal(v, before[k]) for k, v in head.state_dict().items())
    protocol = {
        "representation": {"encoder": "tiny"},
        "validation_manifest": "test",
        "training": {},
    }
    path = tmp_path / "epoch_0001.pt"
    save_checkpoint(
        path,
        head,
        opt,
        epoch=1,
        best_top1=float("-inf"),
        protocol=protocol,
        device=device,
    )
    restored = runtime.classifier(
        embed_dim=64,
        num_heads=4,
        depth=4,
        num_classes=174,
        use_activation_checkpointing=True,
    )
    restored_opt = ProbeOptimization.create(runtime, restored, config, 1)
    assert (
        load_checkpoint(
            path, restored, protocol=protocol, device=device, optimization=restored_opt
        )[0]
        == 1
    )
    assert restored_opt.scheduler._step == opt.scheduler._step
    assert restored_opt.optimizer.state_dict()["state"]
    batch["clips"] = [[None, None, None]]
    prediction_path = tmp_path / "predictions.jsonl"
    a = run_epoch(head, encoder, [batch], training=False, amp_dtype="float32")
    b = run_epoch(
        restored,
        encoder,
        [batch],
        training=False,
        amp_dtype="float32",
        predictions=prediction_path,
    )
    assert a == b and len(prediction_path.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="protocol differs"):
        load_checkpoint(
            path, restored, protocol={**protocol, "representation": {}}, device=device
        )
    assert sorted(i for r in range(8) for i in EvaluationSampler(19, r, 8)) == list(
        range(19)
    )


def test_preflight_and_provenance(tmp_path, monkeypatch):
    config = RunConfig(
        DataConfig(str(tmp_path / "train.csv"), str(tmp_path / "val.csv")),
        model_path=str(tmp_path / "vitl.pt"),
    )
    monkeypatch.setattr(
        "importlib.util.find_spec", lambda name: None if name == "einops" else object()
    )
    issues = preflight(config)
    assert any("einops" in issue for issue in issues)
    assert any("model_path" in issue for issue in issues)
    encoder = SimpleNamespace(
        embed_dim=1024, metadata=lambda: {"weights_sha256": "abc"}
    )
    records = [Sample("a", "/a.webm", 0)]
    protocol = protocol_record(
        config,
        encoder,
        nn.Linear(2, 2),
        SimpleNamespace(provenance={}),
        records,
        records,
        8,
    )
    sources = protocol["representation"]["runner_sources"]
    assert "engine.py" in sources and "baselines/vjepa2/encoder.py" in sources
    assert all("test_" not in name for name in sources)


def test_manifest_directory_cache_preserves_paths_and_symlinks(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "1.webm").touch()
    (folder / "2.webm").touch()
    (tmp_path / "alias").symlink_to(folder, target_is_directory=True)
    (folder / "3.webm").symlink_to(folder / "1.webm")
    path = tmp_path / "train.csv"
    path.write_text("alias/3.webm 0\nvideos/2.webm 1\n")
    config = RunConfig(DataConfig(str(path), str(path)))
    assert read_manifest(path, config) == read_original_manifest(path, config)
    path.write_text("videos/1.webm 0\nvideos/3.webm 0\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path, config)


def test_smoke_loader_stops_without_fetching_extra_batches():
    from .train import LimitedLoader

    fetched = []

    class Source:
        def __len__(self):
            return 660

        def __iter__(self):
            for i in range(660):
                fetched.append(i)
                yield i

    loader = LimitedLoader(Source(), 3)
    assert len(loader) == 3 and list(loader) == [0, 1, 2]
    assert fetched == [0, 1, 2]


def test_smoke_batch_sampler_preserves_full_training_order():
    from .train import make_loader

    config = replace(RunConfig.load(HERE / "ssv2.yaml"), num_workers=0)
    loader, sampler = make_loader(
        range(168913), config, rank=0, world_size=8, training=True, steps=10
    )
    assert len(loader) == 10
    assert (len(sampler) + 31) // 32 == 660
    expected = list(sampler)[:320]
    assert torch.cat(list(loader)).tolist() == expected


def test_parallel_lanes_preserve_order_freezing_and_close():
    serial = FrozenEncoder(TinyWrapper(), device="cpu", extract_batch_size=4)
    parallel = FrozenEncoder(
        TinyWrapper(), device="cpu", extract_batch_size=4, extract_lanes=2
    )
    try:
        parallel.verify_parallel = True
        parallel.measure = True
        batch = fake_batch()
        actual = parallel.encode_view(batch, 1)
        assert torch.equal(serial.encode_view(batch, 1), actual)
        assert parallel.comparison["exact"]
        assert len(parallel.comparison["serial_seconds"]) == 2
        assert parallel.wrappers[0] is not parallel.wrappers[1]
        assert (
            parallel.wrappers[0].scale.data_ptr()
            != parallel.wrappers[1].scale.data_ptr()
        )
        assert not actual.requires_grad
        assert all(
            not p.requires_grad
            for wrapper in parallel.wrappers
            for p in wrapper.parameters()
        )
        nn.Linear(8, 2)(actual.mean(1)).sum().backward()
        assert all(
            p.grad is None
            for wrapper in parallel.wrappers
            for p in wrapper.parameters()
        )
    finally:
        parallel.close()
        serial.close()
    with pytest.raises(RuntimeError, match="closed"):
        parallel.encode_view(fake_batch(), 0)
