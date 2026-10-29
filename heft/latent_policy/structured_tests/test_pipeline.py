"""Input boundaries, time alignment, relation losses and resumable state."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vace_latent_distill.config import (
    HEAD_TASKS,
    TASKS,
    ModelConfig,
    PipelineConfig,
    TrainConfig,
    timeline,
)
from vace_latent_distill.data import StructuredDataset
from vace_latent_distill.losses import (
    augmentation_loss,
    masked_mean,
    relation_loss,
    structured_loss,
)
from vace_latent_distill.model import StructuredStudent
from vace_latent_distill.training import load_checkpoint, optimizer_for, save_checkpoint


class TinyPatch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(3, dim, (2, 16, 16), (2, 16, 16))

    def forward(self, pixels):
        return self.conv(pixels).flatten(2).transpose(1, 2)


class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, tokens, **_):
        return tokens + self.linear(self.norm(tokens))


class TinyBackbone(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_embed = TinyPatch(dim)
        self.blocks = nn.ModuleList([TinyBlock(dim), TinyBlock(dim)])
        self.norm = nn.LayerNorm(dim)


def tiny_config(task="tracking", checkpointing=False):
    return ModelConfig(
        task=task,
        image_size=32,
        grid_size=4,
        dim=24,
        text_dim=12,
        channels=8,
        attention_heads=4,
        language_layers=1,
        decoder_layers=1,
        head_layers=2,
        window_size=2,
        mlp_ratio=1,
        activation_checkpointing=checkpointing,
    )


def make_model(task="tracking", checkpointing=False):
    config = tiny_config(task, checkpointing)
    return StructuredStudent(config, TinyBackbone(config.dim))


def inputs():
    return (
        torch.randint(0, 256, (2, 33, 3, 32, 32), dtype=torch.uint8),
        torch.randn(2, 5, 12),
        torch.tensor([[True, True, True, False, False]] * 2),
    )


def teacher_target(output, task):
    result = {}
    for branch in ("observed", "future"):
        truth = {}
        for name, value in output[branch].items():
            if name == "depth":
                truth[name] = value.detach() * 1.65625 + 4.34375 + 0.1
            elif name == "text_attention":
                distribution = torch.rand_like(value) * (value.detach() > 0)
                truth[name] = distribution / distribution.sum(-1, keepdim=True)
            else:
                truth[name] = F.normalize(torch.randn_like(value), dim=-1)
        if task == "depth":
            truth["seg_k"] = F.normalize(
                torch.randn(2, truth["depth"].shape[1], 16, 8, device=truth["depth"].device), dim=-1
            )
        if task in {"tracking", "shared"}:
            anchors = output["timeline"][branch]["anchor_native"]
            count, temporal = 3, truth["track_q"].shape[1]
            device = truth["track_q"].device
            truth["seeds"] = torch.zeros(2, len(anchors), count, dtype=torch.long, device=device)
            truth["paths"] = torch.zeros(2, len(anchors), count, temporal, 2, device=device)
            truth["path_valid"] = torch.ones(
                2, len(anchors), count, temporal, dtype=torch.bool, device=device
            )
        result[branch] = truth
    return result


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize("task", TASKS)
def test_head_forward_backward(task):
    model = make_model(task, checkpointing=True)
    prediction = model(*inputs())
    keys = {"tracking": {"track_q", "track_k"}, "semantic": {"text_attention"}, "depth": {"depth"}}
    keys["shared"] = {"seg_k_l22h9", "seg_k_l15h7", "track_q", "track_k", "text_attention", "depth"}
    for branch, temporal in (("observed", 12), ("future", 13)):
        assert set(prediction[branch]) == keys.get(task, {"seg_k"})
        for key, value in prediction[branch].items():
            assert value.shape[:3] == (2, temporal, 16)
            assert torch.isfinite(value).all()
            if key == "text_attention":
                assert (value[..., 3:] == 0).all()
    loss, _ = structured_loss(prediction, teacher_target(prediction, task), task=task, samples=4)
    assert torch.isfinite(loss) and loss < 100
    loss.backward()
    assert model.future_query.grad is not None and torch.isfinite(model.future_query.grad).all()
    assert model.backbone.patch_embed.conv.weight.grad.abs().sum() > 0
    assert model.text_projection[1].weight.grad.abs().sum() > 0
    if task == "shared":
        assert set(model.heads) == set(HEAD_TASKS)
        for head in model.heads.values():
            assert head.primary.skip.weight.grad.abs().sum() > 0


def test_joint_head_weights_and_single_backbone():
    model = make_model("shared")
    calls = []
    handle = model.backbone.patch_embed.register_forward_hook(lambda *_: calls.append(1))
    prediction = model(*inputs())
    handle.remove()
    assert len(calls) == 1
    target = teacher_target(prediction, "shared")
    weights = dict.fromkeys(HEAD_TASKS, 0.0)
    weights["depth"] = 0.25
    loss, details = structured_loss(
        prediction, target, task="shared", samples=4, head_weights=weights
    )
    assert torch.allclose(loss.detach(), 0.25 * details["depth/total"])
    for task in HEAD_TASKS[:-1]:
        assert details[f"{task}/weighted"] == 0
    with pytest.raises(ValueError, match="all five heads"):
        TrainConfig(head_weights={"depth": 1.0}).validate()


def test_past_only_and_final_frame_is_encoded():
    model = make_model().eval()
    rgb, text, mask = inputs()
    original = model(rgb, text, mask)["future"]["track_q"]
    with pytest.raises(ValueError, match="exactly33"):
        model(torch.cat([rgb, rgb[:, :16]], 1), text, mask)
    with pytest.raises(TypeError):
        model(rgb, text, mask, future_rgb=rgb[:, :16])
    changed = rgb.clone()
    changed[:, -1] = 255 - changed[:, -1]
    assert not torch.allclose(original, model(changed, text, mask)["future"]["track_q"])
    assert not torch.allclose(
        original, model(rgb, text + torch.randn_like(text), mask)["future"]["track_q"]
    )


def test_relation_identity_and_loss_scale():
    q = torch.randn(2, 5, 16, 8, requires_grad=True)
    k = torch.randn_like(q)
    valid = torch.ones(2, 5, dtype=torch.bool)
    loss = relation_loss(q, k, q.detach(), k.detach(), valid, samples=4, temperature=0.07)
    assert abs(float(loss.detach())) < 1e-6
    assert masked_mean(torch.ones(2, 5, 16), valid) == 1
    empty = torch.zeros_like(valid)
    assert masked_mean(torch.ones(2, 5, 16), empty) == 0


def test_timelines_and_augmentation_identity():
    times = timeline()
    assert times["observed"]["anchor_native"].tolist() == [4, 6, 8, 10]
    assert times["future"]["anchor_rgb"].tolist() == [33, 41]
    assert not times["observed"]["valid"][:4].any()
    assert times["future"]["valid"][8:].all()
    for task in TASKS:
        prediction = make_model(task)(*inputs())
        assert augmentation_loss(prediction, prediction, task) == 0


@pytest.mark.parametrize("task", ["tracking", "shared"])
def test_checkpoint_optimizer_roundtrip(tmp_path, task):
    config = PipelineConfig(model=tiny_config(task), train=TrainConfig(steps=2))
    model = make_model(task)
    optimizer = optimizer_for(model, config.train)
    batch = inputs()
    prediction = model(*batch)
    loss, _ = structured_loss(prediction, teacher_target(prediction, task), task=task, samples=4)
    loss.backward()
    optimizer.step()
    expected = {key: value.detach() for key, value in model(*batch)["future"].items()}
    provenance = {"manifest": "test"}
    path = tmp_path / "check.pt"
    save_checkpoint(
        path, model, optimizer, config=config, step=1, epoch=3, cursor=7, provenance=provenance
    )
    restored = make_model(task)
    restored_optimizer = optimizer_for(restored, config.train)
    assert load_checkpoint(path, restored, restored_optimizer, config, provenance) == (1, 3, 7)
    actual = restored(*batch)["future"]
    assert all(torch.equal(actual[key], value) for key, value in expected.items())
    with pytest.raises(ValueError, match="protocol mismatch"):
        load_checkpoint(path, restored, restored_optimizer, config, {"manifest": "changed"})


def test_random_windows_are_episode_bounded(tmp_path):
    import json

    rows = [
        {
            "id": "e0-image",
            "episode": 0,
            "task": "task",
            "length": 100,
            "episode_start_frame": 20,
            "fps": 10.0,
            "video": "not-decoded",
        }
    ]
    manifest = tmp_path / "episodes.jsonl"
    manifest.write_text(json.dumps(rows[0]) + "\n")
    text = tmp_path / "text.pt"
    torch.save({"task": torch.ones(3, 4096)}, text)
    dataset = StructuredDataset(manifest, text, windows_per_episode_camera=4)
    first = [dataset.window(i) for i in range(len(dataset))]
    assert first == [dataset.window(i) for i in range(len(dataset))]
    for row in first:
        assert len(row["frame_indices"]) == 49
        assert 20 <= min(row["frame_indices"]) < max(row["frame_indices"]) < 120
    dataset.set_epoch(1)
    assert first != [dataset.window(i) for i in range(len(dataset))]
