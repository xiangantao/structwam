"""Clip-coherent appearance changes and label-free partition consistency."""

import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vace_latent_distill.config import HEAD_TASKS, timeline
from vace_latent_distill.seg_invariance import appearance_augmentation, seg_invariance_loss
from test_pipeline import inputs, make_model


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.mark.parametrize("kind", ("color", "texture"))
def test_appearance_transform_has_no_temporal_flicker_and_repeats_with_seed(kind):
    frame = torch.randint(20, 235, (2, 1, 3, 32, 32), dtype=torch.uint8)
    clip = frame.expand(-1, 33, -1, -1, -1).clone()
    original = clip.clone()
    torch.manual_seed(19)
    a = appearance_augmentation(clip, kind)
    torch.manual_seed(19)
    b = appearance_augmentation(clip, kind)
    assert a.dtype == torch.uint8 and a.shape == clip.shape
    assert torch.equal(a, b) and torch.equal(clip, original)
    assert torch.equal(a, a[:, :1].expand_as(a))
    assert not torch.equal(a, clip)


def test_texture_augmentation_suppresses_high_frequency_but_preserves_edge_location():
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    checkerboard = ((xx + yy) % 2 * 100 + 70).to(torch.uint8)
    clip = checkerboard[None, None, None].expand(1, 33, 3, -1, -1).clone()
    torch.manual_seed(3)
    soft = appearance_augmentation(clip, "texture")
    before = (clip.float()[..., 1:] - clip.float()[..., :-1]).abs().mean()
    after = (soft.float()[..., 1:] - soft.float()[..., :-1]).abs().mean()
    assert after < before * 0.95
    edge = torch.full_like(clip, 70)
    edge[..., 16:] = 170
    torch.manual_seed(3)
    softened_edge = appearance_augmentation(edge, "texture").float()
    delta = (softened_edge[..., 1:] - softened_edge[..., :-1]).abs().mean((0, 1, 2, 3))
    assert int(delta.argmax()) == 15
    constant = torch.full_like(clip, 100)
    assert torch.equal(appearance_augmentation(constant, "texture"), constant)


def fields():
    value = {"timeline": timeline()}
    for branch, temporal in (("observed", 12), ("future", 13)):
        value[branch] = {
            name: F.normalize(torch.randn(2, temporal, 4, 8), dim=-1)
            for name in ("seg_k_l22h9", "seg_k_l15h7", "track_q", "track_k", "text_attention")
        }
        value[branch]["depth"] = torch.randn(2, temporal, 4)
    return value


def leaves(value):
    return {"timeline": value["timeline"], **{
        branch: {key: tensor.clone().requires_grad_() for key, tensor in value[branch].items()}
        for branch in ("observed", "future")
    }}


def test_seg_consistency_is_angular_and_excludes_unused_times():
    clean = fields()
    augmented = leaves(clean)
    for branch, times in timeline().items():
        for name in ("seg_k_l22h9", "seg_k_l15h7"):
            with torch.no_grad():
                augmented[branch][name] *= 3
                augmented[branch][name][:, ~times["valid"]] = torch.randn_like(
                    augmented[branch][name][:, ~times["valid"]]
                )
    loss, details = seg_invariance_loss(augmented, clean, "shared")
    assert abs(float(loss.detach())) < 1e-6
    assert set(details) == {"total", "seg_l22h9/observed", "seg_l22h9/future",
                            "seg_l15h7/observed", "seg_l15h7/future"}


def test_only_augmented_partition_heads_receive_direct_gradients():
    clean = leaves(fields())
    augmented = leaves(fields())
    weights = dict.fromkeys(HEAD_TASKS, 1.0)
    weights["seg_l15h7"] = 0.0
    loss, _ = seg_invariance_loss(augmented, clean, "shared", weights)
    loss.backward()
    for branch, times in timeline().items():
        for name, tensor in clean[branch].items():
            assert tensor.grad is None
        for name, tensor in augmented[branch].items():
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
            if name == "seg_k_l22h9":
                assert tensor.grad[:, times["valid"]].abs().sum() > 0
                assert tensor.grad[:, ~times["valid"]].abs().sum() == 0
            else:
                assert tensor.grad.abs().sum() == 0


def test_shared_student_backpropagates_seg_consistency_through_both_timelines():
    model = make_model("shared", checkpointing=True)
    rgb, text, mask = inputs()
    with torch.no_grad():
        clean = model(rgb, text, mask)
    augmented = model(appearance_augmentation(rgb, "color"), text, mask)
    loss, _ = seg_invariance_loss(augmented, clean, "shared")
    assert torch.isfinite(loss)
    loss.backward()
    assert model.backbone.patch_embed.conv.weight.grad.abs().sum() > 0
    assert model.future_query.grad.abs().sum() > 0
    for name in ("seg_l22h9", "seg_l15h7"):
        assert model.heads[name].primary.skip.weight.grad.abs().sum() > 0
    for name in ("tracking", "semantic", "depth"):
        assert model.heads[name].primary.skip.weight.grad.abs().sum() == 0


def test_independent_seg_and_future_zero_weight():
    clean = leaves(fields())
    augmented = leaves(fields())
    for value in (clean, augmented):
        for branch in ("observed", "future"):
            value[branch] = {"seg_k": value[branch]["seg_k_l22h9"]}
    loss, details = seg_invariance_loss(augmented, clean, "seg_l22h9", future_weight=0)
    assert torch.allclose(loss.detach(), details["seg_l22h9/observed"])
    loss.backward()
    assert augmented["observed"]["seg_k"].grad.abs().sum() > 0
    assert augmented["future"]["seg_k"].grad.abs().sum() == 0
    with pytest.raises(ValueError, match="shared or seg"):
        seg_invariance_loss(augmented, clean, "tracking")
