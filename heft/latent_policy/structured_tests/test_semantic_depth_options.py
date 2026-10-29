"""Optional per-word spatial semantic loss and per-camera depth weights."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_pipeline import inputs, make_model, teacher_target
from vace_latent_distill.config import TrainConfig
from vace_latent_distill.losses import structured_loss, word_spatial_loss

CONTENT = torch.tensor([[True, True, False, False, False]] * 2)


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_word_spatial_loss_penalizes_blur_on_content_words_only():
    teacher = torch.full((1, 2, 4, 3), 0.1)
    teacher[..., 1, 0] = 0.9                      # content word 0 is sharp on cell 1
    valid = torch.ones(1, 2, dtype=torch.bool)
    content = torch.tensor([[True, False, False]])
    assert abs(float(word_spatial_loss(teacher, teacher, content, valid))) < 1e-6
    blurred = teacher.clone()
    blurred[..., 0] = 0.25
    loss = word_spatial_loss(blurred, teacher, content, valid)
    assert loss > 0.1
    changed = blurred.clone()
    changed[..., 2] = torch.rand(4)               # a non-content word does not count
    assert torch.allclose(word_spatial_loss(changed, teacher, content, valid), loss)
    with pytest.raises(ValueError, match="No content tokens"):
        word_spatial_loss(teacher, teacher, torch.zeros(1, 3, dtype=torch.bool), valid)


def test_semantic_spatial_term_is_optional():
    model = make_model("semantic")
    prediction = model(*inputs())
    target = teacher_target(prediction, "semantic")
    default, _ = structured_loss(prediction, target, task="semantic")
    zero, _ = structured_loss(prediction, target, task="semantic", semantic_spatial_weight=0.0,
                              content_mask=CONTENT)
    assert torch.equal(default, zero)
    with pytest.raises(ValueError, match="content_mask"):
        structured_loss(prediction, target, task="semantic", semantic_spatial_weight=1.0)
    loss, details = structured_loss(prediction, target, task="semantic",
                                    semantic_spatial_weight=1.0, content_mask=CONTENT)
    assert loss > default and "observed_word_spatial" in details
    loss.backward()
    shared = make_model("shared")
    prediction = shared(*inputs())
    _, details = structured_loss(prediction, teacher_target(prediction, "shared"), task="shared",
                                 samples=4, semantic_spatial_weight=1.0, content_mask=CONTENT)
    assert "semantic/future_word_spatial" in details


def test_depth_sample_weights_drop_zero_weight_samples():
    model = make_model("depth")
    prediction = model(*inputs())
    target = teacher_target(prediction, "depth")
    default, _ = structured_loss(prediction, target, task="depth")
    ones, _ = structured_loss(prediction, target, task="depth",
                              depth_sample_weights=torch.ones(2))
    assert torch.allclose(default, ones)
    first_only = torch.tensor([1.0, 0.0])
    reference, _ = structured_loss(prediction, target, task="depth",
                                   depth_sample_weights=first_only)
    shifted = {branch: dict(values) for branch, values in target.items()}
    for branch in ("observed", "future"):
        depth = shifted[branch]["depth"].clone()
        depth[1] += 100                            # only the zero-weight sample changes
        shifted[branch]["depth"] = depth
    same, _ = structured_loss(prediction, shifted, task="depth", depth_sample_weights=first_only)
    assert torch.allclose(reference, same)
    unweighted, _ = structured_loss(prediction, shifted, task="depth")
    assert unweighted > default + 10
    nothing, _ = structured_loss(prediction, target, task="depth",
                                 depth_sample_weights=torch.zeros(2))
    assert torch.isfinite(nothing)


def test_semantic_and_depth_option_validation():
    TrainConfig(semantic_spatial_weight=1.0,
                depth_camera_weights={"observation.images.image": 0.0,
                                      "observation.images.image2": 1.0}).validate()
    for bad in ({"semantic_spatial_weight": -1.0}, {"depth_camera_weights": {}},
                {"depth_camera_weights": {"observation.images.image": -1.0}}):
        with pytest.raises(ValueError):
            TrainConfig(**bad).validate()
