"""Optional future-branch architecture: motion-warped initialization, observed-field
temporal context and future decoder depth."""

import dataclasses
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_pipeline import TinyBackbone, inputs, make_model, teacher_target, tiny_config
from vace_latent_distill.losses import structured_loss
from vace_latent_distill.model import FutureMotion, StructuredStudent, extrapolate, warp


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def future_model(**changes):
    config = dataclasses.replace(tiny_config("shared"), **changes)
    return StructuredStudent(config, TinyBackbone(config.dim))


def test_warp_identity_and_integer_shift():
    field = torch.randn(2, 5, 5, 3)
    flow = torch.zeros(2, 2, 5, 5, 2)
    flow[:, 1, ..., 0] = 1.0                                   # slot 1: content moves right one cell
    moved = warp(field, flow)
    assert moved.shape == (2, 2, 5, 5, 3)
    torch.testing.assert_close(moved[:, 0], field)
    torch.testing.assert_close(moved[:, 1, :, 1:], field[:, :, :-1])
    torch.testing.assert_close(moved[:, 1, :, 0], field[:, :, 0])   # border clamps, no wrap


def test_future_motion_velocity_recovers_a_shift():
    motion = FutureMotion(dataclasses.replace(tiny_config("shared"), future_motion_radius=2), slots=3)
    grid = 6
    previous = F.normalize(torch.randn(1, grid * grid, 64), dim=-1)
    current = previous.reshape(1, grid, grid, 64).roll(1, dims=2).reshape(1, grid * grid, 64)
    velocity = motion.velocity(previous, current, grid)       # content moved +1 in x
    torch.testing.assert_close(velocity[0, :, 1:, 0], torch.ones(grid, grid - 1), atol=1e-3, rtol=0)
    torch.testing.assert_close(velocity[0, :, 1:, 1], torch.zeros(grid, grid - 1), atol=1e-3, rtol=0)
    field = torch.randn(1, grid, grid, motion.norm.normalized_shape[0])
    flow = motion(field, previous, current, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(flow[0, 0], torch.zeros_like(flow[0, 0]))   # zero residual: pure prior
    assert flow[0, 2, :, 4:, 0].sub(2.0).abs().max() < 0.05     # uniform shift: 2 steps of +1


def test_extrapolation_reads_the_velocity_of_the_content_that_arrives():
    # A 2x2 object at x=4..5 (last observed) moving +1 cell per step on a static background.
    velocity = torch.zeros(1, 12, 12, 2)
    velocity[0, 3:5, 4:6, 0] = 1.0
    flow = extrapolate(velocity, torch.tensor([0, 2]))
    assert flow[0, 0].abs().max() == 0
    # Base = one step later (object at x=5..6); two steps after that it occupies x=7..8.
    # Those target cells must sample the object (flow ~2), not the static background there.
    assert flow[0, 1, 3:5, 7:9, 0].min() > 1.9
    assert flow[0, 1, 7:].abs().max() == 0 and flow[0, 1, :, :3].abs().max() == 0  # background stays
    assert (2 * velocity[0, 3:5, 7:9, 0]).abs().max() == 0                          # naive read: none


def test_default_model_has_no_future_architecture_parameters():
    model = make_model("shared")
    assert model.future_motion is None
    assert len(model.future_blocks) == model.config.decoder_layers
    assert not any("future_motion" in name for name, _ in model.named_parameters())


def test_warp_with_zero_velocity_reproduces_the_default_future_branch(monkeypatch):
    torch.manual_seed(0)
    default = make_model("shared").eval()
    warped = future_model(future_init="warp").eval()
    missing, unexpected = warped.load_state_dict(default.state_dict(), strict=False)
    assert not unexpected and all(name.startswith("future_motion.") for name in missing)
    monkeypatch.setattr(warped.future_motion, "velocity",
                        lambda previous, current, grid: torch.zeros(len(current), grid, grid, 2))
    batch = inputs()
    with torch.no_grad():
        expected, actual = default(*batch), warped(*batch)
    for branch in ("observed", "future"):
        for key, value in expected[branch].items():
            torch.testing.assert_close(actual[branch][key], value)


def test_warp_model_trains_the_residual_flow_without_touching_observed_descriptors():
    model = future_model(future_init="warp")
    prediction = model(*inputs())
    loss, _ = structured_loss(prediction, teacher_target(prediction, "shared"), task="shared", samples=4)
    loss.backward()
    assert model.future_motion.residual[-1].weight.grad.abs().sum() > 0
    assert model.future_motion.gain.grad is not None
    # Observed inputs are detached: a future-only loss cannot retrain the observed decoder.
    model.zero_grad()
    model(*inputs())["future"]["seg_k_l15h7"].sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.observed_blocks.parameters())


@pytest.mark.parametrize("changes", [{"future_context": "memory"}, {"future_context": "observed"},
                                     {"future_init": "warp"}])
def test_future_reads_the_observed_decoder_only_when_enabled(changes):
    torch.manual_seed(0)
    model = future_model(future_layers=3, **changes).eval()
    assert len(model.future_blocks) == 3
    batch = inputs()
    with torch.no_grad():
        before = model(*batch)
        for parameter in model.observed_blocks.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.5)
        after = model(*batch)
    assert before["future"]["track_q"].shape[1] == 13
    changed = not torch.allclose(before["future"]["track_q"], after["future"]["track_q"])
    assert changed == (changes != {"future_context": "memory"})
    model.train()
    model(*batch)["future"]["track_q"].sum().backward()     # never reaches the observed decoder
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.observed_blocks.parameters())


def test_future_architecture_validation():
    for bad in ({"future_init": "flow"}, {"future_context": "both"}, {"future_layers": -1},
                {"future_motion_radius": 0}, {"future_motion_temperature": 0.0}):
        with pytest.raises(ValueError):
            dataclasses.replace(tiny_config("shared"), **bad).validate()
    with pytest.raises(ValueError, match="shared"):
        dataclasses.replace(tiny_config("tracking"), future_init="warp").validate()
