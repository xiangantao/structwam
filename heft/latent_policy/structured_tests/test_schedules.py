"""Fixed-endpoint sparse denoising and the production teacher call contract."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "diffusers/src"))

from diffusers import UniPCMultistepScheduler
from vace_latent_distill.config import PipelineConfig, TeacherConfig
from vace_latent_distill.schedules import sparse_schedule
from vace_latent_distill.teacher import VaceTeacher


def template():
    return UniPCMultistepScheduler(
        use_flow_sigmas=True, flow_shift=3.0, prediction_type="flow_prediction"
    )


@pytest.mark.parametrize("updates,relative", [
    (1, [0, 13]), (2, [0, 6, 13]), (4, [0, 3, 6, 10, 13]),
    (13, list(range(14))),
])
def test_sparse_nodes_and_nonterminal_endpoint(updates, relative):
    scheduler, begin, nodes = sparse_schedule(template(), TeacherConfig(), updates, "cpu")
    assert begin == 17
    assert [node["reference_step"] for node in nodes] == relative
    assert nodes[0]["sigma"] == pytest.approx(0.8019521832466125, abs=1e-7)
    assert nodes[-1]["sigma"] == pytest.approx(0.4996665418148041, abs=1e-7)
    assert nodes[-1]["timestep"] == 499
    original = template()
    original.set_timesteps(40)
    assert torch.equal(scheduler.sigmas[:begin], original.sigmas[:begin])
    assert torch.equal(scheduler.sigmas[begin + updates:], original.sigmas[begin + 13:])
    assert torch.equal(scheduler.timesteps[begin + updates:], original.timesteps[begin + 13:])
    assert len(scheduler.timesteps) > begin + updates + scheduler.config.solver_order


def test_step13_matches_original_solver_and_resets_history():
    original = template()
    original.set_timesteps(40)
    original.set_begin_index(17)
    scheduler, begin, _ = sparse_schedule(original, TeacherConfig(), 13, "cpu")
    left = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16
    right = left.clone()
    for index in range(13):
        timestep = original.timesteps[17 + index]
        left = original.step(left.sin() * 0.1, timestep, left, return_dict=False)[0]
        right = scheduler.step(right.sin() * 0.1, timestep, right, return_dict=False)[0]
    assert torch.equal(left, right)
    assert scheduler.step_index == begin + 13
    fresh, fresh_begin, _ = sparse_schedule(scheduler, TeacherConfig(), 1, "cpu")
    assert scheduler.step_index == begin + 13
    assert fresh.step_index is None and fresh.begin_index == fresh_begin
    assert fresh.model_outputs == [None] * fresh.config.solver_order
    assert fresh.last_sample is None and fresh.lower_order_nums == 0


def test_single_update_integrates_entire_noise_interval():
    scheduler, begin, nodes = sparse_schedule(template(), TeacherConfig(), device="cpu")
    sample, velocity = torch.ones(1, 1, 2, 2), torch.full((1, 1, 2, 2), 0.3)
    result = scheduler.step(velocity, scheduler.timesteps[begin], sample, return_dict=False)[0]
    expected = sample + (nodes[-1]["sigma"] - nodes[0]["sigma"]) * velocity
    assert torch.allclose(result, expected, atol=1e-6, rtol=0)
    assert scheduler.step_index == begin + 1


def test_recipe_defaults_and_invalid_intervals():
    config = PipelineConfig.load(ROOT / "configs/vace_structured.json")
    assert (config.teacher.denoise_updates, config.teacher.denoise_reference_updates) == (1, 13)
    reference = PipelineConfig.load(ROOT / "configs/vace_structured_step13_reference.json")
    assert reference.teacher == dataclasses.replace(config.teacher, denoise_updates=13)
    for updates in (0, 14):
        with pytest.raises(ValueError, match="denoising schedule"):
            TeacherConfig(denoise_updates=updates).validate()
    with pytest.raises(ValueError, match="capture timestep"):
        sparse_schedule(template(), TeacherConfig(sigma=0.1))


@pytest.mark.parametrize("parallel_cfg", ["0", "1"])
def test_extract_batch_one_update_then_endpoint_capture(monkeypatch, parallel_cfg):
    monkeypatch.setenv("VACE_CFG_PARALLEL", parallel_cfg)
    teacher = VaceTeacher.__new__(VaceTeacher)
    teacher.config = TeacherConfig(image_size=16)
    teacher.task, teacher.device = "semantic", torch.device("cpu")
    teacher.model, teacher.protocol = None, {}
    teacher.scheduler = template()
    teacher.negative = torch.zeros(1, 5, 4)

    def encode(videos):
        return [torch.zeros(16, (video.shape[1] + 3) // 4, 2, 2) for video in videos]

    teacher.vae = SimpleNamespace(encode=encode)
    calls = []

    def forward(latent, control, text, timestep, capture=None):
        calls.append((float(timestep), latent.clone(), capture is not None))
        return latent * 0.25

    teacher._forward = forward
    teacher._readout = lambda capture, *_: {
        "text_attention": torch.full((capture.temporal, 1, 5), 0.2)
    }
    targets = teacher.extract_batch(
        torch.zeros(2, 49, 3, 16, 16, dtype=torch.uint8),
        [torch.zeros(5, 4)] * 2, [torch.ones(5, dtype=torch.bool)] * 2,
        [42, 43], [torch.ones(5, dtype=torch.bool)] * 2,
    )
    expected_times = [1000, 801, 499] if parallel_cfg == "1" else [1000, 801, 801, 499]
    assert [call[0] for call in calls] == expected_times
    assert calls[1][1].shape[0] == (4 if parallel_cfg == "1" else 2)
    initial = calls[1][1][:2]
    sigma_delta = teacher.protocol["capture_sigma"] - teacher.protocol["actual_sigma"]
    assert torch.allclose(calls[-1][1], initial + sigma_delta * initial * 0.25, atol=1e-6)
    assert calls[-1][2] and calls[-1][1].shape[0] == 2
    assert len(targets) == 2
    assert [node["reference_step"] for node in teacher.protocol["denoise_schedule"]] == [0, 13]
    assert teacher.protocol["capture_timestep"] == 499
