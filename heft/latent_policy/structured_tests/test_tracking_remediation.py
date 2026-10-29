"""Optional tracking-head remediation: teacher seed rule, seed-stratified relation
queries, per-time future queries, position scale, diagnostics and resume defaults."""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_pipeline import TinyBackbone, inputs, make_model, teacher_target, tiny_config
from vace_latent_distill.config import PipelineConfig, TeacherConfig, TrainConfig, timeline
from vace_latent_distill.losses import (
    match_logits,
    masked_mean,
    relation_loss,
    seed_cells,
    stratified_queries,
    structured_loss,
    tracking_diagnostics,
)
from vace_latent_distill.model import StructuredStudent
from vace_latent_distill.teacher import seed_heat
from vace_latent_distill.training import load_checkpoint, optimizer_for, save_checkpoint

SAMPLING = {"seed_fraction": 0.5, "moving_weight": 3.0, "min_shift": 1.0}


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_seed_heat_spatial_normalization_prefers_concentrated_tokens():
    attention = np.zeros((1, 4, 3), np.float32)
    attention[..., 0] = 0.5                       # bright everywhere (verb-like)
    attention[..., 1] = 0.01
    attention[0, 2, 1] = 0.4                      # concentrated on cell 2
    content = np.array([True, True, False])
    original = seed_heat(attention, content, "max")
    np.testing.assert_allclose(original, 0.5)     # the broad token wins every cell
    normalized = seed_heat(attention, content, "spatial_normalized_max")
    assert int(normalized[0].argmax()) == 2 and normalized[0, 2] > 3
    with pytest.raises(ValueError, match="Unknown seed rule"):
        seed_heat(attention, content, "sum")
    with pytest.raises(ValueError, match="content-token"):
        seed_heat(attention, np.zeros(3, bool))
    with pytest.raises(ValueError, match="seed_rule"):
        TeacherConfig(seed_rule="sum").validate()


def test_seed_cells_marks_only_seeds_whose_teacher_path_moves():
    anchors = torch.tensor([4, 6])
    seeds = torch.tensor([[[5, 9], [5, 20]]])                 # [B, A, P]; cell 5 twice
    paths = torch.zeros(1, 2, 2, 12, 2)
    valid = torch.zeros(1, 2, 2, 12, dtype=torch.bool)
    valid[:, 0, :, 4:7] = True
    valid[:, 1, :, 6:9] = True
    paths[0, 1, 1, 6:9] = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])  # seed 20 moves
    paths[0, 0, 1, 4:7] = torch.tensor([[2.0, 2.0], [2.4, 2.0], [2.2, 2.0]])  # seed 9 < 1 cell
    seeded, moved = seed_cells(seeds, paths, valid, anchors, 32, 1.0)
    assert set(seeded[0].nonzero().flatten().tolist()) == {5, 9, 20}
    assert moved[0].nonzero().flatten().tolist() == [20]


def test_stratified_queries_take_seed_cells_first_without_duplicates():
    torch.manual_seed(0)
    seeded = torch.zeros(3, 64, dtype=torch.bool)
    seeded[:, 10:20] = True
    ids = stratified_queries(seeded, 8, 0.5)
    assert ids.shape == (3, 8)
    assert seeded.gather(1, ids[:, :4]).all()
    assert all(len(set(row.tolist())) == 8 for row in ids)
    # More seed queries requested than seed cells: the remainder comes from other cells.
    ids = stratified_queries(seeded, 16, 1.0)
    assert all(len(set(row.tolist())) == 16 for row in ids)
    assert int(seeded.gather(1, ids).sum(-1).min()) == 10


def reference_relation_loss(query, key, teacher_q, teacher_k, valid, *, samples, temperature):
    """The pre-remediation implementation, kept to pin the default behavior."""
    terms = []
    for step in (1, 2):
        for time in range(query.shape[1] - step):
            mask = valid[:, time] & valid[:, time + step]
            if not mask.any():
                continue
            ids = torch.randperm(query.shape[2], device=query.device)[:samples]
            pred = match_logits(query[:, time, ids], key[:, time + step], temperature)
            reference = match_logits(
                teacher_q[:, time, ids], teacher_k[:, time + step], temperature
            )
            error = F.kl_div(pred.log_softmax(-1), reference.softmax(-1), reduction="none").sum(-1)
            terms.append(masked_mean(error, mask))
    return torch.stack(terms).mean()


def test_uniform_relation_loss_is_unchanged():
    q, k, tq, tk = (torch.randn(2, 5, 16, 8) for _ in range(4))
    valid = torch.tensor([[False, True, True, True, True]] * 2)
    torch.manual_seed(3)
    expected = reference_relation_loss(q, k, tq, tk, valid, samples=4, temperature=0.07)
    torch.manual_seed(3)
    actual = relation_loss(q, k, tq, tk, valid, samples=4, temperature=0.07)
    assert torch.equal(actual, expected)


def test_seeded_relation_loss_weights_moving_seeds_and_backpropagates():
    q = torch.randn(2, 5, 16, 8, requires_grad=True)
    k = torch.randn(2, 5, 16, 8)
    valid = torch.ones(2, 5, dtype=torch.bool)
    seeded = torch.zeros(2, 16, dtype=torch.bool)
    seeded[:, :4] = True
    moved = torch.zeros_like(seeded)
    moved[:, :2] = True
    same = relation_loss(q, k, q.detach(), k, valid, samples=8, temperature=0.07,
                         sampling=SAMPLING, seeds=(seeded, moved))
    assert abs(float(same)) < 1e-5
    teacher_q = torch.randn_like(q)
    values = []
    for weight in (1.0, 5.0):
        torch.manual_seed(1)
        values.append(relation_loss(q, k, teacher_q, k, valid, samples=8, temperature=0.07,
                                    sampling={**SAMPLING, "moving_weight": weight},
                                    seeds=(seeded, moved)))
    assert values[0] > 0 and not torch.isclose(values[0], values[1])
    values[1].backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


@pytest.mark.parametrize("task", ["tracking", "shared"])
def test_structured_loss_with_seed_sampling(task):
    model = make_model(task)
    prediction = model(*inputs())
    target = teacher_target(prediction, task)
    loss, details = structured_loss(prediction, target, task=task, samples=4,
                                    relation_sampling=SAMPLING)
    assert torch.isfinite(loss)
    loss.backward()
    prefix = "tracking/" if task == "shared" else ""
    assert f"{prefix}observed_moving_seed_cells" in details


def one_hot_branch(temporal, grid, shift):
    """Q at t is one-hot of its own cell; K at t+1 is one-hot of the cell `shift` columns left."""
    cells = grid * grid
    eye = torch.eye(cells)
    q = eye.expand(1, temporal, cells, cells).clone()
    k = q.clone()
    if shift:
        index = torch.arange(cells)
        source = torch.where(index % grid >= shift, index - shift, index)
        k = eye[source].expand(1, temporal, cells, cells).clone()
    return q, k


def test_tracking_diagnostics_identity_and_teacher_motion():
    grid = 4
    times = timeline()
    prediction, target = {"timeline": times}, {}
    for branch, temporal in (("observed", 12), ("future", 13)):
        q, k = one_hot_branch(temporal, grid, 0)
        tq, tk = one_hot_branch(temporal, grid, 1)
        anchors = times[branch]["anchor_native"]
        seeds = torch.tensor([[[0, 1]] * len(anchors)])            # cells (0,0) and (1,0)
        paths = torch.zeros(1, len(anchors), 2, temporal, 2)
        paths[..., 0] = torch.tensor([0.0, 1.0])[None, None, :, None]
        valid = torch.zeros(1, len(anchors), 2, temporal, dtype=torch.bool)
        for index, anchor in enumerate(anchors.tolist()):
            paths[0, index, :, anchor + 1:, 0] += 1                  # teacher moves +1 column
            valid[0, index, :, anchor:min(anchor + 3, temporal)] = True
        prediction[branch] = {"track_q": q, "track_k": k}
        target[branch] = {"track_q": tq, "track_k": tk, "seeds": seeds, "paths": paths,
                          "path_valid": valid}
    result = tracking_diagnostics(prediction, target)
    for branch in ("observed", "future"):
        assert result[f"diag/{branch}/self_match_student"] == 1
        # Teacher keys move one column; only first-column cells (ties) stay in place.
        assert result[f"diag/{branch}/self_match_teacher"] == 0.25
        assert result[f"diag/{branch}/seed_step_shift"] == 0
        assert result[f"diag/{branch}/teacher_step_shift"] == 1
        assert result[f"diag/{branch}/seed_step_error"] == 1


def test_default_model_parameters_are_unchanged():
    model = make_model("shared")
    assert model.future_query.shape == (model.config.dim,)
    assert not any("position_scale" in name for name, _ in model.named_parameters())


def test_per_time_future_query_and_learnable_position_scale():
    config = dataclasses.replace(tiny_config("shared"), future_query="per_time",
                                 learnable_position_scale=True, position_scale=0.1)
    model = StructuredStudent(config, TinyBackbone(config.dim))
    assert model.future_query.shape == (13, config.dim)
    assert torch.allclose(model.position_scale, torch.tensor([0.1]))
    prediction = model(*inputs())
    loss, _ = structured_loss(prediction, teacher_target(prediction, "shared"), task="shared",
                              samples=4, relation_sampling=SAMPLING)
    loss.backward()
    assert model.position_scale.grad is not None and torch.isfinite(model.position_scale.grad).all()
    assert model.future_query.grad.abs().sum(-1).gt(0).all()   # every future slot is trained
    fixed = StructuredStudent(dataclasses.replace(tiny_config("shared"), position_scale=0.0),
                              TinyBackbone(config.dim))
    assert torch.isfinite(fixed(*inputs())["future"]["track_q"]).all()
    for bad in ({"future_query": "both"}, {"position_scale": -1.0}):
        with pytest.raises(ValueError):
            dataclasses.replace(tiny_config("shared"), **bad).validate()


def test_train_config_relation_sampling_options_and_validation():
    assert TrainConfig().relation_sampling_options() is None
    options = TrainConfig(relation_sampling="teacher_seeds", relation_moving_weight=4.0)
    assert options.relation_sampling_options() == {"seed_fraction": 0.5, "moving_weight": 4.0,
                                                   "min_shift": 1.0}
    for bad in ({"relation_sampling": "motion"}, {"relation_seed_fraction": 1.5},
                {"relation_moving_weight": -1.0}, {"relation_moving_min_shift": -1.0}):
        with pytest.raises(ValueError):
            TrainConfig(**bad).validate()


def test_resume_fills_options_missing_from_older_checkpoints(tmp_path):
    config = PipelineConfig(model=tiny_config("shared"), train=TrainConfig(steps=2))
    model = make_model("shared")
    optimizer = optimizer_for(model, config.train)
    provenance = {"manifest": "test"}
    path = tmp_path / "old.pt"
    save_checkpoint(path, model, optimizer, config=config, step=1, epoch=0, cursor=1,
                    provenance=provenance)
    saved = torch.load(path, weights_only=False)
    for section, keys in (("model", ("future_query", "position_scale", "learnable_position_scale")),
                          ("teacher", ("seed_rule",)),
                          ("train", ("relation_sampling", "relation_seed_fraction",
                                     "relation_moving_weight", "relation_moving_min_shift",
                                     "tracking_diagnostics"))):
        for key in keys:
            del saved["config"][section][key]
    torch.save(saved, path)
    restored = make_model("shared")
    assert load_checkpoint(path, restored, optimizer_for(restored, config.train), config,
                           provenance) == (1, 0, 1)
    changed = PipelineConfig(model=tiny_config("shared"),
                             train=TrainConfig(steps=2, relation_sampling="teacher_seeds"))
    with pytest.raises(ValueError, match="relation_sampling"):
        load_checkpoint(path, restored, optimizer_for(restored, changed.train), changed, provenance)
    seeded = PipelineConfig(model=tiny_config("shared"),
                            teacher=TeacherConfig(seed_rule="spatial_normalized_max"),
                            train=TrainConfig(steps=2))
    with pytest.raises(ValueError, match="teacher/data protocol"):
        load_checkpoint(path, restored, optimizer_for(restored, seeded.train), seeded, provenance)
    diagnostics = PipelineConfig(model=tiny_config("shared"),
                                 train=TrainConfig(steps=2, tracking_diagnostics=True))
    assert load_checkpoint(path, restored, optimizer_for(restored, diagnostics.train), diagnostics,
                           provenance) == (1, 0, 1)


def test_relation_weight_and_trajectory_sigma_are_optional():
    model = make_model("tracking")
    prediction = model(*inputs())
    target = teacher_target(prediction, "tracking")
    torch.manual_seed(5)
    default, _ = structured_loss(prediction, target, task="tracking", samples=4)
    torch.manual_seed(5)
    same, _ = structured_loss(prediction, target, task="tracking", samples=4,
                              relation_weight=1.0, trajectory_sigma=0.65)
    assert torch.equal(default, same)
    torch.manual_seed(5)
    path_only, details = structured_loss(prediction, target, task="tracking", samples=4,
                                         relation_weight=0.0, cycle_weight=0.0)
    expected = 0.25 * (details["observed_path"] + details["future_path"])
    assert torch.allclose(path_only.detach(), expected)
    path_only.backward()
    torch.manual_seed(5)
    sharp, details = structured_loss(prediction, target, task="tracking", samples=4,
                                     relation_weight=0.5, trajectory_sigma=0.3)
    assert "observed_relation" in details and torch.isfinite(sharp)
    for bad in ({"relation_weight": -1.0}, {"trajectory_sigma": 0.0}):
        with pytest.raises(ValueError):
            TrainConfig(**bad).validate()


def test_descriptor_tracking_target():
    model = make_model("shared")
    prediction = model(*inputs())
    target = teacher_target(prediction, "shared")
    torch.manual_seed(2)
    default, _ = structured_loss(prediction, target, task="shared", samples=4)
    torch.manual_seed(2)
    same, _ = structured_loss(prediction, target, task="shared", samples=4,
                              tracking_target="relation")
    assert torch.equal(default, same)
    copied = {branch: dict(values) for branch, values in target.items()}
    for branch in ("observed", "future"):
        for key in ("track_q", "track_k"):
            copied[branch][key] = prediction[branch][key].detach()
    only_tracking = {"seg_l22h9": 0, "seg_l15h7": 0, "tracking": 1, "semantic": 0, "depth": 0}
    loss, details = structured_loss(prediction, copied, task="shared", samples=4,
                                    tracking_target="descriptor", trajectory_weight=0.0,
                                    cycle_weight=0.0, head_weights=only_tracking)
    assert abs(float(loss)) < 1e-5 and "tracking/observed_descriptor" in details
    loss, _ = structured_loss(prediction, target, task="shared", samples=4,
                              tracking_target="descriptor")
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    with pytest.raises(ValueError):
        TrainConfig(tracking_target="flow").validate()
