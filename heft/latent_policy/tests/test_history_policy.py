import sys
import types
from copy import deepcopy

import pytest
import torch
from conftest import LEROBOT, PLUGIN, definitions, load_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


@pytest.fixture
def plugin_core(core, data_module, monkeypatch):
    memory = definitions(
        (LEROBOT / "src/lerobot/policies/pi05/memory.py").read_text(),
        {"torch": torch, "Tensor": torch.Tensor, "math": __import__("math")},
        {"sample_observation_history", "temporal_sinusoidal_embedding"},
    )
    queue_source = LEROBOT / "src/lerobot/policies/utils.py"
    queues = definitions(queue_source.read_text(), dict(core), {"populate_queues"})
    for name, content in (
        ("lerobot.policies.utils", queues),
        ("lerobot.policies.pi05.memory", memory),
    ):
        module = types.ModuleType(name)
        module.__dict__.update(content)
        monkeypatch.setitem(sys.modules, name, module)
    history = load_file("latent_test_history", PLUGIN / "history.py")
    ns = dict(
        core,
        LatentACTConfig=types.SimpleNamespace,
        CameraHistory=history.CameraHistory,
        canonical_task=data_module.canonical_task,
        checkpoint_wrapper=checkpoint_wrapper,
        temporal_sinusoidal_embedding=memory["temporal_sinusoidal_embedding"],
    )
    return definitions((PLUGIN / "modeling_latent_act.py").read_text(), ns)


def test_history_padding_snapshots_and_reset(plugin_core):
    history = plugin_core["CameraHistory"](["a", "b"], 16)
    for step in range(20):
        image = torch.full((1, 3, 2, 2), float(step))
        batch = {"a": image, "b": image + 100, "task": ["task 0"]}
        history.update(batch)
        image.fill_(-1)
        stacked = history.stack(batch)
        expected = [max(0, i) for i in range(step - 15, step + 1)]
        assert stacked["a"][0, :, 0, 0, 0].tolist() == expected
        assert stacked["b"][0, -1, 0, 0, 0] == step + 100
        assert stacked["a_is_pad"].sum() == max(0, 15 - step)
    with pytest.raises(ValueError, match="Task changed"):
        history.update({"a": image, "b": image, "task": ["task 1"]})
    history.reset()
    history.update({"a": image, "b": image, "task": ["task 1"]})
    assert history.steps_seen == 1


def test_queued_actions_still_update_history(plugin_core, tiny_config):
    config = deepcopy(tiny_config)
    config.camera_keys = ["camera"]
    config.history_frames = 16
    config.task_names = [f"task {i}" for i in range(10)]
    config.suite = "libero_spatial"

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.windows = []

        def forward(self, batch):
            self.windows.append(batch["camera"].clone())
            assert batch["observation.environment_state"].shape == (1, 10)
            return torch.zeros(1, 3, 7), (None, None)

    class Policy(plugin_core["LatentACTPolicy"]):
        def _make_model(self, config):
            return RecordingModel()

    policy = Policy(config)
    for step in range(5):
        batch = {
            "camera": torch.full((1, 3, 2, 2), float(step)),
            "observation.state": torch.zeros(1, 8),
            "task": ["task 0"],
        }
        assert policy.select_action(batch).shape == (1, 7)
    assert len(policy.model.windows) == 3
    assert policy.model.windows[-1][0, -5:, 0, 0, 0].tolist() == [0, 1, 2, 3, 4]
    policy.reset()
    assert not policy._action_queue and policy._history.steps_seen == 0


@pytest.mark.parametrize(
    "encoder,shape", [("wan", (16, 896, 14, 14)), ("vjepa2", (8, 1024, 16, 16)),
                      ("wan_jepa", (8, 1152, 16, 16)), ("distilled", (8, 1152, 16, 16))]
)
def test_visual_projection_freeze_positions_and_checkpoint(
    plugin_core, tiny_config, encoder, shape
):
    config = deepcopy(tiny_config)
    config.visual_dim = shape[1]
    config.encoder = encoder
    config.history_frames = 16
    config.camera_keys = ["a", "b"]
    config.checkpoint_encoder = True
    model = plugin_core["LatentACT"](config)

    class FrozenRuntime:
        def encode(self, images):
            return [torch.ones(1, *shape, requires_grad=True) for _ in images]

        def close(self):
            pass

    model._visual_runtime = FrozenRuntime()
    model._runtime_device = torch.device("cpu")
    images = [torch.zeros(1), torch.zeros(1)]
    tokens, positions = model._visual_tokens(images)
    expected_tokens = 2 * shape[0] * shape[2] * shape[3]
    assert tokens.shape == (expected_tokens, 1, 16)
    assert positions.shape == tokens.shape
    assert not torch.equal(positions[: expected_tokens // 2], positions[expected_tokens // 2 :])
    tokens.square().mean().backward()
    assert model.encoder_img_feat_input_proj.weight.grad is not None
    assert not any("runtime" in name or "backbone" in name for name in model.state_dict())
    restored = plugin_core["LatentACT"](config)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored._visual_runtime = FrozenRuntime()
    restored._runtime_device = torch.device("cpu")
    torch.testing.assert_close(restored._visual_tokens(images)[0], tokens)
    model.close()
    assert model._visual_runtime is None


def test_actual_policy_cpu_update_and_safetensors_resume(plugin_core, tiny_config, tmp_path):
    from safetensors.torch import load_model, save_model

    config = deepcopy(tiny_config)
    config.encoder = "wan"
    config.visual_dim = 896
    config.history_frames = 16
    config.camera_keys = ["a", "b"]
    config.image_features = {"a": None, "b": None}
    config.task_names = [f"task {i}" for i in range(10)]
    config.suite = "libero_spatial"
    config.checkpoint_encoder = True
    config.need_attention_weights = False

    class Runtime:
        def encode(self, images):
            # Deterministic mock of frozen features; the ACT computation itself is real.
            return [torch.full((1, 16, 896, 14, 14), float(i + 1)) for i in range(2)]

        def close(self):
            pass

    policy = plugin_core["LatentACTPolicy"](config)
    policy.model._visual_runtime = Runtime()
    policy.model._runtime_device = torch.device("cpu")
    batch = {
        "a": torch.zeros(1, 16, 3, 1, 1),
        "b": torch.zeros(1, 16, 3, 1, 1),
        "observation.state": torch.randn(1, 8),
        "task": ["task 0"],
        "action": torch.randn(1, 3, 7),
        "action_is_pad": torch.tensor([[False, False, True]]),
    }
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-3)
    previous = policy.model.encoder_img_feat_input_proj.weight.detach().clone()
    loss, metrics = policy(batch)
    assert torch.isfinite(loss) and set(metrics) == {"l1_loss", "kld_loss"}
    loss.backward()
    optimizer.step()
    assert not torch.equal(previous, policy.model.encoder_img_feat_input_proj.weight)
    expected = policy.predict_action_chunk(batch)
    path = tmp_path / "model.safetensors"
    save_model(policy, path)
    restored = plugin_core["LatentACTPolicy"](config)
    restored.model._visual_runtime = Runtime()
    restored.model._runtime_device = torch.device("cpu")
    load_model(restored, path, strict=True)
    torch.testing.assert_close(restored.predict_action_chunk(batch), expected, rtol=0, atol=0)
