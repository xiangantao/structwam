from copy import deepcopy

import pytest
import torch


def test_suite_selection_and_counts(data_module):
    names = [f"Task {i}" for i in range(10)]
    episodes = [{"episode_index": i, "tasks": [f" task {i}. "], "length": i + 1} for i in range(10)]
    episodes.append({"episode_index": 10, "tasks": ["unrelated"], "length": 100})
    selected = data_module.select_suite_episodes(episodes, names)
    assert selected["episodes"] == list(range(10))
    assert selected["num_frames"] == 55
    with pytest.raises(ValueError, match="No local demonstrations"):
        data_module.select_suite_episodes(episodes[1:], names)
    episodes[0]["tasks"].append("unrelated")
    with pytest.raises(ValueError, match="ambiguous"):
        data_module.select_suite_episodes(episodes, names)


def test_suite_duplicate_episode_and_name(data_module):
    names = [f"task {i}" for i in range(10)]
    row = {"episode_index": 0, "tasks": [names[0]], "length": 1}
    with pytest.raises(ValueError, match="Duplicate episode"):
        data_module.select_suite_episodes([row, row], names)
    with pytest.raises(ValueError, match="10 distinct"):
        data_module.select_suite_episodes([], [names[0]] * 10)


def test_rgb_contract_roundtrip_and_reject_normalized(encoder_modules):
    common, _ = encoder_modules
    pixels = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 1, 16, 16).expand(2, 16, 3, -1, -1)
    from_byte, from_float = common.rgb_uint8([pixels, pixels.float() / 255], 16, 16)
    assert torch.equal(from_byte, from_float)
    with pytest.raises(ValueError, match="finite and in"):
        common.rgb_uint8([pixels.float() / 255 - 0.5], 16, 16)
    with pytest.raises(ValueError, match="Expected"):
        common.rgb_uint8([pixels[:, :1]], 16, 16)


def test_strict_checkpoint_loading(encoder_modules):
    _, vjepa = encoder_modules
    model = torch.nn.Linear(4, 3)
    expected = deepcopy(model.state_dict())
    checkpoint = {"target_encoder": {f"module.backbone.{k}": v for k, v in expected.items()}}
    vjepa.load_backbone_state(model, checkpoint)
    assert all(torch.equal(expected[k], v) for k, v in model.state_dict().items())
    del checkpoint["target_encoder"]["module.backbone.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        vjepa.load_backbone_state(model, checkpoint)
    with pytest.raises(RuntimeError, match="size mismatch"):
        vjepa.load_backbone_state(
            model, {"target_encoder": {"weight": torch.zeros(2, 4), "bias": torch.zeros(3)}}
        )


def test_local_checksum(encoder_modules, tmp_path):
    common, _ = encoder_modules
    path = tmp_path / "weights"
    path.write_bytes(b"local weights")
    digest = common.file_sha256(path)
    assert common.checked_local_file(str(path), digest) == path
    with pytest.raises(ValueError, match="checksum"):
        common.checked_local_file(str(path), "0" * 64)
