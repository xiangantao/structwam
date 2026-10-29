"""Real LeRobot imports/serialization. Skipped explicitly when offline dependencies are absent."""

import pytest

pytest.importorskip("draccus", reason="LeRobot integration requires the offline dependency bundle")
pytest.importorskip("gymnasium", reason="LeRobot integration requires gymnasium")
pytest.importorskip("datasets", reason="LeRobot integration requires Hugging Face datasets")

import draccus
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from lerobot_policy_latent_act import LatentACTConfig
from lerobot_policy_latent_act.modeling_latent_act import LatentACTPolicy


def config():
    return LatentACTConfig(
        device="cpu",
        task_names=[f"task {i}" for i in range(10)],
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 256, 256)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )


def test_registration_and_config_roundtrip():
    cfg = config()
    cfg.validate_features()
    restored = draccus.decode(PreTrainedConfig, draccus.encode(cfg, PreTrainedConfig))
    assert isinstance(restored, LatentACTConfig)
    assert restored.image_observation_delta_indices == list(range(-15, 1))
    assert list(restored.image_features) == restored.camera_keys
    assert get_policy_class("latent_act") is LatentACTPolicy


def test_modality_history_contract():
    from types import SimpleNamespace

    cfg = config()
    meta = SimpleNamespace(features=cfg.input_features | cfg.output_features, fps=20)
    delta = resolve_delta_timestamps(cfg, meta)
    assert delta["observation.images.image"] == [i / 20 for i in range(-15, 1)]
    assert "observation.state" not in delta
    assert delta["action"] == [i / 20 for i in range(16)]


def test_processor_roundtrip_and_no_visual_normalization(tmp_path):
    cfg = config()
    stats = {
        key: {"mean": torch.zeros(dim), "std": torch.ones(dim)}
        for key, dim in (("observation.state", 8), ("action", 7))
    }
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    image = torch.rand(1, 16, 3, 256, 256)
    batch = {
        "observation.state": torch.randn(1, 8),
        "action": torch.randn(1, 16, 7),
        "observation.images.image": image,
        "observation.images.image2": image,
        "task": ["task 0"],
    }
    output = pre(batch)
    torch.testing.assert_close(output["observation.images.image"], image)
    torch.testing.assert_close(post(output["action"]), batch["action"])
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)


def test_distilled_config_roundtrip_and_trainable_adapter():
    cfg = config()
    cfg.encoder = "distilled"
    cfg.dim_model = 256
    cfg.n_heads = 8
    cfg.distilled_root = "/local/student"
    cfg.distilled_checkpoint = "/local/student.pt"
    cfg.distilled_checkpoint_sha256 = "1" * 64
    cfg.distilled_runtime_python = "/local/python"
    cfg.validate_features()
    restored = draccus.decode(PreTrainedConfig, draccus.encode(cfg, PreTrainedConfig))
    assert restored.encoder == "distilled"
    assert restored.visual_dim == 1152
    assert restored.dim_model == 256
    assert restored.distilled_checkpoint_sha256 == "1" * 64
    policy = LatentACTPolicy(restored)
    try:
        projection = policy.model.encoder_img_feat_input_proj
        assert tuple(projection.weight.shape) == (256, 1152, 1, 1)
        assert projection.weight.requires_grad
        optim_ids = {id(p) for group in policy.get_optim_params() for p in group["params"]}
        assert id(projection.weight) in optim_ids
        assert not any("backbone" in name or "runtime" in name for name in policy.state_dict())
        assert policy.model._visual_runtime is None
    finally:
        policy.close()
