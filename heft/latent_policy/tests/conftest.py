"""Dependency-light tests execute real source definitions; full integration is separately gated."""

import ast
import importlib.util
import math
import subprocess
import sys
import types
from collections import deque
from collections.abc import Callable
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import einops
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

WORKSPACE = Path(__file__).resolve().parents[3]
PLUGIN = WORKSPACE / "heft/latent_policy/src/lerobot_policy_latent_act"
LEROBOT = WORKSPACE / "repos/lerobot"
ACT_PATH = LEROBOT / "src/lerobot/policies/act/modeling_act.py"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data_module():
    return load_file("latent_test_data", PLUGIN / "data.py")


@pytest.fixture
def encoder_modules():
    package = types.ModuleType("latent_test_encoders")
    package.__path__ = [str(PLUGIN / "encoders")]
    sys.modules[package.__name__] = package
    common = load_file("latent_test_encoders.common", PLUGIN / "encoders/common.py")
    vjepa = load_file("latent_test_encoders.vjepa2", PLUGIN / "encoders/vjepa2.py")
    return common, vjepa


def definitions(source, namespace, names=None):
    """Execute source definitions without importing unavailable LeRobot's CLI/data dependencies."""
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and (names is None or node.name in names)
    ]
    exec(compile(tree, "<real-source-definitions>", "exec"), namespace)  # noqa: S102
    return namespace


@pytest.fixture
def core():
    class PolicyBase(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    namespace = {
        "torch": torch,
        "nn": nn,
        "Tensor": Tensor,
        "F": F,
        "np": np,
        "math": math,
        "einops": einops,
        "chain": chain,
        "deque": deque,
        "torchvision": torchvision,
        "Callable": Callable,
        "IntermediateLayerGetter": IntermediateLayerGetter,
        "FrozenBatchNorm2d": FrozenBatchNorm2d,
        "ACTConfig": SimpleNamespace,
        "PreTrainedPolicy": PolicyBase,
        "ACTION": "action",
        "OBS_STATE": "observation.state",
        "OBS_ENV_STATE": "observation.environment_state",
        "OBS_IMAGES": "observation.images",
    }
    return definitions(ACT_PATH.read_text(), namespace)


@pytest.fixture
def original_core(core):
    source = subprocess.check_output(
        ["git", "-C", str(LEROBOT), "show", "HEAD:src/lerobot/policies/act/modeling_act.py"],
        text=True,
    )
    return definitions(source, dict(core))


@pytest.fixture
def tiny_config():
    return SimpleNamespace(
        use_vae=True,
        dim_model=16,
        n_heads=4,
        dim_feedforward=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=4,
        dropout=0.0,
        pre_norm=False,
        feedforward_activation="relu",
        chunk_size=3,
        n_action_steps=2,
        kl_weight=10.0,
        robot_state_feature=SimpleNamespace(shape=(8,)),
        action_feature=SimpleNamespace(shape=(7,)),
        env_state_feature=SimpleNamespace(shape=(10,)),
        image_features={"camera": None},
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        replace_final_stride_with_dilation=False,
        temporal_ensemble_coeff=None,
        optimizer_lr_backbone=1e-5,
        validate_features=lambda: None,
    )


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)
