#!/usr/bin/env bash
# Source this file; it only selects local source trees and disables remote fetching.
LATENT_POLICY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LATENT_HEFT_ROOT="$(cd -- "$LATENT_POLICY_ROOT/.." && pwd)"
LATENT_WORKSPACE="$(cd -- "$LATENT_HEFT_ROOT/.." && pwd)"
export PYTHONPATH="$LATENT_POLICY_ROOT/src:$LATENT_HEFT_ROOT:$LATENT_HEFT_ROOT/src:$LATENT_WORKSPACE/repos/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
export LATENT_POLICY_PYTHON="$LATENT_POLICY_ROOT/.venv/bin/python"
export LATENT_WAN_PYTHON="$LATENT_HEFT_ROOT/.venv/bin/python"
export LATENT_DISTILLED_PYTHON="${LATENT_DISTILLED_PYTHON:-$LATENT_HEFT_ROOT/.venv/bin/python}"
export LIBERO_CONFIG_PATH="$LATENT_POLICY_ROOT/.runtime/libero"
# Keep Python multiprocessing/torch resource-sharer Unix sockets on the local filesystem.
# The workspace path is long enough to exceed AF_UNIX's pathname limit on DataLoader workers.
export LATENT_ACT_TMPDIR="${LATENT_ACT_TMPDIR:-/tmp/heft-latent-act}"
mkdir -p "$LATENT_ACT_TMPDIR"
export TMPDIR="$LATENT_ACT_TMPDIR"
export XDG_CACHE_HOME="$LATENT_POLICY_ROOT/.runtime/cache"
export NUMBA_CACHE_DIR="$LATENT_POLICY_ROOT/.runtime/cache/numba"
export HF_HOME="$LATENT_POLICY_ROOT/.runtime/cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
