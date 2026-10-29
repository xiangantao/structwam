#!/usr/bin/env bash
# NPU launcher for VACE structured distillation. Usage: launch_vace_npu.sh <nproc> [train args...]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
POLICY="$(cd -- "$SCRIPT_DIR/.." && pwd)"
HEFT="$(cd -- "$POLICY/.." && pwd)"
W="$(cd -- "$HEFT/.." && pwd)"
PY=/home/ma-user/anaconda3/envs/vace-npu/bin/python
NPROC="${1:-8}"; shift || true

export VACE_NPU=1
export PYTHONPATH="$POLICY/src:$HEFT:$HEFT/src:$HEFT/diffusers/src:$W/vjepa2${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export VACE_CFG_PARALLEL="${VACE_CFG_PARALLEL:-1}"
export TMPDIR="${TMPDIR:-/tmp}"
cd "$HEFT"
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  "$SCRIPT_DIR/train_vace_structured.py" \
  --config "$POLICY/configs/vace_structured_npu.json" \
  --task shared "$@"
