#!/usr/bin/env bash
# Ascend 910B launcher; same recipe as launch_vace_online_structured.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
export LATENT_POLICY_PYTHON="${VACE_NPU_PYTHON:-/home/ma-user/anaconda3/envs/vace-npu/bin/python}"
export PYTHONPATH="$LATENT_HEFT_ROOT/diffusers/src:$PYTHONPATH"
export VACE_NPU=1
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
cd "$LATENT_HEFT_ROOT"
export VACE_CFG_PARALLEL="${VACE_CFG_PARALLEL:-1}"
task="${1:-shared}"; if [[ $# -gt 0 ]]; then shift; fi
exec "$LATENT_POLICY_PYTHON" -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-8}" \
  "$SCRIPT_DIR/train_vace_structured.py" \
  --config "$LATENT_POLICY_ROOT/configs/vace_structured_npu.json" \
  --task "$task" \
  --batch-size "${VACE_STUDENT_BATCH_SIZE:-32}" \
  --teacher-batch-size "${VACE_TEACHER_BATCH_SIZE:-8}" \
  "$@"
