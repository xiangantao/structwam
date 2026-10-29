#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/heft/latent_policy/.venv/bin/python"
TRAINER="${ROOT}/heft/latent_policy/scripts/train_vace_tracking_lora.py"
DATA_ROOT="${DATA_ROOT:-${ROOT}/runs/vace_tracking_all_suites_20261004}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/vace_object_group_lora_spatial_heads_20261005}"
GPU_LIST="${GPU_LIST:-7}"
STEPS="${STEPS:-6000}"
NPROC="${NPROC:-1}"
CANDIDATES="${CANDIDATES:-0:1,10:2,10:7,0:11}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${ROOT}/runs/vace_object_group_lora_spatial_heads_20261005/split.json}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export LIBERO_CONFIG_PATH="${ROOT}/heft/latent_policy/.runtime/libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

mkdir -p "${RUN_ROOT}"
LAUNCH=("${PY}" -u)
DIST_ARGS=()
if [[ "${NPROC}" -gt 1 ]]; then
  LAUNCH=("${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29641}")
  DIST_ARGS=(--ddp)
fi
exec "${LAUNCH[@]}" \
  "${TRAINER}" \
  --output "${RUN_ROOT}/training" \
  --video-root "${DATA_ROOT}/videos" \
  --tracks-root "${DATA_ROOT}/tracks" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --windows-per-episode 2 \
  --cameras agentview robot0_eye_in_hand \
  --candidates "${CANDIDATES}" \
  --size 512 --max-frames 129 --skip-frames 8 \
  --points-per-subject 24 --min-motion 0.02 \
  --rank 8 --alpha 16 \
  --spatial-weight 1.0 --temporal-weight 0.01 \
  --rollout-weight 1.0 --separation-weight 0.25 --separation-margin 0.5 \
  --steps "${STEPS}" --lr 0.0001 --save-every 500 \
  --include-static --balance-groups --fresh-noise --lazy-controls --truncate-after-candidates \
  --skip-initial-eval "${DIST_ARGS[@]}"
