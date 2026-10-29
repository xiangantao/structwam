#!/usr/bin/env bash
set -euo pipefail

ROOT="/increase_kairos_vepfs/increase/liwenhao/agent/2337888765"
PYTHON="${ROOT}/heft/latent_policy/.venv/bin/python"
TORCHRUN="${ROOT}/heft/latent_policy/.venv/bin/torchrun"
TRAINER="${ROOT}/heft/latent_policy/scripts/train_vace_tracking_lora.py"
OUTPUT="${ROOT}/runs/vace_tracking_lora_long_20261003"
LOG_DIR="${ROOT}/runs/vace_tracking_logs"
LOG_FILE="${LOG_DIR}/lora_long_20261003.log"
PID_FILE="${LOG_DIR}/lora_long_20261003.pid"

if [[ -e "${OUTPUT}" ]]; then
  echo "Output already exists: ${OUTPUT}" >&2
  echo "Choose a new OUTPUT before launching." >&2
  exit 1
fi

if [[ -e "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Training is already running with PID ${old_pid}." >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MASTER_PORT="29651"
export OMP_NUM_THREADS="1"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export TMPDIR="/tmp"

cd "${ROOT}"

nohup "${TORCHRUN}" \
  --standalone \
  --nproc_per_node=8 \
  "${TRAINER}" \
  --ddp \
  --output "${OUTPUT}" \
  --train-episodes 0 1 \
  --val-episodes 2 \
  --cameras agentview robot0_eye_in_hand \
  --steps 600 \
  --lr 3e-4 \
  --rank 8 \
  --alpha 16 \
  --rollout-weight 1.0 \
  --separation-weight 0.25 \
  --separation-margin 1.0 \
  --save-every 50 \
  >"${LOG_FILE}" 2>&1 &

echo $! >"${PID_FILE}"
echo "Started VACE tracking LoRA training."
echo "PID: ${PID_FILE} ($(cat "${PID_FILE}"))"
echo "Log: ${LOG_FILE}"
echo "Output: ${OUTPUT}"
