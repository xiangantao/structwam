#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HEFT_DIR="$(cd -- "$BASELINE_DIR/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HEFT_DIR/.venv/bin/python}"
CONFIG="${CONFIG:-$BASELINE_DIR/ssv2.yaml}"
LOG_DIR="${LOG_DIR:-$HEFT_DIR/../runs/ssv2_vjepa2_vitl_singlehead/launch_logs}"
mkdir -p -- "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
# Redirect before Python starts so missing dependencies and startup failures persist.
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'status=$?; printf "Launcher exit code: %s\n" "$status"' EXIT
printf 'Launch log: %s\n' "$LOG_FILE"
cd -- "$HEFT_DIR"
unset PYTHONHOME
export PYTHONPATH="$HEFT_DIR:$HEFT_DIR/../vjepa2"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_ERROR_REPORTING=false
export TMPDIR="${VJEPA_TMPDIR:-/dev/shm/heft-vjepa2-$UID/tmp}"
mkdir -p -- "$TMPDIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
unset SLURM_LOCALID
"$PYTHON_BIN" -m latent_video.classification.baselines.vjepa2.train --config "$CONFIG" --check "$@"
for arg in "$@"; do
    if [[ "$arg" == --check ]]; then
        exit 0
    fi
done
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE:-8}" \
    -m latent_video.classification.baselines.vjepa2.train --config "$CONFIG" "$@"
