#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HEFT_DIR="$(cd -- "$BASELINE_DIR/../../../.." && pwd)"
SOURCE_ENV="${VJEPA_SOURCE_ENV:-$HEFT_DIR/.venv}"
SHARED_PYTHON="$SOURCE_ENV/bin/python"
MODEL_PATH="$HEFT_DIR/../models/vjepa2/vitl.pt"
CONFIG="$BASELINE_DIR/ssv2.yaml"
OUTPUT_DIR="${VJEPA_OUTPUT_DIR:-$HEFT_DIR/../runs/ssv2_vjepa2_vitl_singlehead}"
export LOG_DIR="$OUTPUT_DIR/launch_logs"
mkdir -p -- "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/pool_$(date -u +%Y%m%dT%H%M%SZ)_$$.log") 2>&1
trap 'status=$?; printf "Pool entry exit code: %s\n" "$status"' EXIT
printf 'Pool entry: %s\n' "$(date -u +%FT%TZ)"
printf 'Workspace: %s\nOutput: %s\n' "$HEFT_DIR" "$OUTPUT_DIR"

if [[ -n "${TORCHELASTIC_RUN_ID:-}" ]]; then
    printf 'ERROR: run this script once per node, without an outer torchrun.\n' >&2
    exit 1
fi
if [[ ! -x "$SHARED_PYTHON" || ! -r "$MODEL_PATH" ]]; then
    printf 'ERROR: shared Python/weights unavailable. Check the vePFS mount and container user.\n' >&2
    exit 1
fi
CHECK_ONLY=0
PREPARE_ONLY=0
RESUMING=0
TRAIN_ARGS=()
for arg in "$@"; do
    [[ "$arg" != --check ]] || CHECK_ONLY=1
    [[ "$arg" != --resume && "$arg" != --evaluate ]] || RESUMING=1
    if [[ "$arg" == --prepare-only ]]; then
        PREPARE_ONLY=1
    else
        TRAIN_ARGS+=("$arg")
    fi
done
if [[ -f "$OUTPUT_DIR/latest.pt" && "$RESUMING" == 0 && "$CHECK_ONLY" == 0 ]]; then
    printf 'ERROR: output contains latest.pt; supply --resume or VJEPA_OUTPUT_DIR.\n' >&2
    exit 1
fi

cd -- "$HEFT_DIR"
unset PYTHONHOME
export PYTHONPATH="$HEFT_DIR:$HEFT_DIR/../vjepa2"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline WANDB_ERROR_REPORTING=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
unset SLURM_LOCALID
export VJEPA_TMPDIR="${VJEPA_TMPDIR:-/dev/shm/heft-vjepa2-$UID/tmp}"
export TMPDIR="$VJEPA_TMPDIR"
mkdir -p -- "$TMPDIR"

if [[ "$CHECK_ONLY" == 1 ]]; then
    printf 'Checking shared runtime only; no staging, CUDA work or training.\n'
    "$SHARED_PYTHON" -m latent_video.classification.baselines.vjepa2.node_check \
        --config "$CONFIG" --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR" --cpu-only
    exit 0
fi

export PYTHON_BIN="$SHARED_PYTHON"
printf 'Using shared Python: %s\nShared weights: %s\n' "$PYTHON_BIN" "$MODEL_PATH"

printf 'Checking actual node imports, filesystem and CUDA execution...\n'
"$PYTHON_BIN" -m latent_video.classification.baselines.vjepa2.node_check \
    --config "$CONFIG" --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR"
if [[ "$PREPARE_ONLY" == 1 ]]; then
    printf 'Shared runtime checks passed; no training started.\n'
    exit 0
fi
export CONFIG NPROC_PER_NODE=8
printf 'Starting eight DDP ranks: two extraction lanes x 32 clips; 32 videos/rank.\n'
bash "$BASELINE_DIR/launch.sh" --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" --extract-lanes 2 "${TRAIN_ARGS[@]}"
