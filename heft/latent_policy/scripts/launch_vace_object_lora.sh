#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
GPU=${GPU:-7}
BATCH_SIZE=${BATCH_SIZE:-4}
OUTPUT=${OUTPUT:-"$ROOT/runs/vace_object_lora_20261006"}
mkdir -p "$OUTPUT"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR=/tmp
export PYTHONPATH="$ROOT/heft/latent_policy/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
if [[ -f "$OUTPUT/train.pid" ]] && kill -0 "$(<"$OUTPUT/train.pid")" 2>/dev/null; then
    echo "An experiment is already running in $OUTPUT" >&2
    exit 1
fi
nohup setsid "$ROOT/heft/.venv/bin/python" "$ROOT/heft/latent_policy/scripts/train_vace_object_lora.py" \
    --output "$OUTPUT" --batch-size "$BATCH_SIZE" --steps 6000 --lr 0.0001 \
    "$@" > "$OUTPUT/train.log" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$OUTPUT/train.pid"
echo "pid=$TRAIN_PID gpu=$GPU real_batch=$BATCH_SIZE output=$OUTPUT"
