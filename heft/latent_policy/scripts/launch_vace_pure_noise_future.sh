#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
GPU=${GPU:-7}
OUTPUT=${OUTPUT:-"$ROOT/runs/vace_pure_noise_future_20261006"}
mkdir -p "$OUTPUT"
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TMPDIR=/tmp
export PYTHONPATH="$ROOT/heft/.venv/lib/python3.12/site-packages:$ROOT/heft/latent_policy/.venv/lib/python3.12/site-packages"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
if [[ -f "$OUTPUT/generate.pid" ]] && kill -0 "$(<"$OUTPUT/generate.pid")" 2>/dev/null; then
    echo "Generation is already running in $OUTPUT" >&2
    exit 1
fi
nohup setsid "$ROOT/heft/.venv/bin/python" "$ROOT/heft/latent_policy/scripts/evaluate_vace_pure_noise_future.py" \
    --output "$OUTPUT" "$@" > "$OUTPUT/generate.log" 2>&1 < /dev/null &
GENERATION_PID=$!
echo "$GENERATION_PID" > "$OUTPUT/generate.pid"
echo "pid=$GENERATION_PID gpu=$GPU output=$OUTPUT"
