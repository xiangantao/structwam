#!/usr/bin/env bash
# 8-NPU VACE structured distillation, online teacher (no offline cache).
# batch 8/card (global 64), logs every step, speed-ups on (VACE_TEACHER_OPT=1, VACE_VAE_OPT=1).
# Usage: bash run_vace_npu_8card.sh [extra train args]   (default here: --steps 1000)
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
W="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
OUT="${OUT:-$W/runs/vace_structured_npu_1000}"
LOG="$OUT/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT"
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_LAUNCH_BLOCKING
export VACE_TEACHER_OPT=1 VACE_VAE_OPT=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
EXTRA=("$@"); [[ " ${EXTRA[*]} " == *" --steps "* ]] || EXTRA+=(--steps 1000)
nohup "$SCRIPT_DIR/launch_vace_npu.sh" 8 \
  --batch-size 8 --teacher-batch-size 4 --workers 2 --output "$OUT" "${EXTRA[@]}" \
  > "$LOG" 2>&1 &
echo $! > "$OUT/train.pid"
echo "started pid $(cat "$OUT/train.pid")  (1000 steps, global batch 64)"
echo "log:    $LOG"
echo "stop:   kill \$(cat $OUT/train.pid)"
echo "follow: tail -f $LOG | grep -a --line-buffered '^{\"step\"'"
echo "curve:  python3 $SCRIPT_DIR/show_curve.py $OUT"
