#!/usr/bin/env bash
# Full-instruction attn2 head scan: 30 main + 15 VACE blocks x 12 heads, 8 NPUs, then GT scoring.
# Overrides: OUT, BUILD_ARGS (e.g. "--starts 8 --swap-tracking-root ..."), SCAN_ARGS (e.g. "--main-only").
set -euo pipefail
W=/home/ma-user/work/2337888765/workspace
P=$W/heft/latent_policy
OUT=${OUT:-$W/runs/prompt_head_scan_20261019}
TRACK=$W/runs/viz_inputs_20261015/runs/vace_tracking_all_suites_20261004
CACHE=$W/runs/vace_teacher_cache_small_20261015
PY=/home/ma-user/anaconda3/envs/vace-npu/bin/python
cd "$W/heft"
source "$P/scripts/env.sh" >/dev/null 2>&1
export PYTHONPATH="$PWD/diffusers/src:$PYTHONPATH" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"
"$PY" "$P/scripts/scan_vace_full_prompt_heads.py" --output "$OUT" --build-windows "$CACHE/manifest_heldout.jsonl" ${BUILD_ARGS:-}
export VACE_NPU=1 VACE_CFG_PARALLEL=1
for rank in 0 1 2 3 4 5 6 7; do
  "$PY" -u "$P/scripts/scan_vace_full_prompt_heads.py" --output "$OUT" --config "$CACHE/pipeline.json" \
    --rank $rank --world 8 --device cuda:$rank ${SCAN_ARGS:-} > "$OUT/scan_rank$rank.log" 2>&1 &
done
wait
expected=$(wc -l < "$OUT/windows.jsonl")
found=$(ls "$OUT/maps" | grep -c -v tmp || true)
if [ "$found" -ne "$expected" ]; then
  echo "scan incomplete: $found / $expected" > "$OUT/SCAN_FAILED"; exit 1
fi
if "$PY" "$P/scripts/report_vace_full_prompt_heads.py" --scan "$OUT" --tracking-root "$TRACK" > "$OUT/report.log" 2>&1; then
  touch "$OUT/DONE"
else
  touch "$OUT/REPORT_FAILED"
fi
echo "finished at $(date)"
