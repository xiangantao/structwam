#!/usr/bin/env bash
# Closed-loop LIBERO evaluation of both 10k-step libero_spatial policies on 8 NPUs, in parallel:
# DP + maps/tracks/trunk (runs/dp_maps_20261018) and vanilla DP (runs/dp_compare_20261018/rgb).
# 10 tasks x 50 fixed init states per model; SHARDS_PER_NPU processes per model per NPU.
# Usage: bash launch_libero_eval.sh            (defaults: SHARDS_PER_NPU=3, LP_NUM_THREADS=4)
#        SHARDS_PER_NPU=4 LP_NUM_THREADS=3 bash launch_libero_eval.sh
#        OUT=<dir> EXTRA="--compress av1:35" bash launch_libero_eval.sh   (extra eval args for both models)
#        OUT=<dir> MAPS_CKPT=<pt> RGB_CKPT=<pt> bash launch_libero_eval.sh    (other checkpoints)
set -u
W=/home/ma-user/work/2337888765/workspace
P=$W/heft/latent_policy
R=$W/runs
OUT=${OUT:-$R/libero_eval_20261018/full}
NPUS=${NPUS:-8}
PER=${SHARDS_PER_NPU:-3}
SHARDS=$((NPUS * PER))
SUITE=libero_spatial
COMMON=(--suite $SUITE --episodes 50 --save-videos 2 --sampler-steps 10 --num-shards $SHARDS ${EXTRA:-})
declare -A ARGS
ARGS[maps]="--kind maps --checkpoint ${MAPS_CKPT:-$R/dp_maps_20261018/libero_spatial/step_0010000.pt} --student $R/mainline_20261016/g2_full/step_0002000.pt --roles $R/policy_maps_20261018/roles.json --map-workers 2"
ARGS[rgb]="--kind rgb --checkpoint ${RGB_CKPT:-$R/dp_compare_20261018/rgb/step_0010000.pt}"
for model in maps rgb; do
  mkdir -p "$OUT/$model/logs"
  for ((s = 0; s < SHARDS; s++)); do
    npu=$((s % NPUS))
    nohup bash "$P/scripts/eval_libero_shard.sh" $npu 30 ${ARGS[$model]} "${COMMON[@]}" --shard $s --output "$OUT/$model" \
      > "$OUT/$model/logs/shard_$s.log" 2>&1 < /dev/null &
    sleep 1
  done
done
echo "launched $SHARDS shards per model (2 models, $NPUS NPUs, LP_NUM_THREADS=${LP_NUM_THREADS:-4}) -> $OUT"
echo "progress: python $P/scripts/summarize_libero_eval.py $OUT"
