#!/usr/bin/env bash
# Wait until the running LIBERO evaluation (eval_libero_shard.sh processes) has finished, then
# evaluate both policies again with the policy input frames AV1-compressed like the training
# videos (crf 35, calibrated on edge sharpness), into runs/libero_eval_20261018/full_av1.
P=/home/ma-user/work/2337888765/workspace/heft/latent_policy
R=/home/ma-user/work/2337888765/workspace/runs
while ps -eo args | grep -q "[e]val_libero_shard.sh"; do sleep 60; done
echo "previous evaluation finished at $(date); starting the AV1-compressed evaluation"
OUT=$R/libero_eval_20261018/full_av1 EXTRA="--compress av1:35" SHARDS_PER_NPU=${SHARDS_PER_NPU:-4} LP_NUM_THREADS=${LP_NUM_THREADS:-3} \
  bash $P/scripts/launch_libero_eval.sh
