#!/usr/bin/env bash
# Root-cause runs for the DP+A+C+J closed-loop gap (libero_spatial, 10k-step checkpoints):
#  1) open-loop replay of the 10 re-simulated held-out demos: dataset frames vs simulator renders
#     (diagnose_replay_sim_vs_dataset.py, 5 shards on NPUs 0-4);
#  2) closed-loop branch ablation of DP+A+C+J on tasks 1, 2, 5 (largest gaps): maps / tracks /
#     trunk zeroed in turn, 50 init states each, 8 shards per variant over NPUs 0-7.
W=/home/ma-user/work/2337888765/workspace
P=$W/heft/latent_policy
R=$W/runs
OUT=$R/libero_eval_20261018/diagnosis_20261018
MAPS=$R/dp_maps_20261018/libero_spatial/step_0010000.pt
RGB=$R/dp_compare_20261018/rgb/step_0010000.pt
STUDENT=$R/mainline_20261016/g2_full/step_0002000.pt
ROLES=$R/policy_maps_20261018/roles.json
mkdir -p "$OUT/replay/logs"
for s in 0 1 2 3 4; do
  SCRIPT=diagnose_replay_sim_vs_dataset.py LP_NUM_THREADS=4 nohup bash "$P/scripts/eval_libero_shard.sh" $s 20 \
    --episodes-root "$R/viz_inputs_20261015/runs/vace_tracking_all_suites_20261004/videos" --suite libero_spatial \
    --maps-checkpoint "$MAPS" --rgb-checkpoint "$RGB" --student "$STUDENT" --roles "$ROLES" \
    --dataset-root "$W/datasets/libero" --output "$OUT/replay" --shard $s --num-shards 5 \
    > "$OUT/replay/logs/shard_$s.log" 2>&1 < /dev/null &
  sleep 1
done
for branch in maps tracks trunk; do
  mkdir -p "$OUT/drop_$branch/logs"
  for s in 0 1 2 3 4 5 6 7; do
    LP_NUM_THREADS=3 nohup bash "$P/scripts/eval_libero_shard.sh" $s 30 --suite libero_spatial --kind maps \
      --checkpoint "$MAPS" --student "$STUDENT" --roles "$ROLES" --map-workers 2 --drop $branch \
      --tasks 1 2 5 --episodes 50 --save-videos 1 --shard $s --num-shards 8 --output "$OUT/drop_$branch" \
      > "$OUT/drop_$branch/logs/shard_$s.log" 2>&1 < /dev/null &
    sleep 1
  done
done
echo "launched -> $OUT"
