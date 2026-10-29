#!/usr/bin/env bash
# Dev-machine smoke for the A2 / A3 switches on 8 NPUs (not a pool job).
#   smoke_train.sh a2_stop   : A2 overrides, stop_step=3, save_steps=2 -> periodic save @2, stop save @3
#   smoke_train.sh a2_resume : resume the a2_stop run with stop_step=3 -> must exit immediately
#   smoke_train.sh a3        : A3 overrides, 2 steps, no saving
set -euo pipefail
W=/home/ma-user/work/2337888765/workspace
T=$W/runs/structwam_a123_tests_20261027
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
common=(model.architecture.structure.video_detach=[structure] model.architecture.structure.losses.lambda_track=0)
cd $W/StructWAM
case "$1" in
  a2_stop)   RUN_DIR=$T/smoke_a2 bash scripts/structwam/train_libero_8npu.sh "${common[@]}" \
                 model.architecture.structure.action_video_dropout=0.3 training.stop_step=3 training.save_steps=2 ;;
  a2_resume) ckpt=$(dirname "$(ls -d $T/smoke_a2/checkpoints/*/accel_state_step_* | sort -V | tail -1)")
             RUN_DIR=$T/smoke_a2_resume bash scripts/structwam/train_libero_8npu.sh "${common[@]}" \
                 model.architecture.structure.action_video_dropout=0.3 training.stop_step=3 training.save_steps=2 \
                 training.resume_ckpt_path=$ckpt ;;
  a3)        RUN_DIR=$T/smoke_a3 bash scripts/structwam/train_libero_8npu.sh "${common[@]}" \
                 "model.architecture.structure_expert.groups=[map,keypoint,word]" training.max_steps=2 training.save_steps=0 ;;
esac
