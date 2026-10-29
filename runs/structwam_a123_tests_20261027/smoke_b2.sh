#!/usr/bin/env bash
# Dev-machine 8-NPU smoke for b2 (no saving): same overrides as train_arm_modelarts.sh b2, 8 steps.
set -euo pipefail
W=/home/ma-user/work/2337888765/workspace
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
cd $W/StructWAM
RUN_DIR=$W/runs/structwam_a123_tests_20261027/smoke_b2 bash scripts/structwam/train_libero_8npu.sh   model.architecture.structure.video_detach=[structure] model.architecture.structure.losses.lambda_track=0   model.architecture.structure.action_video_dropout=0.3 dataloader.structure_augment.enabled=true   training.dataset_num_workers=4 training.max_steps=8 training.save_steps=0
