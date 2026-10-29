#!/usr/bin/env bash
# ModelArts training-pool job for the StructWAM A1 / A2 / A3 runs: 1 node x 8 NPUs.
# Mount SFS Turbo `sfs-turbo-wangfei` (root /) at /mnt/sfs-turbo-wangfei, the same path as on the
# notebooks (override with SFS_PERSONAL=<mount>/2337888765).  Runs in the foreground.
#   bash train_arm_modelarts.sh a1|a2|a3|b2|b0|c0
#
#   a1  video_detach=[structure], lambda_track=0: the structure expert is trained by the action and
#       phase losses only (video losses no longer reach it; the track-consistency loss is off)
#   a2  a1 + action_video_dropout=0.3: for 30% of the samples the action rows and the structure rows
#       see no video keys, so the action has to come from the structure stream
#   a3  a1 + structure tokens without the V-JEPA2 trunk group (map, keypoint, word only)
#   b2  a2 + consistent image augmentation (openwam/structure/augment.py): head camera zoom / shift /
#       small rotation / occasional blur, one color change for both cameras, applied identically to the
#       video, the first-frame condition and the student's history; 4 loader workers per rank
#   b0  the original s40000 StructWAM recipe (video_detach=[], track consistency on) + the a2 video
#       dropout (0.3) + the b2 image augmentation; 4 loader workers per rank
#   c0  b0 with the cameras side by side (head | wrist, 256x256 each, canvas 256x512, instead of the
#       L-shape where the wrist gets a 128x160 tile) and the augmentation aligned with pi0.5 /
#       Cosmos-Policy: head rotate +-5 deg, zoom <= 1.1, no blur; brightness 0.3, contrast 0.4,
#       saturation 0.5, hue 0.05, drawn independently per camera; 4 loader workers per rank
#
# Everything else is the s40000 recipe (configs/train_struct.yaml): the full 40k steps, cosine LR, so
# the result compares directly with the s40000 StructWAM and OpenWAM-Cosmos checkpoints.
# Resubmitting the same arm resumes from its latest saved state (saved every 2000 steps).
# Optional STOP_STEP=N ends the job at step N with a resumable state (unset by default).
#
# Output: runs/structwam_a123_20261027/<arm>/{checkpoints/<time>/, train.log, provenance_*}
set -euo pipefail
ARM="${1:?usage: train_arm_modelarts.sh a1|a2|a3|b2|b0|c0}"
SFS_PERSONAL="${SFS_PERSONAL:-/mnt/sfs-turbo-wangfei/2337888765}"
LEGACY_PERSONAL=/home/ma-user/work/2337888765
[[ -d "$SFS_PERSONAL/workspace/StructWAM" ]] || {
    echo "Project missing: $SFS_PERSONAL/workspace/StructWAM" >&2
    echo "Mount SFS Turbo sfs-turbo-wangfei (root /) at /mnt/sfs-turbo-wangfei, or set SFS_PERSONAL." >&2
    exit 1
}
# The 4.8T wangfei-sfs-1 holds an older copy of 2337888765 without sysdeps/; refuse to run on it.
[[ -f "$SFS_PERSONAL/sysdeps/hce2_aarch64/env.sh" ]] || {
    echo "$SFS_PERSONAL is not the sfs-turbo-wangfei (30T) copy (no sysdeps/); check the SFS mount." >&2
    exit 1
}
if [[ -e "$LEGACY_PERSONAL" || -L "$LEGACY_PERSONAL" ]]; then
    [[ "$(readlink -f "$LEGACY_PERSONAL")" == "$(readlink -f "$SFS_PERSONAL")" ]] || {
        echo "Existing $LEGACY_PERSONAL is not the mounted personal directory; refusing to overwrite." >&2
        exit 1
    }
else
    mkdir -p "$(dirname "$LEGACY_PERSONAL")"
    ln -s "$SFS_PERSONAL" "$LEGACY_PERSONAL"
fi
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi
R=$LEGACY_PERSONAL
W=$R/workspace
[[ -x "$R/envs/openwam-npu/bin/python" ]] || { echo "Shared openwam-npu environment missing" >&2; exit 1; }

common=(model.architecture.structure.video_detach=[structure]
        model.architecture.structure.losses.lambda_track=0)
case "$ARM" in
    a1) extra=() ;;
    a2) extra=(model.architecture.structure.action_video_dropout=0.3) ;;
    a3) extra=("model.architecture.structure_expert.groups=[map,keypoint,word]") ;;
    b2) extra=(model.architecture.structure.action_video_dropout=0.3
               dataloader.structure_augment.enabled=true training.dataset_num_workers=4) ;;
    b0) common=()
        extra=(model.architecture.structure.action_video_dropout=0.3
               dataloader.structure_augment.enabled=true training.dataset_num_workers=4) ;;
    c0) common=()
        extra=(model.architecture.structure.action_video_dropout=0.3
               model.architecture.structure.canvas_layout=side_by_side
               dataloader.canvas_layout=side_by_side dataloader.height=256 dataloader.width=512
               dataloader.structure_augment.enabled=true dataloader.structure_augment.rotate_deg=5.0
               dataloader.structure_augment.scale_max=1.1 dataloader.structure_augment.blur_prob=0.0
               dataloader.structure_augment.brightness=0.3 dataloader.structure_augment.contrast=0.4
               dataloader.structure_augment.saturation=0.5 dataloader.structure_augment.hue=0.05
               dataloader.structure_augment.independent_color=true training.dataset_num_workers=4) ;;
    *) echo "unknown arm: $ARM (a1|a2|a3|b2|b0|c0)" >&2; exit 2 ;;
esac
STOP_STEP="${STOP_STEP-}"
stop=()
[[ -n "$STOP_STEP" ]] && stop=("training.stop_step=$STOP_STEP")

RUN_DIR=$W/runs/structwam_a123_20261027/$ARM
mkdir -p "$RUN_DIR"
stamp=$(date +%Y%m%d_%H%M%S)
{
    echo "arm=$ARM stop_step=${STOP_STEP:-none} host=$(hostname) date=$(date)"
    git -C "$W/StructWAM" rev-parse HEAD
    git -C "$W/StructWAM" status --short
} > "$RUN_DIR/provenance_$stamp.txt"
git -C "$W/StructWAM" diff > "$RUN_DIR/provenance_$stamp.diff"

# Resume from the latest resumable state of this arm, if any.
resume=()
latest=$(ls -d "$RUN_DIR"/checkpoints/*/accel_state_step_* 2>/dev/null | sort -V | tail -1 || true)
if [[ -n "$latest" ]]; then
    resume=("training.resume_ckpt_path=$(dirname "$latest")")
    echo "Resuming from $latest"
fi
[[ -f "$RUN_DIR/train.log" ]] && mv "$RUN_DIR/train.log" "$RUN_DIR/train.log.$stamp"

cd "$W/StructWAM"
RUN_DIR="$RUN_DIR" exec bash scripts/structwam/train_libero_8npu.sh \
    "${common[@]}" "${extra[@]}" "${stop[@]}" "${resume[@]}"
