#!/usr/bin/env bash
# One LIBERO evaluation shard on one NPU, restarted until it finishes (Mesa's software EGL
# segfaults on roughly 1 in 5 process starts; eval_libero_closed_loop.py resumes from its jsonl).
# Usage: eval_libero_shard.sh <npu> <max_attempts> <eval_libero_closed_loop.py args...>
# Env overrides: LP_NUM_THREADS (llvmpipe render threads), OMP_NUM_THREADS, SCRIPT (another
# script in latent_policy/scripts with the same restart-and-resume behaviour).
W=/home/ma-user/work/2337888765/workspace
P=$W/heft/latent_policy
source /home/ma-user/anaconda3/bin/activate vace-lerobot
cd "$W/heft"
source "$P/scripts/env.sh" >/dev/null 2>&1
# env.sh puts the LeRobot 0.6.2 checkout (Python >= 3.12) on the path; use the installed 0.4.4.
export PYTHONPATH="$PWD/diffusers/src:$(echo "$PYTHONPATH" | tr ":" "\n" | grep -v "repos/lerobot" | paste -sd:)"
export HF_HUB_OFFLINE=1 LIBERO_CONFIG_PATH=$HOME/.libero MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LP_NUM_THREADS=${LP_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export VACE_NPU=1 ASCEND_RT_VISIBLE_DEVICES=$1 PYTHONUNBUFFERED=1 SVT_LOG=1   # SVT-AV1: errors only
attempts=$2; shift 2
cd /tmp
for a in $(seq 1 "$attempts"); do
  python "$P/scripts/${SCRIPT:-eval_libero_closed_loop.py}" "$@" < /dev/null
  status=$?
  echo "[eval_libero_shard] attempt $a exit $status at $(date)"
  [ $status -eq 0 ] && exit 0
done
exit 1
