#!/usr/bin/env bash
# LIBERO client interpreter: vace-lerobot (hf_libero 0.1.4, robosuite 1.4.0, mujoco 3.8.1) plus
# websockets from openwam-npu (same Python 3.11 / aarch64). Mesa software EGL has one device.
EV=/home/ma-user/work/2337888765/workspace/runs/libero10_structwam_s24000_v10
export PYTHONPATH="$EV/pydeps${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_EGL_DEVICE_ID=0 LP_NUM_THREADS="${LP_NUM_THREADS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
exec /home/ma-user/work/2337888765/envs/vace-lerobot/bin/python "$@"
