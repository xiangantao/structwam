#!/usr/bin/env bash
# Policy-server interpreter for benchmarks/libero/scheduler.py on Ascend. The scheduler picks the
# device with CUDA_VISIBLE_DEVICES and calls "<this> scripts/deploy.py ..."; torch_npu reads
# ASCEND_RT_VISIBLE_DEVICES, and npu_eval_entry.py installs torch_npu's CUDA shim (plus the
# deploy-only PhaseGate fix) first.
R=/home/ma-user/work/2337888765
W=$R/workspace
EV=$W/runs/libero10_structwam_s24000_v10
export ASCEND_RT_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$W/StructWAM:$W/heft/latent_policy/src:$W/heft${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
if [[ "${1:-}" == *.py ]]; then
    exec "$R/envs/openwam-npu/bin/python" "$EV/npu_eval_entry.py" "$@"
fi
exec "$R/envs/openwam-npu/bin/python" "$@"
