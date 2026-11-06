#!/usr/bin/env bash
# Policy-server interpreter for OpenWAM's LIBERO-plus scheduler on Ascend. The scheduler picks the
# device with CUDA_VISIBLE_DEVICES and calls "<this> scripts/deploy.py ..."; torch_npu reads
# ASCEND_RT_VISIBLE_DEVICES, and npu_entry.py installs torch_npu's CUDA shim first.
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"
export ASCEND_RT_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
if [[ "${1:-}" == *.py ]]; then
    exec "$OPENWAM_PY" "$STRUCTWAM_REPO/StructWAM/scripts/structwam/npu_entry.py" "$@"
fi
exec "$OPENWAM_PY" "$@"
