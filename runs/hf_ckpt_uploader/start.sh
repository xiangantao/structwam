#!/usr/bin/env bash
# Start the checkpoint uploader in the background on this machine (run it on one machine only).
# Usage: start.sh [extra hf_ckpt_uploader.py args, e.g. --every 4000 or --dry-run --once]
set -euo pipefail
B=/mnt/sfs-turbo-wangfei/2337888765
W=$B/workspace
D=$W/runs/hf_ckpt_uploader
PY=$B/envs/openwam-npu/bin/python
export HF_HOME=$B/.hf_uploader HF_HUB_DISABLE_TELEMETRY=1
if pgrep -f "python.*hf_ckpt_uploader.py" >/dev/null; then
    echo "uploader already running (stop.sh stops it)"; exit 1
fi
ARGS=(--root "$W/runs/structwam_a123_20261027"
      --extra "orig_s40000=$W/runs/structwam_step40000_eval_ckpt"
      --extra "orig_s24000=$W/runs/structwam_step24000_eval_ckpt"
      --state "$D/state.json" --staging "$W/runs/_hf_upload_staging" "$@")
if [[ " $* " == *" --once "* ]]; then
    exec "$PY" "$D/hf_ckpt_uploader.py" "${ARGS[@]}"
fi
nohup setsid "$PY" "$D/hf_ckpt_uploader.py" "${ARGS[@]}" >> "$D/uploader.log" 2>&1 < /dev/null &
echo "uploader started (pid $!), log: $D/uploader.log"
