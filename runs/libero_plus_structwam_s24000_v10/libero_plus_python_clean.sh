#!/usr/bin/env bash
# LIBERO-plus client interpreter that strips the perturbation ids from the instruction
# (clean_prompt_entry.py); every other call (version probes, task enumeration) is plain python.
PV=/home/ma-user/work/2337888765/workspace/runs/libero_plus_structwam_s24000_v10
PY=/home/ma-user/work/2337888765/envs/libero-plus/bin/python
if [[ "${1:-}" == */single_eval.py ]]; then
    exec "$PY" "$PV/clean_prompt_entry.py" "$@"
fi
exec "$PY" "$@"
