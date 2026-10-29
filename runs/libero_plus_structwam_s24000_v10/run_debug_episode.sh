#!/usr/bin/env bash
# One debug LIBERO-plus episode against the debug server (NPU 7, port 9150).
#   bash run_debug_episode.sh NAME SUITE TASK_ID [PROMPT]
set -euo pipefail
R=/home/ma-user/work/2337888765
W=$R/workspace
PV=$W/runs/libero_plus_structwam_s24000_v10
P=$W/repos/LIBERO-plus
B=$W/StructWAM/benchmarks/libero-plus
name="${1:?name}" suite="${2:?suite}" task="${3:?task id}" prompt="${4:-}"
out=$PV/debug/$name
mkdir -p "$out/libero_config"
export PYTHONPATH="$W/StructWAM:$P:$B" LIBERO_PLUS_PATH="$P" LIBERO_PLUS_CONFIG_ROOT="$out/libero_config"
export LIBERO_CONFIG_PATH="$out/libero_config" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
args=(--suite "$suite" --task-id "$task" --port 9150 --out "$out")
[[ -n "$prompt" ]] && args+=(--prompt "$prompt")
cd /tmp
for attempt in 1 2 3 4; do   # Mesa software EGL segfaults on some starts
    "$R/envs/libero-plus/bin/python" "$PV/debug_episode.py" "${args[@]}" > "$out/client_attempt$attempt.log" 2>&1 < /dev/null \
        && break
    echo "attempt $attempt exit $?"
done
grep -h "\[RESULT\]" "$out"/client_attempt*.log
