#!/usr/bin/env bash
# Sealed reset-only LIBERO-10 manifests (benchmarks/libero/eval_manifest.py), one per task, in
# parallel; retried because Mesa's software EGL segfaults on some process starts.
#   bash build_manifests.sh [TOTAL_TRIALS]
set -uo pipefail
R=/home/ma-user/work/2337888765
W=$R/workspace
EV=$W/runs/libero10_structwam_s24000_v10
B=$W/StructWAM/benchmarks/libero
S=$R/envs/vace-lerobot/lib/python3.11/site-packages
total="${1:-50}"
mkdir -p "$EV/manifests"
export PYTHONPATH="$W/StructWAM:$S:$B" LIBERO_PATH="$S" LIBERO_CONFIG_PATH="$EV/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
build() {
    local task=$1 out="$EV/manifests/libero_10_task$(printf %02d "$1").json"
    for attempt in 1 2 3 4 5; do
        [[ -s "$out" ]] && return 0
        "$EV/libero_python.sh" "$B/eval_manifest.py" --config "$B/policy_config.yml" --suite libero_10 \
            --task-id "$task" --total-trials "$total" --result-file "$out" \
            > "$EV/manifests/task$(printf %02d "$task")_attempt$attempt.log" 2>&1 < /dev/null && return 0
    done
    return 1
}
cd /tmp
for task in 0 1 2 3 4 5 6 7 8 9; do build "$task" & done
wait
ls -la "$EV/manifests"/*.json
