#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/heft/latent_policy/.venv/bin/python"
SCRIPTS="${ROOT}/heft/latent_policy/scripts"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/vace_tracking_all_suites_20261004}"
GPU="${GPU:-4}"
STEPS="${STEPS:-10000}"
mkdir -p "${RUN_ROOT}"

export LIBERO_CONFIG_PATH="${ROOT}/heft/latent_policy/.runtime/libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${GPU}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/tmp
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# An already-running collector can be reused while annotations are generated.
if [[ -n "${COLLECTOR_PID:-}" ]]; then
    while kill -0 "${COLLECTOR_PID}" 2>/dev/null && [[ "$(ps -p "${COLLECTOR_PID}" -o stat=)" != Z* ]]; do
        if [[ -f "${RUN_ROOT}/videos/manifest.json" ]]; then
            "${PY}" -u "${SCRIPTS}/render_libero_surface_tracks.py" \
                --input "${RUN_ROOT}/videos" --output "${RUN_ROOT}/tracks" \
                --max-episodes 160 --include-objects --annotations-only \
                --pixel-stride 6 --seed-stride 16 --frame-stride 4 --resume \
                >> "${RUN_ROOT}/annotate.log" 2>&1
        fi
        sleep 5
    done
fi

"${PY}" -u "${SCRIPTS}/collect_libero_long_videos.py" \
    --dataset-root "${ROOT}/datasets/libero" --output "${RUN_ROOT}/videos" \
    --episodes-per-task 4 --max-steps 320 --seed 20261004 --resume \
    >> "${RUN_ROOT}/collect.log" 2>&1

"${PY}" -u "${SCRIPTS}/render_libero_surface_tracks.py" \
    --input "${RUN_ROOT}/videos" --output "${RUN_ROOT}/tracks" \
    --max-episodes 160 --include-objects --annotations-only \
    --pixel-stride 6 --seed-stride 16 --frame-stride 4 --resume \
    >> "${RUN_ROOT}/annotate.log" 2>&1

"${PY}" -u "${SCRIPTS}/build_libero_tracking_split.py" \
    --videos "${RUN_ROOT}/videos" --tracks "${RUN_ROOT}/tracks" \
    --output "${RUN_ROOT}/split.json"

export CUDA_VISIBLE_DEVICES="${GPU}"
exec "${PY}" -u "${SCRIPTS}/train_vace_tracking_lora.py" \
    --output "${RUN_ROOT}/training" \
    --video-root "${RUN_ROOT}/videos" --tracks-root "${RUN_ROOT}/tracks" \
    --split-manifest "${RUN_ROOT}/split.json" --windows-per-episode 2 \
    --resume-lora "${ROOT}/runs/vace_tracking_lora_10x_single_2000_val2_20261004/tracking_lora.pt" \
    --candidates '6:1,10:6,7:11,14:8' --size 512 --max-frames 129 \
    --skip-frames 8 --points-per-subject 24 --min-motion 0.02 \
    --rank 8 --alpha 16 --steps "${STEPS}" --lr 0.0001 --save-every 500 \
    --rollout-weight 1 --separation-weight 0.25 --separation-margin 1 \
    --fresh-noise --balance-groups --lazy-controls --skip-initial-eval \
    > "${RUN_ROOT}/train.log" 2>&1
