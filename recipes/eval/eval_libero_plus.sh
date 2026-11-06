#!/usr/bin/env bash
# Full LIBERO-plus evaluation of a StructWAM checkpoint, one suite per call, on one node of 8 NPUs.
#
#   bash recipes/eval/eval_libero_plus.sh SUITE CKPT_DIR [CKPT_NAME] [OUT_DIR]
#     SUITE      spatial | object | goal | long   (2402 / 2518 / 2591 / 2519 tasks)
#     CKPT_DIR   an eval checkpoint dir from recipes/prepare_eval_ckpt.py, with its caches from
#                recipes/eval/precompute_caches.sh
#     CKPT_NAME  default checkpoint_step_40000.safetensors
#     OUT_DIR    default $STRUCTWAM_ASSETS/eval/$(basename CKPT_DIR); results in OUT_DIR/SUITE/summary.csv
#
# Protocol (OpenWAM's LIBERO-plus runner, pinned and checked at launch): every task once, seed 10000,
# settle 30, max_steps 600 (libero_10: 700), clean instructions (perturbation ids stripped from the
# file-name-derived instruction; Language tasks unchanged), 10 action denoising steps, horizon 10,
# DiT cache off. Re-running the same command resumes; finished tasks are kept.
#
# Fast mode (on by default; why: CPU software rendering makes the NPU sit idle ~80% of each replan):
#   - Reason1 runs on the CPU (OPENWAM_TEXT_ENCODER_DEVICE=cpu) and every prompt comes from the
#     precomputed cache (OPENWAM_PROMPT_CACHE_FILE), so a server takes ~11.6 GB of NPU memory, not ~28;
#   - the structure student reads its UMT5 embeddings for unseen (Language) instructions from
#     OPENWAM_STUDENT_TEXT_EXTRA instead of loading the 11 GB UMT5 encoder onto the NPU;
#   - REPLICAS servers per NPU, each capped at 0.96 / REPLICAS of the card (OPENWAM_NPU_MEM_FRACTION).
#     Default 4 when MemAvailable leaves ~30 GB of host RAM per server + client, else 3, else the old
#     mode (Reason1 on the NPU, 2 per NPU). 4 per NPU measured ~3.4x the 2-per-NPU old mode.
# Overrides: REPLICAS, OPENWAM_TEXT_ENCODER_DEVICE=npu (old mode), GPUS (default 0..7),
# SAMPLE_RATIO=0.1 (a seeded 10% of the suite, for a quick check), MASTER_BASE_PORT (default 9000).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"
EVAL_DIR="$STRUCTWAM_REPO/recipes/eval"

SUITE="${1:?usage: eval_libero_plus.sh SUITE CKPT_DIR [CKPT_NAME] [OUT_DIR]}"
CKPT_DIR="$(cd "${2:?CKPT_DIR}" && pwd)"
CKPT_NAME="${3:-checkpoint_step_40000.safetensors}"
OUT_DIR="${4:-$STRUCTWAM_ASSETS/eval/$(basename "$CKPT_DIR")}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
n_gpus=$(awk -F, '{print NF}' <<< "$GPUS")

for f in "$CKPT_DIR/$CKPT_NAME" "$CKPT_DIR/config.yaml" "$CKPT_DIR/normalization_stats.npy"; do
    [[ -e "$f" ]] || { echo "Missing checkpoint file: $f" >&2; exit 1; }
done
for exe in "$OPENWAM_PY" "$LIBERO_PY"; do
    [[ -x "$exe" ]] || { echo "Missing interpreter: $exe (see recipes/paths.sh)" >&2; exit 1; }
done
[[ -d "$LIBERO_PLUS_REPO/libero" ]] || { echo "Missing LIBERO-plus checkout: $LIBERO_PLUS_REPO" >&2; exit 1; }

if [[ -f "$ASCEND_SET_ENV" ]]; then
    set +eu +o pipefail  # CANN's set_env.sh is not safe under -e / pipefail
    source "$ASCEND_SET_ENV"
    set -euo pipefail
fi
CLIENT_PY="$EVAL_DIR/client_python.sh"
"$CLIENT_PY" -c "import wand.image" || { echo "ImageMagick (python wand) failed to load in the LIBERO-plus env" >&2; exit 1; }
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$CLIENT_PY" -c \
    'import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); c.free(); print("EGL ok")' \
    || { echo "MuJoCo EGL rendering failed in the LIBERO-plus env" >&2; exit 1; }

# ---- fast mode ----
avail_gb=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
te_dev="${OPENWAM_TEXT_ENCODER_DEVICE:-cpu}"
if [[ -z "${REPLICAS:-}" ]]; then
    REPLICAS=0
    if [[ "$te_dev" == cpu ]]; then
        for r in 4 3; do
            (( avail_gb - 100 >= n_gpus * r * 30 )) && { REPLICAS=$r; break; }
        done
    fi
    # At 2 per NPU the old mode is as fast and needs far less host RAM.
    (( REPLICAS == 0 )) && { te_dev=npu; REPLICAS=2; }
fi
if [[ "$te_dev" == cpu ]]; then export OPENWAM_TEXT_ENCODER_DEVICE=cpu; else unset OPENWAM_TEXT_ENCODER_DEVICE; fi
export OPENWAM_PROMPT_CACHE_FILE="${OPENWAM_PROMPT_CACHE_FILE:-$CKPT_DIR/prompt_cache.pt}"
export OPENWAM_STUDENT_TEXT_EXTRA="${OPENWAM_STUDENT_TEXT_EXTRA:-$CKPT_DIR/student_text_extra.pt}"
for f in "$OPENWAM_PROMPT_CACHE_FILE" "$OPENWAM_STUDENT_TEXT_EXTRA"; do
    [[ -f "$f" ]] || { echo "Missing $f: run recipes/eval/precompute_caches.sh $CKPT_DIR first" >&2; exit 1; }
done
export OPENWAM_NPU_MEM_FRACTION="${OPENWAM_NPU_MEM_FRACTION:-$(awk -v r="$REPLICAS" 'BEGIN {printf "%.3f", 0.96 / r}')}"
echo "[fast-mode] MemAvailable=${avail_gb} GB text_encoder=${te_dev} replicas_per_npu=${REPLICAS}" \
     "mem_fraction=${OPENWAM_NPU_MEM_FRACTION} gpus=${GPUS}"
[[ "${FAST_MODE_DRY_RUN:-0}" == 1 ]] && exit 0

sample=()
[[ -n "${SAMPLE_RATIO:-}" ]] && sample=(--task-sample-ratio "$SAMPLE_RATIO" --task-sample-seed 42)
mkdir -p "$OUT_DIR"
cd "$STRUCTWAM_REPO/StructWAM"
exec "$OPENWAM_PY" benchmarks/libero-plus/run_all_suites.py \
    --ckpt-dir "$CKPT_DIR" --ckpt-name "$CKPT_NAME" \
    --server-python "$EVAL_DIR/server_python.sh" \
    --libero-python "$CLIENT_PY" --libero-path "$LIBERO_PLUS_REPO" \
    --policy-config benchmarks/libero-plus/policy_config.yml \
    --suites "$SUITE" --render-gpus 0 --denoise-steps 10 --inference-horizon 10 \
    --compile-enabled false --extra-deploy-arg optimization.dit_cache.enabled=false \
    --client-max-attempts 6 "${sample[@]}" \
    --gpus "$GPUS" --replicas-per-gpu "$REPLICAS" --base-port "${MASTER_BASE_PORT:-9000}" \
    --output-dir "$OUT_DIR/$SUITE"
