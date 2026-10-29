#!/usr/bin/env bash
# ModelArts training-pool job: any OpenWAM / StructWAM checkpoint on LIBERO-plus, same protocol as
# runs/libero_plus_structwam_s24000_v10/sample10_clean: 10% of every suite (task-sample-seed 42),
# clean instructions, 1 trial per task (seed 10000, settle 30), 10 video + 10 action denoising
# steps, inference horizon 10, DiT cache off, one policy server per NPU, 8 NPUs.
# Mount SFS Turbo `sfs-turbo-wangfei` (root /) at /mnt/sfs-turbo-wangfei, the same path as on the
# notebooks (override with SFS_PERSONAL=<mount>/2337888765).  Runs in the foreground; resubmitting the same NAME
# resumes into the same output directory.
#   bash eval_libero_plus_modelarts.sh NAME CKPT_DIR CKPT_NAME
#   e.g. bash eval_libero_plus_modelarts.sh a2_s24000 \
#          /home/ma-user/work/2337888765/workspace/runs/structwam_a123_20261027/a2/checkpoints/<time> \
#          checkpoint_step_24000.safetensors
# CKPT_DIR is a training checkpoint directory (config.yaml, normalization_stats.npy, reason1/, weights).
# Results: runs/libero_plus_sample10_clean/NAME/summary.csv
#
# The ImageMagick / Mesa-EGL libraries stock images lack come from the shared sysdeps/hce2_aarch64,
# loaded into the LIBERO client processes only (stage0_libero_plus_20261027/libero_plus_python_sysdeps.sh).
set -euo pipefail
NAME="${1:?usage: eval_libero_plus_modelarts.sh NAME CKPT_DIR CKPT_NAME}"
CKPT_DIR="${2:?CKPT_DIR}"
CKPT_NAME="${3:?CKPT_NAME}"
SFS_PERSONAL="${SFS_PERSONAL:-/mnt/sfs-turbo-wangfei/2337888765}"
LEGACY_PERSONAL=/home/ma-user/work/2337888765
[[ -d "$SFS_PERSONAL/workspace/StructWAM" ]] || {
    echo "Project missing: $SFS_PERSONAL/workspace/StructWAM" >&2
    echo "Mount SFS Turbo sfs-turbo-wangfei (root /) at /mnt/sfs-turbo-wangfei, or set SFS_PERSONAL." >&2
    exit 1
}
# The 4.8T wangfei-sfs-1 holds an older copy of 2337888765 without sysdeps/; refuse to run on it.
[[ -f "$SFS_PERSONAL/sysdeps/hce2_aarch64/env.sh" ]] || {
    echo "$SFS_PERSONAL is not the sfs-turbo-wangfei (30T) copy (no sysdeps/); check the SFS mount." >&2
    exit 1
}
if [[ -e "$LEGACY_PERSONAL" || -L "$LEGACY_PERSONAL" ]]; then
    [[ "$(readlink -f "$LEGACY_PERSONAL")" == "$(readlink -f "$SFS_PERSONAL")" ]] || {
        echo "Existing $LEGACY_PERSONAL is not the mounted personal directory; refusing to overwrite." >&2
        exit 1
    }
else
    mkdir -p "$(dirname "$LEGACY_PERSONAL")"
    ln -s "$SFS_PERSONAL" "$LEGACY_PERSONAL"
fi
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi
R=$LEGACY_PERSONAL
W=$R/workspace
D=$W/runs/libero_plus_sample10_clean
CLIENT_PY=$W/runs/stage0_libero_plus_20261027/libero_plus_python_sysdeps.sh
for exe in "$R/envs/openwam-npu/bin/python" "$R/envs/libero-plus/bin/python" "$CLIENT_PY"; do
    [[ -x "$exe" ]] || { echo "Missing on the shared disk: $exe" >&2; exit 1; }
done
for f in "$CKPT_DIR/$CKPT_NAME" "$CKPT_DIR/config.yaml" "$CKPT_DIR/normalization_stats.npy"; do
    [[ -e "$f" ]] || { echo "Missing checkpoint file: $f" >&2; exit 1; }
done

"$CLIENT_PY" -c "import wand.image" || {
    echo "ImageMagick (python wand) failed to load with $R/sysdeps/hce2_aarch64." >&2
    exit 1
}
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$CLIENT_PY" -c \
    'import mujoco; c = mujoco.GLContext(64, 64); c.make_current(); c.free(); print("EGL ok")' || {
    echo "MuJoCo EGL rendering failed with $R/sysdeps/hce2_aarch64." >&2
    exit 1
}

mkdir -p "$D"
cd "$W/StructWAM"
exec "$R/envs/openwam-npu/bin/python" "$W/StructWAM/benchmarks/libero-plus/run_all_suites.py" \
    --ckpt-dir "$CKPT_DIR" --ckpt-name "$CKPT_NAME" \
    --server-python "$W/runs/libero10_structwam_s24000_v10/server_python.sh" \
    --libero-python "$CLIENT_PY" --libero-path "$W/repos/LIBERO-plus" \
    --policy-config "$W/StructWAM/benchmarks/libero-plus/policy_config.yml" \
    --suites spatial,object,goal,long --render-gpus 0 --denoise-steps 10 --inference-horizon 10 \
    --compile-enabled false --extra-deploy-arg optimization.dit_cache.enabled=false \
    --client-max-attempts 6 --task-sample-ratio 0.1 --task-sample-seed 42 \
    --gpus 0,1,2,3,4,5,6,7 --replicas-per-gpu 1 --base-port 9000 \
    --output-dir "$D/$NAME"
