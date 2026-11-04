#!/usr/bin/env bash
# ModelArts training-pool job: StructWAM inference timing on one NPU, video steps 10 / 1 / 0 with and
# without the video K/V cache (structure.inference.video_kv_cache), on real LIBERO training samples,
# plus how far each mode's actions are from the uncached ones (bench_kv_cache.py).  Takes ~10 min.
# Mount SFS Turbo `sfs-turbo-wangfei` (root /) at /mnt/sfs-turbo-wangfei, the same path as on the
# notebooks (override with SFS_PERSONAL=<mount>/2337888765).  Runs in the foreground.
#   bash bench_kv_cache_modelarts.sh [CKPT_DIR] [CKPT_NAME] [NPU]
# Defaults: the b2 s40000 eval checkpoint (its 10/10 config; the script switches modes itself), NPU 0.
# Results: runs/structwam_kv_cache_20261104/bench_<ckpt dir name>.{json,log}
set -euo pipefail
CKPT_DIR="${1:-/home/ma-user/work/2337888765/workspace/runs/structwam_a123_s40000_eval_ckpt/b2}"
CKPT_NAME="${2:-checkpoint_step_40000.safetensors}"
NPU="${3:-0}"
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
D=$W/runs/structwam_kv_cache_20261104
for f in "$CKPT_DIR/$CKPT_NAME" "$CKPT_DIR/config.yaml" "$CKPT_DIR/normalization_stats.npy"; do
    [[ -e "$f" ]] || { echo "Missing checkpoint file: $f" >&2; exit 1; }
done
TAG=$(basename "$CKPT_DIR")
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export PYTHONPATH="$W/StructWAM:$W/heft/latent_policy/src:$W/heft${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$W/StructWAM"
"$R/envs/openwam-npu/bin/python" "$D/bench_kv_cache.py" --ckpt-dir "$CKPT_DIR" --ckpt-name "$CKPT_NAME" \
    --out "$D/bench_$TAG.json" 2>&1 | tee "$D/bench_$TAG.log"
