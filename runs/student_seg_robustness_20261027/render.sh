#!/usr/bin/env bash
# Render LIBERO-plus scenes + instance segmentation with N CPU/EGL workers. Usage: render.sh N PER_CATEGORY OUT
set -euo pipefail
R=/home/ma-user/work/2337888765
D=$R/workspace/runs/student_seg_robustness_20261027
LP=$R/workspace/repos/LIBERO-plus
B=$LP/libero/libero
cat > $D/libero_config/config.yaml <<CFG
assets: $B/assets
bddl_files: $B/bddl_files
benchmark_root: $B
datasets: $LP/datasets
init_states: $B/init_files
CFG
export LIBERO_CONFIG_PATH=$D/libero_config LIBERO_PLUS_PATH=$LP PYTHONPATH=$LP
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 LP_NUM_THREADS=4 OMP_NUM_THREADS=2
N=$1; PER=$2; OUT=$3
for i in $(seq 0 $((N-1))); do
  $R/envs/libero-plus/bin/python $D/render_seg.py --output $D/$OUT --per-category $PER --shard $i --shards $N > $D/$OUT.render$i.log 2>&1 &
done
wait
echo RENDER_DONE
