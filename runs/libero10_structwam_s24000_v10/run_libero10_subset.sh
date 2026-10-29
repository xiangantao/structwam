#!/usr/bin/env bash
# Extra managed scheduler run next to the main one (same protocol as run_libero10.sh full).
#   bash run_libero10_subset.sh NAME TASK_IDS NUM_TRIALS GPUS BASE_PORT [REPLICAS_PER_GPU] [TRIAL_START]
# e.g. bash run_libero10_subset.sh quick_t2_8 2 2 5 8965 1 8
# Partial trial ranges need run_eval.py's _validate_protocol bypass.
set -euo pipefail
R=/home/ma-user/work/2337888765
W=$R/workspace
EV=$W/runs/libero10_structwam_s24000_v10
name="${1:?name}" tasks="${2:?task ids}" trials="${3:?num trials}" gpus="${4:?gpus}" port="${5:?base port}"
replicas="${6:-1}" start="${7:-0}"
cd "$W/StructWAM"
nohup "$R/envs/openwam-npu/bin/python" "$EV/run_eval.py" \
    --ckpt-dir "$W/runs/structwam_step24000_eval_ckpt" --ckpt-name checkpoint_step_24000.safetensors \
    --server-python "$EV/server_python.sh" --libero-python "$EV/libero_python.sh" \
    --libero-path "$R/envs/vace-lerobot/lib/python3.11/site-packages" \
    --policy-config "$W/StructWAM/benchmarks/libero/policy_config.yml" \
    --suites long --task-ids "$tasks" --trial-start "$start" --num-trials "$trials" --render-gpus 0 \
    --denoise-steps 10 --inference-horizon 10 --compile-enabled false \
    --extra-deploy-arg optimization.dit_cache.enabled=false --client-max-attempts 6 \
    --gpus "$gpus" --replicas-per-gpu "$replicas" --base-port "$port" \
    --output-dir "$EV/results_$name" > "$EV/${name}_launcher.log" 2>&1 < /dev/null &
echo "$name launcher pid $!"
