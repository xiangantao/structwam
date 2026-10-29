#!/usr/bin/env bash
# One StructWAM policy server on NPU 0, port 8990 (smoke test).
W=/home/ma-user/work/2337888765/workspace
EV=$W/runs/libero10_structwam_s24000_v10
cd "$W/StructWAM"
CUDA_VISIBLE_DEVICES=0 nohup "$EV/server_python.sh" scripts/deploy.py \
    --ckpt-dir "$W/runs/structwam_step24000_eval_ckpt" --ckpt-name checkpoint_step_24000.safetensors \
    --device cuda:0 --host 127.0.0.1 --port 8990 --denoise-steps 10 --denoise-mode sync \
    --inference-mode sync --inference-horizon 10 --compile-enabled false \
    optimization.dit_cache.enabled=false > "$EV/${1:-smoke_server}.log" 2>&1 < /dev/null &
echo "server pid $!"
