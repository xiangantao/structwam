#!/usr/bin/env bash
# Hugging Face login for the uploader only: the token is stored under $HF_HOME, not ~/.cache,
# so other huggingface tools on this shared account do not pick it up.
B=/mnt/sfs-turbo-wangfei/2337888765
export HF_HOME=$B/.hf_uploader
mkdir -p "$HF_HOME" && chmod 700 "$HF_HOME"
exec "$B/envs/openwam-npu/bin/hf" auth login
