#!/usr/bin/env bash
# Stop the uploader; an interrupted upload restarts from its staged hard link on the next start.
pkill -f "python.*hf_ckpt_uploader.py" && echo stopped || echo "not running"
