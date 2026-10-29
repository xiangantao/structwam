#!/usr/bin/env bash
# LIBERO-plus client interpreter for stock pool images: the shared ImageMagick / Mesa-DRI libraries
# (sysdeps/hce2_aarch64) are loaded only into the client process, then the clean-instruction
# client wrapper runs as before.
source /home/ma-user/work/2337888765/sysdeps/hce2_aarch64/env.sh
exec /home/ma-user/work/2337888765/workspace/runs/libero_plus_structwam_s24000_v10/libero_plus_python_clean.sh "$@"
