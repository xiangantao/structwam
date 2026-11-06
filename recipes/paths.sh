# Paths shared by every recipe. Source it, don't run it:  source recipes/paths.sh
#
# Put your machine's values in recipes/paths.local.sh (git-ignored; copy recipes/paths.example.sh).
# Everything the code needs from outside the repo lives under STRUCTWAM_ASSETS:
#
#   $STRUCTWAM_ASSETS/
#     models/Cosmos-Predict2.5-2B/            robot/action-cond weights + tokenizer.pth (video expert)
#     models/Cosmos-Reason1-7B/               live text encoder of the video expert
#     models/Wan2.1-T2V-1.3B-Diffusers/       UMT5 text_encoder + tokenizer (only for precompute_caches.sh)
#     datasets/openwam/libero/                OpenWAM's native-action LIBERO (LeRobot v3)
#     student/step_0002000.pt, text.pt, appearance_stats.npz   frozen structure student
#     checkpoints/b2/                         released StructWAM b2 40k checkpoint (training dir layout)
#     eval_ckpt/b2_v1kv/                      made by recipes/prepare_eval_ckpt.py
#     cache/                                  made by recipes/eval/precompute_caches.sh
#     repos/LIBERO-plus/                      LIBERO-plus checkout with its assets
#
# recipes/download_assets.sh fills the models / dataset / student / checkpoints parts.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "$_here/paths.local.sh" ]] && source "$_here/paths.local.sh"

export STRUCTWAM_REPO="${STRUCTWAM_REPO:-$(cd "$_here/.." && pwd)}"
: "${STRUCTWAM_ASSETS:?set STRUCTWAM_ASSETS (see recipes/paths.example.sh)}"
export STRUCTWAM_ASSETS

# Python environments (see the README): the policy / training env and the LIBERO-plus client env.
export OPENWAM_PY="${OPENWAM_PY:-$STRUCTWAM_ASSETS/envs/openwam-npu/bin/python}"
export LIBERO_PY="${LIBERO_PY:-$STRUCTWAM_ASSETS/envs/libero-plus/bin/python}"
export LIBERO_PLUS_REPO="${LIBERO_PLUS_REPO:-$STRUCTWAM_ASSETS/repos/LIBERO-plus}"

# Ascend toolkit environment, sourced by the launchers when present.
export ASCEND_SET_ENV="${ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
# Optional: extra shared libraries for the LIBERO-plus client (ImageMagick / Mesa-DRI) on stock
# images that lack them; an env.sh that prepends them, e.g. sysdeps/hce2_aarch64/env.sh.
export LIBERO_SYSDEPS_ENV="${LIBERO_SYSDEPS_ENV:-}"

export PYTHONPATH="$STRUCTWAM_REPO/StructWAM:$STRUCTWAM_REPO/heft/latent_policy/src:$STRUCTWAM_REPO/heft${PYTHONPATH:+:$PYTHONPATH}"
unset _here
