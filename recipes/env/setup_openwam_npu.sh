#!/usr/bin/env bash
# Policy / training environment on Ascend (aarch64): Python 3.11, torch 2.7.1 + torch_npu 2.7.1,
# CANN 8.3–8.5 installed on the host. Creates the env at the parent of $OPENWAM_PY if it is missing.
#
#   bash recipes/env/setup_openwam_npu.sh            (needs conda on PATH, or an existing $OPENWAM_PY)
#
# recipes/env/openwam-npu.requirements.txt is a pip freeze of the env every result in the README came
# from, minus the CANN-bundled wheels (te, hccl, ...) that the toolkit provides. torch_npu has to match
# your CANN: if pip can't find torch_npu==2.7.1, install the wheel from the Ascend torch_npu releases first.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"

prefix="$(dirname "$(dirname "$OPENWAM_PY")")"
if [[ ! -x "$OPENWAM_PY" ]]; then
    conda create -y -p "$prefix" python=3.11
fi
"$OPENWAM_PY" -m pip install -r "$STRUCTWAM_REPO/recipes/env/openwam-npu.requirements.txt"

git -C "$STRUCTWAM_REPO" submodule update --init StructWAM/third_party/cosmos-predict2.5
PYBIN="$OPENWAM_PY" bash "$STRUCTWAM_REPO/StructWAM/scripts/install_cosmos_predict25.sh"
"$OPENWAM_PY" -m pip install --no-deps -e "$STRUCTWAM_REPO/StructWAM"

if [[ -f "$ASCEND_SET_ENV" ]]; then
    set +eu +o pipefail
    source "$ASCEND_SET_ENV"
    set -euo pipefail
fi
"$OPENWAM_PY" - <<'EOF'
import torch, torch_npu  # noqa: F401
import openwam, latent_dp  # noqa: F401
print("openwam-npu ready:", torch.__version__, torch_npu.__version__, "NPUs:", torch.npu.device_count())
EOF
