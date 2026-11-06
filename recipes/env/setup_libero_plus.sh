#!/usr/bin/env bash
# LIBERO-plus client environment (Python 3.10, CPU torch, MuJoCo 3.3.2 with EGL) plus the pinned
# LIBERO-plus checkout and its assets. Works on aarch64: usd-core (x86_64-only wheels, never imported
# at runtime) is left out. Creates the env at the parent of $LIBERO_PY if it is missing.
#
#   bash recipes/env/setup_libero_plus.sh            (needs conda, git, curl, sha256sum)
#
# The client also needs ImageMagick (python wand) and Mesa EGL. On images without them, point
# LIBERO_SYSDEPS_ENV at an env.sh that prepends those libraries (sysdeps/hce2_aarch64/env.sh is the one
# used on ModelArts pool images; the libraries themselves are not in this repo).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../paths.sh"

REMOTE=https://github.com/sylvestf/LIBERO-plus.git
COMMIT=4976dc30028e805ff8094b55501d532c48fec182
ASSET_URL=https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip
ASSET_SHA256=96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf
PATCH="$STRUCTWAM_REPO/StructWAM/benchmarks/libero-plus/patches/libero-plus-compatibility.patch"
P="$LIBERO_PLUS_REPO"
PACKAGE_ROOT="$P/libero/libero"
ASSET_ROOT="$PACKAGE_ROOT/assets"
NESTED="$PACKAGE_ROOT/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"

assets_complete() {
    local d
    for d in articulated_objects new_objects scenes stable_hope_objects stable_scanned_objects textures turbosquid_objects; do
        [[ -d "$ASSET_ROOT/$d" ]] || return 1
    done
    for d in serving_region.xml wall_frames.stl wall.xml; do [[ -f "$ASSET_ROOT/$d" ]] || return 1; done
}

# 1. checkout + OpenWAM's compatibility patch
if [[ ! -d "$P/.git" ]]; then
    git clone "$REMOTE" "$P"
    git -C "$P" checkout --detach "$COMMIT"
fi
[[ "$(git -C "$P" rev-parse HEAD)" == "$COMMIT" ]] || { echo "LIBERO-plus must be at $COMMIT" >&2; exit 1; }
if git -C "$P" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "[setup] compatibility patch already applied"
else
    git -C "$P" apply "$PATCH"
fi

# 2. env from the pinned freeze (torch 2.7.1+cpu comes from the PyTorch CPU index)
prefix="$(dirname "$(dirname "$LIBERO_PY")")"
[[ -x "$LIBERO_PY" ]] || conda create -y -p "$prefix" python=3.10
"$LIBERO_PY" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r "$STRUCTWAM_REPO/recipes/env/libero-plus.requirements.txt"
"$LIBERO_PY" -m pip install --no-deps --editable "$P"

# 3. assets (457k small files: extracted in parallel)
if ! assets_complete; then
    [[ -f "$PACKAGE_ROOT/assets.zip" ]] || curl -L --fail --retry 5 -C - -o "$PACKAGE_ROOT/assets.zip" "$ASSET_URL"
    echo "$ASSET_SHA256  $PACKAGE_ROOT/assets.zip" | sha256sum --check --status || { echo "asset checksum mismatch" >&2; exit 1; }
    "$LIBERO_PY" - "$PACKAGE_ROOT/assets.zip" "$PACKAGE_ROOT" 32 <<'EOF'
import os, sys, zipfile
from multiprocessing import Pool

archive, destination, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])


def extract(index):
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()[index::workers]
        for member in members:
            handle.extract(member, destination)
    return len(members)


if __name__ == "__main__":
    with zipfile.ZipFile(archive) as handle:  # create every parent first: extract() races on them
        for d in {os.path.dirname(n.rstrip("/")) for n in handle.namelist()}:
            os.makedirs(os.path.join(destination, d), exist_ok=True)
    with Pool(workers) as pool:
        print("extracted", sum(pool.map(extract, range(workers))), "files")
EOF
    [[ -d "$NESTED" && ! -e "$ASSET_ROOT" ]] && mv "$NESTED" "$ASSET_ROOT"
fi
assets_complete || { echo "LIBERO-plus assets incomplete under $ASSET_ROOT" >&2; exit 1; }

"$STRUCTWAM_REPO/recipes/eval/client_python.sh" -c "import mujoco, wand.image; print('LIBERO-plus env ready, MuJoCo', mujoco.__version__)"
