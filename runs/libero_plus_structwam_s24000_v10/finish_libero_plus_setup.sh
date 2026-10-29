#!/usr/bin/env bash
# Finish the LIBERO-plus setup after setup_libero_plus_aarch64.sh's single-process unzip proved too
# slow on SFS (457,675 small files): parallel extraction, asset check, then restore the pinned
# torch 2.7.1+cpu (robomimic's unpinned torchvision had pulled torch 2.14.1 plus CUDA 13 wheels).
set -euo pipefail
R=/home/ma-user/work/2337888765
W=$R/workspace
E=$R/envs/libero-plus
PY=$E/bin/python
P=$W/repos/LIBERO-plus
PACKAGE_ROOT=$P/libero/libero
ASSET_ROOT=$PACKAGE_ROOT/assets
NESTED_ASSET_ROOT="$PACKAGE_ROOT/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets"
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off
export PIP_INDEX_URL=http://pip.modelarts.private.com:8888/repository/pypi/simple
export PIP_TRUSTED_HOST=pip.modelarts.private.com PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=120

assets_complete() {
    local required
    for required in articulated_objects new_objects scenes stable_hope_objects stable_scanned_objects \
        textures turbosquid_objects; do
        [[ -d "$ASSET_ROOT/$required" ]] || return 1
    done
    for required in serving_region.xml wall_frames.stl wall.xml; do
        [[ -f "$ASSET_ROOT/$required" ]] || return 1
    done
}

if ! assets_complete; then
    echo "[finish] parallel extraction $(date +%T)"
    "$PY" - "$PACKAGE_ROOT/assets.zip" "$PACKAGE_ROOT" 32 <<'EOF'
import os
import sys
import zipfile
from multiprocessing import Pool

archive, destination, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])


def extract(index):
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()[index::workers]
        for member in members:
            try:
                handle.extract(member, destination)  # overwrites anything a killed unzip left half-written
            except FileExistsError:  # another worker created a parent directory in between
                handle.extract(member, destination)
    return len(members)


if __name__ == "__main__":
    # zipfile checks-then-creates parent directories, which races across workers: create them first.
    with zipfile.ZipFile(archive) as handle:
        directories = {os.path.dirname(name.rstrip("/")) for name in handle.namelist()}
        directories |= {name.rstrip("/") for name in handle.namelist() if name.endswith("/")}
    for directory in sorted(d for d in directories if d):
        os.makedirs(os.path.join(destination, directory), exist_ok=True)
    print(f"[finish] created {len(directories)} directories", flush=True)
    with Pool(workers) as pool:
        print(f"[finish] extracted {sum(pool.map(extract, range(workers)))} entries", flush=True)
EOF
    if [[ -d "$NESTED_ASSET_ROOT" && ! -e "$ASSET_ROOT" ]]; then
        mv "$NESTED_ASSET_ROOT" "$ASSET_ROOT"
    fi
fi
assets_complete || { echo "[ERROR] LIBERO-plus assets are incomplete under $ASSET_ROOT" >&2; exit 1; }
echo "[finish] assets complete $(date +%T)"

echo "[finish] restoring torch 2.7.1+cpu (was $("$PY" -c 'import importlib.metadata as m; print(m.version("torch"))'))"
"$PY" -m pip install --no-deps --force-reinstall --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.7.1+cpu" "torchvision==0.22.1+cpu"
cuda=$("$PY" -m pip list --format=freeze | cut -d= -f1 | grep -E '^(nvidia-|cuda-|triton$)' || true)
[[ -n "$cuda" ]] && "$PY" -m pip uninstall -y $cuda
"$PY" -m pip check || true
"$PY" - <<'EOF'
import importlib.metadata as metadata
import mujoco

assert mujoco.__version__ == "3.3.2", mujoco.__version__
expected = {
    "hydra-core": "1.2.0", "numpy": "1.22.4", "wandb": "0.13.1", "easydict": "1.9",
    "transformers": "4.21.1", "opencv-python": "4.6.0.66", "robomimic": "0.2.0", "einops": "0.4.1",
    "thop": "0.1.1-2209072238", "robosuite": "1.4.0", "bddl": "1.0.1", "future": "0.18.2",
    "matplotlib": "3.5.3", "cloudpickle": "2.1.0", "gym": "0.25.2", "torch": "2.7.1+cpu",
    "mujoco": "3.3.2",
}
for package, required in expected.items():
    actual = metadata.version(package)
    assert actual == required, f"{package}: expected {required}, found {actual}"
assert metadata.version("wand") == "0.7.2"
assert metadata.version("scikit-image") == "0.19.3"
import torchvision  # noqa: F401
import wand.image  # noqa: F401  (needs the system ImageMagick)
print(f"LIBERO-plus environment ready (aarch64, no usd-core); MuJoCo {mujoco.__version__}")
EOF
