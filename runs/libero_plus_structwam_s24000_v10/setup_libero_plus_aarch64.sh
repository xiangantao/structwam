#!/usr/bin/env bash
# benchmarks/libero-plus/setup_env.sh, continued by hand for aarch64 (the conda env, checkout and
# compatibility patch already exist).  Same pins, except:
#   - usd-core is skipped: PyPI has x86_64-only Linux wheels, and LIBERO-plus never imports pxr
#     at runtime ("usd" only appears in the paths of the pre-converted MJCF assets);
#   - torch 2.7.1 comes from PyPI if the CPU index has no +cpu aarch64 wheel (aarch64 PyPI
#     torch is CPU-only);
#   - packages come from the ModelArts PyPI mirror: PyPI pages fetched through the proxy are
#     sometimes truncated, which pip reports as "from versions: none".
set -euo pipefail
R=/home/ma-user/work/2337888765
W=$R/workspace
E=$R/envs/libero-plus
P=$W/repos/LIBERO-plus
B=$W/StructWAM/benchmarks/libero-plus
PV=$W/runs/libero_plus_structwam_s24000_v10
PY=$E/bin/python
ASSET_URL="https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip"
ASSET_SHA256="96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
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

# Pip section of environment.yml, minus usd-core, torch (installed first) and the index line.
REQ=$PV/requirements_env_aarch64.txt
sed -n '/- pip:/,$p' "$B/environment.yml" | sed -n 's/^ *- \(.*\)$/\1/p' \
    | grep -v -E '^(pip:|--extra-index-url|usd-core|torch==)' > "$REQ"
if "$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1+cpu"; then
    echo "[setup] torch 2.7.1+cpu from the PyTorch CPU index"
else
    echo "[setup] no torch 2.7.1+cpu aarch64 wheel; using torch==2.7.1 from PyPI"
    "$PY" -m pip install "torch==2.7.1"
fi
echo "[setup] environment.yml pip requirements:"; cat "$REQ"
"$PY" -m pip install -r "$REQ"

# Upstream requirements, minus usd-core.
grep -v -E '^\s*usd-core' "$P/requirements.txt" > "$PV/requirements_upstream_aarch64.txt"
grep -v -E '^\s*usd-core' "$P/extra_requirements.txt" > "$PV/extra_requirements_aarch64.txt"
"$PY" -m pip install -r "$PV/requirements_upstream_aarch64.txt" -r "$PV/extra_requirements_aarch64.txt"
"$PY" -m pip uninstall --yes libero >/dev/null 2>&1 || true
"$PY" -m pip install --no-deps --editable "$P"

if ! assets_complete; then
    curl --location --fail --retry 5 --retry-delay 5 --continue-at - --output "$PACKAGE_ROOT/assets.zip" "$ASSET_URL"
    echo "$ASSET_SHA256  $PACKAGE_ROOT/assets.zip" | sha256sum --check --status || {
        echo "[ERROR] LIBERO-plus asset checksum mismatch" >&2
        exit 1
    }
    unzip -q -o "$PACKAGE_ROOT/assets.zip" -d "$PACKAGE_ROOT"
    if [[ -d "$NESTED_ASSET_ROOT" && ! -e "$ASSET_ROOT" ]]; then
        mv "$NESTED_ASSET_ROOT" "$ASSET_ROOT"
    fi
fi
assets_complete || { echo "[ERROR] LIBERO-plus assets are incomplete under $ASSET_ROOT" >&2; exit 1; }

"$PY" -m pip check || true
"$PY" - <<'EOF'
import importlib.metadata as metadata
import mujoco

assert mujoco.__version__ == "3.3.2", mujoco.__version__
expected = {
    "hydra-core": "1.2.0", "numpy": "1.22.4", "wandb": "0.13.1", "easydict": "1.9",
    "transformers": "4.21.1", "opencv-python": "4.6.0.66", "robomimic": "0.2.0", "einops": "0.4.1",
    "thop": "0.1.1-2209072238", "robosuite": "1.4.0", "bddl": "1.0.1", "future": "0.18.2",
    "matplotlib": "3.5.3", "cloudpickle": "2.1.0", "gym": "0.25.2",
}
for package, required in expected.items():
    actual = metadata.version(package)
    assert actual == required, f"{package}: expected {required}, found {actual}"
assert metadata.version("wand") == "0.7.2"
assert metadata.version("scikit-image") == "0.19.3"
import wand.image  # noqa: F401  (needs the system ImageMagick)
print(f"LIBERO-plus environment ready (aarch64, no usd-core); MuJoCo {mujoco.__version__}")
EOF
