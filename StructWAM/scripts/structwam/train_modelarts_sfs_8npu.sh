#!/usr/bin/env bash
# ModelArts job: mount the SAME SFS root at /mnt/sfs_turbo.
# Run in the foreground; the platform manages the job lifetime.
set -euo pipefail
SFS_PERSONAL=/mnt/sfs_turbo/2337888765
LEGACY_PERSONAL=/home/ma-user/work/2337888765
[[ -d "$SFS_PERSONAL/workspace/StructWAM" ]] || {
    echo "Project missing: $SFS_PERSONAL/workspace/StructWAM" >&2
    echo "Mount the original Notebook SFS root at /mnt/sfs_turbo." >&2
    exit 1
}
# Conda console scripts and project configs contain the original absolute prefix.
# Only alias our personal directory; never replace an existing directory.
if [[ -e "$LEGACY_PERSONAL" || -L "$LEGACY_PERSONAL" ]]; then
    [[ "$(readlink -f "$LEGACY_PERSONAL")" == "$(readlink -f "$SFS_PERSONAL")" ]] || {
        echo "Existing $LEGACY_PERSONAL is not the mounted personal directory; refusing to overwrite." >&2
        exit 1
    }
else
    mkdir -p "$(dirname "$LEGACY_PERSONAL")"
    ln -s "$SFS_PERSONAL" "$LEGACY_PERSONAL"
fi
# Some images initialize CANN themselves; source the conventional installer entry
# when available. Leave nonstandard image initialization to its image entrypoint.
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi
[[ -x "$LEGACY_PERSONAL/envs/openwam-npu/bin/python" ]] || {
    echo "Shared openwam-npu environment missing" >&2
    exit 1
}
cd "$LEGACY_PERSONAL/workspace/StructWAM"
exec bash scripts/structwam/train_libero_8npu.sh "$@"
