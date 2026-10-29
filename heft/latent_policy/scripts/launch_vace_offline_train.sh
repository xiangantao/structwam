#!/usr/bin/env bash
set -euo pipefail

# Student-only launcher. Data extraction/upload is owned by
# cache_vace_teacher.py; this script consumes one fixed uploaded manifest.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$LATENT_HEFT_ROOT"

usage() {
  cat >&2 <<'EOF'
Usage:
  launch_vace_offline_train.sh [options] [-- extra train_vace_structured.py args]

Required:
  --manifest PATH       Fixed, verified uploaded.jsonl snapshot

Common options:
  --output PATH         Student checkpoint/output directory
  --cache-dir PATH      Bounded local cache for TOS targets
  --steps N             Optimizer steps (default: 10000)
  --batch-size N        Per-rank batch size (default: 2)
  --workers N           RGB dataloader workers (default: config value)
  --resume PATH         Resume a checkpoint with the same manifest/protocol
  --min-rows N          Refuse to start below this manifest size (default: 1)
  --nproc N             Number of training ranks (default: 8)
  --master-port N       torchrun rendezvous port (default: 29651)
  --task shared         Shared five-head student (default: shared)
EOF
}

manifest="${VACE_OFFLINE_MANIFEST:-}"
cache_dir="${VACE_TEACHER_CACHE_DIR:-$LATENT_WORKSPACE/cache/vace_teacher_libero8}"
output="${VACE_OUTPUT:-$LATENT_WORKSPACE/runs/vace_structured_offline_libero8}"
steps="${VACE_STEPS:-10000}"
batch_size="${VACE_BATCH_SIZE:-2}"
workers=""
resume=""
min_rows="${VACE_MIN_ROWS:-1}"
nproc="${NPROC_PER_NODE:-8}"
master_port="${MASTER_PORT:-29651}"
task="shared"
extra=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    --cache-dir) cache_dir="$2"; shift 2 ;;
    --steps) steps="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --resume) resume="$2"; shift 2 ;;
    --min-rows) min_rows="$2"; shift 2 ;;
    --nproc) nproc="$2"; shift 2 ;;
    --master-port) master_port="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; extra+=("$@"); break ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$task" != "shared" ]]; then
  printf 'Offline five-head training requires --task shared (got %s).\n' "$task" >&2
  exit 2
fi

if [[ -z "$manifest" ]]; then
  printf '%s\n' 'A fixed --manifest uploaded.jsonl snapshot is required.' >&2
  usage
  exit 2
fi
manifest="$(readlink -f -- "$manifest")"
cache_dir="$(readlink -m -- "$cache_dir")"
output="$(readlink -m -- "$output")"
[[ -f "$manifest" ]] || { printf 'Manifest not found: %s\n' "$manifest" >&2; exit 1; }

rows="$(awk 'NF {count++} END {print count + 0}' "$manifest")"
if (( rows < min_rows )); then
  printf 'Manifest has %s rows; required at least %s.\n' "$rows" "$min_rows" >&2
  exit 1
fi
if ! head -n 1 "$manifest" | grep -q 'target_sha256'; then
  printf '%s\n' 'Manifest is not a verified uploaded.jsonl snapshot (target_sha256 missing).' >&2
  exit 1
fi

mkdir -p "$cache_dir" "$output"
printf 'offline training: rows=%s task=%s world=%s manifest=%s output=%s\n' \
  "$rows" "$task" "$nproc" "$manifest" "$output"

args=(
  "$LATENT_POLICY_ROOT/scripts/train_vace_structured.py"
  --config "$LATENT_POLICY_ROOT/configs/vace_structured.json"
  --task "$task"
  --offline-manifest "$manifest"
  --teacher-cache-dir "$cache_dir"
  --steps "$steps"
  --batch-size "$batch_size"
  --output "$output"
)
[[ -n "$workers" ]] && args+=(--workers "$workers")
[[ -n "$resume" ]] && args+=(--resume "$(readlink -f -- "$resume")")
args+=("${extra[@]}")

export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec "$LATENT_POLICY_PYTHON" -m torch.distributed.run --standalone \
  --nproc_per_node="$nproc" --master_port="$master_port" "${args[@]}"
