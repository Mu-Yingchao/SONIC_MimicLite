#!/usr/bin/env bash
set -euo pipefail

# Download an already-exported Robot/SMPL ONNX pair from world-rank 0 (GPU14)
# into the local deployment-only model directories.  Models remain outside Git.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${BUMI_CLUSTER_CONFIG:-$REPO_ROOT/.local/sonic_bumi_cluster.env}"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_ID STEP" >&2
  exit 2
fi

run_id="$1"
step="$2"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Unsafe RUN_ID" >&2; exit 2; }
[[ "$step" =~ ^[0-9]+$ ]] || { echo "STEP must be an integer" >&2; exit 2; }

# shellcheck disable=SC1090
source "$CONFIG_FILE"
: "${GPU14_SSH:?}" "${GPU14_PORT:?}" "${SSH_KEY:?}" "${RUN_ROOT:?}"

printf -v padded_step "%06d" "$((10#$step))"
remote_base="$RUN_ROOT/$run_id/exported/model_step_$padded_step"
local_base="$REPO_ROOT/models/deployment"
mkdir -p "$local_base/robot" "$local_base/smpl"

for encoder in robot smpl; do
  if [[ "$encoder" == "robot" ]]; then
    remote_suffix="g1"
  else
    remote_suffix="smpl"
  fi
  target="$local_base/$encoder/model_step_$padded_step.onnx"
  if [[ -e "$target" && "${FORCE:-0}" != "1" ]]; then
    echo "Refusing to overwrite $target (set FORCE=1 to replace it)" >&2
    exit 1
  fi
  part="$target.part"
  scp -q -i "$SSH_KEY" -P "$GPU14_PORT" \
    "$GPU14_SSH:${remote_base}_${remote_suffix}.onnx" "$part"
  [[ -s "$part" ]] || { echo "Downloaded file is empty: $part" >&2; exit 1; }
  mv "$part" "$target"
  sha256sum "$target"
done
