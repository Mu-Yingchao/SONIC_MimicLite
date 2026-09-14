#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${BUMI_SIM_PYTHON:-$REPO_ROOT/.venv_sim/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "MuJoCo Python not found: $PYTHON_BIN" >&2; exit 2; }

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export __GL_SYNC_TO_VBLANK="${BUMI_GL_VSYNC:-0}"

exec "$PYTHON_BIN" "$REPO_ROOT/gear_sonic/scripts/run_bumi3_pico_sim2sim.py" "$@"
