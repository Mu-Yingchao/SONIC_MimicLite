#!/usr/bin/env bash
set -euo pipefail

# Force the MuJoCo GLFW viewer onto the discrete NVIDIA GPU on PRIME
# on-demand workstations.  Without these variables this workstation's X11
# screen has no DRI2/DRI3 path and Mesa silently falls back to llvmpipe.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${BUMI_SIM_PYTHON:-$REPO_ROOT/.venv_sim/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "MuJoCo Python not found: $PYTHON_BIN" >&2
  echo "Create .venv_sim as documented, or set BUMI_SIM_PYTHON." >&2
  exit 2
fi

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
# The control loop already enforces 50 Hz.  Avoid blocking viewer.sync() on a
# different display refresh interval; set BUMI_GL_VSYNC=1 to opt back in.
export __GL_SYNC_TO_VBLANK="${BUMI_GL_VSYNC:-0}"

exec "$PYTHON_BIN" "$REPO_ROOT/gear_sonic/scripts/run_bumi3_sim2sim.py" "$@"
