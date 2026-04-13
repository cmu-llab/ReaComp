#!/usr/bin/env bash
# Build the sandbox SIF — no Docker required.
#
# Usage:
#   bash openhands_agents/scripts/build_sandbox.sh [SIF_DIR]
#
# SIF_DIR defaults to /scratch/$USER/sif_images
#
# Strategy (tries in order):
#   1. apptainer build --fakeroot  (preferred; needs user namespaces on the cluster)
#   2. Reuse existing openhands.sif if it already has numpy/scipy/sympy
#      (pull it first with setup_openhands.sh if not already done)

set -euo pipefail

SIF_DIR="${1:-/scratch/$USER/sif_images}"
SIF_PATH="$SIF_DIR/sandbox.sif"
OH_SIF="$SIF_DIR/openhands.sif"
DEF_FILE="$(cd "$(dirname "$0")/.." && pwd)/sandbox.def"

mkdir -p "$SIF_DIR"

# ── Option 1: build from sandbox.def with fakeroot ───────────────────────────
echo "=== Sandbox SIF build ==="
echo "Target: $SIF_PATH"
echo

if apptainer build --fakeroot "$SIF_PATH" "$DEF_FILE" 2>/dev/null; then
    echo "==> Built sandbox.sif via --fakeroot."
    apptainer exec "$SIF_PATH" python -c "import numpy, scipy, sympy; print('  numpy/scipy/sympy: OK')"
    echo "Done: $SIF_PATH"
    exit 0
fi

echo "[fakeroot unavailable or failed — trying fallback]"
echo

# ── Option 2: reuse openhands.sif if it has the needed packages ───────────────
if apptainer inspect "$OH_SIF" &>/dev/null 2>&1; then
    echo "Found existing openhands.sif — checking for numpy/scipy/sympy..."
    if apptainer exec "$OH_SIF" python -c "import numpy, scipy, sympy" &>/dev/null 2>&1; then
        echo "  numpy/scipy/sympy: OK"
        echo "==> Reusing openhands.sif as sandbox (symlinking)."
        ln -sf "$OH_SIF" "$SIF_PATH"
        echo "Done: $SIF_PATH -> $OH_SIF"
        exit 0
    else
        echo "  openhands.sif is missing numpy/scipy/sympy — cannot reuse."
    fi
else
    echo "openhands.sif not found at $OH_SIF."
    echo "Pull it first with:  bash scripts/setup_openhands.sh"
fi

# ── Failed ────────────────────────────────────────────────────────────────────
echo
echo "ERROR: Could not build sandbox.sif."
echo
echo "Options:"
echo "  1. Ask your cluster admin to enable --fakeroot (user namespace mapping)."
echo "  2. Pull openhands.sif and re-run this script:"
echo "       apptainer pull $OH_SIF docker://ghcr.io/all-hands-ai/openhands:main"
echo "     (openhands.sif likely has numpy/scipy but may lack sympy)"
exit 1
