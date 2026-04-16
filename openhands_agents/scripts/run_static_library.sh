#!/usr/bin/env bash
# Run the StaticLibrary baseline — weaker agent using a pre-built fixed library.
#
# Usage:
#   bash openhands_agents/scripts/run_static_library.sh
#
# The library at LIBRARY_PATH must contain LIBRARY.py and PROMPTING_GUIDE.md.

set -euo pipefail

GPU_NODE="${GPU_NODE:-localhost}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-openai/Qwen/Qwen3-Coder-30B-A3B-Instruct}"
DATASET="${DATASET:-data/pbebench_lite/pbebench_lite.jsonl}"
OUTPUT="${OUTPUT:-outputs/oh_static_library_pbebench_lite.jsonl}"
REWARD="${REWARD:-pbebench}"
LIBRARY_PATH="${LIBRARY_PATH:-built_libraries/claude_code/Wed_Apr_15_735_PM}"
SIF_PATH="${SIF_PATH:-${SANDBOX_SIF:-/scratch/$USER/sif_images/sandbox.sif}}"
PKG_DIR="${PKG_DIR:-${HOME}/oh_packages}"
WORKERS="${WORKERS:-4}"
MAX_STEPS="${MAX_STEPS:-100}"
DEBUG_DIR="${DEBUG_DIR:-}"

ARGS=(
    --framework static_library
    --dataset-path "$DATASET"
    --output-path "$OUTPUT"
    --default-reward "$REWARD"
    --library-path "$LIBRARY_PATH"
    --sif-path "$SIF_PATH"
    --pkg-dir "$PKG_DIR"
    --base-url "http://${GPU_NODE}:${PORT}/v1"
    --model "$MODEL"
    --max-steps "$MAX_STEPS"
    --workers "$WORKERS"
    --skip-existing
)

if [[ -n "$DEBUG_DIR" ]]; then
    ARGS+=(--debug-dir "$DEBUG_DIR")
fi

echo "Starting StaticLibrary run..."
echo "  Library: $LIBRARY_PATH"
echo "  Dataset: $DATASET"
echo "  Output:  $OUTPUT"
echo "  Workers: $WORKERS"

python -m openhands_agents.run "${ARGS[@]}"
