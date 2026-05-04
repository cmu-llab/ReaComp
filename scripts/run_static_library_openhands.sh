#!/usr/bin/env bash
# StaticLibrary baseline — pre-built library (LIBRARY.py + PROMPTING_GUIDE.md)
# used by a weaker agent via OpenHands SDK + Apptainer sandbox.
# Edit the variables below, then run from the project root:
#   bash scripts/run_static_library_openhands.sh <PORT>
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
USER="anonymous"
DATASET=data/pbebench/lite_tasks_full.jsonl
OUTPUT=outputs/oh_static_library_pbebench_lite_full.jsonl
REWARD=pbebench
PORT=$1

# Path to the built_libraries directory containing LIBRARY.py + PROMPTING_GUIDE.md
LIBRARY_PATH=built_libraries/claude_code/Wed_Apr_15_735_PM

GPU_NODE=localhost          # hostname of the node running vLLM
BASE_URL=http://${GPU_NODE}:${PORT}/v1
MODEL=openai/Qwen/Qwen3-Coder-30B-A3B-Instruct  # openai/ prefix required by litellm for OpenAI-compat endpoints

SIF_PATH=/scratch/$USER/sif_images/sandbox.sif
PKG_DIR=/scratch/$USER/oh_packages

MAX_TOKENS=4096
MAX_STEPS=100           # max agent steps per conversation
# ─────────────────────────────────────────────────────────────────────────────

python -m openhands_agents.run \
    --framework static_library \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --default-reward "$REWARD" \
    --library-path "$LIBRARY_PATH" \
    --sif-path "$SIF_PATH" \
    --pkg-dir "$PKG_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --workers 4 \
    --max-tokens "$MAX_TOKENS" \
    --max-steps "$MAX_STEPS" \
    --debug-dir "debug_oh_static_library" \
    --skip-existing
