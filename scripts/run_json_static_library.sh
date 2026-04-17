#!/usr/bin/env bash
# Static Library (JSON-mode) — gpt-oss-120b compatible, no OpenHands SDK.
# Edit the variables below, then run from the project root:
#   bash scripts/run_json_static_library.sh <PORT>
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
USER="arnaik"
DATASET=data/pbebench/lite_tasks_full.jsonl
OUTPUT=outputs/json_sl_pbebench_lite_full.jsonl
REWARD=pbebench
PORT=${1:?Usage: $0 <PORT>}

GPU_NODE=localhost
BASE_URL=http://${GPU_NODE}:${PORT}/v1
MODEL=openai/gpt-oss-120b

LIBRARY_PATH=built_libraries/claude_code/Wed_Apr_15_735_PM

SIF_PATH=/scratch/$USER/sif_images/sandbox.sif
PKG_DIR=/scratch/$USER/oh_packages

MAX_TOKENS=8192    # gpt-oss-120b uses reasoning tokens, needs headroom
MAX_ITERS=8
WORKERS=4
# ─────────────────────────────────────────────────────────────────────────────

python -m json_agents.run \
    --framework static_library \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --default-reward "$REWARD" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --library-path "$LIBRARY_PATH" \
    --sif-path "$SIF_PATH" \
    --pkg-dir "$PKG_DIR" \
    --max-tokens "$MAX_TOKENS" \
    --max-iters "$MAX_ITERS" \
    --workers "$WORKERS" \
    --debug-dir "debug_json_sl" \
    --skip-existing
