#!/usr/bin/env bash
# ReAct (JSON-mode, no library) — gpt-oss-120b compatible, no OpenHands SDK.
# Edit the variables below, then run from the project root:
#   bash scripts/run_json_react.sh <PORT>
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
USER="arnaik"
DATASET=data/pbebench/lite_tasks_full.jsonl
OUTPUT=outputs/json_react_pbebench_lite_full.jsonl
REWARD=pbebench
PORT=${1:?Usage: $0 <PORT>}

GPU_NODE=localhost
BASE_URL=http://${GPU_NODE}:${PORT}/v1
MODEL=openai/gpt-oss-120b

SIF_PATH=/scratch/$USER/sif_images/sandbox.sif

MAX_TOKENS=8192
MAX_ITERS=8
WORKERS=4
# ─────────────────────────────────────────────────────────────────────────────

python -m json_agents.run \
    --framework react \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --default-reward "$REWARD" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --sif-path "$SIF_PATH" \
    --max-tokens "$MAX_TOKENS" \
    --max-iters "$MAX_ITERS" \
    --workers "$WORKERS" \
    --debug-dir "debug_json_react" \
    --skip-existing
