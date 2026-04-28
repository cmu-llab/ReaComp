#!/usr/bin/env bash
# DirectSolve baseline — one OpenHands conversation per PBEBench-Lite task.
#
# The agent receives the task (inputs/outputs), has access to the reward
# function via execute_code, and can write and run arbitrary Python code
# up to MAX_STEPS steps before submitting an answer.
#
# Usage (from project root):
#   bash scripts/run_direct_solve_openhands.sh
#   bash scripts/run_direct_solve_openhands.sh --workers 4
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DATASET=${DATASET:-data/pbebench/lite_tasks_full_og.jsonl}
OUTPUT=${OUTPUT:-outputs/lite_direct_solve_openhands.jsonl}

PORT=${PORT:-8000}
GPU_NODE=localhost
BASE_URL=http://${GPU_NODE}:${PORT}/v1
MODEL=openai/Qwen/Qwen3.6-35B-A3B

SIF_PATH=/scratch/$USER/sif_images/sandbox.sif
REWARDS_DIR=$(pwd)/rewards

MAX_STEPS=100
MAX_TOKENS=16384
WORKERS=${WORKERS:-8} # parallel conversations; each gets its own sandbox subprocess — safe to increase
# ─────────────────────────────────────────────────────────────────────────────

# Pass through any extra args (e.g. --workers 4)
EXTRA_ARGS=("$@")

python -m openhands_agents.run \
    --framework direct_solve \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --default-reward "$REWARD" \
    --sif-path "$SIF_PATH" \
    --rewards-dir "$REWARDS_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --max-steps "$MAX_STEPS" \
    --workers "$WORKERS" \
    --skip-existing \
    "${EXTRA_ARGS[@]}"
