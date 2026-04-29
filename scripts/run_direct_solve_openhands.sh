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
#
# Sharding (run two jobs in parallel on separate index ranges):
#   START_INDEX=0   END_INDEX=504  bash scripts/run_direct_solve_openhands.sh
#   START_INDEX=504 END_INDEX=1008 bash scripts/run_direct_solve_openhands.sh
#   Then merge: python scripts/merge_direct_solve_shards.py outputs/lite_direct_solve_openhands_0_504.jsonl outputs/lite_direct_solve_openhands_504_1008.jsonl --output outputs/lite_direct_solve_openhands.jsonl
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DATASET=${DATASET:-data/pbebench/lite_tasks_full_og.jsonl}
START_INDEX=${START_INDEX:-}
END_INDEX=${END_INDEX:-}

# Auto-suffix output filename with shard indices if set
BASE_OUTPUT=${OUTPUT:-outputs/lite_direct_solve_openhands.jsonl}
if [ -n "$START_INDEX" ] && [ -n "$END_INDEX" ]; then
    EXT="${BASE_OUTPUT##*.}"
    BASE="${BASE_OUTPUT%.*}"
    OUTPUT="${BASE}_${START_INDEX}_${END_INDEX}.${EXT}"
else
    OUTPUT="$BASE_OUTPUT"
fi

REWARD=${REWARD:-pbebench}

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

# Build optional shard flags
SHARD_ARGS=()
[ -n "$START_INDEX" ] && SHARD_ARGS+=(--start-index "$START_INDEX")
[ -n "$END_INDEX" ]   && SHARD_ARGS+=(--end-index   "$END_INDEX")

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
    "${SHARD_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
