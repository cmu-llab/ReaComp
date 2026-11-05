#!/usr/bin/env bash
# KinDER DirectSolve — one OpenHands conversation per KinDER instance.
#
# The agent sees the environment description and one instance, runs Python against a
# simulator of that instance (kinder_client), and submits a GeneratedApproach policy that
# the harness rolls out and verifies by exact replay.
#   SPLIT=train → reasoning traces for solver induction (scripts/build_demos.py)
#   SPLIT=test  → LLM baseline and fallback of the solver-first hybrid
#
# Usage (from project root; requires a vLLM server, e.g. scripts/launch_vllm_qwen3.6_35b_a3b.sh):
#   FAMILY=obstruction2d SPLIT=train bash scripts/run_kinder_direct_solve_openhands.sh
#   FAMILY=obstruction2d SPLIT=test  PORT=8004 WORKERS=16 bash scripts/run_kinder_direct_solve_openhands.sh
#
# Families: obstruction2d clutteredretrieval2d clutteredstorage2d motion2d stickbutton2d
#
# Sharding (two jobs on separate index ranges; the output name gets the range):
#   START_INDEX=0  END_INDEX=50  FAMILY=motion2d SPLIT=test bash scripts/run_kinder_direct_solve_openhands.sh
#   START_INDEX=50 END_INDEX=100 FAMILY=motion2d SPLIT=test bash scripts/run_kinder_direct_solve_openhands.sh
#   Then merge: python scripts/merge_direct_solve_shards.py <shard1.jsonl> <shard2.jsonl> --output <merged.jsonl>
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
FAMILY=${FAMILY:-obstruction2d}
SPLIT=${SPLIT:-train}
START_INDEX=${START_INDEX:-}
END_INDEX=${END_INDEX:-}

# Another build of the task files (KINDER_DATA_DIR) has the same task ids: give it its own output file.
SUITE_TAG=${KINDER_DATA_DIR:+_$(basename "$KINDER_DATA_DIR")}
BASE_OUTPUT=${OUTPUT:-outputs/kinder_${FAMILY}_${SPLIT}${SUITE_TAG}_direct_solve_openhands.jsonl}
if [ -n "$START_INDEX" ] && [ -n "$END_INDEX" ]; then
    OUTPUT="${BASE_OUTPUT%.*}_${START_INDEX}_${END_INDEX}.${BASE_OUTPUT##*.}"
else
    OUTPUT="$BASE_OUTPUT"
fi

PORT=${PORT:-8000}
GPU_NODE=${GPU_NODE:-localhost}
BASE_URL=http://${GPU_NODE}:${PORT}/v1
MODEL=${MODEL:-openai/Qwen/Qwen3.6-35B-A3B}  # openai/ prefix required by litellm

MAX_STEPS=${MAX_STEPS:-40}      # LLM turns per task
MAX_TOKENS=${MAX_TOKENS:-16384}
WORKERS=${WORKERS:-8}           # parallel conversations; each runs its own simulator + sandbox processes (CPU)
DEBUG_DIR=${DEBUG_DIR:-}
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_ARGS=("$@")
SHARD_ARGS=()
[ -n "$START_INDEX" ] && SHARD_ARGS+=(--start-index "$START_INDEX")
[ -n "$END_INDEX" ]   && SHARD_ARGS+=(--end-index   "$END_INDEX")
[ -n "$DEBUG_DIR" ]   && SHARD_ARGS+=(--debug-dir   "$DEBUG_DIR")

python -m openhands_agents.run_kinder_direct_solve \
    --domain "kinder_${FAMILY}" \
    --split "$SPLIT" \
    --output-path "$OUTPUT" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --max-steps "$MAX_STEPS" \
    --workers "$WORKERS" \
    --skip-existing \
    "${SHARD_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
