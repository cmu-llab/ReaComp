#!/usr/bin/env bash
# Run the direct-feedback baseline against a local vLLM server.
#
# Each task gets up to --df-k sequential single-turn calls.
# Attempt 1: raw task prompt only.
# Attempt 2+: task prompt + verifier feedback history from prior attempts.
#
# Checkpointing: completed_ids-based — safe to kill and re-launch; already-done
# tasks are skipped even if workers finish out of order.
#
# Usage:
#   bash scripts/run_direct_feedback_vllm.sh                       # defaults
#   bash scripts/run_direct_feedback_vllm.sh --tasks-file <file>   # override tasks file
#   PORT=8003 bash scripts/run_direct_feedback_vllm.sh             # different port
#   WORKERS=8 bash scripts/run_direct_feedback_vllm.sh             # parallelism
#
# Token budget per task (upper bound): df_k × max_tokens
#   default: 32 × 32768 = 1,048,576 tokens max (CoT-heavy model)
#
# df-k=32 gives the model many feedback chances; early-exit on reward=1.0 so
# most tasks will use far fewer than 32 calls.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PORT="${PORT:-8002}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export WORKERS="${WORKERS:-16}"
mkdir -p outputs

TASKS_FILE="${1:-data/pbebench/lite_tasks_full.jsonl}"
STEM="$(basename "${TASKS_FILE%.jsonl}")"
OUT_FILE="outputs/${STEM}_direct_feedback.jsonl"

echo "Tasks   : ${TASKS_FILE}"
echo "Output  : ${OUT_FILE}  (checkpoint: ${OUT_FILE%.jsonl}.ckpt.json)"
echo "Workers : ${WORKERS}"
echo "Port    : ${PORT}"

python main.py \
  --framework        direct_feedback \
  --tasks-file       "${TASKS_FILE}" \
  --base-url         "http://localhost:${PORT}/v1" \
  --model            "openai/gpt-oss-120b" \
  --df-k             32 \
  --max-tokens       32768 \
  --max-reward-iters 32 \
  --default-reward   pbebench \
  --workers          "${WORKERS}" \
  --output-file      "${OUT_FILE}" \
  --debug-dir        debug_direct_feedback \
  "$@"

echo "Done. Output: ${OUT_FILE}"
