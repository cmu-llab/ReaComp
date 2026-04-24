#!/usr/bin/env bash
# Run the direct-feedback-simplify baseline against a local vLLM server.
#
# Uses a single shared budget of --dfs-k attempts per task:
#   Phase 1 (Correctness): all attempts go here first; exits early on reward=1.0.
#   Phase 2 (Simplification): the remaining attempts (budget - phase1_used) try
#     to find a correct solution with lower cascade complexity.
#   If Phase 1 never reaches reward=1.0, Phase 2 is skipped entirely.
#   The lowest-complexity correct solution found across both phases is returned.
#
# Checkpointing: completed_ids-based — safe to kill and re-launch.
#
# Usage:
#   bash scripts/run_direct_feedback_simplify_vllm.sh                       # defaults
#   bash scripts/run_direct_feedback_simplify_vllm.sh --tasks-file <file>   # override tasks
#   PORT=8003 bash scripts/run_direct_feedback_simplify_vllm.sh             # different port
#   WORKERS=8 bash scripts/run_direct_feedback_simplify_vllm.sh             # parallelism
#
# Token budget per task (upper bound): dfs_k × max_tokens
#   default: 32 × 32768 = 1,048,576 tokens max

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PORT="${PORT:-8002}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export WORKERS="${WORKERS:-16}"
mkdir -p outputs

TASKS_FILE="${1:-data/pbebench/lite_tasks_full_og.jsonl}"
STEM="$(basename "${TASKS_FILE%.jsonl}")"
OUT_FILE="outputs/${STEM}_direct_feedback_simplify.jsonl"

echo "Tasks   : ${TASKS_FILE}"
echo "Output  : ${OUT_FILE}  (checkpoint: ${OUT_FILE%.jsonl}.ckpt.json)"
echo "Workers : ${WORKERS}"
echo "Port    : ${PORT}"

python main.py \
  --framework        direct_feedback_simplify \
  --tasks-file       "${TASKS_FILE}" \
  --base-url         "http://localhost:${PORT}/v1" \
  --model            "openai/gpt-oss-120b" \
  --dfs-k            32 \
  --max-tokens       32768 \
  --default-reward   pbebench \
  --max-programs     5 \
  --workers          "${WORKERS}" \
  --output-file      "${OUT_FILE}" \
  --debug-dir        debug_direct_feedback_simplify \
  "$@"

echo "Done. Output: ${OUT_FILE}"
