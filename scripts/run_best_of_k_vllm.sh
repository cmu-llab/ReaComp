#!/usr/bin/env bash
# Run Best-of-K sampling baseline against a local vLLM server.
#
# Usage:
#   bash scripts/run_best_of_k_vllm.sh                       # defaults below
#   bash scripts/run_best_of_k_vllm.sh --tasks-file <file>   # override tasks file
#
# Token budget:
#   K × max_tokens per task (e.g. 5 × 4096 = 20,480 tokens upper bound)
#
# Compute-matching with ssl_bcr (3 iters × 8192 complex tokens = 24,576):
#   Set K=6, max_tokens=4096  →  6 × 4096 = 24,576  ✓
#   Or set K=3, max_tokens=8192  →  3 × 8192 = 24,576  ✓
#
# Temperature > 0 is important for Best-of-K diversity (default: 0.8).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PORT="${PORT:-8002}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p outputs

TASKS_FILE="${1:-data/pbebench/tasks_full_og.jsonl}"
OUT_FILE="outputs/$(basename "${TASKS_FILE%.jsonl}")_best_of_k.jsonl"

python main.py \
  --framework        best_of_k \
  --tasks-file       "${TASKS_FILE}" \
  --base-url         "http://localhost:${PORT}/v1" \
  --model            "openai/gpt-oss-120b" \
  --bok-k            32 \
  --max-tokens       32768 \
  --workers          32 \
  --default-reward   pbebench \
  --output-file      "${OUT_FILE}" \
  --debug-dir        debug_best_of_k \
  "$@"

echo "Output: ${OUT_FILE}"
