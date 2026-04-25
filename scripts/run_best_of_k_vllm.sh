#!/usr/bin/env bash
# Run Best-of-K sampling baseline against a local vLLM server.
#
# Usage:
#   bash scripts/run_best_of_k_vllm.sh                        # PBEBench-Hard (default)
#   bash scripts/run_best_of_k_vllm.sh --tasks-file <file>    # override tasks file
#   DATASET=lite bash scripts/run_best_of_k_vllm.sh           # PBEBench-Lite
#   DATASET=slr  bash scripts/run_best_of_k_vllm.sh           # SLR-Bench
#   MAX_PROGRAMS=5 DATASET=lite bash scripts/run_best_of_k_vllm.sh
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

export PORT="${PORT:-8004}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p outputs

DATASET="${DATASET:-hard}"

case "${DATASET}" in
  slr)
    TASKS_FILE="data/slr_bench/v1_All_full.jsonl"
    OUT_FILE="outputs/slr_bench_best_of_k.jsonl"
    DEFAULT_REWARD="slr_bench"
    DEFAULT_MAX_PROGRAMS=""
    ;;
  lite)
    TASKS_FILE="data/pbebench/lite_tasks_full_og.jsonl"
    OUT_FILE="outputs/lite_tasks_full_og_best_of_k.jsonl"
    DEFAULT_REWARD="pbebench"
    DEFAULT_MAX_PROGRAMS="5"
    ;;
  hard|*)
    TASKS_FILE="data/pbebench/tasks_full_og.jsonl"
    OUT_FILE="outputs/tasks_full_og_best_of_k.jsonl"
    DEFAULT_REWARD="pbebench"
    DEFAULT_MAX_PROGRAMS="20"
    ;;
esac

MAX_PROGRAMS="${MAX_PROGRAMS:-${DEFAULT_MAX_PROGRAMS}}"
PROGRAMS_ARG=""
[[ -n "${MAX_PROGRAMS}" ]] && PROGRAMS_ARG="--max-programs ${MAX_PROGRAMS}"

python main.py \
  --framework        best_of_k \
  --tasks-file       "${TASKS_FILE}" \
  --base-url         "http://localhost:${PORT}/v1" \
  --model            "openai/gpt-oss-120b" \
  --bok-k            32 \
  --max-tokens       32768 \
  --workers          32 \
  --default-reward   "${DEFAULT_REWARD}" \
  ${PROGRAMS_ARG} \
  --output-file      "${OUT_FILE}" \
  "$@"

echo "Output: ${OUT_FILE}"
