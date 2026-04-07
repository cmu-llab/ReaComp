#!/usr/bin/env bash
# Run ReAct+Memory baseline against a local vLLM server.
#
# Usage:
#   bash scripts/run_react_mem_vllm.sh                       # defaults below
#   bash scripts/run_react_mem_vllm.sh --tasks-file <file>   # override tasks file
#
# Token budget:
#   max_reward_iters × max_steps × (2 calls/step) × max_tokens
#   = 3 × 5 × 2 × 4096 = 122,880 tokens (upper bound per task)
#
# To compute-match against ssl_bcr, set --max-tokens so that:
#   K_react × max_steps_react × max_tokens_react
#     == ssl_max_reward_iters × ssl_max_steps × ssl_max_tokens_complex

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PORT="${PORT:-8002}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p outputs

python main.py \
  --framework        react_mem \
  --tasks-file       data/pbebench/lite_tasks_full.jsonl \
  --base-url         "http://localhost:${PORT}/v1" \
  --model            "openai/gpt-oss-120b" \
  --react-mem-k      3 \
  --react-max-steps  5 \
  --max-tokens       4096 \
  --max-reward-iters 3 \
  --default-reward   pbebench \
  --output-file      outputs/pbebench_lite_full_react_mem.jsonl \
  --debug-dir        debug_react_mem \
  "$@"

echo "Output: ${OUT_FILE}"
