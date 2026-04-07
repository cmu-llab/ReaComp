#!/usr/bin/env bash
# Run ReAct+Library baseline against a local vLLM server.
#
# Usage:
#   bash scripts/run_react_library_vllm.sh                       # defaults below
#   bash scripts/run_react_library_vllm.sh --tasks-file <file>   # override tasks file
#
# Token budget (upper bound per task):
#   max_reward_iters × max_steps × (2 calls/step) × max_tokens
#   = 3 × 6 × 2 × 16384 = 589,824 tokens
#
# Key difference from react_mem:
#   react_mem  — retrieves similar past task solutions as few-shot examples.
#   react_library — maintains a shared Python function library; retrieved
#                   functions are loaded into the execution namespace so the
#                   agent calls them by name directly.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PORT="${PORT:-8002}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p outputs

python main.py \
  --framework        react_library \
  --tasks-file       data/pbebench/lite_tasks_full.jsonl \
  --base-url         "http://localhost:${PORT}/v1" \
  --model            "openai/gpt-oss-120b" \
  --react-lib-k      5 \
  --react-max-steps  6 \
  --max-tokens       16384 \
  --max-reward-iters 3 \
  --default-reward   pbebench \
  --output-file      outputs/pbebench_lite_full_react_library.jsonl \
  --debug-dir        debug_react_library \
  "$@"

echo "Output: ${OUT_FILE}"
