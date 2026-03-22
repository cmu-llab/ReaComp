#!/usr/bin/env bash
# Run the agent against a local vLLM server.
# Usage: bash scripts/run_agent_vllm.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p outputs

python main.py \
  --tasks-file     data/pbebench/lite_pilot_tasks.jsonl \
  --base-url       "http://localhost:8002/v1" \
  --model          "openai/gpt-oss-120b" \
  --budget         15.0 \
  --max-reward-iters 3 \
  --output-file    outputs/pbebench_lite_pilot_tasks_with_rewards.jsonl \
  --stats