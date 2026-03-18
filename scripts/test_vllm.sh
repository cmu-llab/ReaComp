#!/usr/bin/env bash
# Run the agent against a local vLLM server using a mock JSONL task file.
# Usage: bash scripts/test_vllm.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

python main.py \
  --tasks-file scripts/mock_tasks.jsonl \
  --base-url   "http://localhost:8002/v1" \
  --model      "openai/gpt-oss-120b" \
  --budget     15.0 \
  --output-dir outputs/vllm_test \
  --debug-dir debug/vllm_test \
  --stats
