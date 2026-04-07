#!/usr/bin/env bash
# Run TroVE baseline against a local vLLM server.
# Usage: bash scripts/run_trove_vllm.sh
#
# For small datasets (≤100 tasks), --trove-trim-every is set high to disable
# trimming (the library never gets large enough for it to matter).
# Set --trove-k 1 for a cheaper run without self-consistency sampling.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PORT=8002
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p outputs

python main.py \
  --framework      trove \
  --tasks-file     data/pbebench/lite_tasks_full.jsonl \
  --base-url       "http://localhost:${PORT}/v1" \
  --model          "openai/gpt-oss-120b" \
  --trove-k        5 \
  --trove-trim-every 9999 \
  --default-reward pbebench \
  --output-file    outputs/pbebench_lite_full_trove.jsonl \
  --debug-dir      debug_trove \
  --stats
