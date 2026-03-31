#!/usr/bin/env bash
# Run TroVE baseline against a local vLLM server.
# Usage: bash scripts/run_trove_vllm.sh
#
# For small datasets (≤100 tasks), --trove-trim-every is set high to disable
# trimming (the library never gets large enough for it to matter).
# Set --trove-k 1 for a cheaper run without self-consistency sampling.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p outputs

python main.py \
  --framework      trove \
  --tasks-file     data/interleaved/pbebench_rg_string_pilot.jsonl \
  --base-url       "http://localhost:8002/v1" \
  --model          "openai/gpt-oss-120b" \
  --trove-k        5 \
  --trove-trim-every 9999 \
  --default-reward pbebench \
  --output-file    outputs/pbebench_rg_string_pilot_trove.jsonl \
  --stats
