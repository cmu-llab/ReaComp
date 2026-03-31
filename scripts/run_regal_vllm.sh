#!/usr/bin/env bash
# Run ReGAL baseline against a local vLLM server.
# Usage: bash scripts/run_regal_vllm.sh
#
# Two-phase: offline training on pilot data, then test run on the same file.
# For test-only mode (no training), remove --regal-train-file.
#
# Dependencies: pip install sentence_transformers scikit-learn scipy

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p outputs outputs/regal_banks

python main.py \
  --framework         regal \
  --tasks-file        data/interleaved/pbebench_rg_string_pilot.jsonl \
  --base-url          "http://localhost:8002/v1" \
  --model             "openai/gpt-oss-120b" \
  --regal-train-file  data/interleaved/pbebench_rg_string_pilot.jsonl \
  --regal-batch-size  4 \
  --regal-retrieval   sentence_transformers \
  --regal-codebank-dir outputs/regal_banks \
  --regal-icl-budget  10 \
  --regal-icl-split   0.5 \
  --default-reward    pbebench \
  --output-file       outputs/pbebench_rg_string_pilot_regal.jsonl \
  --stats
