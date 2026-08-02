#!/usr/bin/env bash
# Run TroVE baseline against a local vLLM server (gpt-oss-20b).
# Usage: bash scripts/run_trove_vllm.sh
#
# Defaults to PBEBench-Lite pilot (50 tasks). Override TASKS_FILE or pass
# extra --flags through the trailing "$@".
#
# For small datasets (<=100 tasks), --trove-trim-every is set high to disable
# trimming (the library never gets large enough for it to matter).
# Set --trove-k 1 for a cheaper run without per-mode K-sampling.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PORT="${PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
mkdir -p outputs

TASKS_FILE="${TASKS_FILE:-data/pbebench/lite_pilot_tasks.jsonl}"
OUT_FILE="${OUT_FILE:-outputs/trove_pbebench_lite_pilot.jsonl}"

echo "Tasks  : ${TASKS_FILE}"
echo "Output : ${OUT_FILE}"
echo "Port   : ${PORT}"

python main.py \
  --framework         trove \
  --tasks-file        "${TASKS_FILE}" \
  --base-url          "http://localhost:${PORT}/v1" \
  --model             "openai/gpt-oss-20b" \
  --trove-task-family pbebench \
  --trove-selection   reward \
  --trove-k           5 \
  --trove-trim-every  9999 \
  --default-reward    pbebench \
  --max-programs      5 \
  --output-file       "${OUT_FILE}" \
  --debug-dir         debug_trove \
  --stats \
  "$@"

echo "Done. Output: ${OUT_FILE}"
echo "Analyze with: python scripts/analyze_trove_run.py ${OUT_FILE}"
