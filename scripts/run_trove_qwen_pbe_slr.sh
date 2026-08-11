#!/usr/bin/env bash
# TroVE library-induction baseline for PBEBench / SLR-Bench, on Qwen3.6-35B-A3B
# via OpenHands sandbox (paper-faithful TroVE algorithm; DSL-aware prompts).
#
# ADDITIVE script (new file). Requires a vLLM server serving Qwen3.6-35B-A3B.
#
# Usage:
#   TASK=lite bash scripts/run_trove_qwen_pbe_slr.sh          # PBEBench-Lite
#   TASK=slr  bash scripts/run_trove_qwen_pbe_slr.sh          # SLR-Bench
#   PORT=8000 K=3 WORKERS=8 TASK=lite bash scripts/run_trove_qwen_pbe_slr.sh
set -euo pipefail

TASK=${TASK:-lite}
PORT=${PORT:-8000}
K=${K:-3}                 # candidates per mode (total 3K LLM calls / task)
WORKERS=${WORKERS:-8}
MAX_TOKENS=${MAX_TOKENS:-4096}
# MATCHED=1 -> compute-matched to Qwen3.6-35B-A3B (OpenHands): chain-of-thought ON,
# 16K-token generations, vLLM-default temperature. Combine with a large K to also
# match total token budget. Output is suffixed so the default run is preserved.
MATCHED=${MATCHED:-0}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-120}
GPU_NODE=${GPU_NODE:-localhost}
BASE_URL=${BASE_URL:-http://${GPU_NODE}:${PORT}/v1}
MODEL=${MODEL:-Qwen/Qwen3.6-35B-A3B}
SIF_PATH=${SIF_PATH:-/scratch/$USER/sif_images/sandbox.sif}

case "$TASK" in
  lite)
    TASK_TYPE=pbe;  REWARD=pbebench;  MAX_PROGRAMS=5
    DATASET=data/pbebench/lite_tasks_full_og.jsonl
    OUT=outputs/oh_trove_qwen_lite.jsonl ;;
  hard)
    TASK_TYPE=pbe;  REWARD=pbebench;  MAX_PROGRAMS=20
    DATASET=data/pbebench/tasks_full_og.jsonl
    OUT=outputs/oh_trove_qwen_hard.jsonl ;;
  slr)
    TASK_TYPE=slr;  REWARD=slr_bench; MAX_PROGRAMS=5
    DATASET=data/slr_bench/v1_All_full.jsonl
    OUT=outputs/oh_trove_qwen_slr.jsonl ;;
  *) echo "Unknown TASK=$TASK (use lite|hard|slr)"; exit 1 ;;
esac

MATCHED_ARGS=()
if [ "$MATCHED" = "1" ]; then
    MATCHED_ARGS+=(--enable-thinking)
    # 16K per-call budget, default temperature (matches the OpenHands agent).
    MAX_TOKENS=${MAX_TOKENS_MATCHED:-16384}
    # Suffix output + pkg dir so the default (K=3, no-CoT) results are preserved.
    OUT=${OUT%.jsonl}_matched.jsonl
    SUFFIX=_matched
fi

PKG_DIR=/scratch/$USER/oh_packages_trove_${TASK}${SUFFIX:-}

echo "TroVE (Qwen) | task=$TASK type=$TASK_TYPE reward=$REWARD max_programs=$MAX_PROGRAMS matched=$MATCHED"
echo "  dataset=$DATASET out=$OUT"
echo "  base_url=$BASE_URL model=$MODEL K=$K workers=$WORKERS max_tokens=$MAX_TOKENS pkg_dir=$PKG_DIR"

python -m openhands_agents.run_trove_pbe_slr \
    --framework-task-type "$TASK_TYPE" \
    --dataset-path "$DATASET" \
    --output-path "$OUT" \
    --default-reward "$REWARD" \
    --max-programs "$MAX_PROGRAMS" \
    --sif-path "$SIF_PATH" \
    --pkg-dir "$PKG_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --k "$K" \
    --workers "$WORKERS" \
    --request-timeout "$REQUEST_TIMEOUT" \
    "${MATCHED_ARGS[@]}" \
    --skip-existing
