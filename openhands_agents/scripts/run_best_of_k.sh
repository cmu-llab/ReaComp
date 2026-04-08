#!/usr/bin/env bash
# Run Best-of-K baseline (async vLLM, two-stage: generate then score).
set -euo pipefail

DATASET="${DATASET:-data/interleaved/pbebench_rg_string_pilot.jsonl}"
OUTPUT="${OUTPUT:-outputs/oh_best_of_k.jsonl}"
SIF_PATH="${SANDBOX_SIF:-/scratch/$USER/sif_images/sandbox.sif}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-480B-A22B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
REWARD="${REWARD:-reasoning_gym}"
K="${K:-8}"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"

python -m openhands_agents.run \
    --framework best_of_k \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --sif-path "$SIF_PATH" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --default-reward "$REWARD" \
    --k "$K" \
    --max-concurrent "$MAX_CONCURRENT" \
    --temperature 0.8 \
    --skip-existing
