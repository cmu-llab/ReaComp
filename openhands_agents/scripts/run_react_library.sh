#!/usr/bin/env bash
# Run ReAct+Library baseline (OpenHands SDK + Apptainer sandbox).
set -euo pipefail

DATASET="${DATASET:-data/interleaved/pbebench_rg_string_pilot.jsonl}"
OUTPUT="${OUTPUT:-outputs/oh_react_library.jsonl}"
SIF_PATH="${SANDBOX_SIF:-/scratch/$USER/sif_images/sandbox.sif}"
PKG_DIR="${PKG_DIR:-/scratch/$USER/oh_packages}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-480B-A22B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
REWARD="${REWARD:-reasoning_gym}"
MAX_STEPS="${MAX_STEPS:-8}"
MAX_REWARD_ITERS="${MAX_REWARD_ITERS:-3}"
LIBRARY_K="${LIBRARY_K:-5}"

python -m openhands_agents.run \
    --framework react_library \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --sif-path "$SIF_PATH" \
    --pkg-dir "$PKG_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --default-reward "$REWARD" \
    --max-steps "$MAX_STEPS" \
    --max-reward-iters "$MAX_REWARD_ITERS" \
    --library-k "$LIBRARY_K" \
    --skip-existing
