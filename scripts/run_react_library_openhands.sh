#!/usr/bin/env bash
# ReAct + Library baseline — OpenHands SDK + Apptainer sandbox.
# Edit the variables below, then run from the project root:
#   bash scripts/run_react_library_openhands.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DATASET=data/interleaved/pbebench_rg_string_pilot.jsonl
OUTPUT=outputs/oh_react_library.jsonl
REWARD=reasoning_gym

GPU_NODE=localhost          # hostname of the node running vLLM
BASE_URL=http://${GPU_NODE}:8000/v1
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct

SIF_PATH=/scratch/$USER/sif_images/sandbox.sif
PKG_DIR=/scratch/$USER/oh_packages

MAX_TOKENS=4096
MAX_STEPS=8           # max agent steps per conversation
MAX_REWARD_ITERS=3    # outer reward-feedback loop iterations
LIBRARY_K=5           # BM25 top-k library functions shown per step
# ─────────────────────────────────────────────────────────────────────────────

python -m openhands_agents.run \
    --framework react_library \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --default-reward "$REWARD" \
    --sif-path "$SIF_PATH" \
    --pkg-dir "$PKG_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --max-steps "$MAX_STEPS" \
    --max-reward-iters "$MAX_REWARD_ITERS" \
    --library-k "$LIBRARY_K" \
    --skip-existing
