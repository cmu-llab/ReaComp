#!/usr/bin/env bash
# Best-of-K baseline — fully async two-stage pipeline with Apptainer sandbox.
# Edit the variables below, then run from the project root:
#   bash scripts/run_best_of_k_openhands.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DATASET=data/interleaved/pbebench_rg_string_pilot.jsonl
OUTPUT=outputs/oh_best_of_k.jsonl
REWARD=reasoning_gym

GPU_NODE=localhost          # hostname of the node running vLLM
BASE_URL=http://${GPU_NODE}:8000/v1
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct

SIF_PATH=/scratch/$USER/sif_images/sandbox.sif

MAX_TOKENS=4096
K=8               # samples per task
TEMPERATURE=0.8
MAX_CONCURRENT=64 # async concurrency cap (tune to vLLM throughput)
# ─────────────────────────────────────────────────────────────────────────────

python -m openhands_agents.run \
    --framework best_of_k \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --default-reward "$REWARD" \
    --sif-path "$SIF_PATH" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --k "$K" \
    --temperature "$TEMPERATURE" \
    --max-concurrent "$MAX_CONCURRENT" \
    --skip-existing
