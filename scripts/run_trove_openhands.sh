#!/usr/bin/env bash
# TroVE baseline — paper-faithful rewrite with Apptainer sandbox.
# Edit the variables below, then run from the project root:
#   bash scripts/run_trove_openhands.sh
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DATASET=data/interleaved/pbebench_rg_string_pilot.jsonl
OUTPUT=outputs/oh_trove.jsonl
REWARD=reasoning_gym

GPU_NODE=localhost          # hostname of the node running vLLM
BASE_URL=http://${GPU_NODE}:8000/v1
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct

SIF_PATH=/scratch/$USER/sif_images/sandbox.sif
PKG_DIR=/scratch/$USER/oh_packages

MAX_TOKENS=4096
K=5               # candidates per mode (total = 3×K per task)
TRIM_EVERY=200    # trim toolbox every N tasks
# ─────────────────────────────────────────────────────────────────────────────

python -m openhands_agents.run \
    --framework trove \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --default-reward "$REWARD" \
    --sif-path "$SIF_PATH" \
    --pkg-dir "$PKG_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --k "$K" \
    --trim-every "$TRIM_EVERY" \
    --skip-existing
