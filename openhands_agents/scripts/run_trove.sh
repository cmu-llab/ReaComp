#!/usr/bin/env bash
# Run TroVE baseline with sandboxed execution.
set -euo pipefail

DATASET="${DATASET:-data/interleaved/pbebench_rg_string_pilot.jsonl}"
OUTPUT="${OUTPUT:-outputs/oh_trove.jsonl}"
SIF_PATH="${SANDBOX_SIF:-/scratch/$USER/sif_images/sandbox.sif}"
PKG_DIR="${PKG_DIR:-/scratch/$USER/oh_packages}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-480B-A22B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
REWARD="${REWARD:-reasoning_gym}"
K="${K:-5}"
TRIM_EVERY="${TRIM_EVERY:-200}"

python -m openhands_agents.run \
    --framework trove \
    --dataset-path "$DATASET" \
    --output-path "$OUTPUT" \
    --sif-path "$SIF_PATH" \
    --pkg-dir "$PKG_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --default-reward "$REWARD" \
    --k "$K" \
    --trim-every "$TRIM_EVERY" \
    --skip-existing
