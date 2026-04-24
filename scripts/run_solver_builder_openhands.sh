#!/usr/bin/env bash
# SolverBuilder — OpenHands coding agent that writes SOLVER.py + SOLVER_ALGORITHM.md.
#
# The agent reads DEMOS.json and building_prompts/SOLVER_BUILDING_PROMPT.md,
# then implements a symbolic PBE solver using only Python stdlib.
#
# Usage (from project root):
#   bash scripts/run_solver_builder_openhands.sh <PORT>
#
# The output SOLVER.py can then be evaluated with scripts/eval_solver.py.
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
USER="${USER:-$(whoami)}"
PORT=${1:-8000}

GPU_NODE=localhost
BASE_URL=http://${GPU_NODE}:${PORT}/v1
MODEL=openai/Qwen/Qwen3-Coder-30B-A3B-Instruct  # openai/ prefix required by litellm

SIF_PATH=/scratch/$USER/sif_images/sandbox.sif

BUILDING_PROMPT=building_prompts/SOLVER_BUILDING_PROMPT.md
DEMOS_PATH=DEMOS.json
REWARDS_DIR=rewards

# Timestamped output dir so reruns don't overwrite each other
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=built_solvers/oh_solver_${TIMESTAMP}
DEBUG_DIR=debug_oh_solver_builder/${TIMESTAMP}

MAX_STEPS=200
MAX_TOKENS=16384
# ─────────────────────────────────────────────────────────────────────────────

echo "SolverBuilder run"
echo "  model      : $MODEL"
echo "  output_dir : $OUTPUT_DIR"
echo "  debug_dir  : $DEBUG_DIR"
echo ""

python -m openhands_agents.build_solver \
    --building-prompt "$BUILDING_PROMPT" \
    --demos-path "$DEMOS_PATH" \
    --rewards-dir "$REWARDS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --sif-path "$SIF_PATH" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-steps "$MAX_STEPS" \
    --max-tokens "$MAX_TOKENS" \
    --debug-dir "$DEBUG_DIR"
