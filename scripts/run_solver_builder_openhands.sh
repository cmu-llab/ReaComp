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
MODEL=openai/Qwen/Qwen3.6-35B-A3B  # openai/ prefix required by litellm

SIF_PATH=/scratch/$USER/sif_images/sandbox.sif

BUILDING_PROMPT=building_prompts/SOLVER_BUILDING_PROMPT.md
DEMOS_PATH=DEMOS.json
REWARDS_DIR=rewards

# Timestamped output dir matching the claude_code naming convention (e.g. Thu_Apr_23_807_PM)
TIMESTAMP=$(date +"%a_%b_%-d_%-I%M_%p")
OUTPUT_DIR=built_solvers/qwen3.6_coder/${TIMESTAMP}
DEBUG_DIR=debug_oh_solver_builder/${TIMESTAMP}

MAX_STEPS=500
MAX_TOKENS=32768
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
