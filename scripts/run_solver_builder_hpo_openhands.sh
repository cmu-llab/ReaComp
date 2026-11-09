#!/usr/bin/env bash
# Autoresearch HPO SolverBuilder — OpenHands coding agent that writes SOLVER_HPO.py +
# SOLVER_HPO_ALGORITHM.md.
#
# The agent reads the HPO demos (decisions of the LLM HPO agent with their reasoning
# and outcome) and building_prompts/SOLVER_BUILDING_PROMPT_HPO.md, then implements a
# symbolic HPO policy solve_hpo(task, history) using only the Python stdlib. It can run
# its solver on the synthetic practice tasks inside the sandbox; no GPU is needed for
# this step apart from the model server.
#
# Usage (from project root):
#   bash scripts/run_solver_builder_hpo_openhands.sh <PORT>
#   DEMOS_PATH=demos/<other demos>.json bash scripts/run_solver_builder_hpo_openhands.sh <PORT>
#
# The output SOLVER_HPO.py can then be evaluated with scripts/run_hpo_eval.sh.
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
USER="${USER:-$(whoami)}"
PORT=${1:-8000}

GPU_NODE=localhost
BASE_URL=http://${GPU_NODE}:${PORT}/v1
MODEL=openai/Qwen/Qwen3.6-35B-A3B  # openai/ prefix required by litellm

SIF_PATH=${SIF_PATH:-/scratch/$USER/sif_images/sandbox.sif}

BUILDING_PROMPT=building_prompts/SOLVER_BUILDING_PROMPT_HPO.md
_DEFAULT_DEMOS=demos/DEMOS_AUTORESEARCH_HPO_seed_42_100_examples_with_CoT.json
DEMOS_PATH="${DEMOS_PATH:-${_DEFAULT_DEMOS}}"
SMOKE_TASKS=data/autoresearch_hpo/synthetic_tasks.jsonl

REWARDS_DIR=rewards

# Timestamped output dir matching the claude_code naming convention (e.g. Thu_Apr_23_807_PM)
# DEMOS_TAG encodes the demos variant (basename without extension) for non-default files.
TIMESTAMP=$(date +"%a_%b_%-d_%-I%M_%p")
if [[ "${DEMOS_PATH}" != "${_DEFAULT_DEMOS}" ]]; then
    DEMOS_TAG="_$(basename "${DEMOS_PATH}" .json)"
else
    DEMOS_TAG=""
fi
OUTPUT_DIR=built_solvers/qwen3.6_35b_a3b/${TIMESTAMP}_hpo${DEMOS_TAG}
DEBUG_DIR=debug_oh_solver_builder/${TIMESTAMP}_hpo${DEMOS_TAG}

MAX_STEPS=500
MAX_TOKENS=32768
SANDBOX_TIMEOUT=120   # seconds per execute call of the agent
# ─────────────────────────────────────────────────────────────────────────────

echo "Autoresearch HPO SolverBuilder run"
echo "  model      : $MODEL"
echo "  demos      : $DEMOS_PATH"
echo "  output_dir : $OUTPUT_DIR"
echo "  debug_dir  : $DEBUG_DIR"
echo ""

python -m openhands_agents.build_solver_hpo \
    --building-prompt "$BUILDING_PROMPT" \
    --demos-path "$DEMOS_PATH" \
    --rewards-dir "$REWARDS_DIR" \
    --smoke-tasks "$SMOKE_TASKS" \
    --output-dir "$OUTPUT_DIR" \
    --sif-path "$SIF_PATH" \
    --sandbox-timeout "$SANDBOX_TIMEOUT" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-steps "$MAX_STEPS" \
    --max-tokens "$MAX_TOKENS" \
    --debug-dir "$DEBUG_DIR"
