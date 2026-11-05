#!/usr/bin/env bash
# SolverBuilder for a KinDER family — OpenHands coding agent that writes
# SOLVER_KINDER.py + SOLVER_KINDER_ALGORITHM.md.
#
# The agent reads DEMOS.json (per-instance DirectSolve traces) and
# building_prompts/SOLVER_BUILDING_PROMPT_KINDER.md (plus the family's environment
# description), and tests its solver against a simulator of the family through
# execute_code (training instances and fresh instances of any count). No Apptainer image
# is needed: the KinDER domain brings its own sandbox.
#
# Usage (from project root; requires a vLLM server at PORT):
#   FAMILY=obstruction2d bash scripts/run_kinder_solver_builder_openhands.sh <PORT>
#   (or PORT=<PORT> in the environment; default 8000)
#
# The output can then be evaluated with scripts/run_all_kinder_evals.sh.
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PORT=${1:-${PORT:-8000}}
FAMILY=${FAMILY:-obstruction2d}
FAMILY_UC=$(echo "$FAMILY" | tr '[:lower:]' '[:upper:]')

GPU_NODE=${GPU_NODE:-localhost}
BASE_URL=http://${GPU_NODE}:${PORT}/v1
MODEL=${MODEL:-openai/Qwen/Qwen3.6-35B-A3B}  # openai/ prefix required by litellm
MODEL_TAG=${MODEL_TAG:-qwen3.6_35b_a3b}

_DEFAULT_DEMOS=demos/DEMOS_KINDER_${FAMILY_UC}_seed_42_100_examples_with_CoT.json
DEMOS_PATH="${DEMOS_PATH:-${_DEFAULT_DEMOS}}"
REWARDS_DIR=rewards

# Timestamped output dir matching the claude_code naming convention (e.g. Thu_Apr_23_807_PM),
# tagged with the family; DEMOS_TAG encodes a non-default demos file.
TIMESTAMP=$(date +"%a_%b_%-d_%-I%M_%p")
if [[ "${DEMOS_PATH}" != "${_DEFAULT_DEMOS}" ]]; then
    DEMOS_TAG="_$(basename "${DEMOS_PATH}" .json)"
else
    DEMOS_TAG=""
fi
OUTPUT_DIR=built_solvers/${MODEL_TAG}/${TIMESTAMP}_kinder_${FAMILY}${DEMOS_TAG}
DEBUG_DIR=debug_oh_solver_builder/${TIMESTAMP}_kinder_${FAMILY}${DEMOS_TAG}

MAX_STEPS=${MAX_STEPS:-500}
MAX_TOKENS=${MAX_TOKENS:-32768}
# ─────────────────────────────────────────────────────────────────────────────

echo "SolverBuilder run (KinDER ${FAMILY})"
echo "  model      : $MODEL"
echo "  demos      : $DEMOS_PATH"
echo "  output_dir : $OUTPUT_DIR"
echo "  debug_dir  : $DEBUG_DIR"
echo ""

python -m openhands_agents.build_solver \
    --domain "kinder_${FAMILY}" \
    --demos-path "$DEMOS_PATH" \
    --rewards-dir "$REWARDS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --max-steps "$MAX_STEPS" \
    --max-tokens "$MAX_TOKENS" \
    --debug-dir "$DEBUG_DIR"
