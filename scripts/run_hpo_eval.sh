#!/usr/bin/env bash
# Autoresearch HPO — evaluate an induced SOLVER_HPO.py or a classical baseline.
#
# One episode per task record on the real objective: every trial trains nanochat for
# 300 s on the local GPU, so the pilot task (1 h) takes a bit over an hour and each
# task of tasks_full.jsonl 24 h. No LLM is involved. Re-running the same command
# resumes unfinished episodes.
#
# Prerequisite: bash scripts/setup_autoresearch_hpo.sh
#
# Usage (from project root):
#   SOLVER=built_solvers/claude_code/<timestamp>/SOLVER_HPO.py bash scripts/run_hpo_eval.sh
#   POLICY=tpe bash scripts/run_hpo_eval.sh              # random | tpe | cmaes
#
# The paper's protocol (24 h, seeds 0-2), one seed per GPU:
#   for s in 0 1 2; do
#     TRAIN_GPU=$s TASKS=data/autoresearch_hpo/tasks_full.jsonl SOLVER=... \
#       bash scripts/run_hpo_eval.sh --task-id autoresearch_hpo_24h_seed$s &
#   done; wait
#
# Dry run without a GPU (synthetic objective, NOT benchmark numbers):
#   POLICY=random TASKS=data/autoresearch_hpo/synthetic_tasks.jsonl bash scripts/run_hpo_eval.sh
#
# Results table: python scripts/summarize_hpo_results.py <output dirs>
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
TASKS=${TASKS:-data/autoresearch_hpo/pilot_tasks.jsonl}

# Exactly one of the two:
SOLVER=${SOLVER:-}   # path to a SOLVER_HPO.py
POLICY=${POLICY:-}   # random | tpe | cmaes  (tpe, cmaes: pip install optuna cmaes)

# GPU index for the training runs (empty = whatever CUDA_VISIBLE_DEVICES already says)
TRAIN_GPU=${TRAIN_GPU:-}

TASKS_TAG=autoresearch_hpo_$(basename "${TASKS}" .jsonl)
if [ -n "${SOLVER}" ] && [ -z "${POLICY}" ]; then
    # built_solvers/<model>/<run>/SOLVER_HPO.py -> evals/solver_results/<model>/<run>/<tasks>
    SOLVER_DIR=$(cd "$(dirname "${SOLVER}")" && pwd)
    DEFAULT_OUTPUT_DIR=evals/solver_results/$(basename "$(dirname "${SOLVER_DIR}")")/$(basename "${SOLVER_DIR}")/${TASKS_TAG}
    POLICY_ARGS=(--solver "${SOLVER}")
elif [ -n "${POLICY}" ] && [ -z "${SOLVER}" ]; then
    DEFAULT_OUTPUT_DIR=outputs/${TASKS_TAG}_${POLICY}
    POLICY_ARGS=(--policy "${POLICY}")
else
    echo "Set exactly one of SOLVER=<path to SOLVER_HPO.py> or POLICY=<random|tpe|cmaes>." >&2
    exit 2
fi
OUTPUT_DIR=${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}
# ─────────────────────────────────────────────────────────────────────────────

[ -n "${TRAIN_GPU}" ] && export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"

echo "Autoresearch HPO evaluation"
echo "  tasks      : $TASKS"
echo "  policy     : ${SOLVER:-$POLICY}"
echo "  train GPU  : ${CUDA_VISIBLE_DEVICES:-<all visible>}"
echo "  output_dir : $OUTPUT_DIR"
echo ""

python scripts/eval_hpo_solver.py \
    "${POLICY_ARGS[@]}" \
    --tasks-file "$TASKS" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
