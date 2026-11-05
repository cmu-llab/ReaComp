#!/usr/bin/env bash
# Evaluate a KinDER solver on the test split and build the solver-first hybrid.
#
# Inputs: a solver directory (SOLVER_KINDER.py) and, optionally, the DirectSolve baseline
# on the test split (scripts/run_kinder_direct_solve_openhands.sh with SPLIT=test).
#
# Produces:
#   evals/solver_results/<MODEL_TAG>/<run>/test.jsonl        solver rows (zero LLM tokens)
#   outputs/kinder_<family>_test_hybrid_<run>.jsonl           solver first, DirectSolve fallback
#   metrics/kinder_<family>_test_<run>.json                   pass rate by count and regime, tokens
# With KINDER_DATA_DIR=<dir> (another build of the task files) "test" becomes "test_<basename of dir>"
# and BASELINE must be set explicitly for the hybrid.
#
# Usage:
#   FAMILY=obstruction2d bash scripts/run_all_kinder_evals.sh built_solvers/qwen3.6_35b_a3b/<run>
#   FAMILY=obstruction2d WORKERS=16 bash scripts/run_all_kinder_evals.sh built_solvers/claude_code/<run>
#   KINDER_DATA_DIR=<dir> BASELINE=<rows on that suite> FAMILY=obstruction2d bash scripts/run_all_kinder_evals.sh <run dir>
set -euo pipefail

SOLVER_DIR=${1:?usage: FAMILY=<family> bash scripts/run_all_kinder_evals.sh <solver_dir>}
FAMILY=${FAMILY:-obstruction2d}
WORKERS=${WORKERS:-8}
TASK_TIMEOUT=${TASK_TIMEOUT:-600}      # wall clock per instance; policy time is capped at 60 s inside

RUN=$(basename "$SOLVER_DIR")
MODEL_TAG=$(basename "$(dirname "$SOLVER_DIR")")
EVAL_DIR=evals/solver_results/${MODEL_TAG}/${RUN}

# Another test suite (KINDER_DATA_DIR, e.g. robocode's evaluation seed) gets its own file names, so it
# never resumes from or overwrites the public-split results, and its baseline must be given explicitly.
if [ -n "${KINDER_DATA_DIR:-}" ]; then
    SUITE=test_$(basename "$KINDER_DATA_DIR")
    BASELINE=${BASELINE:-}
else
    SUITE=test
    BASELINE=${BASELINE:-outputs/kinder_${FAMILY}_test_direct_solve_openhands.jsonl}
fi
HYBRID=outputs/kinder_${FAMILY}_${SUITE}_hybrid_${RUN}.jsonl

echo "=== Solver: ${SOLVER_DIR} on kinder_${FAMILY}/${SUITE} ==="
python scripts/eval_domain_solver.py --domain "kinder_${FAMILY}" --split test --tag "$SUITE" \
    --solver "${SOLVER_DIR}/SOLVER_KINDER.py" --workers "$WORKERS" --task-timeout "$TASK_TIMEOUT" \
    --output-dir "$EVAL_DIR"

REPORT_FILES=("${EVAL_DIR}/${SUITE}.jsonl")
if [ -n "$BASELINE" ] && [ -f "$BASELINE" ]; then
    echo "=== Hybrid: solver first, DirectSolve fallback ==="
    python scripts/hybrid_gate.py --domain "kinder_${FAMILY}" --solver "${EVAL_DIR}/${SUITE}.jsonl" \
        --fallback "$BASELINE" --out "$HYBRID"
    REPORT_FILES+=("$BASELINE" "$HYBRID")
else
    echo "(no DirectSolve baseline for this suite${BASELINE:+ at $BASELINE}; skipping the hybrid; set BASELINE=...)"
fi

echo "=== Report ==="
python scripts/kinder_report.py "${REPORT_FILES[@]}" --metrics-json "metrics/kinder_${FAMILY}_${SUITE}_${RUN}.json"
