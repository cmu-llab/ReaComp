#!/usr/bin/env bash
# Run all PBEBench-Hard ensemble evals and generate comparison plots.
#
# Note: DF (gpt-oss-120b) is NOT available for PBEBench-Hard due to compute cost.
#       Only BoK (gpt-oss-120b), CC Solver, and OH Qwen Solver are included.
#
# Individual systems:
#   BoK (gpt-oss-120b), CC Solver, OH Qwen Solver
#
# Ensembles produced (standard + effi):
#   BoK+CC, BoK+Qwen, BoK+CC+Qwen
#   ... and effi variants
#
# Figures produced (in figures/):
#   pbebench_hard_comparison_passrate.png
#   pbebench_hard_comparison_meanreward.png
#   pbebench_hard_comparison_complexity.png
#
# Usage:
#   bash scripts/run_all_pbebench_hard_evals.sh
#   bash scripts/run_all_pbebench_hard_evals.sh --metrics-json metrics/pbebench_hard_all.json

set -euo pipefail

METRICS_JSON="metrics/pbebench_hard_all.json"
for arg in "$@"; do
  case $arg in
    --metrics-json=*) METRICS_JSON="${arg#*=}" ;;
    --metrics-json)   shift; METRICS_JSON="$1"; shift ;;
  esac
done

# --- Input files ---
BOK=outputs/hard_bok_converted.jsonl
CC=evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl
QWEN=evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/hard.jsonl
TASKS=data/pbebench/tasks_full_og.jsonl
OUT=outputs

ENSEMBLE="python scripts/ensemble_outputs.py"
EVAL="python scripts/quick_eval.py"

echo "=== Building PBEBench-Hard ensembles ==="

# --- Standard ensembles ---
$ENSEMBLE --sources $BOK $CC          --out $OUT/hard_ensemble_bok_cc.jsonl
$ENSEMBLE --sources $BOK $QWEN        --out $OUT/hard_ensemble_bok_qwen.jsonl
$ENSEMBLE --sources $BOK $CC $QWEN    --out $OUT/hard_ensemble_bok_cc_qwen.jsonl

# --- Effi ensembles (CC solver) ---
$ENSEMBLE --effi --symbolic $CC \
    --sources $BOK                     --out $OUT/hard_ensemble_effi_bok_cc.jsonl

# --- Effi ensembles (OH Qwen solver) ---
$ENSEMBLE --effi --symbolic $QWEN \
    --sources $BOK                     --out $OUT/hard_ensemble_effi_bok_qwen.jsonl

# --- Effi ensembles (CC + OH Qwen: CC as primary symbolic) ---
$ENSEMBLE --effi --symbolic $CC \
    --sources $BOK $QWEN               --out $OUT/hard_ensemble_effi_bok_cc_qwen.jsonl

echo ""
echo "=== Running quick_eval ==="

EVAL_ARGS=(
    $CC
    $QWEN
    $BOK
    $OUT/hard_ensemble_bok_cc.jsonl
    $OUT/hard_ensemble_bok_qwen.jsonl
    $OUT/hard_ensemble_bok_cc_qwen.jsonl
    $OUT/hard_ensemble_effi_bok_cc.jsonl
    $OUT/hard_ensemble_effi_bok_qwen.jsonl
    $OUT/hard_ensemble_effi_bok_cc_qwen.jsonl
)

if [[ -n "$METRICS_JSON" ]]; then
    $EVAL "${EVAL_ARGS[@]}" --tasks-file $TASKS --metrics-json "$METRICS_JSON"
else
    $EVAL "${EVAL_ARGS[@]}" --tasks-file $TASKS
fi

echo ""
echo "=== Generating comparison plots ==="

PLOT_ARGS=(--split hard --plot
    --bok  $BOK
    --cc-solver   $CC
    --qwen-solver $QWEN
    --tasks $TASKS
)
if [[ -n "$METRICS_JSON" ]]; then
    python scripts/plot_pbebench_comparison.py "${PLOT_ARGS[@]}" \
        --metrics-json "${METRICS_JSON%.json}_plot.json"
else
    python scripts/plot_pbebench_comparison.py "${PLOT_ARGS[@]}"
fi

echo ""
echo "Done. Figures written to figures/pbebench_hard_comparison_*.png"
