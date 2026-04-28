#!/usr/bin/env bash
# Run all PBEBench-Lite ensemble evals and generate comparison plots.
#
# Individual systems:
#   DF (gpt-oss-120b), BoK (gpt-oss-120b), CC Solver, OH Qwen Solver
#
# Ensembles produced (standard + effi):
#   DF+BoK, DF+CC, BoK+CC, DF+BoK+CC
#   DF+Qwen, BoK+Qwen, DF+BoK+Qwen
#   DF+CC+Qwen, BoK+CC+Qwen, DF+BoK+CC+Qwen
#   ... and effi variants of all solver-including ensembles
#
# Figures produced (in figures/):
#   pbebench_lite_comparison_passrate.png
#   pbebench_lite_comparison_meanreward.png
#   pbebench_lite_comparison_complexity.png
#
# Usage:
#   bash scripts/run_all_pbebench_lite_evals.sh
#   bash scripts/run_all_pbebench_lite_evals.sh --metrics-json metrics/pbebench_lite_all.json

set -euo pipefail

METRICS_JSON="metrics/pbebench_lite_all.json"
for arg in "$@"; do
  case $arg in
    --metrics-json=*) METRICS_JSON="${arg#*=}" ;;
    --metrics-json)   shift; METRICS_JSON="$1"; shift ;;
  esac
done

# --- Input files ---
DF=outputs/lite_tasks_full_og_direct_feedback_stripped.jsonl
BOK=outputs/lite_tasks_full_og_best_of_k_stripped.jsonl
CC=evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl
QWEN=evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/lite.jsonl
TASKS=data/pbebench/lite_tasks_full_og.jsonl
OUT=outputs

ENSEMBLE="python scripts/ensemble_outputs.py"
EVAL="python scripts/quick_eval.py"

echo "=== Building PBEBench-Lite ensembles ==="

# --- LLM-only ---
$ENSEMBLE \
    --sources $DF $BOK \
    --out $OUT/lite_ensemble_df_bok.jsonl

# --- Standard ensembles (CC solver) ---
$ENSEMBLE --sources $DF  $CC          --out $OUT/lite_ensemble_df_cc.jsonl
$ENSEMBLE --sources $BOK $CC          --out $OUT/lite_ensemble_bok_cc.jsonl
$ENSEMBLE --sources $DF $BOK $CC      --out $OUT/lite_ensemble_df_bok_cc.jsonl

# --- Standard ensembles (OH Qwen solver) ---
$ENSEMBLE --sources $DF  $QWEN        --out $OUT/lite_ensemble_df_qwen.jsonl
$ENSEMBLE --sources $BOK $QWEN        --out $OUT/lite_ensemble_bok_qwen.jsonl
$ENSEMBLE --sources $DF $BOK $QWEN    --out $OUT/lite_ensemble_df_bok_qwen.jsonl

# --- Standard ensembles (CC + OH Qwen) ---
$ENSEMBLE --sources $DF  $CC $QWEN         --out $OUT/lite_ensemble_df_cc_qwen.jsonl
$ENSEMBLE --sources $BOK $CC $QWEN         --out $OUT/lite_ensemble_bok_cc_qwen.jsonl
$ENSEMBLE --sources $DF $BOK $CC $QWEN     --out $OUT/lite_ensemble_df_bok_cc_qwen.jsonl

# --- Effi ensembles (CC solver) ---
$ENSEMBLE --effi --symbolic $CC \
    --sources $DF                      --out $OUT/lite_ensemble_effi_df_cc.jsonl
$ENSEMBLE --effi --symbolic $CC \
    --sources $BOK                     --out $OUT/lite_ensemble_effi_bok_cc.jsonl
$ENSEMBLE --effi --symbolic $CC \
    --sources $DF $BOK                 --out $OUT/lite_ensemble_effi_df_bok_cc.jsonl

# --- Effi ensembles (OH Qwen solver) ---
$ENSEMBLE --effi --symbolic $QWEN \
    --sources $DF                      --out $OUT/lite_ensemble_effi_df_qwen.jsonl
$ENSEMBLE --effi --symbolic $QWEN \
    --sources $BOK                     --out $OUT/lite_ensemble_effi_bok_qwen.jsonl
$ENSEMBLE --effi --symbolic $QWEN \
    --sources $DF $BOK                 --out $OUT/lite_ensemble_effi_df_bok_qwen.jsonl

# --- Effi ensembles (CC + OH Qwen: CC as primary symbolic) ---
$ENSEMBLE --effi --symbolic $CC \
    --sources $DF $QWEN                --out $OUT/lite_ensemble_effi_df_cc_qwen.jsonl
$ENSEMBLE --effi --symbolic $CC \
    --sources $BOK $QWEN               --out $OUT/lite_ensemble_effi_bok_cc_qwen.jsonl
$ENSEMBLE --effi --symbolic $CC \
    --sources $DF $BOK $QWEN           --out $OUT/lite_ensemble_effi_df_bok_cc_qwen.jsonl

echo ""
echo "=== Running quick_eval ==="

EVAL_ARGS=(
    $CC
    $QWEN
    $DF
    $BOK
    $OUT/lite_ensemble_df_bok.jsonl
    $OUT/lite_ensemble_df_cc.jsonl
    $OUT/lite_ensemble_bok_cc.jsonl
    $OUT/lite_ensemble_df_bok_cc.jsonl
    $OUT/lite_ensemble_df_qwen.jsonl
    $OUT/lite_ensemble_bok_qwen.jsonl
    $OUT/lite_ensemble_df_bok_qwen.jsonl
    $OUT/lite_ensemble_df_cc_qwen.jsonl
    $OUT/lite_ensemble_bok_cc_qwen.jsonl
    $OUT/lite_ensemble_df_bok_cc_qwen.jsonl
    $OUT/lite_ensemble_effi_df_cc.jsonl
    $OUT/lite_ensemble_effi_bok_cc.jsonl
    $OUT/lite_ensemble_effi_df_bok_cc.jsonl
    $OUT/lite_ensemble_effi_df_qwen.jsonl
    $OUT/lite_ensemble_effi_bok_qwen.jsonl
    $OUT/lite_ensemble_effi_df_bok_qwen.jsonl
    $OUT/lite_ensemble_effi_df_cc_qwen.jsonl
    $OUT/lite_ensemble_effi_bok_cc_qwen.jsonl
    $OUT/lite_ensemble_effi_df_bok_cc_qwen.jsonl
)

if [[ -n "$METRICS_JSON" ]]; then
    $EVAL "${EVAL_ARGS[@]}" --tasks-file $TASKS --metrics-json "$METRICS_JSON"
else
    $EVAL "${EVAL_ARGS[@]}" --tasks-file $TASKS
fi

echo ""
echo "=== Generating comparison plots ==="

PLOT_ARGS=(--split lite --plot
    --df   $DF
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
echo "Done. Figures written to figures/pbebench_lite_comparison_*.png"
