#!/usr/bin/env bash
# Run all SLR-Bench ensemble evals (standard + effi) and quick_eval.
#
# Ensembles produced:
#   DF, BoK, DF+BoK
#   DF+CC, BoK+CC
#   DF+Qwen, BoK+Qwen
#   DF+CC+Qwen, BoK+CC+Qwen
#   DF+BoK+CC+Qwen
#   ... and effi variants of all solver-including ensembles
#
# Usage:
#   bash scripts/run_all_slr_evals.sh
#   bash scripts/run_all_slr_evals.sh --metrics-json metrics/slr_all.json

set -euo pipefail

METRICS_JSON="metrics/slr_all.json"
for arg in "$@"; do
  case $arg in
    --metrics-json) shift; METRICS_JSON="$1"; shift ;;
    --metrics-json=*) METRICS_JSON="${arg#*=}" ;;
  esac
done

# --- Input files ---
BOK=outputs/slr_bench_best_of_k_stripped.jsonl
DF=outputs/slr_bench_direct_feedback_stripped.jsonl
CC=evals/solver_results/slr_claude_code/slr.jsonl
QWEN=evals/solver_results/slr_qwen3.6_35b_a3b/Sun_Apr_26_131_PM/slr.jsonl
DS=outputs/slr_bench_direct_solve_openhands.jsonl   # DirectSolve baseline (Qwen3.6-35B-A3B OpenHands, may be partial)
TASKS=data/slr_bench/v1_All_full.jsonl
OUT=outputs

ENSEMBLE="python scripts/ensemble_outputs.py"
EVAL="python scripts/quick_eval.py"

echo "=== Building SLR-Bench ensembles ==="

# --- LLM-only (no solver) ---
# DF and BoK standalone are already raw outputs; DF+BoK needs an ensemble
$ENSEMBLE \
    --sources $DF $BOK \
    --out $OUT/slr_ensemble_df_bok.jsonl

# --- Standard ensembles (CC solver) ---
$ENSEMBLE --sources $DF $CC               --out $OUT/slr_ensemble_df_cc.jsonl
$ENSEMBLE --sources $BOK $CC              --out $OUT/slr_ensemble_bok_cc.jsonl
$ENSEMBLE --sources $DF $BOK $CC          --out $OUT/slr_ensemble_df_bok_cc.jsonl

# --- Standard ensembles (OH Qwen solver) ---
$ENSEMBLE --sources $DF $QWEN             --out $OUT/slr_ensemble_df_qwen.jsonl
$ENSEMBLE --sources $BOK $QWEN            --out $OUT/slr_ensemble_bok_qwen.jsonl
$ENSEMBLE --sources $DF $BOK $QWEN        --out $OUT/slr_ensemble_df_bok_qwen.jsonl

# --- Standard ensembles (CC + OH Qwen) ---
$ENSEMBLE --sources $DF $CC $QWEN         --out $OUT/slr_ensemble_df_cc_qwen.jsonl
$ENSEMBLE --sources $BOK $CC $QWEN        --out $OUT/slr_ensemble_bok_cc_qwen.jsonl
$ENSEMBLE --sources $DF $BOK $CC $QWEN    --out $OUT/slr_ensemble_df_bok_cc_qwen.jsonl

# --- Effi ensembles (CC solver) ---
$ENSEMBLE --effi --symbolic $CC \
    --sources $DF                         --out $OUT/slr_ensemble_effi_df_cc.jsonl
$ENSEMBLE --effi --symbolic $CC \
    --sources $BOK                        --out $OUT/slr_ensemble_effi_bok_cc.jsonl
$ENSEMBLE --effi --symbolic $CC \
    --sources $DF $BOK                    --out $OUT/slr_ensemble_effi_df_bok_cc.jsonl

# --- Effi ensembles (OH Qwen solver) ---
$ENSEMBLE --effi --symbolic $QWEN \
    --sources $DF                         --out $OUT/slr_ensemble_effi_df_qwen.jsonl
$ENSEMBLE --effi --symbolic $QWEN \
    --sources $BOK                        --out $OUT/slr_ensemble_effi_bok_qwen.jsonl
$ENSEMBLE --effi --symbolic $QWEN \
    --sources $DF $BOK                    --out $OUT/slr_ensemble_effi_df_bok_qwen.jsonl

# --- Effi ensembles (CC + OH Qwen: use CC as primary symbolic, Qwen as LLM fallback) ---
$ENSEMBLE --effi --symbolic $CC \
    --sources $DF $QWEN                   --out $OUT/slr_ensemble_effi_df_cc_qwen.jsonl
$ENSEMBLE --effi --symbolic $CC \
    --sources $BOK $QWEN                  --out $OUT/slr_ensemble_effi_bok_cc_qwen.jsonl
$ENSEMBLE --effi --symbolic $CC \
    --sources $DF $BOK $QWEN              --out $OUT/slr_ensemble_effi_df_bok_cc_qwen.jsonl

echo ""
echo "=== Running quick_eval ==="

EVAL_ARGS=(
    $CC
    $QWEN
    $DF
    $BOK
    $DS
    $OUT/slr_ensemble_df_bok.jsonl
    $OUT/slr_ensemble_df_cc.jsonl
    $OUT/slr_ensemble_bok_cc.jsonl
    $OUT/slr_ensemble_df_bok_cc.jsonl
    $OUT/slr_ensemble_df_qwen.jsonl
    $OUT/slr_ensemble_bok_qwen.jsonl
    $OUT/slr_ensemble_df_bok_qwen.jsonl
    $OUT/slr_ensemble_df_cc_qwen.jsonl
    $OUT/slr_ensemble_bok_cc_qwen.jsonl
    $OUT/slr_ensemble_df_bok_cc_qwen.jsonl
    $OUT/slr_ensemble_effi_df_cc.jsonl
    $OUT/slr_ensemble_effi_bok_cc.jsonl
    $OUT/slr_ensemble_effi_df_bok_cc.jsonl
    $OUT/slr_ensemble_effi_df_qwen.jsonl
    $OUT/slr_ensemble_effi_bok_qwen.jsonl
    $OUT/slr_ensemble_effi_df_bok_qwen.jsonl
    $OUT/slr_ensemble_effi_df_cc_qwen.jsonl
    $OUT/slr_ensemble_effi_bok_cc_qwen.jsonl
    $OUT/slr_ensemble_effi_df_bok_cc_qwen.jsonl
)

if [[ -n "$METRICS_JSON" ]]; then
    $EVAL "${EVAL_ARGS[@]}" --tasks-file $TASKS --metrics-json "$METRICS_JSON"
else
    $EVAL "${EVAL_ARGS[@]}" --tasks-file $TASKS
fi

echo ""
echo "=== Generating comparison plots ==="

python scripts/plot_slr_comparison.py --plot \
    --df            $DF \
    --bok           $BOK \
    --claude-solver $CC \
    --qwen-solver   $QWEN \
    --direct-solve  $DS \
    --dataset       $TASKS

echo ""
echo "Done. Figures written to figures/slr_comparison_*.png"
