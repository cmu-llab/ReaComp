#!/usr/bin/env bash
# Run all PBEBench-Lite ensemble combinations.
#
# Two strategies:
#   standard  — pick best by reward, break ties by complexity, then source order
#   effi      — if symbolic solver is perfect use it (zero LLM tokens);
#               otherwise fall back to best LLM by reward then complexity
#
# Usage (from project root):
#   bash scripts/run_ensembles.sh            # run everything
#   bash scripts/run_ensembles.sh standard   # standard only
#   bash scripts/run_ensembles.sh effi       # effi only
#
# LLM output files (large, live on the cluster):
#   outputs/lite_tasks_full_og_best_of_k.jsonl
#   outputs/lite_tasks_full_og_direct_feedback.jsonl
#
# Symbolic solver results (local):
#   evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl
#   evals/solver_results/qwen3.6_coder/Fri_Apr_24_200_AM/lite.jsonl

set -euo pipefail

MODE=${1:-all}

CLAUDE_SOLVER=evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl
QWEN_SOLVER=evals/solver_results/qwen3.6_coder/Fri_Apr_24_200_AM/lite.jsonl
BOK=outputs/lite_tasks_full_og_best_of_k.jsonl
DF=outputs/lite_tasks_full_og_direct_feedback.jsonl

run_standard() {
    echo "=== Standard ensembles ==="

    # Symbolic-only
    python scripts/ensemble_outputs.py \
        --sources "$CLAUDE_SOLVER" "$QWEN_SOLVER" \
        --out outputs/ensemble_claude_solver_qwen_solver.jsonl

    # BoK + solver
    python scripts/ensemble_outputs.py \
        --sources "$BOK" "$CLAUDE_SOLVER" \
        --out outputs/ensemble_bok_claude_solver.jsonl

    python scripts/ensemble_outputs.py \
        --sources "$BOK" "$QWEN_SOLVER" \
        --out outputs/ensemble_bok_qwen_solver.jsonl

    python scripts/ensemble_outputs.py \
        --sources "$BOK" "$CLAUDE_SOLVER" "$QWEN_SOLVER" \
        --out outputs/ensemble_bok_claude_solver_qwen_solver.jsonl

    # DF + solver
    python scripts/ensemble_outputs.py \
        --sources "$DF" "$CLAUDE_SOLVER" \
        --out outputs/ensemble_df_claude_solver.jsonl

    python scripts/ensemble_outputs.py \
        --sources "$DF" "$QWEN_SOLVER" \
        --out outputs/ensemble_df_qwen_solver.jsonl

    python scripts/ensemble_outputs.py \
        --sources "$DF" "$CLAUDE_SOLVER" "$QWEN_SOLVER" \
        --out outputs/ensemble_df_claude_solver_qwen_solver.jsonl

    # DF + BoK + solver
    python scripts/ensemble_outputs.py \
        --sources "$DF" "$BOK" "$CLAUDE_SOLVER" \
        --out outputs/ensemble_df_bok_claude_solver.jsonl

    python scripts/ensemble_outputs.py \
        --sources "$DF" "$BOK" "$QWEN_SOLVER" \
        --out outputs/ensemble_df_bok_qwen_solver.jsonl

    python scripts/ensemble_outputs.py \
        --sources "$DF" "$BOK" "$CLAUDE_SOLVER" "$QWEN_SOLVER" \
        --out outputs/ensemble_df_bok_claude_solver_qwen_solver.jsonl
}

run_effi() {
    echo "=== Efficiency ensembles ==="

    # BoK + solver (effi)
    python scripts/ensemble_outputs.py --effi \
        --symbolic "$CLAUDE_SOLVER" \
        --sources "$BOK" \
        --out outputs/ensemble_effi_bok_claude_solver.jsonl

    python scripts/ensemble_outputs.py --effi \
        --symbolic "$QWEN_SOLVER" \
        --sources "$BOK" \
        --out outputs/ensemble_effi_bok_qwen_solver.jsonl

    python scripts/ensemble_outputs.py --effi \
        --symbolic "$CLAUDE_SOLVER" \
        --sources "$BOK" "$QWEN_SOLVER" \
        --out outputs/ensemble_effi_bok_claude_solver_qwen_solver.jsonl

    # DF + solver (effi)
    python scripts/ensemble_outputs.py --effi \
        --symbolic "$CLAUDE_SOLVER" \
        --sources "$DF" \
        --out outputs/ensemble_effi_df_claude_solver.jsonl

    python scripts/ensemble_outputs.py --effi \
        --symbolic "$QWEN_SOLVER" \
        --sources "$DF" \
        --out outputs/ensemble_effi_df_qwen_solver.jsonl

    python scripts/ensemble_outputs.py --effi \
        --symbolic "$CLAUDE_SOLVER" \
        --sources "$DF" "$QWEN_SOLVER" \
        --out outputs/ensemble_effi_df_claude_solver_qwen_solver.jsonl

    # DF + BoK + solver (effi)
    python scripts/ensemble_outputs.py --effi \
        --symbolic "$CLAUDE_SOLVER" \
        --sources "$DF" "$BOK" \
        --out outputs/ensemble_effi_df_bok_claude_solver.jsonl

    python scripts/ensemble_outputs.py --effi \
        --symbolic "$QWEN_SOLVER" \
        --sources "$DF" "$BOK" \
        --out outputs/ensemble_effi_df_bok_qwen_solver.jsonl

    python scripts/ensemble_outputs.py --effi \
        --symbolic "$CLAUDE_SOLVER" \
        --sources "$DF" "$BOK" "$QWEN_SOLVER" \
        --out outputs/ensemble_effi_df_bok_claude_solver_qwen_solver.jsonl
}

case "$MODE" in
    standard) run_standard ;;
    effi)     run_effi ;;
    all)      run_standard; run_effi ;;
    *) echo "Usage: $0 [all|standard|effi]"; exit 1 ;;
esac

echo ""
echo "Done."
