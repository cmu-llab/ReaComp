#!/usr/bin/env bash
# run_eval.sh — run eval.py + plot_eval.py on one or more output JSONL files.
#
# Usage:
#   bash scripts/run_eval.sh                                   # all outputs/*.jsonl
#   bash scripts/run_eval.sh outputs/pbebench_lite*.jsonl      # specific file(s)
#   bash scripts/run_eval.sh outputs/pbebench*.jsonl --csv results/eval.csv
#
# All extra arguments after the file globs are forwarded to eval.py.
# Plots are always saved to outputs/eval_plots.png.
#
# Requirements: pandas, matplotlib, numpy  (pip install pandas matplotlib numpy)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── collect JSONL file arguments ───────────────────────────────────────────────
FILES=()
EXTRA_ARGS=()
# Ensure arrays are always set (needed for set -u with empty arrays in bash < 4.4)
FILES+=()
EXTRA_ARGS+=()

for arg in "$@"; do
    if [[ "$arg" == *.jsonl ]]; then
        FILES+=("$arg")
    else
        EXTRA_ARGS+=("$arg")
    fi
done

# Default: all JSONL files in outputs/ that look like run outputs
if [[ ${#FILES[@]} -eq 0 ]]; then
    while IFS= read -r -d '' f; do
        FILES+=("$f")
    done < <(find outputs -maxdepth 1 -name "*.jsonl" ! -name "eval.csv" -print0 2>/dev/null | sort -z)
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No output JSONL files found. Run the agent first with --output-file."
    exit 1
fi

echo "=== Evaluating ${#FILES[@]} file(s) ==="
for f in "${FILES[@]}"; do echo "  $f"; done
echo

# ── text summary ───────────────────────────────────────────────────────────────
python3 scripts/eval.py "${FILES[@]}" \
    --combined \
    --csv outputs/eval.csv \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# ── plots ──────────────────────────────────────────────────────────────────────
echo
echo "=== Generating plots ==="
python3 scripts/plot_eval.py "${FILES[@]}" --out outputs/eval_plots.png

echo
echo "Done."
echo "  Summary CSV : outputs/eval.csv"
echo "  Plot        : outputs/eval_plots.png"
