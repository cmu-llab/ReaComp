#!/usr/bin/env bash
# run_eval.sh — run eval.py + plot_eval.py on one or more output JSONL files.
#
# Usage:
#   bash scripts/run_eval.sh                                        # all outputs/*.jsonl
#   bash scripts/run_eval.sh outputs/pbebench_lite*.jsonl           # specific file(s)
#   bash scripts/run_eval.sh outputs/pbebench*.jsonl \
#       --csv results/eval.csv --plots-dir results/plots
#
# Options:
#   --csv PATH       Path for the per-instance CSV summary (default: evals/eval.csv)
#   --plots-dir DIR  Directory for individual panel PNGs  (default: evals/)
#
# Any other non-.jsonl arguments are forwarded to eval.py.
#
# Requirements: pandas, matplotlib, numpy  (pip install pandas matplotlib numpy)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── parse arguments ────────────────────────────────────────────────────────────
FILES=()
EXTRA_ARGS=()
CSV_PATH="evals/eval.csv"
PLOTS_DIR="evals"
# Ensure arrays are always set (needed for set -u with empty arrays in bash < 4.4)
FILES+=()
EXTRA_ARGS+=()

_next_is_value=0
for arg in "$@"; do
    if [[ $_next_is_value -eq 1 ]]; then
        # previous arg was --csv
        CSV_PATH="$arg"
        _next_is_value=0
    elif [[ $_next_is_value -eq 2 ]]; then
        # previous arg was --plots-dir
        PLOTS_DIR="$arg"
        _next_is_value=0
    elif [[ "$arg" == "--csv" ]]; then
        _next_is_value=1
    elif [[ "$arg" == --csv=* ]]; then
        CSV_PATH="${arg#--csv=}"
    elif [[ "$arg" == "--plots-dir" ]]; then
        _next_is_value=2
    elif [[ "$arg" == --plots-dir=* ]]; then
        PLOTS_DIR="${arg#--plots-dir=}"
    elif [[ "$arg" == *.jsonl ]]; then
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
echo "  CSV     -> $CSV_PATH"
echo "  Plots   -> $PLOTS_DIR/"
echo

# ── text summary ───────────────────────────────────────────────────────────────
python3 scripts/eval.py "${FILES[@]}" \
    --combined \
    --csv "$CSV_PATH" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

# ── plots ──────────────────────────────────────────────────────────────────────
echo
echo "=== Generating plots ==="
python3 scripts/plot_eval.py "${FILES[@]}" --out-dir "$PLOTS_DIR"

echo
echo "Done."
echo "  Summary CSV : $CSV_PATH"
echo "  Plots       : $PLOTS_DIR/"
