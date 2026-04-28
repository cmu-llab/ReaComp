#!/usr/bin/env bash
# Sweep max-programs-ratio on Real-FR using Qwen run 2 solver.
#
# Tests how enforcing a compression ratio (max_programs = ceil(n_examples / ratio))
# affects pass rate and avg program count — diagnosing whether the solver is
# memorising training examples or discovering generalisable rules.
#
# Ratios tested:
#   1.0  → 1 program per example  (strong compression)
#   2.0  → 1 program per 2 examples
#   3.0  → 1 program per 3 examples
#   5.0  → 1 program per 5 examples
#   unconstrained → max_programs=100 (baseline)
#
# Usage (from project root):
#   bash scripts/sweep_real_fr_compression.sh
#   bash scripts/sweep_real_fr_compression.sh --workers 8
set -euo pipefail

SOLVER=built_solvers/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/SOLVER.py
BASE_OUTPUT_DIR=evals/solver_results/real_fr/qwen_run2_compression_sweep
WORKERS=${1:-8}
EXTRA_ARGS=("${@:2}")

echo "Solver : $SOLVER"
echo "Output : $BASE_OUTPUT_DIR"
echo "Workers: $WORKERS"
echo

# Baseline: unconstrained (max_programs=100, matches existing qwen_k100 run)
echo "===== Baseline: max_programs=100 (no ratio) ====="
python scripts/eval_solver.py \
    --solver "$SOLVER" \
    --dataset real \
    --max-programs 100 \
    --output-dir "$BASE_OUTPUT_DIR/k100" \
    --workers "$WORKERS" \
    "${EXTRA_ARGS[@]}"

# Ratio sweep
for RATIO in 1.0 2.0 3.0 5.0; do
    echo
    echo "===== Ratio: $RATIO (max_programs = ceil(n_examples / $RATIO), min 2) ====="
    python scripts/eval_solver.py \
        --solver "$SOLVER" \
        --dataset real \
        --max-programs 100 \
        --max-programs-ratio "$RATIO" \
        --output-dir "$BASE_OUTPUT_DIR/ratio${RATIO//./_}" \
        --workers "$WORKERS" \
        "${EXTRA_ARGS[@]}"
done

echo
echo "===== Sweep complete. Summarising results ====="
python3 - <<'EOF'
import json, os, glob

base = "evals/solver_results/real_fr/qwen_run2_compression_sweep"
runs = []
for summary_path in sorted(glob.glob(f"{base}/*/*.json")):
    with open(summary_path) as f:
        s = json.load(f)
    run_name = os.path.basename(os.path.dirname(summary_path))
    runs.append((run_name, s))

# Also compute avg programs per task for each run
def avg_programs(jsonl_path):
    if not os.path.exists(jsonl_path):
        return None
    total, n = 0, 0
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            ans = r.get("answer") or []
            total += len(ans)
            n += 1
    return total / n if n else None

print(f"{'Run':<20}  {'N':>5}  {'Solved':>6}  {'Pass%':>7}  {'AvgProgs':>9}")
print("-" * 58)
for run_name, s in runs:
    jsonl = f"{base}/{run_name}/real_ratio*.jsonl"
    candidates = sorted(glob.glob(jsonl))
    if not candidates:
        candidates = sorted(glob.glob(f"{base}/{run_name}/real_k*.jsonl"))
    avg_p = avg_programs(candidates[0]) if candidates else None
    avg_p_str = f"{avg_p:.2f}" if avg_p is not None else "—"
    print(f"{run_name:<20}  {s['n']:>5}  {s['solved']:>6}  {s['pass_rate']:>6.1%}  {avg_p_str:>9}")
EOF
