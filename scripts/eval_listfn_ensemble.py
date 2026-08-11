"""
Ensemble evaluation for List Functions solvers (additive, this-session file).

Runs multiple induced solvers on the full task set and reports both the per-solver
held-out solve rate and the UNION ensemble (a task counts solved if ANY solver
solves its held-out pairs exactly). Mirrors the paper's "All Symbolic" aggregation
that recovers induction variance.

Usage:
  python scripts/eval_listfn_ensemble.py \
    --solvers built_solvers/listfn_pilot/SOLVER.py \
              built_solvers/listfn_pilot_run1/SOLVER.py \
              built_solvers/listfn_pilot_run2/SOLVER.py \
              built_solvers/listfn_pilot_run3/SOLVER.py \
    --tasks-file data/list_functions/full_tasks.jsonl \
    --out outputs/listfn_ensemble_eval.jsonl
"""

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rewards.list_functions import _extract_callable, score_program  # noqa: E402


def load_solver(path):
    spec = importlib.util.spec_from_file_location(f"solver_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "solve_listfn"):
        raise AttributeError(f"{path} has no solve_listfn")
    return mod.solve_listfn


def solver_heldout_solved(solve_fn, rec):
    """Return True iff this solver's program (fit on shown pairs) reproduces all
    held-out pairs exactly."""
    examples = list(zip([list(x) for x in rec["held_in_inputs"]],
                        [list(y) for y in rec["held_in_outputs"]]))
    try:
        result = solve_fn(examples)
    except Exception:
        return False
    program = result.get("program") if isinstance(result, dict) else result
    fn = _extract_callable(program) if program is not None else None
    if fn is None:
        return False
    ho, _ = score_program(fn, rec["held_out_inputs"], rec["held_out_outputs"])
    return ho >= 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvers", nargs="+", required=True)
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.tasks_file) if l.strip()]
    solvers = [(p, load_solver(p)) for p in args.solvers]
    print(f"{len(solvers)} solvers x {len(recs)} tasks")

    per_solver = {p: 0 for p, _ in solvers}
    union_solved = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, "w")

    for rec in recs:
        flags = {}
        for p, fn in solvers:
            ok = solver_heldout_solved(fn, rec)
            flags[p] = ok
            per_solver[p] += int(ok)
        union = any(flags.values())
        union_solved += int(union)
        out_f.write(json.dumps({
            "task_id": rec["task_id"],
            "per_solver": {os.path.basename(os.path.dirname(p)): flags[p] for p in flags},
            "union_solved": union,
        }) + "\n")
        out_f.flush()
    out_f.close()

    n = len(recs)
    print("\n=== Per-solver held-out solve rate (full set) ===")
    for p, _ in solvers:
        c = per_solver[p]
        print(f"  {os.path.basename(os.path.dirname(p)):<24} {c}/{n} ({100*c/n:.1f}%)")
    print(f"\n=== UNION ensemble ({len(solvers)} solvers) ===")
    print(f"  {union_solved}/{n} ({100*union_solved/n:.1f}%)")


if __name__ == "__main__":
    main()
