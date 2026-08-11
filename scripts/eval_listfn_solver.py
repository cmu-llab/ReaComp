#!/usr/bin/env python3
"""Evaluate an induced SOLVER_LISTFN.py on the List Functions pilot tasks.

New-files-only. For each pilot task, calls the solver's solve_listfn(examples) on the
SHOWN (held-in) pairs only, then scores the returned program on the HELD-OUT pairs. A
task counts as "solved" iff the induced program reproduces every held-out pair exactly
(generalization, not memorization of the shown pairs). This is the discriminator that
answers the meta-review (d.4) point directly.

Usage:
    python scripts/eval_listfn_solver.py \
        --solver built_solvers/listfn_pilot/SOLVER_LISTFN.py \
        --tasks-file data/list_functions/pilot_tasks.jsonl \
        --targets data/list_functions/pilot_targets.json \
        --out outputs/listfn_solver_eval.jsonl
"""
import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.list_functions import _extract_callable, score_program


def load_solver(path):
    spec = importlib.util.spec_from_file_location("solver_listfn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "solve_listfn"):
        raise AttributeError(f"{path} does not define solve_listfn(examples)")
    return mod.solve_listfn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="built_solvers/listfn_pilot/SOLVER_LISTFN.py")
    ap.add_argument("--tasks-file", default="data/list_functions/pilot_tasks.jsonl")
    ap.add_argument("--targets", default="data/list_functions/pilot_targets.json")
    ap.add_argument("--out", default="outputs/listfn_solver_eval.jsonl")
    args = ap.parse_args()

    solve_listfn = load_solver(args.solver)
    recs = [json.loads(l) for l in open(args.tasks_file) if l.strip()]
    targets = {}
    if os.path.exists(args.targets):
        targets = json.load(open(args.targets))

    rows = []
    n_solved = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # Stream: write each row as it is computed and flush, so progress is visible
    # and a killed run leaves partial (usable) output rather than nothing.
    out_f = open(args.out, "w")
    for i, rec in enumerate(recs):
        examples = list(zip(
            [list(x) for x in rec["held_in_inputs"]],
            [list(y) for y in rec["held_in_outputs"]],
        ))
        result, err = None, None
        try:
            result = solve_listfn(examples)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        program = result.get("program") if isinstance(result, dict) else result
        fn = _extract_callable(program) if program is not None else None

        if fn is None:
            shown_score = heldout_score = 0.0
            mismatches = ["solver returned no usable program"] if err is None else [err]
        else:
            shown_score, _ = score_program(
                fn, rec["held_in_inputs"], rec["held_in_outputs"])
            heldout_score, mismatches = score_program(
                fn, rec["held_out_inputs"], rec["held_out_outputs"])

        solved = heldout_score >= 1.0
        n_solved += int(solved)
        row = {
            "task_id": rec["task_id"],
            "target": targets.get(rec["task_id"], ""),
            "shown_score": round(shown_score, 4),
            "heldout_score": round(heldout_score, 4),
            "solved": solved,
            "reported_success": bool(result.get("success")) if isinstance(result, dict) else None,
            "mismatches": mismatches[:3],
            "error": err,
        }
        rows.append(row)
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
    out_f.close()

    print(f"{'task':<8} {'shown':>6} {'heldout':>8} {'solved':>7}  target")
    for r in rows:
        print(f"{r['task_id']:<8} {r['shown_score']:>6.2f} {r['heldout_score']:>8.2f} "
              f"{str(r['solved']):>7}  {r['target'][:60]}")
    print(f"\nSolved (generalizes to held-out): {n_solved}/{len(rows)}")
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
