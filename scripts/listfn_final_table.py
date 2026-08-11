"""
Final List Functions comparison table (this-session, additive).

Combines already-computed per-task results into one table, all on the full 250
tasks, held-out solve = the induced program / any BoK candidate reproduces every
held-out pair exactly. No new inference; pure set logic over saved outputs.

Rows:
  - BoK@K (gpt-oss)                : task solved if ANY of the K candidates passes held-out
  - Symbolic solver (each run)     : per-run held-out solve
  - Symbolic ensemble (union of N) : task solved if ANY induced solver passes held-out
  - Hybrid (ensemble OR BoK)        : solver-first with BoK fallback (union of the two)
"""

import argparse
import json
from collections import defaultdict


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def bok_solved_set(path):
    by = defaultdict(list)
    for r in load_jsonl(path):
        by[r["task_id"]].append(r)
    return {t for t, cs in by.items() if any(c.get("heldout_score") == 1.0 for c in cs)}, set(by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bok", default="outputs/listfn_full_bok_traces.jsonl")
    ap.add_argument("--ensemble", default="outputs/listfn_ensemble_eval.jsonl")
    args = ap.parse_args()

    ens = load_jsonl(args.ensemble)
    all_tasks = {r["task_id"] for r in ens}
    n = len(all_tasks)

    # per-solver + union from the ensemble eval
    solver_names = list(ens[0]["per_solver"].keys())
    per_solver = {s: {r["task_id"] for r in ens if r["per_solver"].get(s)} for s in solver_names}
    ens_union = {r["task_id"] for r in ens if r["union_solved"]}

    bok_set, bok_tasks = bok_solved_set(args.bok)
    # guard: BoK should cover the same task set
    missing = all_tasks - bok_tasks
    if missing:
        print(f"note: {len(missing)} tasks missing from BoK output (partial run?)")

    hybrid = ens_union | bok_set

    def pct(s):
        return f"{len(s)}/{n} ({100*len(s)/n:.1f}%)"

    print(f"\n=== List Functions, full {n} tasks, held-out generalization ===\n")
    print(f"{'System':<38} {'Held-out solved':>18}")
    print("-" * 58)
    print(f"{'BoK@8 (gpt-oss-120b)':<38} {pct(bok_set):>18}")
    for s in solver_names:
        print(f"{'Symbolic solver: ' + s:<38} {pct(per_solver[s]):>18}")
    print(f"{'Symbolic ensemble (union of ' + str(len(solver_names)) + ')':<38} {pct(ens_union):>18}")
    print(f"{'Hybrid (symbolic ensemble OR BoK)':<38} {pct(hybrid):>18}")
    print()
    # complementarity detail
    only_sym = ens_union - bok_set
    only_bok = bok_set - ens_union
    print(f"solver-only (BoK missed): {len(only_sym)} | BoK-only (solver missed): {len(only_bok)} | both: {len(ens_union & bok_set)}")


if __name__ == "__main__":
    main()
