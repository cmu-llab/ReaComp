"""
Token-cost analysis for the List Functions hybrid (this-session, additive).

Compares BoK-alone token usage against the solver-first hybrid, where the symbolic
solver ensemble runs at zero LLM cost and BoK is invoked only on the tasks the
ensemble does not solve.

BoK token accounting: sum input+output tokens over the K candidates the run
actually generated. Hybrid: same, but only over tasks NOT solved by the ensemble
(ensemble-solved tasks contribute zero LLM tokens).
"""

import argparse
import json
from collections import defaultdict


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bok-traces", default="outputs/listfn_full_bok_traces_tok.jsonl",
                    help="BoK candidate rows carrying token_usage")
    ap.add_argument("--ensemble", default="outputs/listfn_ensemble_eval.jsonl",
                    help="symbolic ensemble eval (per-task union_solved)")
    args = ap.parse_args()

    bok = load_jsonl(args.bok_traces)
    by_task = defaultdict(list)
    for r in bok:
        by_task[r["task_id"]].append(r)

    ens = load_jsonl(args.ensemble)
    ens_solved = {r["task_id"] for r in ens if r["union_solved"]}

    def toks(cands):
        return sum((c.get("token_usage") or {}).get("input", 0) +
                   (c.get("token_usage") or {}).get("output", 0) for c in cands)

    tasks = sorted(by_task)
    bok_total = sum(toks(by_task[t]) for t in tasks)
    # hybrid: BoK tokens only on tasks the ensemble did NOT solve
    hybrid_total = sum(toks(by_task[t]) for t in tasks if t not in ens_solved)

    n = len(tasks)
    n_ens = len(ens_solved & set(tasks))
    print(f"=== List Functions token cost: BoK-alone vs solver-first hybrid ===")
    print(f"tasks: {n} | ensemble-solved (zero LLM): {n_ens} | BoK-fallback tasks: {n - n_ens}\n")
    print(f"BoK-alone total tokens : {bok_total/1e6:.2f}M")
    print(f"Hybrid total tokens    : {hybrid_total/1e6:.2f}M")
    if bok_total:
        print(f"Token savings          : {100*(1-hybrid_total/bok_total):.1f}%")
    print(f"(task coverage by solver: {100*n_ens/n:.1f}% handled at zero LLM cost)")


if __name__ == "__main__":
    main()
