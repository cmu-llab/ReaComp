"""
Solver-first gate: combine solver result files with a fallback (LLM/harness) result file.

For every task the solver's answer is used at zero LLM cost when its reward reaches
--threshold (default 1.0); otherwise the fallback row is used and its tokens are
counted.  This is the "effi" hybrid of the paper, made domain-agnostic: any JSONL
with quick_eval-style rows (task_index, best_reward, answer, token_usage) works, so
the fallback can be Best-of-K, Direct Feedback, an agent, or someone else's harness.

    python scripts/hybrid_gate.py --domain pbebench \
        --solver evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl \
        --fallback outputs/lite_tasks_full_og_best_of_k_stripped.jsonl \
        --out outputs/lite_hybrid_bok_cc.jsonl

Several --solver files form a union: the best-reward row wins, ties broken by the
domain's complexity measure (lower is better), then by file order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _load(path):
    rows = OrderedDict()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = r.get("task_index", r.get("task_id"))
            rows[key] = r
    return rows


def _reward(r):
    if r.get("best_reward") is not None:
        return float(r["best_reward"])
    if "score" in r:
        return float(r["score"])
    return 1.0 if r.get("solved") or r.get("success") else 0.0


def _tokens(r):
    tu = r.get("token_usage") or {}
    return int(tu.get("input", 0) or 0) + int(tu.get("output", 0) or 0) + int(tu.get("reasoning", 0) or 0)


def _zero_tokens(r):
    r = dict(r)
    r["token_usage"] = {"input": 0, "output": 0, "reasoning": 0}
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", nargs="+", required=True, help="solver result JSONL file(s); union if several")
    ap.add_argument("--fallback", required=True, help="LLM / harness result JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default=None, help="registered domain, used for complexity tie-breaks")
    ap.add_argument("--threshold", type=float, default=1.0)
    args = ap.parse_args()

    dom = None
    if args.domain:
        from domains import get_domain
        dom = get_domain(args.domain)

    def complexity(r):
        if dom is None:
            return float("inf")
        c = dom.complexity(r.get("answer"))
        return float("inf") if c is None else c

    solver_files = [_load(p) for p in args.solver]
    fallback = _load(args.fallback)
    keys = list(fallback.keys())
    for sf in solver_files:
        for k in sf:
            if k not in fallback:
                keys.append(k)

    out_rows, n_solver, n_hybrid_solved, n_fb_solved = [], 0, 0, 0
    tok_fb, tok_hybrid = 0, 0
    for k in keys:
        cands = [sf[k] for sf in solver_files if k in sf]
        best = None
        for c in cands:
            if best is None or (_reward(c), -complexity(c)) > (_reward(best), -complexity(best)):
                best = c
        fb = fallback.get(k)
        if fb is not None:
            tok_fb += _tokens(fb)
            n_fb_solved += int(_reward(fb) >= 1.0)
        if best is not None and _reward(best) >= args.threshold:
            row = _zero_tokens(best); row["source"] = "solver"; n_solver += 1
        elif fb is not None:
            row = dict(fb); row["source"] = "fallback"; tok_hybrid += _tokens(fb)
        else:
            row = dict(best) if best else {"task_index": k, "best_reward": 0.0, "solved": False}
            row["source"] = "solver"
        row["solved"] = _reward(row) >= 1.0
        n_hybrid_solved += int(row["solved"])
        out_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, default=str) + "\n")

    n = len(out_rows)
    summary = {
        "n": n,
        "solver_coverage": n_solver / max(n, 1),
        "fallback_pass_rate": n_fb_solved / max(len(fallback), 1),
        "hybrid_pass_rate": n_hybrid_solved / max(n, 1),
        "fallback_tokens": tok_fb,
        "hybrid_tokens": tok_hybrid,
        "token_savings": (1 - tok_hybrid / tok_fb) if tok_fb else None,
        "out": args.out,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
