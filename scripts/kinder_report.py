#!/usr/bin/env python3
"""Pass rate of KinDER result files by regime (design vs extrapolation counts) and by count.

Works on any quick_eval-style rows that carry ``count`` and ``regime`` (solver rows from
scripts/eval_domain_solver.py, DirectSolve rows from openhands_agents.run_kinder_direct_solve,
hybrid rows from scripts/hybrid_gate.py).  Tokens are summed over input, output and reasoning.

    python scripts/kinder_report.py evals/solver_results/<model>/<run>/test.jsonl \\
        outputs/kinder_obstruction2d_test_direct_solve_openhands.jsonl \\
        outputs/kinder_obstruction2d_test_hybrid_<run>.jsonl --metrics-json metrics/kinder_obstruction2d.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def load(path: str) -> list:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def tokens(row: dict) -> int:
    tu = row.get("token_usage") or {}
    return sum(int(tu.get(k, 0) or 0) for k in ("input", "output", "reasoning"))


def summarise(rows: list) -> dict:
    def rate(rs):
        return {"n": len(rs), "solved": sum(bool(r.get("solved")) for r in rs),
                "pass_rate": (sum(bool(r.get("solved")) for r in rs) / len(rs)) if rs else None}

    by_regime, by_count = defaultdict(list), defaultdict(list)
    for r in rows:
        by_regime[r.get("regime", "?")].append(r)
        by_count[r.get("count")].append(r)
    out = {"all": rate(rows),
           "by_regime": {k: rate(v) for k, v in sorted(by_regime.items())},
           "by_count": {str(k): rate(v) for k, v in sorted(by_count.items(), key=lambda kv: (kv[0] is None, kv[0]))},
           "tokens": sum(tokens(r) for r in rows)}
    sources = [r["source"] for r in rows if "source" in r]
    if sources:
        out["solver_coverage"] = sources.count("solver") / len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--metrics-json", default=None)
    args = ap.parse_args()

    results = {path: summarise(load(path)) for path in args.files}
    counts = sorted({c for s in results.values() for c in s["by_count"]}, key=lambda c: (c == "None", float(c) if c != "None" else 0))
    header = ["file", "all", "design", "extrapolation", *[f"c={c}" for c in counts], "tokens", "solver cov."]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))

    def cell(d):
        return "-" if not d or not d["n"] else f"{d['pass_rate']:.0%} ({d['solved']}/{d['n']})"

    for path, s in results.items():
        cov = s.get("solver_coverage")
        print("| " + " | ".join([os.path.basename(path), cell(s["all"]), cell(s["by_regime"].get("design")),
                                 cell(s["by_regime"].get("extrapolation")),
                                 *[cell(s["by_count"].get(c)) for c in counts], f"{s['tokens']:,}",
                                 "-" if cov is None else f"{cov:.0%}"]) + " |")
    if args.metrics_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_json)) or ".", exist_ok=True)
        with open(args.metrics_json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nwrote {args.metrics_json}")


if __name__ == "__main__":
    main()
