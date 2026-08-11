"""
Compare solver-seeded DF vs plain DF on the solver-failed subset (additive).

Reports headline accuracy and a per-difficulty breakdown with n per bucket, on
exactly the tasks where the symbolic solver failed (best_reward < 1.0). Plain-DF
numbers are read from the existing DF output file; nothing is re-run.

Usage:
  python scripts/analyze_seeded_df.py \
    --benchmark slr \
    --solver-results evals/solver_results/slr_claude_code/slr.jsonl \
    --seeded outputs/seeded_df_slr_cc.jsonl \
    --plain-df outputs/slr_bench_direct_feedback.jsonl
"""

import argparse
import json
from collections import defaultdict


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _plain_reward_by_index(plain_rows):
    out = {}
    for r in plain_rows:
        ti = r.get("task_index")
        if ti is not None:
            out[ti] = float(r.get("best_reward", 0.0))
    return out


def _difficulty(benchmark, srow):
    if benchmark == "slr":
        return srow.get("curriculum_tier") or "?"
    return srow.get("cascade_length")


def _acc(vals):
    n = len(vals)
    solved = sum(1 for v in vals if v >= 1.0)
    mr = sum(vals) / n if n else 0.0
    return solved, n, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["lite", "slr"])
    ap.add_argument("--solver-results", required=True)
    ap.add_argument("--seeded", required=True)
    ap.add_argument("--plain-df", required=True)
    args = ap.parse_args()

    solver_rows = _load(args.solver_results)
    seeded_rows = _load(args.seeded)
    plain = _plain_reward_by_index(_load(args.plain_df))

    failed = {r["task_index"]: r for r in solver_rows if float(r.get("best_reward", 0.0)) < 1.0}
    seeded = {r["task_index"]: r for r in seeded_rows}

    # Restrict to solver-failed tasks that have BOTH a seeded result and a plain-DF
    # entry, so the comparison is like-for-like.
    idxs = [ti for ti in failed if ti in seeded and ti in plain]
    missing = [ti for ti in failed if ti not in seeded]
    if missing:
        print(f"note: {len(missing)} solver-failed tasks not yet in seeded output "
              f"(run still in progress?)")

    overall_seeded, overall_plain = [], []
    by_diff = defaultdict(lambda: {"seeded": [], "plain": []})

    for ti in idxs:
        s = float(seeded[ti].get("seeded_best_reward", 0.0))
        p = plain[ti]
        d = _difficulty(args.benchmark, failed[ti])
        overall_seeded.append(s)
        overall_plain.append(p)
        by_diff[d]["seeded"].append(s)
        by_diff[d]["plain"].append(p)

    print(f"\n=== Solver-seeded DF vs plain DF on solver-FAILED {args.benchmark} tasks ===")
    print(f"(CC solver failures with both seeded + plain-DF results; N={len(idxs)})\n")

    ss, sn, smr = _acc(overall_seeded)
    ps, pn, pmr = _acc(overall_plain)
    print("HEADLINE (accuracy = solved / N on the solver-failed subset):")
    print(f"  plain DF     : {ps}/{pn} = {100*ps/pn:.1f}%  (mean reward {pmr:.3f})")
    print(f"  seeded DF    : {ss}/{sn} = {100*ss/sn:.1f}%  (mean reward {smr:.3f})")
    print(f"  delta        : {100*(ss-ps)/pn:+.1f} pp accuracy, {smr-pmr:+.3f} mean reward\n")

    print("BY DIFFICULTY (n reported per bucket):")
    def _key(k):
        return (str(k)) if args.benchmark == "slr" else (k if k is not None else -1)
    for d in sorted(by_diff, key=_key):
        b = by_diff[d]
        ps2, pn2, pmr2 = _acc(b["plain"])
        ss2, sn2, smr2 = _acc(b["seeded"])
        label = f"tier={d}" if args.benchmark == "slr" else f"cascade_len={d}"
        print(f"  {label:>18} (n={pn2:>3}): "
              f"plain {100*ps2/pn2:5.1f}%  ->  seeded {100*ss2/sn2:5.1f}%  "
              f"({100*(ss2-ps2)/pn2:+5.1f} pp;  reward {pmr2:.3f} -> {smr2:.3f})")


if __name__ == "__main__":
    main()
