"""
Build solver-induction demos for any registered domain from trace files.

    # PBEBench: Best-of-K / Direct Feedback output rows that still carry agent_messages
    python scripts/build_demos.py --domain pbebench --split hard \
        --traces outputs/tasks_full_og_best_of_k.jsonl --n 100 --seed 42 --out demos/DEMOS_PBEBENCH_new.json

    # List Functions: the paper's protocol (25 tasks x 4 attempts, hardness-balanced)
    python scripts/build_demos.py --domain list_functions --split full \
        --traces outputs/listfn_full_bok_traces.jsonl --n 100 --per-task 4 --out demos/DEMOS_LISTFN_balanced.json

    # Environment setup: rows written by domains.envsetup.agent
    python scripts/build_demos.py --domain envsetup --split setupbench \
        --traces outputs/envsetup/agent_setupbench.jsonl --n 100 --out demos/DEMOS_ENVSETUP.json

    # KinDER: rows written by openhands_agents.run_kinder_direct_solve on the train split
    python scripts/build_demos.py --domain kinder_obstruction2d --split train \
        --traces outputs/kinder_obstruction2d_train_direct_solve_openhands.jsonl --n 100 \
        --out demos/DEMOS_KINDER_OBSTRUCTION2D_seed_42_100_examples_with_CoT.json

``--no-cot`` drops the ``cot`` field (the "without CoT" ablation of the paper's demo files).

Traces are matched to tasks by task_index or task_id; each domain's ``demo_from_trace``
defines the demo schema and ``demo_group_key`` the balancing (difficulty x success by default).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from domains import get_domain, list_domains  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=list_domains())
    ap.add_argument("--split", required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-task", type=int, default=None, help="max demos per task (domain-specific meaning)")
    ap.add_argument("--no-cot", action="store_true", help="drop the reasoning trace (cot) from every demo")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dom = get_domain(args.domain)
    demos = dom.build_demos(args.traces, args.split, n=args.n, seed=args.seed, out=args.out, per_task=args.per_task)
    if args.no_cot:
        import json
        for d in demos:
            d.pop("cot", None)
        with open(args.out, "w") as fh:
            json.dump(demos, fh, indent=1)
    groups = Counter(dom.demo_group_key(d) for d in demos)
    print(f"{len(demos)} demos -> {args.out}")
    for k in sorted(groups, key=repr):
        print(f"  {k}: {groups[k]}")


if __name__ == "__main__":
    main()
