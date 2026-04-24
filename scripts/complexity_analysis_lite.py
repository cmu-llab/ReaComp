"""
Complexity analysis for PBEBench-Lite.

Systems compared (all in standard quick_eval JSONL format):
  - BoK∪CC Solver ensemble  (outputs/ensemble_bok_claude_solver.jsonl)
  - BoK∪Qwen Solver ensemble (outputs/ensemble_bok_qwen_solver.jsonl)
  - Symbolic Solver (Claude Code)  (evals/solver_results/claude_code/.../lite.jsonl)
  - Symbolic Solver (Qwen3.6-35B-A3B) (evals/solver_results/qwen3.6_35b_a3b/.../lite.jsonl)

Selection policy: max reward first, min complexity as tiebreak.
Complexity reported for solved tasks only (best_reward >= 1.0), compared to GT.

Usage:
    python scripts/complexity_analysis_lite.py
    python scripts/complexity_analysis_lite.py --plot figures/complexity_lite.png
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rewards.pbebench import _parse_programs, cascade_complexity
from scripts.complexity_analysis_hard import (
    load_gt, load_standard, load_cc_solver, load_qwen_solver,
    complexity_stats, make_complexity_plot,
)

_MAX_PROGRAMS = 5  # PBEBench-Lite constraint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks",      default="data/pbebench/lite_tasks_full_og.jsonl")
    parser.add_argument("--bok-cc",     default="outputs/ensemble_bok_claude_solver.jsonl")
    parser.add_argument("--bok-qwen",   default="outputs/ensemble_bok_qwen_solver.jsonl")
    parser.add_argument("--solver-cc",  default="evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl")
    parser.add_argument("--solver-qw",  default="evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/lite.jsonl")
    parser.add_argument("--plot",          default=None, metavar="FILE")
    parser.add_argument("--metrics-json",  default=None, metavar="FILE")
    args = parser.parse_args()

    print("Loading GT complexity (Lite)…")
    gt = load_gt(Path(args.tasks))
    print(f"  {len(gt)} tasks")

    print("Loading BoK∪CC Solver ensemble…")
    bok_cc   = load_standard(Path(args.bok_cc))
    print("Loading BoK∪Qwen Solver ensemble…")
    bok_qwen = load_standard(Path(args.bok_qwen))
    print("Loading Symbolic Solver (Claude Code) Lite…")
    cc       = load_standard(Path(args.solver_cc))
    print("Loading Symbolic Solver (Qwen3.6-35B-A3B) Lite…")
    qwen     = load_standard(Path(args.solver_qw))

    # Attach cascade_length from tasks file (solver lite files may already have it,
    # but ensemble files might not — patch from GT task metadata)
    task_cl: dict[int, int] = {}
    with open(args.tasks) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                task_cl[i] = json.loads(line).get("cascade_length")
    for recs in (bok_cc, bok_qwen, cc, qwen):
        for tid, r in recs.items():
            if r["cascade_length"] is None and tid in task_cl:
                r["cascade_length"] = task_cl[tid]

    systems = [
        ("BoK ∪ Solver (Claude Code)",        bok_cc),
        ("BoK ∪ Solver (Qwen3.6)",            bok_qwen),
        ("Symbolic Solver (Claude Code)",      cc),
        ("Symbolic Solver (Qwen3.6-35B-A3B)", qwen),
    ]

    print(f"\n{'='*58}")
    print("  Complexity vs Ground Truth  (solved tasks only, Lite)")
    print(f"{'='*58}")
    all_stats = [complexity_stats(recs, gt, label) for label, recs in systems]

    if args.metrics_json:
        with open(args.metrics_json, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nMetrics written to {args.metrics_json}")

    if args.plot:
        make_complexity_plot(systems, gt, Path(args.plot),
                             subtitle="PBEBench-Lite, solved tasks only, CL 2–5")


if __name__ == "__main__":
    main()
