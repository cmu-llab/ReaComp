"""
Evaluate gpt_oss_120b_pbebench_outputs.jsonl (best-of-k BoK format) against
the PBEBench hard tasks and compare with symbolic solvers.

Format of BoK file:
  Each record: {"input": {inputs, outputs, programs, cascade_length, index, ...},
                "outputs": [<32 candidate strings>]}
  index = "<task_id>_<j>" where task_id is the task index into tasks_full_og.jsonl.

Usage:
    python scripts/eval_bok_hard.py
    python scripts/eval_bok_hard.py --bok outputs/gpt_oss_120b_pbebench_outputs.jsonl
                                    --tasks data/pbebench/tasks_full_og.jsonl
                                    --solver-cc evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl
                                    --solver-qw evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rewards.pbebench import _parse_programs, _validate_programs, cascade_complexity

_REPLACE_RE = re.compile(
    r"""replace\(\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\)""",
    re.IGNORECASE,
)
MAX_PROGRAMS = 20  # full PBEBench hard


def score_candidate(candidate: str, inputs: list, outputs: list) -> float:
    """Return fraction of I/O pairs correctly mapped by this candidate program sequence."""
    programs, err = _parse_programs(candidate)
    if programs is None:
        return 0.0
    violations = _validate_programs(programs, max_programs=MAX_PROGRAMS)
    if violations:
        return 0.0
    correct = sum(
        1 for inp, exp in zip(inputs, outputs)
        if _apply(programs, inp) == exp
    )
    return correct / len(inputs)


def _apply(programs, inp: str) -> str:
    s = inp
    for pred, transform in programs:
        s = s.replace(pred, transform)
    return s


def best_score_and_program(candidates: list, inputs: list, outputs: list):
    """Return (best_score, best_candidate_str) across all candidates."""
    best_s, best_c = 0.0, None
    for c in candidates:
        s = score_candidate(c, inputs, outputs)
        if s > best_s:
            best_s, best_c = s, c
        if best_s >= 1.0:
            break
    return best_s, best_c


def load_bok(path: str) -> dict[int, dict]:
    """Return {task_id: record} with best_score and best_program added."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx_str = rec["input"]["index"]
            task_id = int(idx_str.split("_")[0])
            inp = rec["input"]
            candidates = rec["outputs"]
            best_s, best_c = best_score_and_program(
                candidates, inp["inputs"], inp["outputs"]
            )
            records[task_id] = {
                "task_id": task_id,
                "cascade_length": inp.get("cascade_length"),
                "best_score": best_s,
                "solved": best_s >= 1.0,
                "best_program": best_c,
                "n_candidates": len(candidates),
            }
    return records


def load_solver(path: str) -> dict[int, dict]:
    """Load solver JSONL — handles both eval_solver format (score/success/program)
    and standard main.py format (best_reward/solved)."""
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_index")
            if tid is None:
                continue
            # standard format
            if rec.get("best_reward") is not None:
                score = float(rec["best_reward"])
            elif rec.get("solved") is not None:
                score = 1.0 if rec["solved"] else 0.0
            else:
                score = rec.get("score", 1.0 if rec.get("success") else 0.0)
            records[tid] = {
                "task_id": tid,
                "cascade_length": rec.get("cascade_length"),
                "best_score": float(score),
                "solved": float(score) >= 1.0,
            }
    return records


def summarise(label: str, records: dict[int, dict]) -> None:
    recs = list(records.values())
    n = len(recs)
    solved = sum(1 for r in recs if r["solved"])
    mean_r = sum(r["best_score"] for r in recs) / n if n else 0.0
    print(f"\n{'='*55}")
    print(f"  {label}  (n={n})")
    print(f"{'='*55}")
    print(f"  Pass rate   : {solved}/{n} = {100*solved/n:.1f}%")
    print(f"  Mean reward : {mean_r:.4f}")


def breakdown_by_cl(label: str, records: dict[int, dict]) -> list[dict]:
    """Print and return pass rate by cascade length."""
    by_cl: dict[int, list] = defaultdict(list)
    for r in records.values():
        cl = r.get("cascade_length")
        if cl is not None:
            by_cl[cl].append(r)

    print(f"\n  By cascade length ({label})")
    print(f"    {'CL':>4}  {'n':>5}  {'solved':>6}  {'pass%':>6}  {'mean_r':>7}")
    rows = []
    for cl in sorted(by_cl.keys()):
        recs = by_cl[cl]
        n = len(recs)
        s = sum(1 for r in recs if r["solved"])
        mean_r = sum(r["best_score"] for r in recs) / n
        print(f"    {cl:>4}  {n:>5}  {s:>6}  {100*s/n:>5.1f}%  {mean_r:>7.4f}")
        rows.append({"cl": cl, "n": n, "solved": s, "pass_pct": round(100*s/n, 2), "mean_reward": round(mean_r, 4)})
    return rows


def compare_cl(systems: dict[str, dict[int, dict]]) -> None:
    """Side-by-side pass% by cascade length for all systems."""
    all_cls = sorted(set(
        r["cascade_length"]
        for recs in systems.values()
        for r in recs.values()
        if r.get("cascade_length") is not None
    ))

    names = list(systems.keys())
    col_w = max(len(n) for n in names)
    header = f"    {'CL':>4}  {'n':>5}  " + "  ".join(f"{n:>{max(col_w,6)}}" for n in names)
    print(f"\n{'='*55}")
    print(f"  Pass% by cascade length — side-by-side comparison")
    print(f"{'='*55}")
    print(header)

    for cl in all_cls:
        cols = []
        n_col = None
        for name, recs in systems.items():
            cl_recs = [r for r in recs.values() if r.get("cascade_length") == cl]
            if not cl_recs:
                cols.append(f"{'—':>{max(col_w,6)}}")
                continue
            n = len(cl_recs)
            n_col = n
            s = sum(1 for r in cl_recs if r["solved"])
            cols.append(f"{100*s/n:>{max(col_w,6)}.1f}%")
        n_str = f"{n_col:>5}" if n_col is not None else "    ?"
        print(f"    {cl:>4}  {n_str}  " + "  ".join(cols))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bok", default="outputs/gpt_oss_120b_pbebench_outputs.jsonl")
    parser.add_argument("--tasks", default="data/pbebench/tasks_full_og.jsonl")
    parser.add_argument("--solver-cc", default="evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl")
    parser.add_argument("--solver-qw", default="evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl")
    parser.add_argument("--metrics-json", default=None)
    args = parser.parse_args()

    print(f"Loading BoK outputs: {args.bok}")
    bok = load_bok(args.bok)
    print(f"  {len(bok)} tasks loaded")

    systems: dict[str, dict] = {"BoK-32 (gpt-oss-120b)": bok}

    if Path(args.solver_cc).exists():
        print(f"Loading Claude Code solver: {args.solver_cc}")
        cc = load_solver(args.solver_cc)
        print(f"  {len(cc)} tasks loaded")
        systems["Solver (Claude Code)"] = cc

    if Path(args.solver_qw).exists():
        print(f"Loading Qwen3.6 solver: {args.solver_qw}")
        qw = load_solver(args.solver_qw)
        print(f"  {len(qw)} tasks loaded")
        systems["Solver (Qwen3.6)"] = qw

    for label, recs in systems.items():
        summarise(label, recs)
        breakdown_by_cl(label, recs)

    if len(systems) > 1:
        compare_cl(systems)

    if args.metrics_json:
        out = {}
        for label, recs in systems.items():
            rlist = list(recs.values())
            n = len(rlist)
            solved = sum(1 for r in rlist if r["solved"])
            out[label] = {
                "n": n, "solved": solved,
                "pass_pct": round(100*solved/n, 2) if n else 0,
                "mean_reward": round(sum(r["best_score"] for r in rlist)/n, 4) if n else 0,
            }
        with open(args.metrics_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Metrics written to {args.metrics_json}")


if __name__ == "__main__":
    main()
