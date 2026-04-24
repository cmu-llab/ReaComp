"""
Complexity analysis for PBEBench-Hard across BoK-32, Symbolic Solver (CC),
and Symbolic Solver (Qwen3.6-35B-A3B), compared against ground-truth complexity.

Selection policy (consistent across all systems):
  - Primary:   max reward
  - Tiebreak:  min complexity

For BoK-32 this means: among all 32 candidates, pick the one with the highest
score; if tied, pick the simplest one.

Usage:
    python scripts/complexity_analysis_hard.py
    python scripts/complexity_analysis_hard.py --plot figures/complexity_hard.png
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rewards.pbebench import _parse_programs, _validate_programs, cascade_complexity

_MAX_PROGRAMS = 20

# ── helpers ───────────────────────────────────────────────────────────────────

def _apply(programs, s: str) -> str:
    for pred, transform in programs:
        s = s.replace(pred, transform)
    return s


def _score_programs(programs, inputs, outputs) -> float:
    correct = sum(1 for i, o in zip(inputs, outputs) if _apply(programs, i) == o)
    return correct / len(inputs) if inputs else 0.0


def _candidate_score_complexity(candidate: str, inputs: list, outputs: list):
    """Return (score, complexity) for a candidate string, or (0.0, inf) if unparseable."""
    programs, _ = _parse_programs(candidate)
    if programs is None or _validate_programs(programs, max_programs=_MAX_PROGRAMS):
        return 0.0, float("inf")
    return _score_programs(programs, inputs, outputs), cascade_complexity(programs)


def _best_candidate(candidates, inputs, outputs):
    """Select candidate by max reward, tiebreak min complexity. Returns (score, complexity)."""
    best_score, best_complexity = -1.0, float("inf")
    for cand in candidates:
        score, complexity = _candidate_score_complexity(cand, inputs, outputs)
        if score > best_score or (score == best_score and complexity < best_complexity):
            best_score, best_complexity = score, complexity
    return best_score, best_complexity if best_complexity != float("inf") else None


# ── loaders ───────────────────────────────────────────────────────────────────

def load_gt(tasks_path: Path) -> dict[int, int]:
    """Return {task_id: gt_complexity}."""
    gt = {}
    with open(tasks_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            progs_raw = rec.get("original_programs") or rec.get("programs")
            programs, _ = _parse_programs(progs_raw)
            if programs:
                gt[i] = cascade_complexity(programs)
    return gt


def load_bok(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            inp = rec["input"]
            task_id = int(inp["index"].split("_")[0])
            inputs, outputs = inp["inputs"], inp["outputs"]
            score, complexity = _best_candidate(rec.get("outputs", []), inputs, outputs)
            records[task_id] = {
                "cascade_length": inp.get("cascade_length"),
                "best_score": score,
                "complexity": complexity,
            }
    return records


def load_cc_solver(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_index")
            if tid is None:
                continue
            score = float(rec.get("score", 1.0 if rec.get("success") else 0.0))
            programs, _ = _parse_programs(rec.get("program"))
            complexity = cascade_complexity(programs) if programs else None
            records[tid] = {
                "cascade_length": rec.get("cascade_length"),
                "best_score": score,
                "complexity": complexity,
            }
    return records


def load_standard(path: Path) -> dict[int, dict]:
    """Load any standard quick_eval-compatible JSONL (best_reward + answer).
    One record per task_index; answer parsed via _parse_programs."""
    records: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_index")
            if tid is None:
                continue
            score = float(rec.get("best_reward", 1.0 if rec.get("solved") else 0.0))
            programs, _ = _parse_programs(rec.get("answer"))
            complexity = cascade_complexity(programs) if programs else None
            cl = rec.get("cascade_length")
            prev = records.get(tid)
            if prev is None or score > prev["best_score"] or (
                score == prev["best_score"] and complexity is not None
                and (prev["complexity"] is None or complexity < prev["complexity"])
            ):
                records[tid] = {"cascade_length": cl, "best_score": score, "complexity": complexity}
    return records


def load_qwen_solver(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_index")
            if tid is None:
                continue
            score = float(rec.get("best_reward", 1.0 if rec.get("solved") else 0.0))
            answer = rec.get("answer")
            complexity = None
            if answer and isinstance(answer, list):
                if answer and isinstance(answer[0], list):
                    # already list of [pred, transform] pairs
                    programs = [(p, t) for p, t in answer]
                else:
                    programs, _ = _parse_programs(answer)
                if programs:
                    complexity = cascade_complexity(programs)
            records[tid] = {
                "cascade_length": rec.get("cascade_length"),
                "best_score": score,
                "complexity": complexity,
            }
    return records


# ── stats ─────────────────────────────────────────────────────────────────────

def complexity_stats(records: dict[int, dict], gt: dict[int, int], label: str) -> dict:
    """Compute complexity stats over solved tasks only, compared to GT."""
    pairs = []
    for tid, r in records.items():
        if r["best_score"] < 1.0:
            continue
        pred_c = r.get("complexity")
        gt_c = gt.get(tid)
        if pred_c is None or gt_c is None:
            continue
        pairs.append((pred_c, gt_c, r["cascade_length"]))

    n = len(pairs)
    if n == 0:
        return {"label": label, "n": 0}

    mean_pred = sum(p for p, _, _ in pairs) / n
    mean_gt   = sum(g for _, g, _ in pairs) / n
    mean_delta = sum(p - g for p, g, _ in pairs) / n
    simpler    = sum(1 for p, g, _ in pairs if p < g)
    equal      = sum(1 for p, g, _ in pairs if p == g)
    harder     = sum(1 for p, g, _ in pairs if p > g)

    print(f"\n  {label}  (n_solved={n})")
    print(f"    Mean pred complexity : {mean_pred:.2f}")
    print(f"    Mean GT complexity   : {mean_gt:.2f}")
    print(f"    Mean delta (pred−GT) : {mean_delta:+.2f}")
    print(f"    Simpler than GT      : {simpler}/{n} ({100*simpler/n:.1f}%)")
    print(f"    Equal to GT          : {equal}/{n} ({100*equal/n:.1f}%)")
    print(f"    More complex than GT : {harder}/{n} ({100*harder/n:.1f}%)")

    return {
        "label": label, "n": n,
        "mean_pred": round(mean_pred, 2), "mean_gt": round(mean_gt, 2),
        "mean_delta": round(mean_delta, 2),
        "simpler": simpler, "equal": equal, "harder": harder,
    }


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = [
    ("#D97706", "#FDE68A"),  # amber  — BoK-32
    ("#2563EB", "#BFDBFE"),  # blue   — CC solver
    ("#DC2626", "#FECACA"),  # red    — Qwen solver
    ("#6B7280", "#E5E7EB"),  # grey   — GT
]


def make_complexity_plot(
    systems: list[tuple[str, dict[int, dict]]],
    gt: dict[int, int],
    out: Path,
    subtitle: str = "PBEBench-Hard, solved tasks only, 64 tasks per level",
) -> None:
    """Mean complexity vs cascade length (solved tasks only), with GT as reference."""
    all_cls = sorted(set(
        r["cascade_length"]
        for _, recs in systems for r in recs.values()
        if r["cascade_length"] is not None
    ))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # GT line first (grey, dashed)
    gt_by_cl: defaultdict = defaultdict(list)
    for tid, c in gt.items():
        # need CL — get from any system that has this task
        cl = next((recs[tid]["cascade_length"] for _, recs in systems if tid in recs and recs[tid]["cascade_length"] is not None), None)
        if cl is not None:
            gt_by_cl[cl].append(c)
    gt_vals = [sum(gt_by_cl[cl]) / len(gt_by_cl[cl]) if gt_by_cl[cl] else float("nan") for cl in all_cls]
    ax.plot(all_cls, gt_vals, marker="s", markersize=5, linewidth=1.8,
            linestyle="--", color="#9CA3AF", zorder=2, label="Ground Truth")
    ax.fill_between(all_cls, gt_vals, alpha=0.06, color="#E5E7EB")

    for i, (label, recs) in enumerate(systems):
        color_line, color_fill = COLORS[i % len(COLORS)]
        by_cl: defaultdict = defaultdict(list)
        for r in recs.values():
            if r["best_score"] >= 1.0 and r.get("complexity") is not None and r["cascade_length"] is not None:
                by_cl[r["cascade_length"]].append(r["complexity"])
        vals = [sum(by_cl[cl]) / len(by_cl[cl]) if by_cl[cl] else float("nan") for cl in all_cls]

        ax.fill_between(all_cls, vals, alpha=0.10, color=color_fill)
        ax.plot(all_cls, vals, marker="o", markersize=6, linewidth=2.0,
                color=color_line, zorder=3, label=label)

    ax.set_xlabel("Cascade length (number of replace() programs)", fontsize=12)
    ax.set_ylabel("Mean cascade complexity", fontsize=12)
    ax.set_title(
        f"Mean complexity of solutions vs cascade length\n({subtitle})",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xticks(all_cls)
    ax.set_xlim(all_cls[0] - 0.5, all_cls[-1] + 0.5)
    ax.grid(axis="y", linewidth=0.5, color="#E5E7EB")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bok",       default="outputs/gpt_oss_120b_pbebench_outputs.jsonl")
    parser.add_argument("--solver-cc", default="evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl")
    parser.add_argument("--solver-qw", default="evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl")
    parser.add_argument("--tasks",     default="data/pbebench/tasks_full_og.jsonl")
    parser.add_argument("--plot",      default=None, metavar="FILE")
    parser.add_argument("--metrics-json", default=None, metavar="FILE")
    args = parser.parse_args()

    print("Loading GT complexity…")
    gt = load_gt(Path(args.tasks))
    print(f"  {len(gt)} tasks")

    print("Loading BoK-32…")
    bok  = load_bok(Path(args.bok))
    print("Loading Symbolic Solver (Claude Code)…")
    cc   = load_cc_solver(Path(args.solver_cc))
    print("Loading Symbolic Solver (Qwen3.6-35B-A3B)…")
    qwen = load_qwen_solver(Path(args.solver_qw))

    systems = [
        ("BoK-32 (gpt-oss-120b)",             bok),
        ("Symbolic Solver (Claude Code)",      cc),
        ("Symbolic Solver (Qwen3.6-35B-A3B)", qwen),
    ]

    print(f"\n{'='*58}")
    print("  Complexity vs Ground Truth  (solved tasks only)")
    print(f"{'='*58}")
    all_stats = [complexity_stats(recs, gt, label) for label, recs in systems]

    if args.metrics_json:
        with open(args.metrics_json, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nMetrics written to {args.metrics_json}")

    if args.plot:
        make_complexity_plot(systems, gt, Path(args.plot),
                             subtitle="PBEBench-Hard, solved tasks only, 64 tasks per level")


if __name__ == "__main__":
    main()
