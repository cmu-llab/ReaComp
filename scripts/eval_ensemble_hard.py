"""
Ensemble evaluation for PBEBench-Hard: union of BoK-32, Symbolic Solver (Claude Code),
and Symbolic Solver (Qwen3.6-35B-A3B).

Union ensemble = per task, take the best score across all included systems.

Usage:
    python scripts/eval_ensemble_hard.py
    python scripts/eval_ensemble_hard.py --metrics-json figures/ensemble_hard_metrics.json
    python scripts/eval_ensemble_hard.py --plot figures/ensemble_hard_cascade.png
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

from rewards.pbebench import _parse_programs, _validate_programs

_BOK_MAX_PROGRAMS = 20

# ── loaders ──────────────────────────────────────────────────────────────────

def _score_candidate(candidate: str, inputs: list, outputs: list) -> float:
    programs, _ = _parse_programs(candidate)
    if programs is None:
        return 0.0
    if _validate_programs(programs, max_programs=_BOK_MAX_PROGRAMS):
        return 0.0
    correct = sum(
        1 for inp, exp in zip(inputs, outputs)
        if _apply(programs, inp) == exp
    )
    return correct / len(inputs) if inputs else 0.0


def _apply(programs, s: str) -> str:
    for pred, transform in programs:
        s = s.replace(pred, transform)
    return s


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
            cl = inp.get("cascade_length")
            inputs, outputs = inp["inputs"], inp["outputs"]
            best = max((_score_candidate(c, inputs, outputs) for c in rec.get("outputs", [])), default=0.0)
            records[task_id] = {"cascade_length": cl, "best_score": best, "solved": best >= 1.0}
    return records


def load_cc_solver(path: Path) -> dict[int, dict]:
    """CC solver JSONL: task_index, inputs, outputs, program (list of strings), score."""
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
            records[tid] = {
                "cascade_length": rec.get("cascade_length"),
                "best_score": score,
                "solved": score >= 1.0,
            }
    return records


def load_qwen_solver(path: Path) -> dict[int, dict]:
    """Qwen solver JSONL: task_index, answer (list of [pred,transform] tuples), best_reward."""
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
            records[tid] = {
                "cascade_length": rec.get("cascade_length"),
                "best_score": score,
                "solved": score >= 1.0,
            }
    return records


# ── ensemble ─────────────────────────────────────────────────────────────────

def ensemble(*systems: dict[int, dict]) -> dict[int, dict]:
    """Union ensemble: per task take max best_score across all systems."""
    all_ids = set(tid for s in systems for tid in s)
    result: dict[int, dict] = {}
    for tid in all_ids:
        recs = [s[tid] for s in systems if tid in s]
        best = max(r["best_score"] for r in recs)
        cl = next((r["cascade_length"] for r in recs if r["cascade_length"] is not None), None)
        result[tid] = {"cascade_length": cl, "best_score": best, "solved": best >= 1.0}
    return result


# ── reporting ─────────────────────────────────────────────────────────────────

def summarise(label: str, records: dict[int, dict]) -> dict:
    recs = list(records.values())
    n = len(recs)
    solved = sum(1 for r in recs if r["solved"])
    mean_r = sum(r["best_score"] for r in recs) / n if n else 0.0
    return {"label": label, "n": n, "solved": solved,
            "pass_pct": round(100 * solved / n, 2) if n else 0.0,
            "mean_reward": round(mean_r, 4)}


def print_table(metrics: list[dict]) -> None:
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {'System':<38}  {'n':>5}  {'Pass%':>6}  {'MeanR':>6}")
    print(sep)
    for m in metrics:
        mark = "  ←" if "∪" in m["label"] else ""
        print(f"  {m['label']:<38}  {m['n']:>5}  {m['pass_pct']:>5.1f}%  {m['mean_reward']:>6.4f}{mark}")
    print(sep)


def by_cl_dict(records: dict[int, dict]) -> defaultdict:
    d: defaultdict = defaultdict(list)
    for r in records.values():
        if r["cascade_length"] is not None:
            d[r["cascade_length"]].append(r["best_score"])
    return d


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = [
    # individual systems (lighter / dashed)
    ("#93C5FD", "#EFF6FF"),  # light blue  — BoK-32
    ("#6EE7B7", "#ECFDF5"),  # light green — CC solver
    ("#FCA5A5", "#FEF2F2"),  # light red   — Qwen solver
    # ensembles (bold / solid)
    ("#1D4ED8", "#DBEAFE"),  # deep blue   — BoK∪CC
    ("#7C3AED", "#EDE9FE"),  # purple      — BoK∪Qwen
    ("#B45309", "#FEF3C7"),  # amber       — BoK∪CC∪Qwen
]


def make_plot(series: list[tuple[str, defaultdict, bool]], out: Path, metric: str = "pass_rate") -> None:
    """
    series: list of (label, by_cl_dict, is_ensemble)
    metric: 'pass_rate' or 'mean_reward'
    """
    all_cls = sorted(set(cl for _, d, _ in series for cl in d))
    use_mr = metric == "mean_reward"

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, by_cl, is_ens) in enumerate(series):
        color_line, color_fill = COLORS[i % len(COLORS)]
        if use_mr:
            vals = [sum(by_cl[cl]) / len(by_cl[cl]) if by_cl[cl] else float("nan") for cl in all_cls]
        else:
            vals = [100 * sum(1 for v in by_cl[cl] if v >= 1.0) / len(by_cl[cl]) if by_cl[cl] else float("nan") for cl in all_cls]

        lw = 2.2 if is_ens else 1.4
        ls = "-" if is_ens else "--"
        alpha_fill = 0.12 if is_ens else 0.06
        ms = 6 if is_ens else 4
        zorder = 4 if is_ens else 2
        ax.fill_between(all_cls, vals, alpha=alpha_fill, color=color_fill)
        ax.plot(all_cls, vals, marker="o", markersize=ms, linewidth=lw,
                linestyle=ls, color=color_line, zorder=zorder, label=label)

    # Reference lines
    top = 1.12 if use_mr else 112
    ref_top = 1.0 if use_mr else 100
    ref_mid = 0.5 if use_mr else 50
    ax.axhline(ref_top, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(ref_mid, color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)

    ylabel = "Mean reward" if use_mr else "Pass rate (%)"
    title_metric = "Mean reward" if use_mr else "Pass rate"
    ax.set_xlabel("Cascade length (number of replace() programs)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"{title_metric} vs cascade length — individual & ensemble\n(PBEBench-Hard, 64 tasks per level)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower left", ncol=2)
    ax.set_xticks(all_cls)
    ax.set_xlim(all_cls[0] - 0.5, all_cls[-1] + 0.5)
    ax.set_ylim(0, top)
    if use_mr:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    else:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.grid(axis="y", linewidth=0.5, color="#E5E7EB")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bok",       default="outputs/gpt_oss_120b_pbebench_outputs.jsonl")
    parser.add_argument("--solver-cc", default="evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl")
    parser.add_argument("--solver-qw", default="evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl")
    parser.add_argument("--plot",          default=None, metavar="FILE",
                        help="Save pass-rate plot to this path")
    parser.add_argument("--plot-mr",       default=None, metavar="FILE",
                        help="Save mean-reward plot to this path")
    parser.add_argument("--metrics-json",  default=None, metavar="FILE")
    args = parser.parse_args()

    print("Loading BoK-32 (gpt-oss-120b)…")
    bok  = load_bok(Path(args.bok))
    print("Loading Symbolic Solver (Claude Code)…")
    cc   = load_cc_solver(Path(args.solver_cc))
    print("Loading Symbolic Solver (Qwen3.6-35B-A3B)…")
    qwen = load_qwen_solver(Path(args.solver_qw))

    print("Computing ensembles…")
    ens_bok_cc   = ensemble(bok, cc)
    ens_bok_qw   = ensemble(bok, qwen)
    ens_all      = ensemble(bok, cc, qwen)

    all_systems = [
        ("BoK-32 (gpt-oss-120b)",                bok,        False),
        ("Symbolic Solver (Claude Code)",          cc,         False),
        ("Symbolic Solver (Qwen3.6-35B-A3B)",     qwen,       False),
        ("BoK-32 ∪ Symbolic Solver (Claude Code)", ens_bok_cc, True),
        ("BoK-32 ∪ Symbolic Solver (Qwen3.6)",    ens_bok_qw, True),
        ("BoK-32 ∪ CC Solver ∪ Qwen Solver",      ens_all,    True),
    ]

    metrics = [summarise(label, recs) for label, recs, _ in all_systems]
    print_table(metrics)

    if args.metrics_json:
        with open(args.metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to {args.metrics_json}")

    series = [(label, by_cl_dict(recs), is_ens) for label, recs, is_ens in all_systems]

    if args.plot:
        make_plot(series, Path(args.plot), metric="pass_rate")
    if args.plot_mr:
        make_plot(series, Path(args.plot_mr), metric="mean_reward")


if __name__ == "__main__":
    main()
