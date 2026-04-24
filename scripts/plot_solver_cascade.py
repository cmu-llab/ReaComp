"""
Plot symbolic solver pass rate vs cascade length on PBEBench-Hard.
Supports overlaying multiple solver outputs for comparison, including
best-of-k (BoK) sampling outputs.

Usage:
    # Single solver (original behaviour)
    python scripts/plot_solver_cascade.py
    python scripts/plot_solver_cascade.py --input evals/solver_results/hard.jsonl

    # Compare two solvers
    python scripts/plot_solver_cascade.py \\
        --input evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl \\
                evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl \\
        --labels "Claude Code" "Qwen3.6-Coder (OH)" \\
        --out figures/solver_cascade_passrate_comparison.png

    # Add BoK line
    python scripts/plot_solver_cascade.py \\
        --input evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl \\
                evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl \\
        --labels "Claude Code" "Qwen3.6-Coder (OH)" \\
        --bok outputs/gpt_oss_120b_pbebench_outputs.jsonl \\
        --bok-labels "BoK-32 (gpt-oss-120b)" \\
        --out figures/solver_cascade_passrate_with_bok.png
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rewards.pbebench import _parse_programs, _validate_programs

DEFAULT_INPUT = [REPO_ROOT / "evals/solver_results/hard.jsonl"]
DEFAULT_OUT   = REPO_ROOT / "figures/solver_cascade_passrate.png"

COLORS = [
    ("#2563EB", "#BFDBFE"),  # blue  (Claude Code)
    ("#DC2626", "#FECACA"),  # red   (Qwen)
    ("#D97706", "#FDE68A"),  # amber (BoK / third system)
    ("#16A34A", "#BBF7D0"),  # green (fourth if needed)
]

_BOK_MAX_PROGRAMS = 20  # full PBEBench hard


def load(path: Path) -> defaultdict:
    """Load standard solver JSONL {cascade_length, solved/success/best_reward}."""
    by_cl: defaultdict = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cl = r.get("cascade_length")
            if cl is None:
                continue
            if r.get("best_reward") is not None:
                score = float(r["best_reward"])
            elif "solved" in r:
                score = 1.0 if r["solved"] else 0.0
            else:
                score = 1.0 if r.get("success") else 0.0
            by_cl[cl].append(score)
    return by_cl


def _score_candidate(candidate: str, inputs: list, outputs: list) -> float:
    """Return fraction of I/O pairs correctly mapped (1.0 = fully solved)."""
    programs, err = _parse_programs(candidate)
    if programs is None:
        return 0.0
    if _validate_programs(programs, max_programs=_BOK_MAX_PROGRAMS):
        return 0.0
    correct = 0
    for inp, exp in zip(inputs, outputs):
        s = inp
        for pred, transform in programs:
            s = s.replace(pred, transform)
        if s == exp:
            correct += 1
    return correct / len(inputs) if inputs else 0.0


def load_bok(path: Path) -> defaultdict:
    """Load BoK sampling JSONL {input: {inputs, outputs, cascade_length, index}, outputs: [candidates]}.
    Stores the best score across all candidates per task."""
    by_cl: defaultdict = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            inp = rec["input"]
            cl = inp.get("cascade_length")
            if cl is None:
                continue
            inputs = inp["inputs"]
            outputs = inp["outputs"]
            best = max((_score_candidate(c, inputs, outputs) for c in rec.get("outputs", [])), default=0.0)
            by_cl[cl].append(best)
    return by_cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=None, metavar="FILE",
                        help="One or more standard solver JSONL files to plot")
    parser.add_argument("--labels", nargs="+", default=None, metavar="LABEL",
                        help="Legend labels for --input files")
    parser.add_argument("--bok", nargs="+", default=None, metavar="FILE",
                        help="One or more BoK sampling JSONL files to overlay")
    parser.add_argument("--bok-labels", nargs="+", default=None, metavar="LABEL",
                        help="Legend labels for --bok files")
    parser.add_argument("--metric", choices=["pass_rate", "mean_reward"], default="pass_rate",
                        help="Metric to plot on y-axis (default: pass_rate)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), metavar="FILE")
    args = parser.parse_args()
    use_mean_reward = args.metric == "mean_reward"

    inputs = [Path(p) for p in args.input] if args.input else DEFAULT_INPUT
    labels = args.labels if args.labels else [p.parent.parent.name or p.stem for p in inputs]
    if len(labels) < len(inputs):
        labels += [inputs[i].parent.parent.name or inputs[i].stem for i in range(len(labels), len(inputs))]

    bok_inputs = [Path(p) for p in args.bok] if args.bok else []
    bok_labels = args.bok_labels if args.bok_labels else [p.stem for p in bok_inputs]
    if len(bok_labels) < len(bok_inputs):
        bok_labels += [bok_inputs[i].stem for i in range(len(bok_labels), len(bok_inputs))]

    solver_data = [load(p) for p in inputs]
    bok_data = []
    for p in bok_inputs:
        print(f"Loading BoK file (may take a moment): {p}")
        bok_data.append(load_bok(p))

    all_data_combined = solver_data + bok_data
    all_cascade_lengths = sorted(set(cl for d in all_data_combined for cl in d))

    all_series = list(zip(solver_data, labels, [False] * len(solver_data))) + \
                 list(zip(bok_data, bok_labels, [True] * len(bok_data)))
    total_series = len(all_series)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Keep per-series value arrays for crossover detection
    series_vals: list[tuple[bool, list[float]]] = []

    for i, (by_cl, label, is_bok) in enumerate(all_series):
        color_line, color_fill = COLORS[i % len(COLORS)]
        if use_mean_reward:
            values = [
                sum(by_cl[cl]) / len(by_cl[cl]) if by_cl[cl] else float("nan")
                for cl in all_cascade_lengths
            ]
        else:
            values = [
                100 * sum(1 for v in by_cl[cl] if v >= 1.0) / len(by_cl[cl]) if by_cl[cl] else float("nan")
                for cl in all_cascade_lengths
            ]
        series_vals.append((is_bok, values))

        linestyle = "--" if is_bok else "-"
        ax.fill_between(all_cascade_lengths, values, alpha=0.10, color=color_fill)
        ax.plot(all_cascade_lengths, values, marker="o", markersize=6,
                linewidth=2.0, color=color_line, linestyle=linestyle, zorder=3, label=label)

        # Annotate only for single-series plots
        if total_series == 1:
            for cl, v in zip(all_cascade_lengths, values):
                if not np.isnan(v):
                    txt = f"{v:.3f}" if use_mean_reward else f"{v:.0f}%"
                    ax.annotate(
                        txt,
                        xy=(cl, v),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.5, color="#1E3A5F",
                    )

    # Crossover annotation: first CL where best solver exceeds best BoK
    solver_val_lists = [vals for is_bok, vals in series_vals if not is_bok]
    bok_val_lists    = [vals for is_bok, vals in series_vals if is_bok]
    if solver_val_lists and bok_val_lists:
        crossover_cl = None
        crossover_y = None
        for j, cl in enumerate(all_cascade_lengths):
            best_solver = max((v[j] for v in solver_val_lists if not np.isnan(v[j])), default=float("nan"))
            best_bok    = max((v[j] for v in bok_val_lists    if not np.isnan(v[j])), default=float("nan"))
            if not np.isnan(best_solver) and not np.isnan(best_bok) and best_solver > best_bok:
                crossover_cl = cl
                crossover_y  = (best_solver + best_bok) / 2
                break
        if crossover_cl is not None:
            ax.axvline(crossover_cl, color="#6B7280", linewidth=1.2, linestyle=":", zorder=2)
            ax.annotate(
                f"solver leads\nfrom CL {crossover_cl}",
                xy=(crossover_cl, crossover_y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center", ha="left",
                fontsize=8.5, color="#374151",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", lw=0.8),
                arrowprops=dict(arrowstyle="-|>", color="#6B7280", lw=1.0),
            )

    # Reference lines
    if use_mean_reward:
        ax.axhline(1.0, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
        ax.axhline(0.5, color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)
    else:
        ax.axhline(100, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
        ax.axhline(50,  color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)

    ax.set_xlabel("Cascade length (number of replace() programs)", fontsize=12)

    if use_mean_reward:
        ax.set_ylabel("Mean reward", fontsize=12)
        metric_label = "mean reward"
        y_top = 1.12
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    else:
        ax.set_ylabel("Pass rate (%)", fontsize=12)
        metric_label = "pass rate"
        y_top = 112
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    if total_series == 1:
        ax.set_title(
            f"Symbolic solver {metric_label} vs cascade length\n(PBEBench-Hard, 64 tasks per level)",
            fontsize=13, fontweight="bold",
        )
    else:
        ax.set_title(
            f"{metric_label.capitalize()} vs cascade length — comparison\n(PBEBench-Hard, 64 tasks per level)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10, loc="lower left")

    ax.set_xticks(all_cascade_lengths)
    ax.set_xlim(all_cascade_lengths[0] - 0.5, all_cascade_lengths[-1] + 0.5)
    ax.set_ylim(0, y_top)
    ax.grid(axis="y", linewidth=0.5, color="#E5E7EB")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
