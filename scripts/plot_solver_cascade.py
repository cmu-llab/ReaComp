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
                evals/solver_results/qwen3.6_coder/Fri_Apr_24_200_AM/hard.jsonl \\
        --labels "Claude Code" "Qwen3.6-Coder (OH)" \\
        --out figures/solver_cascade_passrate_comparison.png

    # Add BoK line
    python scripts/plot_solver_cascade.py \\
        --input evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl \\
                evals/solver_results/qwen3.6_coder/Fri_Apr_24_200_AM/hard.jsonl \\
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
                solved = float(r["best_reward"]) >= 1.0
            elif "solved" in r:
                solved = bool(r["solved"])
            else:
                solved = bool(r.get("success"))
            by_cl[cl].append(solved)
    return by_cl


def _score_candidate(candidate: str, inputs: list, outputs: list) -> bool:
    programs, err = _parse_programs(candidate)
    if programs is None:
        return False
    if _validate_programs(programs, max_programs=_BOK_MAX_PROGRAMS):
        return False
    for inp, exp in zip(inputs, outputs):
        s = inp
        for pred, transform in programs:
            s = s.replace(pred, transform)
        if s != exp:
            return False
    return True


def load_bok(path: Path) -> defaultdict:
    """Load BoK sampling JSONL {input: {inputs, outputs, cascade_length, index}, outputs: [candidates]}."""
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
            solved = any(_score_candidate(c, inputs, outputs) for c in rec.get("outputs", []))
            by_cl[cl].append(solved)
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
    parser.add_argument("--out", default=str(DEFAULT_OUT), metavar="FILE")
    args = parser.parse_args()

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

    for i, (by_cl, label, is_bok) in enumerate(all_series):
        color_line, color_fill = COLORS[i % len(COLORS)]
        pass_rates = [
            100 * sum(by_cl[cl]) / len(by_cl[cl]) if by_cl[cl] else float("nan")
            for cl in all_cascade_lengths
        ]

        linestyle = "--" if is_bok else "-"
        ax.fill_between(all_cascade_lengths, pass_rates, alpha=0.10, color=color_fill)
        ax.plot(all_cascade_lengths, pass_rates, marker="o", markersize=6,
                linewidth=2.0, color=color_line, linestyle=linestyle, zorder=3, label=label)

        # Annotate only for single-series plots
        if total_series == 1:
            for cl, pr in zip(all_cascade_lengths, pass_rates):
                if not np.isnan(pr):
                    ax.annotate(
                        f"{pr:.0f}%",
                        xy=(cl, pr),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=7.5, color="#1E3A5F",
                    )

    # Reference lines
    ax.axhline(100, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(50,  color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)

    ax.set_xlabel("Cascade length (number of replace() programs)", fontsize=12)
    ax.set_ylabel("Pass rate (%)", fontsize=12)

    if total_series == 1:
        ax.set_title(
            "Symbolic solver pass rate vs cascade length\n(PBEBench-Hard, 64 tasks per level)",
            fontsize=13, fontweight="bold",
        )
    else:
        ax.set_title(
            "Pass rate vs cascade length — comparison\n(PBEBench-Hard, 64 tasks per level)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=11, loc="upper right")

    ax.set_xticks(all_cascade_lengths)
    ax.set_xlim(all_cascade_lengths[0] - 0.5, all_cascade_lengths[-1] + 0.5)
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
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
