"""
Plot symbolic solver pass rate vs cascade length on PBEBench-Hard.
Supports overlaying multiple solver outputs for comparison.

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
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = [REPO_ROOT / "evals/solver_results/hard.jsonl"]
DEFAULT_OUT   = REPO_ROOT / "figures/solver_cascade_passrate.png"

COLORS = [
    ("#2563EB", "#BFDBFE"),  # blue (Claude Code)
    ("#DC2626", "#FECACA"),  # red  (Qwen)
    ("#16A34A", "#BBF7D0"),  # green (third solver if needed)
]


def load(path: Path):
    by_cl = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cl = r.get("cascade_length")
            success = r.get("solved") if "solved" in r else r.get("success")
            if cl is not None:
                by_cl[cl].append(bool(success))
    return by_cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=None, metavar="FILE",
                        help="One or more JSONL result files to plot (default: evals/solver_results/hard.jsonl)")
    parser.add_argument("--labels", nargs="+", default=None, metavar="LABEL",
                        help="Legend labels for each input file (default: filename stems)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), metavar="FILE")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.input] if args.input else DEFAULT_INPUT
    labels = args.labels if args.labels else [p.parent.parent.name or p.stem for p in inputs]

    if len(labels) < len(inputs):
        labels += [inputs[i].parent.parent.name or inputs[i].stem for i in range(len(labels), len(inputs))]

    # Collect all cascade lengths across files so x-axis is consistent
    all_data = [load(p) for p in inputs]
    all_cascade_lengths = sorted(set(cl for d in all_data for cl in d))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (by_cl, label) in enumerate(zip(all_data, labels)):
        color_line, color_fill = COLORS[i % len(COLORS)]
        pass_rates = [
            100 * sum(by_cl[cl]) / len(by_cl[cl]) if by_cl[cl] else float("nan")
            for cl in all_cascade_lengths
        ]
        ns = [len(by_cl[cl]) for cl in all_cascade_lengths]

        ax.fill_between(all_cascade_lengths, pass_rates, alpha=0.12, color=color_fill)
        ax.plot(all_cascade_lengths, pass_rates, marker="o", markersize=6,
                linewidth=2.0, color=color_line, zorder=3, label=label)

        # Annotate only when a single solver (avoid clutter in comparison mode)
        if len(inputs) == 1:
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

    if len(inputs) == 1:
        ax.set_title(
            "Symbolic solver pass rate vs cascade length\n(PBEBench-Hard, 64 tasks per level)",
            fontsize=13, fontweight="bold",
        )
    else:
        ax.set_title(
            "Symbolic solver pass rate vs cascade length — comparison\n(PBEBench-Hard, 64 tasks per level)",
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
