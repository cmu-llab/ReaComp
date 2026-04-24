"""
Plot symbolic solver pass rate vs cascade length on PBEBench-Hard.

Usage:
    python scripts/plot_solver_cascade.py
    python scripts/plot_solver_cascade.py --input evals/solver_results/hard.jsonl
    python scripts/plot_solver_cascade.py --out figures/solver_cascade.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = REPO_ROOT / "evals/solver_results/hard.jsonl"
DEFAULT_OUT   = REPO_ROOT / "figures/solver_cascade_passrate.png"


def load(path: Path):
    by_cl = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cl = r.get("cascade_length")
            if cl is not None:
                by_cl[cl].append(r)
    return by_cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT), metavar="FILE")
    parser.add_argument("--out",   default=str(DEFAULT_OUT),   metavar="FILE")
    args = parser.parse_args()

    by_cl = load(Path(args.input))
    cascade_lengths = sorted(by_cl)
    pass_rates = [
        100 * sum(r["success"] for r in by_cl[cl]) / len(by_cl[cl])
        for cl in cascade_lengths
    ]
    ns = [len(by_cl[cl]) for cl in cascade_lengths]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    color_line  = "#2563EB"   # blue
    color_fill  = "#BFDBFE"   # light blue

    ax.fill_between(cascade_lengths, pass_rates, alpha=0.18, color=color_fill)
    ax.plot(cascade_lengths, pass_rates, marker="o", markersize=6,
            linewidth=2.0, color=color_line, zorder=3)

    # Annotate each point with pass%
    for cl, pr in zip(cascade_lengths, pass_rates):
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
    ax.set_title("Symbolic solver pass rate vs cascade length\n(PBEBench-Hard, 64 tasks per level)",
                 fontsize=13, fontweight="bold")

    ax.set_xticks(cascade_lengths)
    ax.set_xlim(cascade_lengths[0] - 0.5, cascade_lengths[-1] + 0.5)
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
