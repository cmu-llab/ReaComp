"""
Plotting script for symbolic-library-agent eval results.

Generates a 2x3 figure with six panels:
  1. Reward distribution (histogram + KDE)
  2. Pass rate by source dataset (bar chart)
  3. Mean reward by source dataset (bar chart)
  4. Reward-loop iteration breakdown (stacked bar: first-perfect-iter vs never)
  5. Blame sequence heatmap (top sequences x runs)
  6. Cost breakdown (total_cost + objective distributions)

Usage:
    python scripts/plot_eval.py outputs/pbebench_lite_pilot_tasks_with_rewards.jsonl
    python scripts/plot_eval.py outputs/pbebench_lite_pilot_tasks_with_rewards.jsonl \
                                outputs/reasoning_gym_easy_pilot_with_rewards.jsonl
    python scripts/plot_eval.py outputs/*.jsonl --out results/eval_plots.png
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── data loading (mirrors eval.py) ────────────────────────────────────────────

def _source_dataset(record: dict) -> str:
    for h in record.get("reward_history", []):
        msg = h.get("message", "")
        if " for " in msg:
            return msg.split(" for ")[1].split(".")[0].strip()
    return record.get("task_type", "unknown")


def _first_perfect_iter(rh: list) -> int | None:
    for h in rh:
        if h.get("reward", 0.0) >= 1.0:
            return h["iteration"]
    return None


def record_to_row(rec: dict) -> dict:
    best_reward = rec.get("best_reward")
    if best_reward is None:
        best_reward = 1.0 if rec.get("solved") else 0.0
    cost = rec.get("cost_summary") or {}
    rh = rec.get("reward_history", [])
    blames = [h.get("blame", "?") for h in rh]
    return {
        "source_dataset":     _source_dataset(rec),
        "best_reward":        best_reward,
        "task_loss":          1.0 - best_reward,
        "solved":             int(bool(rec.get("solved"))),
        "num_iters":          len(rh),
        "first_perfect_iter": _first_perfect_iter(rh),
        "blame_sequence":     "→".join(blames) if blames else "(no retries)",
        "total_cost":         cost.get("total_cost", np.nan),
        "objective":          cost.get("objective", np.nan),
        "num_new_functions":  cost.get("num_new_functions", np.nan),
        "reuse_count":        cost.get("reuse_count", np.nan),
    }


def load_file(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(record_to_row(json.loads(line)))
    df = pd.DataFrame(rows)
    df["run"] = Path(path).stem
    return df


# ── plot helpers ───────────────────────────────────────────────────────────────

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def _ax_style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


# ── individual panels ──────────────────────────────────────────────────────────

def plot_reward_hist(ax, dfs: list[pd.DataFrame], labels: list[str]):
    """Panel 1: reward distribution histogram per run."""
    bins = np.linspace(0, 1, 21)
    for df, label, color in zip(dfs, labels, PALETTE):
        ax.hist(df["best_reward"], bins=bins, alpha=0.6, label=label,
                color=color, edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=7, framealpha=0.7)
    _ax_style(ax, "Reward Distribution", xlabel="Best Reward", ylabel="Count")


def plot_pass_rate_by_dataset(ax, dfs: list[pd.DataFrame], labels: list[str]):
    """Panel 2: pass rate grouped by source dataset."""
    all_df = pd.concat(
        [df.assign(run=label) for df, label in zip(dfs, labels)],
        ignore_index=True,
    )
    datasets = sorted(all_df["source_dataset"].unique())
    runs = labels
    x = np.arange(len(datasets))
    width = 0.8 / max(len(runs), 1)

    for i, (run, color) in enumerate(zip(runs, PALETTE)):
        sub = all_df[all_df["run"] == run]
        rates = [sub[sub["source_dataset"] == ds]["solved"].mean()
                 if len(sub[sub["source_dataset"] == ds]) else np.nan
                 for ds in datasets]
        offset = (i - len(runs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=run, color=color, alpha=0.85)
        for bar, rate in zip(bars, rates):
            if not np.isnan(rate):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{rate:.0%}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=7, framealpha=0.7)
    _ax_style(ax, "Pass Rate by Dataset", ylabel="Pass Rate")


def plot_mean_reward_by_dataset(ax, dfs: list[pd.DataFrame], labels: list[str]):
    """Panel 3: mean reward grouped by source dataset."""
    all_df = pd.concat(
        [df.assign(run=label) for df, label in zip(dfs, labels)],
        ignore_index=True,
    )
    datasets = sorted(all_df["source_dataset"].unique())
    runs = labels
    x = np.arange(len(datasets))
    width = 0.8 / max(len(runs), 1)

    for i, (run, color) in enumerate(zip(runs, PALETTE)):
        sub = all_df[all_df["run"] == run]
        means = [sub[sub["source_dataset"] == ds]["best_reward"].mean()
                 if len(sub[sub["source_dataset"] == ds]) else np.nan
                 for ds in datasets]
        offset = (i - len(runs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=run, color=color, alpha=0.85)
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{m:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7, framealpha=0.7)
    _ax_style(ax, "Mean Reward by Dataset", ylabel="Mean Best Reward")


def plot_iter_breakdown(ax, dfs: list[pd.DataFrame], labels: list[str]):
    """Panel 4: stacked bar of first-perfect-iter distribution per run."""
    max_iters = max(
        int(df["first_perfect_iter"].max())
        for df in dfs
        if df["first_perfect_iter"].notna().any()
    )
    iter_labels = [f"iter {i}" for i in range(max_iters + 1)] + ["never solved"]
    x = np.arange(len(labels))
    bottoms = np.zeros(len(labels))

    colors = PALETTE[:len(iter_labels)]
    for j, it_label in enumerate(iter_labels):
        heights = []
        for df in dfs:
            n = len(df)
            if it_label == "never solved":
                cnt = df["first_perfect_iter"].isna().sum()
            else:
                it = int(it_label.split()[1])
                cnt = (df["first_perfect_iter"] == it).sum()
            heights.append(cnt / n)
        bars = ax.bar(x, heights, bottom=bottoms, label=it_label,
                      color=colors[j], alpha=0.9, edgecolor="white", linewidth=0.4)
        for bar, h in zip(bars, heights):
            if h > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{h:.0%}", ha="center", va="center", fontsize=7, color="white")
        bottoms += np.array(heights)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=7, framealpha=0.7, loc="lower right")
    _ax_style(ax, "Reward-Loop Iteration Breakdown", ylabel="Fraction of Tasks")


def plot_blame_heatmap(ax, dfs: list[pd.DataFrame], labels: list[str]):
    """Panel 5: blame-sequence heatmap (top sequences vs runs)."""
    all_seqs = Counter()
    for df in dfs:
        all_seqs.update(df["blame_sequence"].value_counts().to_dict())
    top_seqs = [s for s, _ in all_seqs.most_common(8)]

    matrix = np.zeros((len(top_seqs), len(labels)))
    for j, (df, label) in enumerate(zip(dfs, labels)):
        counts = df["blame_sequence"].value_counts()
        n = len(df)
        for i, seq in enumerate(top_seqs):
            matrix[i, j] = counts.get(seq, 0) / n

    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_seqs)))
    ax.set_yticklabels(top_seqs, fontsize=7)
    for i in range(len(top_seqs)):
        for j in range(len(labels)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    _ax_style(ax, "Blame Sequence Distribution")


def plot_cost(ax, dfs: list[pd.DataFrame], labels: list[str]):
    """Panel 6: total_cost and objective box plots per run."""
    data_cost = [df["total_cost"].dropna().values for df in dfs]
    data_obj  = [df["objective"].dropna().values  for df in dfs]

    x = np.arange(len(labels))
    width = 0.35

    bp1 = ax.boxplot(data_cost, positions=x - width / 2, widths=width * 0.9,
                     patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
    bp2 = ax.boxplot(data_obj,  positions=x + width / 2, widths=width * 0.9,
                     patch_artist=True, medianprops=dict(color="black", linewidth=1.5))

    for patch in bp1["boxes"]:
        patch.set_facecolor(PALETTE[0])
        patch.set_alpha(0.7)
    for patch in bp2["boxes"]:
        patch.set_facecolor(PALETTE[1])
        patch.set_alpha(0.7)

    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["total_cost", "objective"],
              fontsize=7, framealpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    _ax_style(ax, "Cost Distribution", ylabel="Value")


# ── individual plot saving ─────────────────────────────────────────────────────

PANELS = [
    ("reward_hist",            plot_reward_hist,            (6, 4)),
    ("pass_rate_by_dataset",   plot_pass_rate_by_dataset,   (7, 4)),
    ("mean_reward_by_dataset", plot_mean_reward_by_dataset, (7, 4)),
    ("iter_breakdown",         plot_iter_breakdown,         (6, 4)),
    ("blame_heatmap",          plot_blame_heatmap,          (7, 4)),
    ("cost",                   plot_cost,                   (6, 4)),
]


def save_individual_plots(dfs, labels, out_dir: str, dpi: int = 150):
    """Save each panel as a separate PNG in out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, fn, figsize in PANELS:
        fig, ax = plt.subplots(figsize=figsize)
        fn(ax, dfs, labels)
        plt.tight_layout()
        dest = out_path / f"{name}.png"
        fig.savefig(dest, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {dest}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot eval results for symbolic-library-agent runs.")
    parser.add_argument("files", nargs="+", metavar="FILE", help="One or more output JSONL files")
    parser.add_argument("--out", default=None, metavar="FILE",
                        help="Save combined 2x3 figure to this path")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help="Save each panel as a separate PNG in this directory")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if args.out is None and args.out_dir is None:
        parser.error("Provide --out FILE and/or --out-dir DIR")

    dfs = [load_file(p) for p in args.files]
    labels = [Path(p).stem for p in args.files]
    # Shorten long stem names for readability
    short = []
    for lb in labels:
        lb = lb.replace("_pilot_tasks_with_rewards", "").replace("_easy", "")
        lb = lb.replace("_lite", "")
        short.append(lb)
    labels = short

    if args.out_dir:
        print(f"Saving individual panels to {args.out_dir}/")
        save_individual_plots(dfs, labels, args.out_dir, dpi=args.dpi)

    if args.out:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle("Symbolic Library Agent — Eval Summary", fontsize=13, fontweight="bold", y=0.98)
        plot_reward_hist(axes[0, 0], dfs, labels)
        plot_pass_rate_by_dataset(axes[0, 1], dfs, labels)
        plot_mean_reward_by_dataset(axes[0, 2], dfs, labels)
        plot_iter_breakdown(axes[1, 0], dfs, labels)
        plot_blame_heatmap(axes[1, 1], dfs, labels)
        plot_cost(axes[1, 2], dfs, labels)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved combined plot to {args.out}")


if __name__ == "__main__":
    main()
