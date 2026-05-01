"""
Pareto frontier plot: cost vs accuracy for PBEBench-Hard systems.

Costs:
  - BoK: inference only (no-cache, reasoning billed as output)
  - Solvers: build cost only (zero per-task LLM calls)
  - Effi hybrids: build cost + reduced BoK inference cost
  - DirectSolve (Qwen3.6-35B-A3B OpenHands): per-task inference at AtlasCloud pricing

Usage:
    python scripts/plot_hard_pareto.py
    python scripts/plot_hard_pareto.py --pareto-line
    python scripts/plot_hard_pareto.py --out-dir figures/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Pricing (gpt-oss-120b @ DeepInfra, no-cache, reasoning billed as output)
# ---------------------------------------------------------------------------
INPUT_PRICE  = 0.039 / 1e6
OUTPUT_PRICE = 0.19  / 1e6

# AtlasCloud pricing for Qwen3.6-35B-A3B (DirectSolve / solver build)
ATLAS_INPUT_PRICE  = 0.1612 / 1e6
ATLAS_OUTPUT_PRICE = 0.9653 / 1e6

# Solver build costs (native Qwen3.6-35B-A3B tokenizer, AtlasCloud pricing)
CC_BUILD        = 10.00   # Claude Code PBE session (estimated)
QWEN_R2_BUILD   =  0.85   # Qwen run 2 (Sun_Apr_26_402, exact from trajectory)
QWEN_ALL_BUILD  =  5.11   # All 6 Qwen runs combined (exact from trajectories)


def infer_cost(path):
    """Sum inference cost from a JSONL file (input + output + reasoning as output)."""
    total = 0.0
    for line in open(path):
        r = json.loads(line)
        tok = r.get("token_usage") or {}
        inp = tok.get("input", 0) or 0
        out = tok.get("output", 0) or 0
        rea = tok.get("reasoning", 0) or 0
        total += inp * INPUT_PRICE + (out + rea) * OUTPUT_PRICE
    return total


def infer_cost_atlas(path):
    """Sum inference cost using AtlasCloud Qwen3.6-35B-A3B pricing."""
    total = 0.0
    for line in open(path):
        r = json.loads(line)
        tok = r.get("token_usage") or {}
        inp = tok.get("input", 0) or 0
        out = tok.get("output", 0) or 0
        total += inp * ATLAS_INPUT_PRICE + out * ATLAS_OUTPUT_PRICE
    return total


def accuracy(path):
    rows = [json.loads(l) for l in open(path)]
    return sum(1 for r in rows if r.get("solved") or r.get("success")) / len(rows) * 100


def build_points():
    """Return list of point dicts with name, legend_label, cost, overall, color, marker, size, zorder."""
    out = REPO_ROOT / "outputs"
    sol = REPO_ROOT / "evals" / "solver_results"

    CC_PATH    = sol / "claude_code/Thu_Apr_23_807_PM/hard.jsonl"
    QR2_PATH   = sol / "qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/hard.jsonl"
    BOK_PATH   = out / "hard_bok_converted.jsonl"
    DS_PATH    = out / "hard_direct_solve_openhands.jsonl"

    bok_infer = infer_cost(BOK_PATH)

    raw = [
        # (name, legend_label, cost, acc, color, marker, size, zorder)
        ("CC",          "CC Solver",
         CC_BUILD,      accuracy(CC_PATH),
         "#e6550d", "*", 280, 6),

        ("QO",          "QO Solver",
         QWEN_R2_BUILD, accuracy(QR2_PATH),
         "#fd8d3c", "*", 280, 6),

        ("CC+QO",       "CC Solver + QO Solver",
         CC_BUILD + QWEN_R2_BUILD,
         accuracy(out / "hard_union_cc_qwen_run2.jsonl"),
         "#e6550d", "P", 200, 6),

        ("All solvers", "All solvers",
         CC_BUILD + QWEN_ALL_BUILD,
         accuracy(out / "hard_union_all_solvers.jsonl"),
         "#a63603", "P", 200, 6),

        ("BoK",         "BoK (gpt-oss-120b)",
         bok_infer,     accuracy(BOK_PATH),
         "#3182bd", "o", 100, 5),

        ("BoK+CC",      "BoK + CC Solver",
         infer_cost(out / "hard_effi_bok_cc.jsonl") + CC_BUILD,
         accuracy(out / "hard_effi_bok_cc.jsonl"),
         "#31a354", "D", 120, 5),

        ("BoK+QO",      "BoK + QO Solver",
         infer_cost(out / "hard_effi_bok_qwen_run2.jsonl") + QWEN_R2_BUILD,
         accuracy(out / "hard_effi_bok_qwen_run2.jsonl"),
         "#74c476", "D", 120, 5),

        ("BoK+CC+QO",   "BoK + CC Solver + QO Solver",
         infer_cost(out / "hard_effi_bok_cc_qwen_run2.jsonl") + CC_BUILD + QWEN_R2_BUILD,
         accuracy(out / "hard_effi_bok_cc_qwen_run2.jsonl"),
         "#006d2c", "D", 120, 5),

        ("BoK+All",     "BoK + All solvers",
         infer_cost(out / "hard_effi_bok_all_solvers.jsonl") + CC_BUILD + QWEN_ALL_BUILD,
         accuracy(out / "hard_effi_bok_all_solvers.jsonl"),
         "#00441b", "D", 140, 5),
    ]

    # DirectSolve baseline — only included if output file exists (may be partial)
    if DS_PATH.exists():
        raw.append((
            "DirectSolve", "QO Agent",
            infer_cost_atlas(DS_PATH), accuracy(DS_PATH),
            "#7C3AED", "s", 140, 5,
        ))

    return [
        dict(name=name, legend_label=ll, cost=cost, overall=acc,
             color=color, marker=marker, size=size, zorder=zorder)
        for name, ll, cost, acc, color, marker, size, zorder in raw
    ]


def _pareto_frontier(points):
    valid = sorted([(p["cost"], p["overall"]) for p in points], key=lambda t: t[0])
    frontier, best_y = [], -float("inf")
    for cost, y in valid:
        if y > best_y:
            frontier.append((cost, y))
            best_y = y
    return frontier


def make_figure(points, out_path, dpi=200, draw_pareto=False):
    fig, ax = plt.subplots(figsize=(7, 5))

    if draw_pareto:
        frontier = _pareto_frontier(points)
        if frontier:
            xs = [t[0] for t in frontier] + [max(p["cost"] for p in points) * 1.05]
            ys = [t[1] for t in frontier] + [frontier[-1][1]]
            ax.step(xs, ys, where="post",
                    color="#aaaaaa", linewidth=1.2, zorder=2, linestyle="--")

    seen = set()
    for p in points:
        label = p["legend_label"] if p["legend_label"] not in seen else "_nolegend_"
        seen.add(p["legend_label"])
        ax.scatter(p["cost"], p["overall"],
                   c=p["color"], marker=p["marker"],
                   s=p["size"], zorder=p["zorder"],
                   edgecolors="white", linewidths=0.5,
                   label=label)

    ax.set_xlabel("Total cost ($)", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # Legend below the plot so marker symbols don't visually overlap the data
    leg = ax.legend(fontsize=8, frameon=False,
                    loc="upper center", bbox_to_anchor=(0.5, -0.18),
                    ncol=2, borderaxespad=0.)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pareto-line", action="store_true",
                        help="Draw the Pareto frontier step line")
    args = parser.parse_args()

    points = build_points()
    out_dir = Path(args.out_dir)
    make_figure(points,
                out_dir / "pbebench_hard_pareto.png",
                dpi=args.dpi, draw_pareto=args.pareto_line)


if __name__ == "__main__":
    main()
