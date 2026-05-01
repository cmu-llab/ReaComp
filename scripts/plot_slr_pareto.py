"""
Pareto frontier plot: cost vs performance for SLR-Bench systems.

Two separate figures:
  1. slr_pareto_overall.png  — overall accuracy
  2. slr_pareto_hard.png     — hard-tier accuracy

Usage:
    python scripts/plot_slr_pareto.py
    python scripts/plot_slr_pareto.py --out-dir figures/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
INPUT_PRICE  = 0.039 / 1e6   # gpt-oss-120b @ DeepInfra
OUTPUT_PRICE = 0.19  / 1e6

# AtlasCloud pricing for Qwen3.6-35B-A3B (DirectSolve / solver build)
ATLAS_INPUT_PRICE  = 0.1612 / 1e6
ATLAS_OUTPUT_PRICE = 0.9653 / 1e6

CC_SLR_BUILD   = 24.00   # estimated (see findings.md)
QWEN_SLR_BUILD =  1.28   # exact from trajectory (native Qwen3.6-35B-A3B tokenizer)

DS_SLR_PATH = REPO_ROOT / "outputs" / "slr_bench_direct_solve_openhands.jsonl"


def infer_cost(d):
    tok = d.get("token_usage", {})
    return tok.get("input_total", 0) * INPUT_PRICE + tok.get("output_total", 0) * OUTPUT_PRICE


def infer_cost_atlas_jsonl(path):
    """Sum inference cost for a raw JSONL file using AtlasCloud Qwen pricing."""
    total = 0.0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tok = r.get("token_usage") or {}
            inp = tok.get("input", 0) or 0
            out = tok.get("output", 0) or 0
            total += inp * ATLAS_INPUT_PRICE + out * ATLAS_OUTPUT_PRICE
    return total


def accuracy_jsonl(path):
    rows = [json.loads(l) for l in open(path)]
    return sum(1 for r in rows if r.get("solved") or r.get("success")) / len(rows) * 100


def get_hard_jsonl(path):
    """Compute hard-tier accuracy from a raw JSONL (requires dataset for tier lookup)."""
    # Hard tier in SLR-Bench = task indices 750-999 (250 tasks per tier, 4 tiers)
    rows = [json.loads(l) for l in open(path)]
    hard = [r for r in rows if 750 <= (r.get("task_index") or r.get("task_id") or 0) < 1000]
    if not hard:
        return None
    return sum(1 for r in hard if r.get("solved") or r.get("success")) / len(hard) * 100


def get_hard(d):
    for t in (d.get("slr_breakdown_by_tier") or []):
        if t["tier"] == "hard":
            return t["pass_pct"]
    return None


def build_points(data):
    """
    Returns list of dicts:
      name, cost, overall, hard, color, marker, size, zorder, legend_label
    """
    points = []
    slr_count = 0

    INCLUDE = {
        "slr_bench_direct_feedback_stripped" : ("DF",          False, False),
        "slr_bench_best_of_k_stripped"       : ("BoK",         False, False),
        "slr_ensemble_effi_df_cc"            : ("DF + CC",     True,  False),
        "slr_ensemble_effi_bok_cc"           : ("BoK + CC",    True,  False),
        "slr_ensemble_effi_df_qwen"          : ("DF + QO",     False, True),
        "slr_ensemble_effi_bok_qwen"         : ("BoK + QO",    False, True),
        "slr_ensemble_effi_df_cc_qwen"       : ("DF + CC + QO",  True, True),
        "slr_ensemble_effi_bok_cc_qwen"      : ("BoK + CC + QO", True, True),
    }

    for d in data:
        label = d["label"]
        if label == "COMBINED":
            continue

        if label == "slr":
            slr_count += 1
            name = "CC" if slr_count == 1 else "QO"
            build = CC_SLR_BUILD if slr_count == 1 else QWEN_SLR_BUILD
            legend_label = "CC Solver" if slr_count == 1 else "QO Solver"
            points.append(dict(
                name=name, legend_label=legend_label,
                cost=build + infer_cost(d),
                overall=d["pass_rate"], hard=get_hard(d),
                color="#e6550d" if slr_count == 1 else "#fd8d3c",
                marker="*", size=280, zorder=6,
            ))
            continue

        if label not in INCLUDE:
            continue

        name, add_cc, add_qwen = INCLUDE[label]
        cost = infer_cost(d)
        if add_cc:   cost += CC_SLR_BUILD
        if add_qwen: cost += QWEN_SLR_BUILD

        if "CC + QO" in name:
            color = "#006d2c"
        elif "CC" in name:
            color = "#31a354"
        elif "QO" in name:
            color = "#74c476"
        else:
            color = "#3182bd"

        marker = "o" if name in ("DF", "BoK") else "D"
        size   = 100 if name in ("DF", "BoK") else 120

        legend_map = {
            "DF":          "DF (gpt-oss-120b)",
            "BoK":         "BoK (gpt-oss-120b)",
            "DF + CC":     "DF + CC Solver",
            "BoK + CC":    "BoK + CC Solver",
            "DF + QO":     "DF + QO Solver",
            "BoK + QO":    "BoK + QO Solver",
            "DF + CC + QO":  "DF + CC Solver + QO Solver",
            "BoK + CC + QO": "BoK + CC Solver + QO Solver",
        }

        points.append(dict(
            name=name, legend_label=legend_map[name],
            cost=cost, overall=d["pass_rate"], hard=get_hard(d),
            color=color, marker=marker, size=size, zorder=5,
        ))

    for name, cost, overall, hard in [
        ("o3 (LLM-only)",    207.24, 77.75, 45.0),
        ("gpt-5 (LLM-only)", 103.13, 77.0,  46.0),
    ]:
        points.append(dict(
            name=name, legend_label=name,
            cost=cost, overall=overall, hard=hard,
            color="#756bb1", marker="^", size=160, zorder=5,
        ))

    # DirectSolve baseline — included if output file exists (may be partial)
    if DS_SLR_PATH.exists():
        ds_cost = infer_cost_atlas_jsonl(DS_SLR_PATH)
        ds_overall = accuracy_jsonl(DS_SLR_PATH)
        ds_hard = get_hard_jsonl(DS_SLR_PATH)
        points.append(dict(
            name="DirectSolve",
            legend_label="QO Agent",
            cost=ds_cost, overall=ds_overall, hard=ds_hard,
            color="#7C3AED", marker="s", size=140, zorder=5,
        ))

    return points


def _pareto_frontier(points, y_key):
    """Return non-dominated points sorted by cost (lower cost, higher y is better)."""
    valid = [(p["cost"], p[y_key]) for p in points if p[y_key] is not None]
    valid.sort(key=lambda t: t[0])
    frontier = []
    best_y = -float("inf")
    for cost, y in valid:
        if y > best_y:
            frontier.append((cost, y))
            best_y = y
    return frontier


def make_figure(points, y_key, ylabel, out_path, dpi=200, draw_pareto=False):
    import matplotlib.ticker as mticker

    fig, ax = plt.subplots(figsize=(7, 5))

    if draw_pareto:
        frontier = _pareto_frontier(points, y_key)
        if frontier:
            xs = [t[0] for t in frontier]
            ys = [t[1] for t in frontier]
            # extend the step line to the right edge of the plot
            xs_step = xs + [max(p["cost"] for p in points) * 1.05]
            ys_step = ys + [ys[-1]]
            ax.step(xs_step, ys_step, where="post",
                    color="#aaaaaa", linewidth=1.2, zorder=2, linestyle="--")

    seen_labels = set()
    for p in points:
        y = p[y_key]
        if y is None:
            continue
        label = p["legend_label"] if p["legend_label"] not in seen_labels else "_nolegend_"
        seen_labels.add(p["legend_label"])
        ax.scatter(p["cost"], y,
                   c=p["color"], marker=p["marker"],
                   s=p["size"], zorder=p["zorder"],
                   edgecolors="white", linewidths=0.5,
                   label=label)

    ax.set_xlabel("Total cost ($)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

    # Legend below the plot so marker symbols don't visually overlap the data
    ax.legend(fontsize=8, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, borderaxespad=0.)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-json", default=str(REPO_ROOT / "metrics" / "slr_all.json"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figures"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pareto-line", action="store_true",
                        help="Draw the Pareto frontier step line")
    args = parser.parse_args()

    data = json.load(open(args.metrics_json))
    points = build_points(data)
    out_dir = Path(args.out_dir)

    make_figure(points, "overall", "Overall accuracy (%)",
                out_dir / "slr_pareto_overall.png", dpi=args.dpi,
                draw_pareto=args.pareto_line)
    make_figure(points, "hard",    "Hard-tier accuracy (%)",
                out_dir / "slr_pareto_hard.png", dpi=args.dpi,
                draw_pareto=args.pareto_line)


if __name__ == "__main__":
    main()
