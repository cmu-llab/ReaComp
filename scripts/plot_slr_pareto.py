"""
Pareto frontier plot: cost vs performance for SLR-Bench systems.

Two separate figures:
  1. slr_pareto_overall.png  — overall pass rate
  2. slr_pareto_hard.png     — hard-tier pass rate

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

CC_SLR_BUILD   = 24.00   # estimated (see findings.md)
QWEN_SLR_BUILD =  1.25   # measured from trajectory


def infer_cost(d):
    tok = d.get("token_usage", {})
    return tok.get("input_total", 0) * INPUT_PRICE + tok.get("output_total", 0) * OUTPUT_PRICE


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
            legend_label = "CC (symbolic solver)" if slr_count == 1 else "QO (symbolic solver)"
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
            "DF": "DF (LLM-only)", "BoK": "BoK (LLM-only)",
            "DF + CC": "DF + CC (hybrid)", "BoK + CC": "BoK + CC (hybrid)",
            "DF + QO": "DF + QO (hybrid)", "BoK + QO": "BoK + QO (hybrid)",
            "DF + CC + QO": "DF + CC + QO (hybrid)", "BoK + CC + QO": "BoK + CC + QO (hybrid)",
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

    ax.legend(fontsize=8, frameon=False, loc="lower right")

    plt.tight_layout()
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

    make_figure(points, "overall", "Overall pass rate (%)",
                out_dir / "slr_pareto_overall.png", dpi=args.dpi,
                draw_pareto=args.pareto_line)
    make_figure(points, "hard",    "Hard-tier pass rate (%)",
                out_dir / "slr_pareto_hard.png", dpi=args.dpi,
                draw_pareto=args.pareto_line)


if __name__ == "__main__":
    main()
