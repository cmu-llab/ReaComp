"""
5-way comparison plots for SLR-Bench: DF, BoK, CC solver, OH Qwen solver,
Qwen3.6-35B-A3B (OpenHands) [DirectSolve coding-agent baseline].
No ensembles — individual systems only, for readability.

Produces 6 figures in the style of figures/solver_cascade_meanreward_with_bok.png:
  By curriculum tier (basic/easy/medium/hard):
    1. slr_comparison_passrate.png     — pass rate vs curriculum tier
    2. slr_comparison_meanreward.png   — mean reward vs curriculum tier
    3. slr_comparison_complexity.png   — mean rule complexity vs curriculum tier
  By curriculum level (1-20, analogous to cascade length in PBEBench):
    4. slr_comparison_level_passrate.png
    5. slr_comparison_level_meanreward.png
    6. slr_comparison_level_complexity.png

Usage:
    python scripts/plot_slr_comparison.py --plot
    python scripts/plot_slr_comparison.py --plot --figures-dir figures/
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rewards.slr_bench import _extract_rule, rule_complexity

# ── paths ─────────────────────────────────────────────────────────────────────

DATASET         = REPO_ROOT / "data/slr_bench/v1_All_full.jsonl"
CLAUDE_SOLVER   = REPO_ROOT / "evals/solver_results/slr_claude_code/slr.jsonl"
QWEN_SOLVER     = REPO_ROOT / "evals/solver_results/slr_qwen3.6_35b_a3b/Sun_Apr_26_131_PM/slr.jsonl"
BOK_OUTPUT      = REPO_ROOT / "outputs/slr_bench_best_of_k_stripped.jsonl"
DF_OUTPUT       = REPO_ROOT / "outputs/slr_bench_direct_feedback_stripped.jsonl"
DS_OUTPUT       = REPO_ROOT / "outputs/slr_bench_direct_solve_openhands.jsonl"
FIGURES_DIR     = REPO_ROOT / "figures"

TIER_ORDER  = ["basic", "easy", "medium", "hard"]
LEVEL_ORDER = list(range(1, 21))

SYSTEMS = [
    ("DF (gpt-oss-120b)",   DF_OUTPUT,     "#16A34A", False),
    ("BoK (gpt-oss-120b)",  BOK_OUTPUT,    "#D97706", False),
    ("CC Solver",           CLAUDE_SOLVER, "#2563EB", True),
    ("QO Solver",           QWEN_SOLVER,   "#DC2626", True),
    ("QO Agent",            DS_OUTPUT,     "#7C3AED", False),
]

# ── loaders ───────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> dict[int, dict]:
    meta = {}
    with open(path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            gt_rule = rec.get("ground-truth rule", "")
            gt_c = None
            if gt_rule:
                try:
                    gt_c = rule_complexity(gt_rule)
                except Exception:
                    pass
            meta[i] = {
                "curriculum_tier": rec.get("curriculum tier"),
                "curriculum_level": rec.get("curriculum level"),
                "gt_complexity": gt_c,
            }
    return meta


def load_results(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.exists():
        print(f"  Warning: {path} not found, skipping.")
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_index")
            if tid is None:
                continue
            prev = records.get(tid)
            br = float(rec.get("best_reward") or (1.0 if rec.get("solved") or rec.get("success") else 0.0))
            if prev is None or br > float(prev.get("best_reward") or 0):
                rec["_best_reward"] = br
                records[tid] = rec
    return records


def _pred_complexity(rec: dict):
    ans = rec.get("answer") or rec.get("program")
    if ans is None:
        return None
    rule, _ = _extract_rule(str(ans))
    if rule is None:
        return None
    try:
        return rule_complexity(rule)
    except Exception:
        return None


# ── GT complexity from full dataset ───────────────────────────────────────────

def gt_complexity_by_tier(meta: dict[int, dict]) -> dict[str, float]:
    groups: dict[str, list] = defaultdict(list)
    for m in meta.values():
        tier = m.get("curriculum_tier")
        gc   = m.get("gt_complexity")
        if tier and gc is not None:
            groups[tier].append(gc)
    return {tier: sum(vs) / len(vs) for tier, vs in groups.items()}


def gt_complexity_by_level(meta: dict[int, dict]) -> dict[int, float]:
    groups: dict[int, list] = defaultdict(list)
    for m in meta.values():
        level = m.get("curriculum_level")
        gc    = m.get("gt_complexity")
        if level is not None and gc is not None:
            groups[int(level)].append(gc)
    return {level: sum(vs) / len(vs) for level, vs in groups.items()}


# ── per-tier stats ────────────────────────────────────────────────────────────

def stats_by_tier(results: dict[int, dict], meta: dict[int, dict]) -> dict[str, dict]:
    """Return pass_rate, mean_reward, mean_pred_complexity, mean_gt_complexity per tier."""
    groups: dict[str, list] = defaultdict(list)
    for tid, rec in results.items():
        tier = (meta.get(tid) or {}).get("curriculum_tier")
        if tier:
            groups[tier].append((tid, rec))

    out = {}
    for tier, items in groups.items():
        n = len(items)
        br_list = [float(rec.get("_best_reward", 0)) for _, rec in items]
        solved = sum(1 for v in br_list if v >= 1.0)
        mean_r = sum(br_list) / n

        # complexity on correct solutions only
        pred_cs, gt_cs = [], []
        for tid, rec in items:
            if float(rec.get("_best_reward", 0)) < 1.0:
                continue
            pc = _pred_complexity(rec)
            gc = (meta.get(tid) or {}).get("gt_complexity")
            if pc is not None:
                pred_cs.append(pc)
            if gc is not None:
                gt_cs.append(gc)

        out[tier] = {
            "n": n,
            "pass_rate": solved / n,
            "mean_reward": mean_r,
            "mean_pred_complexity": sum(pred_cs) / len(pred_cs) if pred_cs else None,
            "mean_gt_complexity":   sum(gt_cs)   / len(gt_cs)   if gt_cs   else None,
            "n_correct": solved,
        }
    return out


def stats_by_level(results: dict[int, dict], meta: dict[int, dict]) -> dict[int, dict]:
    """Return per-level stats (analogous to stats_by_tier but keyed by int level 1-20)."""
    groups: dict[int, list] = defaultdict(list)
    for tid, rec in results.items():
        level = (meta.get(tid) or {}).get("curriculum_level")
        if level is not None:
            groups[int(level)].append((tid, rec))

    out = {}
    for level, items in groups.items():
        n = len(items)
        br_list = [float(rec.get("_best_reward", 0)) for _, rec in items]
        solved = sum(1 for v in br_list if v >= 1.0)
        mean_r = sum(br_list) / n

        pred_cs, gt_cs = [], []
        for tid, rec in items:
            if float(rec.get("_best_reward", 0)) < 1.0:
                continue
            pc = _pred_complexity(rec)
            gc = (meta.get(tid) or {}).get("gt_complexity")
            if pc is not None:
                pred_cs.append(pc)
            if gc is not None:
                gt_cs.append(gc)

        out[level] = {
            "n": n,
            "pass_rate": solved / n,
            "mean_reward": mean_r,
            "mean_pred_complexity": sum(pred_cs) / len(pred_cs) if pred_cs else None,
            "mean_gt_complexity":   sum(gt_cs)   / len(gt_cs)   if gt_cs   else None,
            "n_correct": solved,
        }
    return out


# ── plot helpers ──────────────────────────────────────────────────────────────

def _base_fig():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=0.5, color="#E5E7EB")
    return fig, ax


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _x(tiers):
    return list(range(len(tiers)))


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_pass_rate(tier_stats: dict[str, dict[str, dict]], out: Path):
    fig, ax = _base_fig()
    xs = _x(TIER_ORDER)
    for label, by_tier in tier_stats.items():
        color = next(c for n, _, c, _ in SYSTEMS if n == label)
        is_solver = next(s for n, _, _, s in SYSTEMS if n == label)
        vals = [by_tier.get(t, {}).get("pass_rate", float("nan")) * 100 for t in TIER_ORDER]
        ns   = [by_tier.get(t, {}).get("n", 0) for t in TIER_ORDER]
        ax.fill_between(xs, vals, alpha=0.08, color=color)
        ax.plot(xs, vals, marker="o", markersize=7, linewidth=2.0, color=color,
                linestyle="--" if is_solver else "-", label=label, zorder=3)
        # annotate n per tier
        for x, v, n in zip(xs, vals, ns):
            if not np.isnan(v):
                ax.annotate(f"n={n}", xy=(x, v), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7, color="#6B7280")

    ax.axhline(100, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(50,  color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(TIER_ORDER, fontsize=11)
    ax.set_xlabel("Curriculum tier", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_title("Accuracy vs curriculum tier\n(SLR-Bench, 250 tasks per tier)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    _save(fig, out)


def plot_mean_reward(tier_stats: dict[str, dict[str, dict]], out: Path):
    fig, ax = _base_fig()
    xs = _x(TIER_ORDER)
    for label, by_tier in tier_stats.items():
        color = next(c for n, _, c, _ in SYSTEMS if n == label)
        is_solver = next(s for n, _, _, s in SYSTEMS if n == label)
        vals = [by_tier.get(t, {}).get("mean_reward", float("nan")) for t in TIER_ORDER]
        ax.fill_between(xs, vals, alpha=0.08, color=color)
        ax.plot(xs, vals, marker="o", markersize=7, linewidth=2.0, color=color,
                linestyle="--" if is_solver else "-", label=label, zorder=3)
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", xy=(x, v), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7.5, color="#374151")

    ax.axhline(1.0, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(0.5, color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(TIER_ORDER, fontsize=11)
    ax.set_xlabel("Curriculum tier", fontsize=12)
    ax.set_ylabel("Mean reward", fontsize=12)
    ax.set_ylim(0.4, 1.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Mean reward vs curriculum tier\n(SLR-Bench, 250 tasks per tier)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    _save(fig, out)


def plot_complexity(tier_stats: dict[str, dict[str, dict]], gt_by_tier: dict[str, float], out: Path):
    fig, ax = _base_fig()
    xs = _x(TIER_ORDER)

    gt_vals = [gt_by_tier.get(t, float("nan")) for t in TIER_ORDER]
    if any(not np.isnan(v) for v in gt_vals):
        ax.plot(xs, gt_vals, marker="s", markersize=6, linewidth=1.5,
                color="#9CA3AF", linestyle=":", label="GT", zorder=2)

    for label, by_tier in tier_stats.items():
        color = next(c for n, _, c, _ in SYSTEMS if n == label)
        is_solver = next(s for n, _, _, s in SYSTEMS if n == label)
        vals = [by_tier.get(t, {}).get("mean_pred_complexity", float("nan")) for t in TIER_ORDER]
        # replace None with nan
        vals = [v if v is not None else float("nan") for v in vals]
        ax.fill_between(xs, vals, alpha=0.08, color=color)
        ax.plot(xs, vals, marker="o", markersize=7, linewidth=2.0, color=color,
                linestyle="--" if is_solver else "-", label=label, zorder=3)
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.2f}", xy=(x, v), xytext=(0, 8),
                            textcoords="offset points", ha="center", fontsize=7.5, color="#374151")

    ax.set_xticks(xs)
    ax.set_xticklabels(TIER_ORDER, fontsize=11)
    ax.set_xlabel("Curriculum tier", fontsize=12)
    ax.set_ylabel("Mean rule complexity (correct solutions)", fontsize=12)
    ax.set_title("Rule complexity vs curriculum tier\n(SLR-Bench, correct solutions only)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, out)


def plot_level_pass_rate(level_stats: dict[str, dict[int, dict]], out: Path):
    fig, ax = _base_fig()
    xs = list(range(len(LEVEL_ORDER)))
    for label, by_level in level_stats.items():
        color = next(c for n, _, c, _ in SYSTEMS if n == label)
        is_solver = next(s for n, _, _, s in SYSTEMS if n == label)
        vals = [by_level.get(lv, {}).get("pass_rate", float("nan")) * 100 for lv in LEVEL_ORDER]
        ax.fill_between(xs, vals, alpha=0.08, color=color)
        ax.plot(xs, vals, marker="o", markersize=5, linewidth=2.0, color=color,
                linestyle="--" if is_solver else "-", label=label, zorder=3)

    ax.axhline(100, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(50,  color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(lv) for lv in LEVEL_ORDER], fontsize=9)
    ax.set_xlabel("Curriculum level", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.set_title("Accuracy vs curriculum level\n(SLR-Bench, 50 tasks per level)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    _save(fig, out)


def plot_level_mean_reward(level_stats: dict[str, dict[int, dict]], out: Path):
    fig, ax = _base_fig()
    xs = list(range(len(LEVEL_ORDER)))
    for label, by_level in level_stats.items():
        color = next(c for n, _, c, _ in SYSTEMS if n == label)
        is_solver = next(s for n, _, _, s in SYSTEMS if n == label)
        vals = [by_level.get(lv, {}).get("mean_reward", float("nan")) for lv in LEVEL_ORDER]
        ax.fill_between(xs, vals, alpha=0.08, color=color)
        ax.plot(xs, vals, marker="o", markersize=5, linewidth=2.0, color=color,
                linestyle="--" if is_solver else "-", label=label, zorder=3)

    ax.axhline(1.0, color="#9CA3AF", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(0.5, color="#9CA3AF", linewidth=0.8, linestyle=":",  zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(lv) for lv in LEVEL_ORDER], fontsize=9)
    ax.set_xlabel("Curriculum level", fontsize=12)
    ax.set_ylabel("Mean reward", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_title("Mean reward vs curriculum level\n(SLR-Bench, 50 tasks per level)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    _save(fig, out)


def plot_level_complexity(level_stats: dict[str, dict[int, dict]], gt_by_level: dict[int, float], out: Path):
    fig, ax = _base_fig()
    xs = list(range(len(LEVEL_ORDER)))

    gt_vals = [gt_by_level.get(lv, float("nan")) for lv in LEVEL_ORDER]
    if any(not np.isnan(v) for v in gt_vals):
        ax.plot(xs, gt_vals, marker="s", markersize=5, linewidth=1.5,
                color="#9CA3AF", linestyle=":", label="GT", zorder=2)

    for label, by_level in level_stats.items():
        color = next(c for n, _, c, _ in SYSTEMS if n == label)
        is_solver = next(s for n, _, _, s in SYSTEMS if n == label)
        vals = [by_level.get(lv, {}).get("mean_pred_complexity", float("nan")) for lv in LEVEL_ORDER]
        vals = [v if v is not None else float("nan") for v in vals]
        ax.fill_between(xs, vals, alpha=0.08, color=color)
        ax.plot(xs, vals, marker="o", markersize=5, linewidth=2.0, color=color,
                linestyle="--" if is_solver else "-", label=label, zorder=3)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(lv) for lv in LEVEL_ORDER], fontsize=9)
    ax.set_xlabel("Curriculum level", fontsize=12)
    ax.set_ylabel("Mean rule complexity (correct solutions)", fontsize=12)
    ax.set_title("Rule complexity vs curriculum level\n(SLR-Bench, correct solutions only)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _save(fig, out)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot",           action="store_true")
    parser.add_argument("--figures-dir",    default=str(FIGURES_DIR))
    parser.add_argument("--claude-solver",  default=str(CLAUDE_SOLVER))
    parser.add_argument("--qwen-solver",    default=str(QWEN_SOLVER))
    parser.add_argument("--bok",            default=str(BOK_OUTPUT))
    parser.add_argument("--df",             default=str(DF_OUTPUT))
    parser.add_argument("--direct-solve",   default=str(DS_OUTPUT),
                        help="DirectSolve (Qwen3.6-35B-A3B OpenHands) output JSONL")
    parser.add_argument("--dataset",        default=str(DATASET))
    args = parser.parse_args()

    fdir = Path(args.figures_dir)
    if args.plot:
        fdir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    meta = load_dataset(Path(args.dataset))

    paths = {
        "DF (gpt-oss-120b)":  Path(args.df),
        "BoK (gpt-oss-120b)": Path(args.bok),
        "CC Solver":           Path(args.claude_solver),
        "QO Solver":           Path(args.qwen_solver),
        "QO Agent":            Path(args.direct_solve),
    }
    print("Loading results...")
    all_results = {name: load_results(p) for name, p in paths.items()}
    all_results = {k: v for k, v in all_results.items() if v}

    print("Computing per-tier stats...")
    tier_stats  = {name: stats_by_tier(res, meta)  for name, res in all_results.items()}
    level_stats = {name: stats_by_level(res, meta) for name, res in all_results.items()}
    gt_by_tier  = gt_complexity_by_tier(meta)
    gt_by_level = gt_complexity_by_level(meta)

    # Print summary table (by tier)
    print(f"\n{'System':<20} {'Tier':<8} {'N':>5} {'Pass%':>7} {'MeanRew':>9} {'MeanComplexity':>15}")
    for name, by_tier in tier_stats.items():
        for tier in TIER_ORDER:
            s = by_tier.get(tier)
            if not s:
                continue
            mc = f"{s['mean_pred_complexity']:.3f}" if s["mean_pred_complexity"] is not None else "—"
            print(f"  {name:<18} {tier:<8} {s['n']:>5} {s['pass_rate']*100:>6.1f}% {s['mean_reward']:>9.4f} {mc:>15}")
        print()

    if args.plot:
        print("Generating figures...")
        plot_pass_rate(tier_stats,   fdir / "slr_comparison_passrate.png")
        plot_mean_reward(tier_stats, fdir / "slr_comparison_meanreward.png")
        plot_complexity(tier_stats,  gt_by_tier,  fdir / "slr_comparison_complexity.png")
        plot_level_pass_rate(level_stats,   fdir / "slr_comparison_level_passrate.png")
        plot_level_mean_reward(level_stats, fdir / "slr_comparison_level_meanreward.png")
        plot_level_complexity(level_stats,  gt_by_level, fdir / "slr_comparison_level_complexity.png")


if __name__ == "__main__":
    main()
