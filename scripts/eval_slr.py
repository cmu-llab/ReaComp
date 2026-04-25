"""
SLR-Bench evaluation: pass rate, mean reward, rule complexity, and token usage
across symbolic solvers (Claude Code, Qwen3.6) and LLM baselines (BoK, DF).

Generates figures analogous to those produced for PBEBench-Hard:
  - Pass rate vs curriculum level  (line chart, per system + ensembles)
  - Mean reward vs curriculum level (line chart)
  - Pass rate vs curriculum tier   (bar chart)
  - Pass rate vs rule complexity   (bar chart)
  - Rule complexity vs GT          (bar chart, correct solutions only)
  - Token usage vs curriculum level (line chart, LLM baselines only)

Usage:
    python scripts/eval_slr.py
    python scripts/eval_slr.py --plot --metrics-json figures/slr_metrics.json
    python scripts/eval_slr.py --no-ensemble   # skip ensemble lines
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

# ── default file paths ────────────────────────────────────────────────────────

DATASET       = REPO_ROOT / "data/slr_bench/v1_All_full.jsonl"
CLAUDE_SOLVER = REPO_ROOT / "evals/solver_results/slr_claude_code/slr.jsonl"
QWEN_SOLVER   = REPO_ROOT / "evals/solver_results/slr_qwen3.6_35b_a3b/Sat_Apr_25_643_AM/slr.jsonl"
BOK_OUTPUT    = REPO_ROOT / "outputs/slr_bench_best_of_k.jsonl"
DF_OUTPUT     = REPO_ROOT / "outputs/slr_bench_direct_feedback.jsonl"

FIGURES_DIR   = REPO_ROOT / "figures"

COLORS = {
    "Claude solver":  "#2563EB",
    "Qwen solver":    "#DC2626",
    "BoK":            "#D97706",
    "DF":             "#16A34A",
    "BoK ∪ Claude":   "#7C3AED",
    "DF ∪ Claude":    "#0891B2",
    "BoK ∪ Qwen":     "#F59E0B",
    "DF ∪ Qwen":      "#10B981",
    "BoK ∪ CC ∪ Qwen":"#6D28D9",
}

TIER_ORDER = ["basic", "easy", "medium", "hard"]

# ── loaders ───────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> dict[int, dict]:
    """Load dataset metadata keyed by 0-based task index."""
    meta = {}
    with open(path) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            meta[i] = {
                "curriculum_level": rec.get("curriculum level"),
                "curriculum_tier":  rec.get("curriculum tier"),
                "rule_complexity":  rec.get("rule complexity"),
                "gt_rule":          rec.get("ground-truth rule", ""),
            }
    return meta


def _gt_complexity(gt_rule: str) -> int | None:
    if not gt_rule:
        return None
    return rule_complexity(gt_rule)


def load_results(path: Path) -> dict[int, dict]:
    """Load eval JSONL keyed by task_index. Keeps highest-reward record per task."""
    records: dict[int, dict] = {}
    if not path.exists():
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
            if prev is None or rec.get("best_reward", 0) > prev.get("best_reward", 0):
                records[tid] = rec
    return records


def _best_reward(rec: dict) -> float:
    v = rec.get("best_reward")
    if v is not None:
        return float(v)
    return 1.0 if rec.get("solved") else 0.0


def _pred_complexity(rec: dict) -> int | None:
    return _rc(rec.get("answer"))


def _rc(answer) -> int | None:
    if answer is None:
        return None
    rule, _ = _extract_rule(str(answer))
    if rule is None:
        return None
    return rule_complexity(rule)


def _token_usage(rec: dict) -> int:
    tu = rec.get("token_usage") or {}
    return (tu.get("input", 0) or 0) + (tu.get("output", 0) or 0)


# ── ensemble ──────────────────────────────────────────────────────────────────

def union_ensemble(*result_dicts: dict[int, dict]) -> dict[int, dict]:
    """Per task: take record with highest reward; tiebreak by lower pred_complexity."""
    all_ids = set()
    for d in result_dicts:
        all_ids.update(d.keys())
    out = {}
    for tid in all_ids:
        candidates = [d[tid] for d in result_dicts if tid in d]
        best = max(candidates, key=lambda r: (_best_reward(r), -(_pred_complexity(r) or 999)))
        out[tid] = best
    return out


# ── per-stratum stats ─────────────────────────────────────────────────────────

def by_level(results: dict[int, dict], meta: dict[int, dict]) -> dict[int, dict]:
    """Group records by curriculum_level."""
    groups: dict[int, list] = defaultdict(list)
    for tid, rec in results.items():
        lv = (meta.get(tid) or {}).get("curriculum_level")
        if lv is not None:
            groups[int(lv)].append(rec)
    out = {}
    for lv, recs in groups.items():
        n = len(recs)
        solved = sum(1 for r in recs if _best_reward(r) >= 1.0)
        mr = sum(_best_reward(r) for r in recs) / n
        tokens = sum(_token_usage(r) for r in recs) / n
        out[lv] = {"n": n, "pass_rate": solved / n, "mean_reward": mr, "avg_tokens": tokens}
    return out


def by_tier(results: dict[int, dict], meta: dict[int, dict]) -> dict[str, dict]:
    groups: dict[str, list] = defaultdict(list)
    for tid, rec in results.items():
        tier = (meta.get(tid) or {}).get("curriculum_tier")
        if tier:
            groups[tier].append(rec)
    out = {}
    for tier, recs in groups.items():
        n = len(recs)
        solved = sum(1 for r in recs if _best_reward(r) >= 1.0)
        mr = sum(_best_reward(r) for r in recs) / n
        out[tier] = {"n": n, "pass_rate": solved / n, "mean_reward": mr}
    return out


def by_rule_complexity(results: dict[int, dict], meta: dict[int, dict]) -> dict[str, dict]:
    groups: dict[str, list] = defaultdict(list)
    for tid, rec in results.items():
        rc = (meta.get(tid) or {}).get("rule_complexity")
        if rc is not None:
            groups[str(rc)].append(rec)
    out = {}
    for rc, recs in groups.items():
        n = len(recs)
        solved = sum(1 for r in recs if _best_reward(r) >= 1.0)
        mr = sum(_best_reward(r) for r in recs) / n
        out[rc] = {"n": n, "pass_rate": solved / n, "mean_reward": mr}
    return out


def complexity_vs_gt(results: dict[int, dict], meta: dict[int, dict]) -> dict:
    pairs = []
    for tid, rec in results.items():
        if _best_reward(rec) < 1.0:
            continue
        pred_c = _pred_complexity(rec)
        gt_rule = (meta.get(tid) or {}).get("gt_rule", "")
        gt_c = _gt_complexity(gt_rule)
        if pred_c is not None and gt_c is not None:
            pairs.append((pred_c, gt_c))
    if not pairs:
        return {}
    n = len(pairs)
    simpler      = sum(1 for p, g in pairs if p < g)
    equal        = sum(1 for p, g in pairs if p == g)
    more_complex = sum(1 for p, g in pairs if p > g)
    return {
        "n": n,
        "mean_pred":  round(sum(p for p, _ in pairs) / n, 3),
        "mean_gt":    round(sum(g for _, g in pairs) / n, 3),
        "mean_delta": round(sum(p - g for p, g in pairs) / n, 3),
        "simpler": simpler, "equal": equal, "more_complex": more_complex,
    }


# ── summary stats ──────────────────────────────────────────────────────────────

def overall_stats(results: dict[int, dict], n_total: int) -> dict:
    n = len(results)
    solved = sum(1 for r in results.values() if _best_reward(r) >= 1.0)
    mr = sum(_best_reward(r) for r in results.values()) / n if n else 0.0
    avg_tok = sum(_token_usage(r) for r in results.values()) / n if n else 0.0
    return {
        "n_evaluated": n, "n_total": n_total,
        "solved": solved,
        "pass_rate": round(solved / n, 4) if n else 0.0,
        "pass_rate_of_total": round(solved / n_total, 4) if n_total else 0.0,
        "mean_reward": round(mr, 4),
        "avg_tokens": round(avg_tok, 1),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def _levels_xy(by_lv: dict[int, dict], key: str):
    levels = sorted(by_lv)
    return levels, [by_lv[lv][key] for lv in levels]


def plot_pass_rate_by_level(systems: dict[str, dict[int, dict]], out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, by_lv in systems.items():
        lvs, vals = _levels_xy(by_lv, "pass_rate")
        ax.plot(lvs, [v * 100 for v in vals], marker="o", markersize=4,
                label=label, color=COLORS.get(label))
    ax.set_xlabel("Curriculum level")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("SLR-Bench: pass rate vs curriculum level")
    ax.set_xticks(range(1, 21))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_mean_reward_by_level(systems: dict[str, dict[int, dict]], out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, by_lv in systems.items():
        lvs, vals = _levels_xy(by_lv, "mean_reward")
        ax.plot(lvs, vals, marker="o", markersize=4,
                label=label, color=COLORS.get(label))
    ax.set_xlabel("Curriculum level")
    ax.set_ylabel("Mean reward")
    ax.set_title("SLR-Bench: mean reward vs curriculum level")
    ax.set_xticks(range(1, 21))
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_pass_rate_by_tier(systems: dict[str, dict[str, dict]], out: Path):
    tiers = TIER_ORDER
    labels = list(systems.keys())
    x = np.arange(len(tiers))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, by_tier_data) in enumerate(systems.items()):
        vals = [by_tier_data.get(t, {}).get("pass_rate", 0) * 100 for t in tiers]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=COLORS.get(label), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_xlabel("Curriculum tier")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("SLR-Bench: pass rate by curriculum tier")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_pass_rate_by_rule_complexity(systems: dict[str, dict[str, dict]], out: Path):
    # Collect all rule_complexity labels and sort them
    all_rc = set()
    for d in systems.values():
        all_rc.update(d.keys())
    rc_order = sorted(all_rc, key=lambda x: float(x.split("-")[0]))

    labels = list(systems.keys())
    x = np.arange(len(rc_order))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, by_rc) in enumerate(systems.items()):
        vals = [by_rc.get(rc, {}).get("pass_rate", 0) * 100 for rc in rc_order]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=COLORS.get(label), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(rc_order)
    ax.set_xlabel("Rule complexity (dataset label)")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("SLR-Bench: pass rate by rule complexity")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_complexity_vs_gt(systems: dict[str, dict], out: Path):
    """Bar chart: mean pred complexity and mean GT complexity per system (correct only)."""
    labels = list(systems.keys())
    mean_preds = [systems[l].get("mean_pred", 0) for l in labels]
    mean_gts   = [systems[l].get("mean_gt", 0)   for l in labels]
    deltas     = [systems[l].get("mean_delta", 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_gt   = ax.bar(x - width / 2, mean_gts,   width, label="GT complexity",   color="#94A3B8", alpha=0.9)
    bars_pred = ax.bar(x + width / 2, mean_preds, width, label="Pred complexity", color="#3B82F6", alpha=0.9)

    for bar, delta in zip(bars_pred, deltas):
        sign = "+" if delta >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{sign}{delta:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean rule complexity")
    ax.set_title("SLR-Bench: predicted vs GT rule complexity (correct solutions)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_tokens_by_level(systems: dict[str, dict[int, dict]], out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, by_lv in systems.items():
        lvs, vals = _levels_xy(by_lv, "avg_tokens")
        ax.plot(lvs, vals, marker="o", markersize=4,
                label=label, color=COLORS.get(label))
    ax.set_xlabel("Curriculum level")
    ax.set_ylabel("Avg tokens / task")
    ax.set_title("SLR-Bench: token usage vs curriculum level")
    ax.set_xticks(range(1, 21))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLR-Bench evaluation and plotting.")
    parser.add_argument("--claude-solver", default=str(CLAUDE_SOLVER))
    parser.add_argument("--qwen-solver",   default=str(QWEN_SOLVER))
    parser.add_argument("--bok",           default=str(BOK_OUTPUT))
    parser.add_argument("--df",            default=str(DF_OUTPUT))
    parser.add_argument("--dataset",       default=str(DATASET))
    parser.add_argument("--plot",          action="store_true", help="Save figures to figures/")
    parser.add_argument("--figures-dir",   default=str(FIGURES_DIR))
    parser.add_argument("--metrics-json",  default=None, help="Write all metrics to this JSON file")
    parser.add_argument("--no-ensemble",   action="store_true", help="Skip ensemble computation")
    args = parser.parse_args()

    fdir = Path(args.figures_dir)
    if args.plot:
        fdir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset metadata...")
    meta = load_dataset(Path(args.dataset))
    n_total = len(meta)

    print("Loading result files...")
    claude_res = load_results(Path(args.claude_solver))
    qwen_res   = load_results(Path(args.qwen_solver))
    bok_res    = load_results(Path(args.bok))
    df_res     = load_results(Path(args.df))

    available = {k: v for k, v in {
        "Claude solver": claude_res,
        "Qwen solver":   qwen_res,
        "BoK":           bok_res,
        "DF":            df_res,
    }.items() if v}

    ensembles = {}
    if not args.no_ensemble:
        if claude_res and bok_res:
            ensembles["BoK ∪ Claude"] = union_ensemble(bok_res, claude_res)
        if claude_res and df_res:
            ensembles["DF ∪ Claude"]  = union_ensemble(df_res,  claude_res)
        if qwen_res and bok_res:
            ensembles["BoK ∪ Qwen"]   = union_ensemble(bok_res, qwen_res)
        if qwen_res and df_res:
            ensembles["DF ∪ Qwen"]    = union_ensemble(df_res,  qwen_res)
        if claude_res and qwen_res and bok_res:
            ensembles["BoK ∪ CC ∪ Qwen"] = union_ensemble(bok_res, claude_res, qwen_res)

    all_systems = {**available, **ensembles}

    # ── overall stats ─────────────────────────────────────────────────────────
    print("\n=== Overall stats ===")
    all_stats = {}
    for name, res in all_systems.items():
        s = overall_stats(res, n_total)
        all_stats[name] = s
        note = " (partial)" if s["n_evaluated"] < n_total else ""
        print(f"  {name:<22} {s['solved']:>4}/{s['n_evaluated']:<4}  "
              f"pass={s['pass_rate']:.1%}  mean_reward={s['mean_reward']:.4f}"
              f"  avg_tok={s['avg_tokens']:,.0f}{note}")

    # ── per-level ─────────────────────────────────────────────────────────────
    by_lv_all = {name: by_level(res, meta) for name, res in all_systems.items()}

    # ── per-tier ──────────────────────────────────────────────────────────────
    by_tier_all = {name: by_tier(res, meta) for name, res in all_systems.items()}

    # ── per-rule-complexity ───────────────────────────────────────────────────
    by_rc_all = {name: by_rule_complexity(res, meta) for name, res in all_systems.items()}

    # ── complexity vs GT ──────────────────────────────────────────────────────
    cplx_vs_gt = {}
    for name, res in all_systems.items():
        cplx_vs_gt[name] = complexity_vs_gt(res, meta)
    print("\n=== Complexity vs GT (correct solutions only) ===")
    for name, c in cplx_vs_gt.items():
        if not c:
            continue
        print(f"  {name:<22} n={c['n']:<5} mean_pred={c['mean_pred']:.3f}  "
              f"mean_gt={c['mean_gt']:.3f}  delta={c['mean_delta']:+.3f}  "
              f"simpler={c['simpler']}  equal={c['equal']}  harder={c['more_complex']}")

    # ── plots ──────────────────────────────────────────────────────────────────
    if args.plot:
        print("\n=== Generating figures ===")
        plot_pass_rate_by_level(by_lv_all, fdir / "slr_passrate_by_level.png")
        plot_mean_reward_by_level(by_lv_all, fdir / "slr_meanreward_by_level.png")
        plot_pass_rate_by_tier(by_tier_all, fdir / "slr_passrate_by_tier.png")
        plot_pass_rate_by_rule_complexity(by_rc_all, fdir / "slr_passrate_by_rule_complexity.png")
        if cplx_vs_gt:
            plot_complexity_vs_gt(
                {k: v for k, v in cplx_vs_gt.items() if v},
                fdir / "slr_complexity_vs_gt.png",
            )
        llm_systems = {k: by_lv_all[k] for k in ["BoK", "DF"] if k in by_lv_all}
        if llm_systems:
            plot_tokens_by_level(llm_systems, fdir / "slr_tokens_by_level.png")

    # ── metrics JSON ──────────────────────────────────────────────────────────
    metrics = {
        "overall": all_stats,
        "by_level": {name: {str(lv): v for lv, v in d.items()} for name, d in by_lv_all.items()},
        "by_tier":           by_tier_all,
        "by_rule_complexity": by_rc_all,
        "complexity_vs_gt":  cplx_vs_gt,
    }
    if args.metrics_json:
        out_path = Path(args.metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics written to {out_path}")

    return metrics


if __name__ == "__main__":
    main()
