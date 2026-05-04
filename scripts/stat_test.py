"""
Statistical significance testing for PBEBench-Lite and PBEBench-Hard.

Per-task accuracy (binary): McNemar's exact test (two-sided).
Per-task mean reward (continuous): Wilcoxon signed-rank test (two-sided).

Usage:
    python scripts/stat_test.py [--split {lite,hard,both}] [--alpha FLOAT]

Output: a LaTeX table written to evals/stat_tests_{split}.tex
        and a text summary printed to stdout.
"""

import argparse
import json
from pathlib import Path
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
import numpy as np

REPO = Path(__file__).resolve().parent.parent

# ── file manifests ────────────────────────────────────────────────────────────

LITE_SYSTEMS = {
    "BoK":               REPO / "outputs/lite_tasks_full_og_best_of_k_stripped.jsonl",
    "DF":                REPO / "outputs/lite_tasks_full_og_direct_feedback_stripped.jsonl",
    "QO Agent":          REPO / "outputs/lite_direct_solve_openhands.jsonl",
    "CC Solver":         REPO / "evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl",
    "QO Solver":         REPO / "evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/lite.jsonl",
    "All Symbolic":      REPO / "outputs/lite_ensemble_all_solvers.jsonl",
    "BoK + QO":          REPO / "outputs/lite_effi_bok_qwen_run2.jsonl",
    "BoK + CC":          REPO / "outputs/lite_effi_bok_cc.jsonl",
    "BoK + All Sym":     REPO / "outputs/lite_effi_bok_all_solvers.jsonl",
    "DF + QO":           REPO / "outputs/lite_effi_df_qwen_run2.jsonl",
    "DF + CC":           REPO / "outputs/lite_effi_df_cc.jsonl",
    "DF + All Sym":      REPO / "outputs/lite_effi_df_all_solvers.jsonl",
}

HARD_SYSTEMS = {
    "BoK":               REPO / "outputs/hard_bok_converted.jsonl",
    "CC Solver":         REPO / "evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl",
    "QO Solver":         REPO / "evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/hard.jsonl",
    "All Symbolic":      REPO / "outputs/hard_union_all_solvers.jsonl",
    "BoK + CC":          REPO / "outputs/hard_effi_bok_cc.jsonl",
    "BoK + QO":          REPO / "outputs/hard_effi_bok_qwen_run2.jsonl",
    "BoK + CC + QO":     REPO / "outputs/hard_effi_bok_cc_qwen_run2.jsonl",
    "BoK + All Sym":     REPO / "outputs/hard_effi_bok_all_solvers.jsonl",
}

SLR_SYSTEMS = {
    "BoK":               REPO / "outputs/slr_bench_best_of_k_stripped.jsonl",
    "DF":                REPO / "outputs/slr_bench_direct_feedback_stripped.jsonl",
    "CC Solver":         REPO / "evals/solver_results/slr_claude_code/slr.jsonl",
    "QO Solver":         REPO / "evals/solver_results/slr_qwen3.6_35b_a3b/Sun_Apr_26_131_PM/slr.jsonl",
    "BoK + QO":          REPO / "outputs/slr_effi_bok_qwen.jsonl",
    "BoK + CC":          REPO / "outputs/slr_effi_bok_cc.jsonl",
    "BoK + CC + QO":     REPO / "outputs/slr_effi_bok_cc_qwen.jsonl",
    "DF + QO":           REPO / "outputs/slr_effi_df_qwen.jsonl",
    "DF + CC":           REPO / "outputs/slr_effi_df_cc.jsonl",
    "DF + CC + QO":      REPO / "outputs/slr_effi_df_cc_qwen.jsonl",
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_results(path: Path) -> dict[int, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["task_index"]] = r
    return rows


def load_system(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (solved_binary, reward) arrays sorted by task_index."""
    rows = load_results(path)
    task_ids = sorted(rows)
    solved = np.array([1 if rows[t]["best_reward"] >= 1.0 else 0 for t in task_ids])
    reward = np.array([rows[t]["best_reward"] for t in task_ids])
    return solved, reward


# ── tests ─────────────────────────────────────────────────────────────────────

def mcnemar_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided McNemar exact test. Returns (statistic, p-value)."""
    n01 = int(np.sum((a == 0) & (b == 1)))   # A wrong, B right
    n10 = int(np.sum((a == 1) & (b == 0)))   # A right, B wrong
    table = [[int(np.sum((a == 0) & (b == 0))), n01],
             [n10,                              int(np.sum((a == 1) & (b == 1)))]]
    result = mcnemar(table, exact=True)
    return result.statistic, result.pvalue


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on reward differences."""
    diff = b - a
    if np.all(diff == 0):
        return 0.0, 1.0
    stat, p = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    return float(stat), float(p)


# ── formatting ────────────────────────────────────────────────────────────────

def fmt_p(p: float, alpha: float) -> str:
    if p < 0.001:
        return r"$<\!0.001^{***}$"
    elif p < 0.01:
        return f"{p:.3f}$^{{**}}$"
    elif p < alpha:
        return f"{p:.3f}$^{{*}}$"
    else:
        return f"{p:.3f}"


def run_suite(systems: dict, split: str, alpha: float):
    print(f"\n{'='*70}")
    print(f"  {split.upper()}  —  McNemar (accuracy) + Wilcoxon (mean reward)")
    print(f"  α = {alpha}")
    print(f"{'='*70}")

    # Load all
    data = {}
    for name, path in systems.items():
        if not path.exists():
            print(f"  [SKIP] {name}: {path.name} not found")
            continue
        solved, reward = load_system(path)
        acc = solved.mean() * 100
        mr  = reward.mean()
        data[name] = (solved, reward)
        print(f"  {name:<20}  acc={acc:5.1f}%  reward={mr:.4f}  n={len(solved)}")

    names = list(data.keys())
    pairs = list(combinations(names, 2))

    rows_tex = []
    print(f"\n  {'System A':<20} {'System B':<20} {'ΔAcc%':>7} {'p(McN)':>10} {'ΔRew':>8} {'p(Wilc)':>10}")
    print(f"  {'-'*80}")

    for a_name, b_name in pairs:
        sol_a, rew_a = data[a_name]
        sol_b, rew_b = data[b_name]
        if len(sol_a) != len(sol_b):
            print(f"  [SKIP] {a_name} vs {b_name}: length mismatch {len(sol_a)} vs {len(sol_b)}")
            continue

        delta_acc = (sol_b.mean() - sol_a.mean()) * 100
        _, p_mcn  = mcnemar_test(sol_a, sol_b)
        delta_rew = rew_b.mean() - rew_a.mean()
        _, p_wil  = wilcoxon_test(rew_a, rew_b)

        sig_mcn = "***" if p_mcn < 0.001 else ("**" if p_mcn < 0.01 else ("*" if p_mcn < alpha else ""))
        sig_wil = "***" if p_wil < 0.001 else ("**" if p_wil < 0.01 else ("*" if p_wil < alpha else ""))
        print(f"  {a_name:<20} {b_name:<20} {delta_acc:+7.2f} {p_mcn:>10.4f}{sig_mcn:<3} {delta_rew:+8.4f} {p_wil:>10.4f}{sig_wil:<3}")

        rows_tex.append((a_name, b_name, delta_acc, p_mcn, delta_rew, p_wil))

    # Write LaTeX
    tex_path = REPO / f"evals/stat_tests_{split.lower()}.tex"
    with open(tex_path, "w") as f:
        f.write(r"\begin{table*}[!tbh]" + "\n")
        f.write(r"\centering" + "\n")
        cap = f"Pairwise significance tests on PBEBench-{split.capitalize()}. "
        cap += r"$\Delta$Acc = B$-$A accuracy difference (pp). "
        cap += r"$\Delta$Rew = B$-$A mean reward difference. "
        cap += r"\textit{p}(McNemar) uses the exact two-sided McNemar test on per-task binary correctness. "
        cap += r"\textit{p}(Wilcoxon) uses the two-sided Wilcoxon signed-rank test on per-task rewards. "
        cap += r"$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
        f.write(r"\caption{" + cap + "}\n")
        f.write(r"\smallskip" + "\n")
        f.write(r"\label{tab:stat_tests_" + split.lower() + "}\n")
        f.write(r"\resizebox{\textwidth}{!}{%" + "\n")
        f.write(r"\begin{tabular}{llrrrr}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{System A} & \textbf{System B} & $\boldsymbol{\Delta}$\textbf{Acc\%} & \textit{p}(McNemar) & $\boldsymbol{\Delta}$\textbf{Rew} & \textit{p}(Wilcoxon) \\" + "\n")
        f.write(r"\midrule" + "\n")

        # Group rows by category transitions for readability
        for a_name, b_name, da, pm, dr, pw in rows_tex:
            da_s = f"{da:+.2f}"
            dr_s = f"{dr:+.4f}"
            f.write(f"{a_name} & {b_name} & {da_s} & {fmt_p(pm, alpha)} & {dr_s} & {fmt_p(pw, alpha)} \\\\\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}%" + "\n")
        f.write("}\n")
        f.write(r"\end{table*}" + "\n")

    print(f"\n  LaTeX written to {tex_path.relative_to(REPO)}")
    return rows_tex


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["lite", "hard", "slr", "both", "all"], default="both")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if args.split in ("lite", "both", "all"):
        run_suite(LITE_SYSTEMS, "Lite", args.alpha)
    if args.split in ("hard", "both", "all"):
        run_suite(HARD_SYSTEMS, "Hard", args.alpha)
    if args.split in ("slr", "all"):
        run_suite(SLR_SYSTEMS, "SLR", args.alpha)


if __name__ == "__main__":
    main()
