"""
Evaluation script for symbolic-library-agent output JSONL files.

Computes per-instance and summary metrics for:
  - Reward        : best_reward (0-1), task_loss (1 - best_reward)
  - Pass rate     : whether solved=True
  - Cost          : total_cost, objective, num_new_functions, reuse_count
  - Reward iters  : how many iterations were needed, blame distribution

Usage:
    python scripts/eval.py outputs/pbebench_lite_pilot_tasks_with_rewards.jsonl
    python scripts/eval.py outputs/pbebench_lite_pilot_tasks_with_rewards.jsonl \
                           outputs/reasoning_gym_easy_pilot_with_rewards.jsonl
    python scripts/eval.py outputs/*.jsonl --csv results/eval.csv
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ── record parsing ─────────────────────────────────────────────────────────────

def _source_dataset(record: dict) -> str:
    """Best-effort extraction of source dataset name from a record."""
    # reasoning_gym reward messages contain "for <dataset>."
    for h in record.get("reward_history", []):
        msg = h.get("message", "")
        if " for " in msg:
            return msg.split(" for ")[1].split(".")[0].strip()
    # fallback: task_type or unknown
    return record.get("task_type", "unknown")


def _first_perfect_iter(reward_history: list) -> int | None:
    """Return the 0-based iteration index where reward first hit 1.0, or None."""
    for h in reward_history:
        if h.get("reward", 0.0) >= 1.0:
            return h["iteration"]
    return None


def _blame_sequence(reward_history: list) -> str:
    """e.g. 'execution→partial→logic'"""
    return "→".join(h.get("blame", "?") for h in reward_history)


def record_to_row(rec: dict) -> dict:
    best_reward = rec.get("best_reward")
    if best_reward is None:
        best_reward = 1.0 if rec.get("solved") else 0.0

    cost = rec.get("cost_summary") or {}
    rh = rec.get("reward_history", [])

    return {
        "task_index":         rec.get("task_index"),
        "source_dataset":     _source_dataset(rec),
        # reward / loss
        "best_reward":        best_reward,
        "task_loss":          round(1.0 - best_reward, 6),
        # pass
        "solved":             int(bool(rec.get("solved"))),
        # reward loop
        "num_iters":          len(rh),
        "first_perfect_iter": _first_perfect_iter(rh),
        "blame_sequence":     _blame_sequence(rh),
        # cost
        "total_cost":         cost.get("total_cost"),
        "objective":          cost.get("objective"),
        "num_new_functions":  cost.get("num_new_functions"),
        "reuse_count":        cost.get("reuse_count"),
        "redundancy_penalty": cost.get("redundancy_penalty"),
        "reuse_reward":       cost.get("reuse_reward"),
        # misc
        "steps_taken":        rec.get("steps_taken"),
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


# ── summary stats ──────────────────────────────────────────────────────────────

def _num_fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def summarise(df: pd.DataFrame, label: str = "") -> None:
    n = len(df)
    hdr = f"  {label}" if label else ""
    print(f"\n{'='*62}")
    print(f"  Summary{hdr}  (n={n})")
    print(f"{'='*62}")

    # --- reward / loss ---
    print("\n  Reward & Loss")
    print(f"  {'metric':<28}  {'mean':>8}  {'median':>8}  {'std':>8}  {'min':>8}  {'max':>8}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for col in ["best_reward", "task_loss"]:
        s = df[col].dropna()
        print(f"  {col:<28}  {s.mean():>8.4f}  {s.median():>8.4f}  "
              f"{s.std():>8.4f}  {s.min():>8.4f}  {s.max():>8.4f}")

    # --- pass rate ---
    pass_rate = df["solved"].mean()
    n_solved = df["solved"].sum()
    print(f"\n  Pass rate: {n_solved}/{n} = {pass_rate:.1%}")

    # --- reward by iteration ---
    fpi = df["first_perfect_iter"].dropna()
    if len(fpi):
        counts = Counter(int(x) for x in fpi)
        print(f"\n  First-perfect-iter distribution (of {len(fpi)} solved):")
        for it in sorted(counts):
            print(f"    iter {it}: {counts[it]:3d}  ({counts[it]/n:.1%} of all tasks)")
    no_perfect = df["first_perfect_iter"].isna().sum()
    print(f"    never : {no_perfect:3d}  ({no_perfect/n:.1%} of all tasks)")

    # --- blame ---
    blame_counts = Counter(df["blame_sequence"])
    print(f"\n  Blame sequences (top 5):")
    for seq, cnt in blame_counts.most_common(5):
        seq_str = seq if seq else "(no retries)"
        print(f"    {cnt:3d}  {seq_str}")

    # --- cost ---
    print(f"\n  Cost")
    print(f"  {'metric':<28}  {'mean':>8}  {'median':>8}  {'total':>10}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*10}")
    for col in ["total_cost", "objective", "num_new_functions", "reuse_count"]:
        s = df[col].dropna()
        if len(s):
            print(f"  {col:<28}  {s.mean():>8.4f}  {s.median():>8.4f}  {s.sum():>10.2f}")

    # --- per-dataset breakdown (if multiple) ---
    datasets = df["source_dataset"].unique()
    if len(datasets) > 1:
        print(f"\n  Per-dataset breakdown:")
        print(f"  {'dataset':<30}  {'n':>4}  {'pass_rate':>9}  {'mean_reward':>11}  {'mean_cost':>9}")
        print(f"  {'-'*30}  {'-'*4}  {'-'*9}  {'-'*11}  {'-'*9}")
        for ds in sorted(datasets):
            sub = df[df["source_dataset"] == ds]
            print(f"  {ds:<30}  {len(sub):>4}  "
                  f"{sub['solved'].mean():>9.1%}  "
                  f"{sub['best_reward'].mean():>11.4f}  "
                  f"{sub['total_cost'].mean():>9.4f}")

    print()


# ── per-instance table ─────────────────────────────────────────────────────────

def print_per_instance(df: pd.DataFrame) -> None:
    cols = ["task_index", "source_dataset", "solved", "best_reward",
            "task_loss", "num_iters", "first_perfect_iter", "total_cost", "objective"]
    out = df[cols].copy()
    out["best_reward"] = out["best_reward"].map(lambda x: f"{x:.4f}")
    out["task_loss"]   = out["task_loss"].map(lambda x: f"{x:.4f}")
    out["total_cost"]  = out["total_cost"].map(lambda x: f"{x:.4f}" if x is not None else "N/A")
    out["objective"]   = out["objective"].map(lambda x: f"{x:.4f}" if x is not None else "N/A")
    print(out.to_string(index=False))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate symbolic-library-agent output JSONL files.")
    parser.add_argument("files", nargs="+", metavar="FILE", help="One or more output JSONL files")
    parser.add_argument("--csv",      default=None, metavar="FILE", help="Save per-instance table to CSV")
    parser.add_argument("--per-instance", action="store_true", help="Print full per-instance table")
    parser.add_argument("--combined", action="store_true", help="Also print summary over all files combined")
    args = parser.parse_args()

    dfs = []
    for path in args.files:
        df = load_file(path)
        print(f"\nLoaded {len(df)} records from {path}")
        summarise(df, label=f"— {Path(path).stem}")
        dfs.append(df)

    if len(dfs) > 1 and args.combined:
        combined = pd.concat(dfs, ignore_index=True)
        summarise(combined, label="— COMBINED")

    if args.per_instance:
        all_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        print("\n--- Per-instance table ---")
        print_per_instance(all_df)

    if args.csv:
        all_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        all_df.to_csv(args.csv, index=False)
        print(f"\nPer-instance CSV saved to {args.csv}")


if __name__ == "__main__":
    main()
