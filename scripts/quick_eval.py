"""
Quick evaluation of output JSONL files.

Usage:
    python scripts/quick_eval.py outputs/file.jsonl
    python scripts/quick_eval.py outputs/a.jsonl outputs/b.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load(path: str) -> list[dict]:
    tasks: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("task_index", len(tasks))
            rh = rec.get("reward_history") or []
            prev = tasks.get(tid)
            if prev is None or len(rh) > len(prev.get("reward_history") or []):
                tasks[tid] = rec
    return list(tasks.values())


def best_reward(rec: dict) -> float:
    v = rec.get("best_reward")
    if v is not None:
        return float(v)
    return 1.0 if rec.get("solved") else 0.0


def reward_seq(rec: dict) -> list[float]:
    return [h.get("reward", 0.0) for h in (rec.get("reward_history") or [])]


def summarise(records: list[dict], label: str) -> None:
    n = len(records)
    rewards = [best_reward(r) for r in records]
    solved = sum(1 for v in rewards if v >= 1.0)

    attempt_counts = [len(rec.get("reward_history") or []) for rec in records]
    single = sum(1 for c in attempt_counts if c <= 1)
    multi = n - single
    multi_solved = sum(
        1 for r in records
        if len(r.get("reward_history") or []) > 1 and best_reward(r) >= 1.0
    )

    total_calls = sum(attempt_counts)
    mean_reward = sum(rewards) / n if n else 0.0

    # first-perfect-iter distribution
    first_perfect: Counter = Counter()
    never_perfect = 0
    for rec in records:
        rh = rec.get("reward_history") or []
        hit = next((h.get("iteration", i) for i, h in enumerate(rh) if h.get("reward", 0.0) >= 1.0), None)
        if hit is None:
            never_perfect += 1
        else:
            first_perfect[hit] += 1

    # blame
    blame_counter: Counter = Counter()
    for rec in records:
        rh = rec.get("reward_history") or []
        for h in rh:
            b = h.get("blame")
            if b:
                blame_counter[b] += 1

    # attempt distribution
    attempt_dist: Counter = Counter(attempt_counts)

    print(f"\n{'='*60}")
    print(f"  {label}  (n={n})")
    print(f"{'='*60}")

    print(f"\n  Pass rate   : {solved}/{n} = {100*solved/n:.1f}%")
    print(f"  Mean reward : {mean_reward:.4f}")
    print(f"  Task loss   : {1 - mean_reward:.4f}  (sum={sum(1-v for v in rewards):.2f})")

    print(f"\n  Feedback usage")
    print(f"    No feedback (1 attempt) : {single}/{n} = {100*single/n:.1f}%")
    print(f"    Used feedback (>=2)     : {multi}/{n} = {100*multi/n:.1f}%")
    if multi:
        print(f"      Of those, solved     : {multi_solved}/{multi} = {100*multi_solved/multi:.1f}%")
    print(f"    Total LLM calls         : {total_calls}")
    print(f"    Avg attempts / task     : {total_calls/n:.2f}")

    attempt_buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, None)]
    print(f"\n  Attempt distribution")
    for lo, hi in attempt_buckets:
        count = sum(v for k, v in attempt_dist.items() if lo <= k <= (hi if hi else 10**9))
        if not count:
            continue
        label_k = f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")
        label_k = f"{label_k} attempt{'s' if lo != 1 or hi != 1 else ''}"
        bar = "#" * min(count, 40)
        print(f"    {label_k:>12} : {count:3d}  {bar}")

    iter_buckets = [(0, 0), (1, 1), (2, 2), (3, None)]
    print(f"\n  First solved at iteration")
    for lo, hi in iter_buckets:
        count = sum(v for it, v in first_perfect.items() if lo <= it <= (hi if hi else 10**9))
        if not count:
            continue
        label_it = f"iter {lo}" if lo == hi else f"iter {lo}+"
        print(f"    {label_it:>8} : {count:3d}  ({100*count/n:.1f}%)")
    print(f"    {'never':>8} : {never_perfect:3d}  ({100*never_perfect/n:.1f}%)")

    if blame_counter:
        print(f"\n  Blame distribution (across all iters)")
        for blame, cnt in blame_counter.most_common():
            print(f"    {blame:<20} : {cnt}")

    # token usage
    n_with_tokens = sum(1 for r in records if r.get("token_usage"))
    if n_with_tokens:
        total_in    = sum((r.get("token_usage") or {}).get("input",     0) or 0 for r in records)
        total_out   = sum((r.get("token_usage") or {}).get("output",    0) or 0 for r in records)
        total_reas  = sum((r.get("token_usage") or {}).get("reasoning", 0) or 0 for r in records)
        total_toks  = total_in + total_out
        print(f"\n  Token usage  ({n_with_tokens}/{n} tasks have data)")
        print(f"    Input     : {total_in:>12,}  (avg {total_in/n_with_tokens:>8,.1f}/task)")
        print(f"    Output    : {total_out:>12,}  (avg {total_out/n_with_tokens:>8,.1f}/task)")
        if total_reas:
            print(f"    Reasoning : {total_reas:>12,}  (avg {total_reas/n_with_tokens:>8,.1f}/task)")
        print(f"    Total     : {total_toks:>12,}  (avg {total_toks/n_with_tokens:>8,.1f}/task)")

    # unsolved
    unsolved = [(r.get("task_index"), best_reward(r), len(r.get("reward_history") or []))
                for r in records if best_reward(r) < 1.0]
    if unsolved:
        print(f"\n  Unsolved tasks ({len(unsolved)})")
        for tid, br, na in sorted(unsolved):
            print(f"    task {tid:>4} : best_reward={br:.2f}, attempts={na}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", metavar="FILE")
    args = parser.parse_args()

    all_records = []
    for path in args.files:
        records = load(path)
        summarise(records, Path(path).stem)
        all_records.extend(records)

    if len(args.files) > 1:
        summarise(all_records, "COMBINED")


if __name__ == "__main__":
    main()
