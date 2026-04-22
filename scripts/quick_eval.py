"""
Quick evaluation of output JSONL files.

Usage:
    python scripts/quick_eval.py outputs/file.jsonl
    python scripts/quick_eval.py outputs/a.jsonl outputs/b.jsonl
    python scripts/quick_eval.py outputs/file.jsonl --tasks-file data/pbebench/lite_tasks_full_og.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPLACE_RE = re.compile(
    r"""replace\(\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\)""",
    re.IGNORECASE,
)


def _parse_complexity(answer) -> int | None:
    """
    Parse the agent's answer into replace() programs and return cascade complexity
    (sum of all predicate + transform string lengths), or None if no programs found.
    """
    if answer is None:
        return None
    if isinstance(answer, list):
        if len(answer) == 1 and isinstance(answer[0], list):
            answer = answer[0]
        raw = "\n".join(str(x) for x in answer)
    elif isinstance(answer, str):
        raw = answer.strip()
        raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip("` \n")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                raw = "\n".join(str(x) for x in parsed)
            else:
                raw = str(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        return None
    programs = _REPLACE_RE.findall(raw)
    if not programs:
        return None
    return sum(len(pred) + len(transform) for pred, transform in programs)


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


def load_task_metadata(path: str) -> dict[int, dict]:
    """Load task data file and return {task_index: {cascade_length, bfcc_dag_len, gt_complexity}} dict."""
    meta: dict[int, dict] = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bfcc_dag = rec.get("bfcc_dag")
            dag_len = None
            if bfcc_dag is not None:
                try:
                    dag = json.loads(bfcc_dag) if isinstance(bfcc_dag, str) else bfcc_dag
                    dag_len = len(dag)
                except (json.JSONDecodeError, TypeError):
                    pass
            gt_complexity = _parse_complexity(rec.get("original_programs"))
            meta[i] = {
                "cascade_length": rec.get("cascade_length"),
                "bfcc_dag_len": dag_len,
                "gt_complexity": gt_complexity,
            }
    return meta


def _make_buckets(values: list[int]) -> list[tuple[int, int | None]]:
    """Return bucket boundaries. If <=5 distinct values, one bucket per value; else ~5 equal-width buckets."""
    distinct = sorted(set(values))
    if len(distinct) <= 5:
        return [(v, v) for v in distinct]
    lo, hi = distinct[0], distinct[-1]
    width = max(1, (hi - lo + 1) // 5)
    buckets, start = [], lo
    while start <= hi:
        end = start + width - 1
        buckets.append((start, end if end < hi else None))
        start += width
    return buckets


def _bucket_label(lo: int, hi: int | None) -> str:
    return f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")


def _print_breakdown(label: str, key_fn, records: list[dict], n_total: int) -> None:
    """Print pass rate and mean attempts broken down by a per-record integer key."""
    by_key: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        k = key_fn(rec)
        if k is not None:
            by_key[k].append(rec)

    if not by_key:
        return

    all_vals = list(by_key.keys())
    buckets = _make_buckets(all_vals)

    print(f"\n  By {label}")
    header = f"    {'value':>8}  {'n':>5}  {'pass%':>6}  {'mean_reward':>11}  {'avg_attempts':>12}"
    print(header)
    for lo, hi in buckets:
        recs = [r for k, rs in by_key.items() for r in rs if lo <= k <= (hi if hi is not None else 10**9)]
        if not recs:
            continue
        n = len(recs)
        solved = sum(1 for r in recs if best_reward(r) >= 1.0)
        mean_r = sum(best_reward(r) for r in recs) / n
        mean_att = sum(len(r.get("reward_history") or []) for r in recs) / n
        lbl = _bucket_label(lo, hi)
        print(f"    {lbl:>8}  {n:>5}  {100*solved/n:>5.1f}%  {mean_r:>11.4f}  {mean_att:>12.2f}")


def best_reward(rec: dict) -> float:
    v = rec.get("best_reward")
    if v is not None:
        return float(v)
    return 1.0 if rec.get("solved") else 0.0


def reward_seq(rec: dict) -> list[float]:
    return [h.get("reward", 0.0) for h in (rec.get("reward_history") or [])]


def summarise(records: list[dict], label: str, task_meta: dict[int, dict] | None = None) -> None:
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

    # PBEBench complexity — computed from the best answer (replace() cascade)
    complexities = []
    for rec in records:
        c = _parse_complexity(rec.get("answer"))
        if c is not None:
            complexities.append(c)
    if complexities:
        nc = len(complexities)
        mean_c = sum(complexities) / nc
        buckets = [(0, 4), (5, 8), (9, 12), (13, 20), (21, None)]
        print(f"\n  PBEBench cascade complexity  ({nc}/{n} tasks)")
        print(f"    Mean complexity : {mean_c:.1f}")
        print(f"    Distribution")
        for lo, hi in buckets:
            cnt = sum(1 for c in complexities if lo <= c <= (hi if hi is not None else 10**9))
            if not cnt:
                continue
            label_c = f"{lo}+" if hi is None else f"{lo}-{hi}"
            bar = "#" * min(cnt, 40)
            print(f"      {label_c:>6} : {cnt:3d}  {bar}")

    # complexity vs ground truth (requires --tasks-file join)
    if task_meta:
        pairs = []
        for rec in records:
            pred_c = _parse_complexity(rec.get("answer"))
            gt_c = (task_meta.get(rec.get("task_index")) or {}).get("gt_complexity")
            if pred_c is not None and gt_c is not None and best_reward(rec) >= 1.0:
                pairs.append((pred_c, gt_c))
        if pairs:
            n_pairs = len(pairs)
            simpler   = sum(1 for p, g in pairs if p < g)
            equal     = sum(1 for p, g in pairs if p == g)
            more_complex = sum(1 for p, g in pairs if p > g)
            mean_pred = sum(p for p, _ in pairs) / n_pairs
            mean_gt   = sum(g for _, g in pairs) / n_pairs
            mean_delta = sum(p - g for p, g in pairs) / n_pairs
            print(f"\n  Complexity vs ground truth  (correct solutions only, n={n_pairs})")
            print(f"    Mean predicted complexity : {mean_pred:.2f}")
            print(f"    Mean GT complexity        : {mean_gt:.2f}")
            print(f"    Mean delta (pred − GT)    : {mean_delta:+.2f}")
            print(f"    Simpler than GT  (pred<GT): {simpler:>4} / {n_pairs}  ({100*simpler/n_pairs:.1f}%)")
            print(f"    Equal to GT      (pred=GT): {equal:>4} / {n_pairs}  ({100*equal/n_pairs:.1f}%)")
            print(f"    More complex     (pred>GT): {more_complex:>4} / {n_pairs}  ({100*more_complex/n_pairs:.1f}%)")

    # cascade length / BFCC breakdowns (requires --tasks-file join)
    if task_meta:
        def _cascade_len(rec):
            return (task_meta.get(rec.get("task_index")) or {}).get("cascade_length")

        def _bfcc_dag_len(rec):
            return (task_meta.get(rec.get("task_index")) or {}).get("bfcc_dag_len")

        _print_breakdown("cascade length", _cascade_len, records, n)
        _print_breakdown("BFCC relation count", _bfcc_dag_len, records, n)

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
    parser.add_argument(
        "--tasks-file",
        metavar="FILE",
        help="Task data JSONL (same order as used during eval). "
             "Enables per-cascade-length and per-BFCC-relation-count breakdowns.",
    )
    args = parser.parse_args()

    task_meta = load_task_metadata(args.tasks_file) if args.tasks_file else None

    all_records = []
    for path in args.files:
        records = load(path)
        summarise(records, Path(path).stem, task_meta=task_meta)
        all_records.extend(records)

    if len(args.files) > 1:
        summarise(all_records, "COMBINED", task_meta=task_meta)


if __name__ == "__main__":
    main()
