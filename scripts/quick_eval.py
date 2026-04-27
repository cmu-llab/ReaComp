"""
Quick evaluation of output JSONL files.

Usage:
    python scripts/quick_eval.py outputs/file.jsonl
    python scripts/quick_eval.py outputs/a.jsonl outputs/b.jsonl
    python scripts/quick_eval.py outputs/file.jsonl --tasks-file data/pbebench/lite_tasks_full_og.jsonl
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import editdistance as _editdistance
    _EDITDISTANCE_AVAILABLE = True
except ImportError:
    _EDITDISTANCE_AVAILABLE = False

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_REPLACE_RE = re.compile(
    r"""replace\(\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\)""",
    re.IGNORECASE,
)


def _parse_slr_complexity(answer) -> int | None:
    """Extract a Prolog eastbound rule from answer and return its rule_complexity."""
    try:
        from rewards.slr_bench import rule_complexity, _extract_rule
        rule, _ = _extract_rule(answer)
        if rule is None:
            return None
        return rule_complexity(rule)
    except Exception:
        return None


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


def _apply_programs(programs: list[tuple[str, str]], inputs: list[str]) -> list[str]:
    """Apply a replace() cascade to each input string."""
    results = []
    for inp in inputs:
        cur = inp
        for pred, transform in programs:
            cur = cur.replace(pred, transform)
        results.append(cur)
    return results


def _parse_programs_any(answer) -> list[tuple[str, str]] | None:
    """Parse programs from any answer format: replace() strings or [[pred,transform],...] pairs."""
    if answer is None:
        return None
    # Qwen-style: [[pred, transform], ...] or [pred, transform] (single pair)
    if isinstance(answer, list) and answer:
        first = answer[0]
        if isinstance(first, (list, tuple)) and len(first) == 2 and all(isinstance(x, str) for x in first):
            return [(str(p), str(t)) for p, t in answer]
    from rewards.pbebench import _parse_programs as _pbebench_parse
    programs, _ = _pbebench_parse(answer)
    return programs


def _edit_sim(pred_outputs: list[str], inputs: list[str], targets: list[str]) -> float | None:
    """
    Edit similarity: 1 - (edit_dist(pred, target) / edit_dist(input, target)).
    Tokenised by whitespace, summed over all pairs. Returns None if denominator is 0.
    """
    if not _EDITDISTANCE_AVAILABLE:
        return None
    pred_tok    = [s.split() for s in pred_outputs]
    inputs_tok  = [s.split() for s in inputs]
    targets_tok = [s.split() for s in targets]
    num = sum(_editdistance.eval(p, t) for p, t in zip(pred_tok, targets_tok))
    den = sum(_editdistance.eval(i, t) for i, t in zip(inputs_tok, targets_tok))
    if den == 0:
        return None
    return 1.0 - num / den


def _syntax_valid_slr(answer, validation_program: str) -> bool | None:
    """
    Return True if the rule is syntactically valid Prolog, False if not, None on error.
    Only called for records with best_reward == 0.0 (positive reward implies valid syntax).
    """
    try:
        from rewards.slr_bench import _get_judge, _extract_rule
        rule, _ = _extract_rule(answer)
        if rule is None:
            return False
        judge = _get_judge()
        results = judge.compute(
            predictions=[rule],
            references=[{
                "validation_program": validation_program,
                "evaluation_config": {
                    "positive_predicate": "eastbound",
                    "negative_predicate": "westbound",
                },
            }],
        )
        detail = results["detailed_results"][0]
        return bool(detail.get("syntax_valid", False))
    except Exception:
        return None


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
    """Load task data file and return {task_index: metadata} dict.

    Supports both PBEBench (cascade_length, bfcc_dag, original_programs) and
    SLR-Bench (ground-truth rule, rule complexity, curriculum level/tier) formats.
    """
    meta: dict[int, dict] = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # PBEBench fields
            bfcc_dag = rec.get("bfcc_dag")
            dag_len = None
            if bfcc_dag is not None:
                try:
                    dag = json.loads(bfcc_dag) if isinstance(bfcc_dag, str) else bfcc_dag
                    dag_len = len(dag)
                except (json.JSONDecodeError, TypeError):
                    pass
            gt_complexity = _parse_complexity(rec.get("original_programs"))
            # SLR-Bench fields
            gt_rule = rec.get("ground-truth rule")
            slr_gt_complexity = _parse_slr_complexity(gt_rule) if gt_rule else None
            meta[i] = {
                "cascade_length": rec.get("cascade_length"),
                "bfcc_dag_len": dag_len,
                "gt_complexity": gt_complexity,
                "inputs": rec.get("inputs"),
                "outputs": rec.get("outputs"),
                "slr_gt_rule": gt_rule,
                "slr_gt_complexity": slr_gt_complexity,
                "validation_program": rec.get("validation program"),
                "rule_complexity": rec.get("rule complexity"),
                "curriculum_level": rec.get("curriculum level"),
                "curriculum_tier": rec.get("curriculum tier"),
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


def _print_breakdown(label: str, key_fn, records: list[dict], n_total: int) -> list[dict]:
    """Print pass rate and mean attempts broken down by a per-record integer key. Returns row data."""
    by_key: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        k = key_fn(rec)
        if k is not None:
            by_key[k].append(rec)

    if not by_key:
        return []

    all_vals = list(by_key.keys())
    buckets = _make_buckets(all_vals)

    print(f"\n  By {label}")
    header = f"    {'value':>8}  {'n':>5}  {'pass%':>6}  {'mean_reward':>11}  {'avg_attempts':>12}"
    print(header)
    rows = []
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
        rows.append({"value": lbl, "n": n, "pass_pct": round(100 * solved / n, 2), "mean_reward": round(mean_r, 4), "avg_attempts": round(mean_att, 2)})
    return rows


def best_reward(rec: dict) -> float:
    v = rec.get("best_reward")
    if v is not None:
        return float(v)
    return 1.0 if rec.get("solved") else 0.0


def reward_seq(rec: dict) -> list[float]:
    return [h.get("reward", 0.0) for h in (rec.get("reward_history") or [])]


def summarise(records: list[dict], label: str, task_meta: dict[int, dict] | None = None) -> dict:
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
    attempt_dist_rows = []
    print(f"\n  Attempt distribution")
    for lo, hi in attempt_buckets:
        count = sum(v for k, v in attempt_dist.items() if lo <= k <= (hi if hi else 10**9))
        if not count:
            continue
        label_k = f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")
        label_k = f"{label_k} attempt{'s' if lo != 1 or hi != 1 else ''}"
        bar = "#" * min(count, 40)
        print(f"    {label_k:>12} : {count:3d}  {bar}")
        attempt_dist_rows.append({"bucket": label_k, "count": count})

    iter_dist_rows = []
    iter_buckets = [(0, 0), (1, 1), (2, 2), (3, None)]
    print(f"\n  First solved at iteration")
    for lo, hi in iter_buckets:
        count = sum(v for it, v in first_perfect.items() if lo <= it <= (hi if hi else 10**9))
        if not count:
            continue
        label_it = f"iter {lo}" if lo == hi else f"iter {lo}+"
        print(f"    {label_it:>8} : {count:3d}  ({100*count/n:.1f}%)")
        iter_dist_rows.append({"bucket": label_it, "count": count, "pct": round(100 * count / n, 2)})
    print(f"    {'never':>8} : {never_perfect:3d}  ({100*never_perfect/n:.1f}%)")
    iter_dist_rows.append({"bucket": "never", "count": never_perfect, "pct": round(100 * never_perfect / n, 2)})

    if blame_counter:
        print(f"\n  Blame distribution (across all iters)")
        for blame, cnt in blame_counter.most_common():
            print(f"    {blame:<20} : {cnt}")

    # token usage
    token_metrics: dict = {}
    n_with_tokens = sum(1 for r in records if r.get("token_usage"))
    if n_with_tokens:
        total_in    = sum((r.get("token_usage") or {}).get("input",     0) or 0 for r in records)
        total_out   = sum((r.get("token_usage") or {}).get("output",    0) or 0 for r in records)
        total_reas  = sum((r.get("token_usage") or {}).get("reasoning", 0) or 0 for r in records)
        total_toks  = total_in + total_out + total_reas
        print(f"\n  Token usage  ({n_with_tokens}/{n} tasks have data)")
        print(f"    Input     : {total_in:>12,}  (avg {total_in/n_with_tokens:>8,.1f}/task)")
        print(f"    Output    : {total_out:>12,}  (avg {total_out/n_with_tokens:>8,.1f}/task)")
        if total_reas:
            print(f"    Reasoning : {total_reas:>12,}  (avg {total_reas/n_with_tokens:>8,.1f}/task)")
        print(f"    Total     : {total_toks:>12,}  (avg {total_toks/n_with_tokens:>8,.1f}/task)")
        token_metrics = {
            "tasks_with_data": n_with_tokens,
            "input_total": total_in, "input_avg": round(total_in / n_with_tokens, 1),
            "output_total": total_out, "output_avg": round(total_out / n_with_tokens, 1),
            "reasoning_total": total_reas, "reasoning_avg": round(total_reas / n_with_tokens, 1) if total_reas else None,
            "total": total_toks, "total_avg": round(total_toks / n_with_tokens, 1),
        }

    # PBEBench cascade complexity — computed from the best answer (replace() cascade)
    complexity_metrics: dict = {}
    complexities = []
    for rec in records:
        c = _parse_complexity(rec.get("answer"))
        if c is not None:
            complexities.append(c)
    if complexities:
        nc = len(complexities)
        mean_c = sum(complexities) / nc
        buckets = [(0, 4), (5, 8), (9, 12), (13, 20), (21, None)]
        complexity_dist_rows = []
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
            complexity_dist_rows.append({"bucket": label_c, "count": cnt})
        complexity_metrics = {"n": nc, "mean": round(mean_c, 2), "distribution": complexity_dist_rows}

    # complexity vs ground truth (requires --tasks-file join)
    complexity_vs_gt: dict = {}
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
            complexity_vs_gt = {
                "n": n_pairs,
                "mean_pred": round(mean_pred, 2), "mean_gt": round(mean_gt, 2),
                "mean_delta": round(mean_delta, 2),
                "simpler": simpler, "equal": equal, "more_complex": more_complex,
            }

    # PBEBench edit similarity (requires --tasks-file with inputs/outputs)
    edit_sim_metrics: dict = {}
    if task_meta and _EDITDISTANCE_AVAILABLE:
        sim_scores = []
        for rec in records:
            m = task_meta.get(rec.get("task_index")) or {}
            task_inputs  = m.get("inputs")
            task_outputs = m.get("outputs")
            if not task_inputs or not task_outputs:
                continue
            programs = _parse_programs_any(rec.get("answer"))
            if programs is None:
                continue
            pred_outputs = _apply_programs(programs, task_inputs)
            sim = _edit_sim(pred_outputs, task_inputs, task_outputs)
            if sim is not None:
                sim_scores.append(sim)
        if sim_scores:
            mean_sim = sum(sim_scores) / len(sim_scores)
            solved_sim_scores = []
            for rec in records:
                if best_reward(rec) < 1.0:
                    continue
                m = task_meta.get(rec.get("task_index")) or {}
                task_inputs  = m.get("inputs")
                task_outputs = m.get("outputs")
                if not task_inputs or not task_outputs:
                    continue
                programs = _parse_programs_any(rec.get("answer"))
                if programs is None:
                    continue
                pred_outputs = _apply_programs(programs, task_inputs)
                sim = _edit_sim(pred_outputs, task_inputs, task_outputs)
                if sim is not None:
                    solved_sim_scores.append(sim)
            mean_sim_solved = sum(solved_sim_scores) / len(solved_sim_scores) if solved_sim_scores else None
            print(f"\n  Edit similarity  ({len(sim_scores)}/{n} tasks)")
            print(f"    Mean (all tasks)     : {mean_sim:.4f}")
            if mean_sim_solved is not None:
                print(f"    Mean (solved only)   : {mean_sim_solved:.4f}")
            edit_sim_metrics = {
                "n": len(sim_scores),
                "mean": round(mean_sim, 4),
                "mean_solved": round(mean_sim_solved, 4) if mean_sim_solved is not None else None,
            }

    # SLR-Bench syntax rate
    # Invariant (validated): partial_score > 0 ↔ syntax_valid=True.
    # So records with best_reward > 0 are always syntax-valid; only re-run the judge
    # for best_reward == 0.0 records to distinguish syntax errors from logic failures.
    slr_syntax_metrics: dict = {}
    _is_slr = any(r.get("curriculum_tier") is not None or r.get("dataset") == "SLR-Bench" for r in records)
    if _is_slr:
        n_positive = sum(1 for r in records if best_reward(r) > 0.0)
        zero_records = [r for r in records if best_reward(r) == 0.0]
        n_zero_syntax_valid = 0
        if zero_records and task_meta:
            print(f"\n  SLR syntax check: re-running judge on {len(zero_records)} zero-reward records...")
            for rec in zero_records:
                m = task_meta.get(rec.get("task_index")) or {}
                vp = m.get("validation_program")
                if not vp:
                    continue
                sv = _syntax_valid_slr(rec.get("answer"), vp)
                if sv:
                    n_zero_syntax_valid += 1
        n_syntax_valid = n_positive + n_zero_syntax_valid
        syntax_rate = 100.0 * n_syntax_valid / n if n else 0.0
        print(f"\n  SLR syntax rate")
        print(f"    Syntax-valid (reward>0)  : {n_positive}/{n}")
        if zero_records:
            print(f"    Syntax-valid (reward=0)  : {n_zero_syntax_valid}/{len(zero_records)}")
        print(f"    Overall syntax rate      : {n_syntax_valid}/{n} = {syntax_rate:.1f}%")
        slr_syntax_metrics = {
            "n_syntax_valid": n_syntax_valid,
            "n_total": n,
            "syntax_rate": round(syntax_rate, 2),
            "n_positive_reward": n_positive,
            "n_zero_reward_syntax_valid": n_zero_syntax_valid,
            "n_zero_reward_syntax_invalid": len(zero_records) - n_zero_syntax_valid,
        }

    # cascade length / BFCC breakdowns (requires --tasks-file join)
    breakdown_cascade: list[dict] = []
    breakdown_bfcc: list[dict] = []
    if task_meta:
        def _cascade_len(rec):
            return (task_meta.get(rec.get("task_index")) or {}).get("cascade_length")

        def _bfcc_dag_len(rec):
            return (task_meta.get(rec.get("task_index")) or {}).get("bfcc_dag_len")

        breakdown_cascade = _print_breakdown("cascade length", _cascade_len, records, n)
        breakdown_bfcc = _print_breakdown("BFCC relation count", _bfcc_dag_len, records, n)

    # SLR-Bench rule complexity — analogous to PBEBench cascade complexity
    slr_complexity_metrics: dict = {}
    slr_complexity_vs_gt: dict = {}
    _slr_complexities = []
    for rec in records:
        c = _parse_slr_complexity(rec.get("answer"))
        if c is not None:
            _slr_complexities.append(c)
    if _slr_complexities:
        nc = len(_slr_complexities)
        mean_c = sum(_slr_complexities) / nc
        buckets = [(1, 1), (2, 2), (3, 3), (4, 4), (5, None)]
        slr_dist_rows = []
        print(f"\n  SLR rule complexity  ({nc}/{n} tasks)")
        print(f"    Mean complexity : {mean_c:.2f}")
        print(f"    Distribution")
        for lo, hi in buckets:
            cnt = sum(1 for c in _slr_complexities if lo <= c <= (hi if hi is not None else 10**9))
            if not cnt:
                continue
            label_c = f"{lo}+" if hi is None else f"{lo}"
            bar = "#" * min(cnt, 40)
            print(f"      {label_c:>4} : {cnt:3d}  {bar}")
            slr_dist_rows.append({"bucket": label_c, "count": cnt})
        slr_complexity_metrics = {"n": nc, "mean": round(mean_c, 3), "distribution": slr_dist_rows}

        # vs ground truth (requires --tasks-file with SLR data)
        if task_meta and any(v.get("slr_gt_complexity") is not None for v in task_meta.values()):
            pairs = []
            for rec in records:
                pred_c = _parse_slr_complexity(rec.get("answer"))
                gt_c = (task_meta.get(rec.get("task_index")) or {}).get("slr_gt_complexity")
                if pred_c is not None and gt_c is not None and best_reward(rec) >= 1.0:
                    pairs.append((pred_c, gt_c))
            if pairs:
                n_pairs = len(pairs)
                simpler = sum(1 for p, g in pairs if p < g)
                equal   = sum(1 for p, g in pairs if p == g)
                more_complex = sum(1 for p, g in pairs if p > g)
                mean_pred  = sum(p for p, _ in pairs) / n_pairs
                mean_gt    = sum(g for _, g in pairs) / n_pairs
                mean_delta = sum(p - g for p, g in pairs) / n_pairs
                print(f"\n  SLR rule complexity vs GT  (correct solutions only, n={n_pairs})")
                print(f"    Mean predicted complexity : {mean_pred:.3f}")
                print(f"    Mean GT complexity        : {mean_gt:.3f}")
                print(f"    Mean delta (pred − GT)    : {mean_delta:+.3f}")
                print(f"    Simpler than GT  (pred<GT): {simpler:>4} / {n_pairs}  ({100*simpler/n_pairs:.1f}%)")
                print(f"    Equal to GT      (pred=GT): {equal:>4} / {n_pairs}  ({100*equal/n_pairs:.1f}%)")
                print(f"    More complex     (pred>GT): {more_complex:>4} / {n_pairs}  ({100*more_complex/n_pairs:.1f}%)")
                slr_complexity_vs_gt = {
                    "n": n_pairs,
                    "mean_pred": round(mean_pred, 3), "mean_gt": round(mean_gt, 3),
                    "mean_delta": round(mean_delta, 3),
                    "simpler": simpler, "equal": equal, "more_complex": more_complex,
                }

    # SLR-Bench breakdowns — triggered by presence of curriculum_level / curriculum_tier fields
    slr_breakdown_tier: list[dict] = []
    slr_breakdown_level: list[dict] = []
    slr_breakdown_complexity: list[dict] = []
    _slr_records = [r for r in records if r.get("curriculum_tier") is not None or r.get("curriculum_level") is not None]
    if _slr_records:
        # By curriculum tier (ordered)
        _tier_order = ["basic", "easy", "medium", "hard"]
        _by_tier: dict = defaultdict(list)
        for rec in _slr_records:
            t = rec.get("curriculum_tier")
            if t:
                _by_tier[t].append(rec)
        if _by_tier:
            print(f"\n  By curriculum tier")
            print(f"    {'tier':>8}  {'n':>5}  {'pass%':>6}  {'mean_reward':>11}")
            for tier in _tier_order:
                recs = _by_tier.get(tier, [])
                if not recs:
                    continue
                nt = len(recs)
                sol = sum(1 for r in recs if best_reward(r) >= 1.0)
                mr = sum(best_reward(r) for r in recs) / nt
                print(f"    {tier:>8}  {nt:>5}  {100*sol/nt:>5.1f}%  {mr:>11.4f}")
                slr_breakdown_tier.append({"tier": tier, "n": nt, "pass_pct": round(100*sol/nt, 2), "mean_reward": round(mr, 4)})

        # By curriculum level (grouped into 5-level bands)
        _by_level: dict = defaultdict(list)
        for rec in _slr_records:
            lv = rec.get("curriculum_level")
            if lv is not None:
                _by_level[int(lv)].append(rec)
        if _by_level:
            _level_buckets = [(1, 5), (6, 10), (11, 15), (16, 20)]
            print(f"\n  By curriculum level")
            print(f"    {'levels':>8}  {'n':>5}  {'pass%':>6}  {'mean_reward':>11}")
            for lo, hi in _level_buckets:
                recs = [r for lv, rs in _by_level.items() for r in rs if lo <= lv <= hi]
                if not recs:
                    continue
                nt = len(recs)
                sol = sum(1 for r in recs if best_reward(r) >= 1.0)
                mr = sum(best_reward(r) for r in recs) / nt
                lbl = f"{lo}-{hi}"
                print(f"    {lbl:>8}  {nt:>5}  {100*sol/nt:>5.1f}%  {mr:>11.4f}")
                slr_breakdown_level.append({"levels": lbl, "n": nt, "pass_pct": round(100*sol/nt, 2), "mean_reward": round(mr, 4)})

        # By rule complexity (string labels like '1', '1-2', '2-3', etc.)
        _by_rc: dict = defaultdict(list)
        for rec in _slr_records:
            rc = rec.get("rule_complexity")
            if rc is not None:
                _by_rc[str(rc)].append(rec)
        if _by_rc:
            _rc_order = sorted(_by_rc.keys(), key=lambda x: float(x.split("-")[0]))
            print(f"\n  By rule complexity")
            print(f"    {'complexity':>10}  {'n':>5}  {'pass%':>6}  {'mean_reward':>11}")
            for rc in _rc_order:
                recs = _by_rc[rc]
                nt = len(recs)
                sol = sum(1 for r in recs if best_reward(r) >= 1.0)
                mr = sum(best_reward(r) for r in recs) / nt
                print(f"    {rc:>10}  {nt:>5}  {100*sol/nt:>5.1f}%  {mr:>11.4f}")
                slr_breakdown_complexity.append({"rule_complexity": rc, "n": nt, "pass_pct": round(100*sol/nt, 2), "mean_reward": round(mr, 4)})

    # unsolved
    unsolved = [
        {"task_index": r.get("task_index"), "best_reward": round(best_reward(r), 4), "attempts": len(r.get("reward_history") or [])}
        for r in records if best_reward(r) < 1.0
    ]
    if unsolved:
        # print(f"\n  Unsolved tasks ({len(unsolved)})")
        # for u in sorted(unsolved, key=lambda x: x["task_index"] or 0):
        #     print(f"    task {u['task_index']:>4} : best_reward={u['best_reward']:.2f}, attempts={u['attempts']}")
        print(f"\n  Unsolved tasks : {len(unsolved)}")

    print()

    return {
        "label": label,
        "n": n,
        "pass_rate": round(100 * solved / n, 2) if n else 0.0,
        "solved": solved,
        "mean_reward": round(mean_reward, 4),
        "task_loss_mean": round(1 - mean_reward, 4),
        "task_loss_sum": round(sum(1 - v for v in rewards), 4),
        "feedback": {
            "no_feedback": single, "used_feedback": multi,
            "feedback_solved": multi_solved,
            "total_llm_calls": total_calls,
            "avg_attempts_per_task": round(total_calls / n, 2) if n else 0.0,
        },
        "attempt_distribution": attempt_dist_rows,
        "first_solved_at_iter": iter_dist_rows,
        "blame": dict(blame_counter.most_common()),
        "token_usage": token_metrics or None,
        "pbe_complexity": complexity_metrics or None,
        "complexity_vs_gt": complexity_vs_gt or None,
        "edit_sim": edit_sim_metrics or None,
        "breakdown_by_cascade_length": breakdown_cascade or None,
        "breakdown_by_bfcc_relations": breakdown_bfcc or None,
        "slr_complexity": slr_complexity_metrics or None,
        "slr_complexity_vs_gt": slr_complexity_vs_gt or None,
        "slr_breakdown_by_tier": slr_breakdown_tier or None,
        "slr_breakdown_by_level": slr_breakdown_level or None,
        "slr_breakdown_by_rule_complexity": slr_breakdown_complexity or None,
        "slr_syntax": slr_syntax_metrics or None,
        "unsolved": sorted(unsolved, key=lambda x: x["task_index"] or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", metavar="FILE")
    parser.add_argument(
        "--tasks-file",
        metavar="FILE",
        help="Task data JSONL (same order as used during eval). "
             "Enables per-cascade-length and per-BFCC-relation-count breakdowns.",
    )
    parser.add_argument(
        "--metrics-json",
        metavar="FILE",
        help="Write all metrics to this JSON file (one entry per input file, plus COMBINED if multiple).",
    )
    args = parser.parse_args()

    task_meta = load_task_metadata(args.tasks_file) if args.tasks_file else None

    all_records = []
    all_metrics = []
    for path in args.files:
        records = load(path)
        metrics = summarise(records, Path(path).stem, task_meta=task_meta)
        all_metrics.append(metrics)
        all_records.extend(records)

    if len(args.files) > 1:
        combined = summarise(all_records, "COMBINED", task_meta=task_meta)
        all_metrics.append(combined)

    if args.metrics_json:
        out = all_metrics[0] if len(all_metrics) == 1 else all_metrics
        with open(args.metrics_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Metrics written to {args.metrics_json}")


if __name__ == "__main__":
    main()
