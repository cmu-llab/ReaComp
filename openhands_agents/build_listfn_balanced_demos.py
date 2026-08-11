"""
Build a balanced List Functions demos file from existing BoK candidate traces.

This-session, additive. Selects N tasks balanced across an intrinsic, model-free
hardness axis, and within each task keeps a few candidate traces balanced across
success/failure, so the coding agent sees both working and broken reasoning across
the difficulty range.

Hardness (intrinsic): mean normalized Levenshtein edit distance between each shown
input list and its output list, averaged over the task's shown pairs. This measures
"how much the list changes" independent of any model. Tasks are split into
hard/medium/easy by terciles of this score over the full task set.

Within a task we keep up to --attempts candidate traces, split evenly across
success (solved held-out) and failure. Output matches the demos schema SolverBuilder
consumes; no held-out data is written. Reproducible via --seed.
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _lev(a, b):
    """Levenshtein distance between two sequences (lists of ints)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def task_hardness(rec):
    """Mean normalized edit distance over shown (input, output) pairs."""
    ins, outs = rec["held_in_inputs"], rec["held_in_outputs"]
    scores = []
    for a, b in zip(ins, outs):
        d = _lev(a, b)
        denom = max(len(a), len(b), 1)
        scores.append(d / denom)
    return sum(scores) / len(scores) if scores else 0.0


def build_prompt(rec):
    from openhands_agents.run_listfn_traces import build_prompt as bp
    return bp(rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="outputs/listfn_full_bok_traces.jsonl")
    ap.add_argument("--tasks-file", default="data/list_functions/full_tasks.jsonl")
    ap.add_argument("--n-tasks", type=int, default=25)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="demos/DEMOS_LISTFN_balanced.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = {r["task_id"]: r for r in (json.loads(l) for l in open(args.tasks_file) if l.strip())}
    cand_by_task = defaultdict(list)
    for l in open(args.traces):
        if l.strip():
            c = json.loads(l)
            cand_by_task[c["task_id"]].append(c)

    # intrinsic hardness per task, terciled over all tasks that have candidates
    hard_score = {t: task_hardness(tasks[t]) for t in cand_by_task if t in tasks}
    ordered = sorted(hard_score, key=lambda t: hard_score[t])
    n = len(ordered)
    third = n // 3
    bucket = {}
    for i, t in enumerate(ordered):
        bucket[t] = "easy" if i < third else ("medium" if i < 2 * third else "hard")

    buckets = defaultdict(list)
    for t, b in bucket.items():
        buckets[b].append(t)

    per_bucket = max(1, args.n_tasks // 3)
    picked = []
    for b in ("hard", "medium", "easy"):
        ids = buckets.get(b, [])[:]
        rng.shuffle(ids)
        picked.extend(ids[:per_bucket])
    remaining = [t for t in cand_by_task if t not in picked]
    rng.shuffle(remaining)
    picked = (picked + remaining)[:args.n_tasks]

    demos = []
    stats = {"success": 0, "failure": 0, "hard": 0, "medium": 0, "easy": 0}
    for tid in picked:
        cs = cand_by_task[tid]
        stats[bucket[tid]] += 1
        succ = [c for c in cs if c.get("solved") or c.get("heldout_score") == 1.0]
        fail = [c for c in cs if not (c.get("solved") or c.get("heldout_score") == 1.0)]
        rng.shuffle(succ); rng.shuffle(fail)
        half = args.attempts // 2
        chosen = succ[:half] + fail[: args.attempts - half]
        if len(chosen) < args.attempts:  # backfill if one side is short
            for c in succ + fail:
                if c not in chosen:
                    chosen.append(c)
                if len(chosen) >= args.attempts:
                    break
        rec = tasks[tid]
        for c in chosen:
            is_succ = bool(c.get("solved") or c.get("heldout_score") == 1.0)
            stats["success" if is_succ else "failure"] += 1
            demos.append({
                "prompt": build_prompt(rec),
                "input_examples": rec["held_in_inputs"],
                "output_examples": rec["held_in_outputs"],
                "final_response": c.get("code") or "",
                "cot": c.get("reasoning") or "",
                "success": is_succ,
                "task_id": tid,
                "shown_score": c.get("shown_score"),
                "heldout_score": c.get("heldout_score"),
                "hardness": round(hard_score[tid], 3),
                "hardness_bucket": bucket[tid],
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(demos, f, indent=2)
    print(f"wrote {len(demos)} demo traces over {len(picked)} tasks -> {args.out}")
    print(f"  success={stats['success']} failure={stats['failure']} | "
          f"tasks: hard={stats['hard']} medium={stats['medium']} easy={stats['easy']}")


if __name__ == "__main__":
    main()
