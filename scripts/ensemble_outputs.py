"""
Ensemble multiple run outputs (LLM and/or symbolic solver) into a single JSONL
suitable for quick_eval.py.

Selection rule per task:
  1. Pick the candidate(s) with the highest best_reward.
  2. Among ties, pick the least complex program (sum of len(A)+len(B) for all replace(A,B)).
  3. If complexity is also tied, prefer the first source listed on the command line.

Usage:
    python scripts/ensemble_outputs.py \\
        --sources outputs/lite_tasks_full_og_best_of_k.jsonl \\
                  evals/solver_results/lite.jsonl \\
        --out outputs/ensemble_bok_solver.jsonl

    python scripts/ensemble_outputs.py \\
        --sources outputs/lite_tasks_full_og_direct_feedback.jsonl \\
                  evals/solver_results/lite.jsonl \\
        --out outputs/ensemble_df_solver.jsonl

    python scripts/ensemble_outputs.py \\
        --sources outputs/lite_tasks_full_og_direct_feedback.jsonl \\
                  outputs/lite_tasks_full_og_best_of_k.jsonl \\
                  evals/solver_results/lite.jsonl \\
        --out outputs/ensemble_df_bok_solver.jsonl

Output fields (minimal set for quick_eval.py):
    task_index, solved, answer, best_reward, reward_history, token_usage, cost_summary
"""

import argparse
import json
import re
from pathlib import Path

_REPLACE_RE = re.compile(
    r"""replace\(\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\)""",
    re.IGNORECASE,
)


def _complexity(answer) -> float:
    """Return cascade complexity (sum of len(A)+len(B)), or inf if unparseable."""
    if answer is None:
        return float("inf")
    if isinstance(answer, list):
        raw = "\n".join(str(x) for x in answer)
    else:
        raw = str(answer).strip()
        raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip("` \n")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                raw = "\n".join(str(x) for x in parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    programs = _REPLACE_RE.findall(raw)
    if not programs:
        return float("inf")
    return sum(len(pred) + len(transform) for pred, transform in programs)


def _best_reward(rec: dict) -> float:
    v = rec.get("best_reward")
    if v is not None:
        return float(v)
    # Solver format: uses 'score' and 'success'
    if "score" in rec:
        return float(rec["score"])
    return 1.0 if rec.get("solved") or rec.get("success") else 0.0


def _answer(rec: dict):
    """Normalise answer field — solver uses 'program' list, LLM uses 'answer'."""
    if "answer" in rec:
        return rec["answer"]
    if "program" in rec:
        return rec["program"]  # already a list of replace(...) strings
    return None


def load_source(path: str) -> dict[int, dict]:
    """Load a JSONL file, return {task_index: record}. Keeps highest-reward record per task."""
    records: dict[int, dict] = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Determine task index: explicit field, or line number for solver files
            tid = rec.get("task_index")
            if tid is None:
                tid = i
            prev = records.get(tid)
            if prev is None or _best_reward(rec) > _best_reward(prev):
                records[tid] = rec
    return records


def ensemble(sources: list[dict[int, dict]]) -> list[dict]:
    """
    Merge sources into one record per task_index using the selection rule:
      1. Highest best_reward wins.
      2. Ties broken by lowest complexity.
      3. Ties broken by source order (first listed wins).
    """
    all_ids = sorted(set(tid for src in sources for tid in src))
    output = []

    for tid in all_ids:
        candidates = []
        for src in sources:
            rec = src.get(tid)
            if rec is not None:
                candidates.append(rec)

        if not candidates:
            continue

        best_r = max(_best_reward(c) for c in candidates)
        finalists = [c for c in candidates if _best_reward(c) >= best_r - 1e-9]
        # Break ties by complexity, then source order (stable since candidates are ordered)
        winner = min(finalists, key=lambda c: (_complexity(_answer(c)), candidates.index(c)))

        answer = _answer(winner)
        reward = _best_reward(winner)

        # Build a minimal reward_history that quick_eval can parse:
        #   - attempt count (for attempt distribution and feedback usage)
        #   - reward value (for first_solved_at_iter)
        #   - no blame needed for ensemble output
        rh = winner.get("reward_history")
        if rh is None:
            # Solver records have no reward_history — synthesise a single entry
            rh = [{"iteration": 0, "reward": reward}]

        # Sum token usage across all LLM sources for this task (solver contributes 0).
        combined_tokens: dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}
        for c in candidates:
            tu = c.get("token_usage") or {}
            for k in combined_tokens:
                combined_tokens[k] += int(tu.get(k, 0) or 0)

        output.append({
            "task_index": tid,
            "solved": reward >= 1.0,
            "answer": answer,
            "best_reward": reward,
            "reward_history": rh,
            "token_usage": combined_tokens,
            "cost_summary": winner.get("cost_summary", {}),
        })

    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", required=True, metavar="FILE",
                        help="Input JSONL files to ensemble (order determines tie-breaking priority)")
    parser.add_argument("--out", required=True, metavar="FILE",
                        help="Output JSONL path")
    args = parser.parse_args()

    sources = [load_source(p) for p in args.sources]
    total_tasks = len(set(tid for src in sources for tid in src))
    print(f"Loaded {len(args.sources)} sources, {total_tasks} unique task indices")
    for p, src in zip(args.sources, sources):
        print(f"  {Path(p).name}: {len(src)} tasks")

    records = ensemble(sources)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    solved = sum(1 for r in records if r["best_reward"] >= 1.0)
    print(f"\nEnsemble: {solved}/{len(records)} solved = {100*solved/len(records):.2f}%")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
