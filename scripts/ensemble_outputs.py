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

    # Efficiency mode: symbolic solver takes priority when correct; LLM tokens
    # are only counted when the symbolic solver is imperfect.
    python scripts/ensemble_outputs.py --effi \\
        --symbolic evals/solver_results/claude_code/.../lite.jsonl \\
        --sources outputs/lite_tasks_full_og_direct_feedback.jsonl \\
        --out outputs/ensemble_effi_df_claude_solver.jsonl

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


def ensemble_effi(symbolic_src: dict[int, dict], llm_sources: list[dict[int, dict]]) -> list[dict]:
    """
    Efficiency-focused ensemble strategy.

    Per task:
      - If the symbolic solver produced a perfect program (reward == 1.0):
          use it as-is; token_usage = 0 (LLM never needed).
      - Otherwise:
          pick the best LLM candidate by (highest reward, then lowest complexity,
          then source order); token_usage = sum across all LLM sources for that task.
    """
    all_ids = sorted(set(list(symbolic_src.keys()) + [tid for src in llm_sources for tid in src]))
    output = []

    for tid in all_ids:
        sym_rec = symbolic_src.get(tid)
        sym_reward = _best_reward(sym_rec) if sym_rec is not None else 0.0

        if sym_reward >= 1.0:
            # Symbolic solver is correct — use it, zero token cost.
            rh = sym_rec.get("reward_history") or [{"iteration": 0, "reward": 1.0}]
            output.append({
                "task_index": tid,
                "solved": True,
                "answer": _answer(sym_rec),
                "best_reward": 1.0,
                "reward_history": rh,
                "token_usage": {"input": 0, "output": 0, "reasoning": 0},
                "cost_summary": sym_rec.get("cost_summary", {}),
            })
        else:
            # Symbolic solver imperfect — pick best LLM candidate.
            llm_candidates = [src[tid] for src in llm_sources if tid in src]
            if not llm_candidates:
                # No LLM source either — fall back to symbolic if available.
                if sym_rec is not None:
                    rh = sym_rec.get("reward_history") or [{"iteration": 0, "reward": sym_reward}]
                    output.append({
                        "task_index": tid,
                        "solved": sym_reward >= 1.0,
                        "answer": _answer(sym_rec),
                        "best_reward": sym_reward,
                        "reward_history": rh,
                        "token_usage": {"input": 0, "output": 0, "reasoning": 0},
                        "cost_summary": sym_rec.get("cost_summary", {}),
                    })
                continue

            best_llm_r = max(_best_reward(c) for c in llm_candidates)
            finalists = [c for c in llm_candidates if _best_reward(c) >= best_llm_r - 1e-9]
            winner = min(finalists, key=lambda c: (_complexity(_answer(c)), llm_candidates.index(c)))

            answer = _answer(winner)
            reward = _best_reward(winner)
            rh = winner.get("reward_history") or [{"iteration": 0, "reward": reward}]

            # Sum token usage across ALL LLM sources for this task.
            combined_tokens: dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}
            for c in llm_candidates:
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
    parser.add_argument("--effi", action="store_true",
                        help="Efficiency mode: use --symbolic as the symbolic solver source; "
                             "LLM token cost is zeroed out for tasks the symbolic solver solves perfectly.")
    parser.add_argument("--symbolic", default="", metavar="FILE",
                        help="(--effi only) Path to symbolic solver JSONL.")
    args = parser.parse_args()

    if args.effi:
        if not args.symbolic:
            parser.error("--effi requires --symbolic FILE")
        symbolic_src = load_source(args.symbolic)
        llm_sources = [load_source(p) for p in args.sources]
        total_tasks = len(set(list(symbolic_src.keys()) + [tid for src in llm_sources for tid in src]))
        print(f"Efficiency ensemble: symbolic={Path(args.symbolic).name} + {len(args.sources)} LLM source(s), {total_tasks} unique tasks")
        print(f"  {Path(args.symbolic).name}: {len(symbolic_src)} tasks (symbolic)")
        for p, src in zip(args.sources, llm_sources):
            print(f"  {Path(p).name}: {len(src)} tasks (LLM)")
        records = ensemble_effi(symbolic_src, llm_sources)
    else:
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
