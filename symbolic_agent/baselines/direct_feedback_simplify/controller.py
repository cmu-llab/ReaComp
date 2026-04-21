"""
Direct-feedback + simplification baseline.

Two-phase variant of the direct_feedback baseline with a single shared attempt
budget:

  Phase 1 — Correctness: identical to DirectFeedbackController.  Consumes
    attempts one by one until reward == 1.0 or the full budget is exhausted.
    Stops as soon as a correct solution is found — all remaining attempts are
    preserved for Phase 2.

  Phase 2 — Simplification: only entered if Phase 1 found a correct solution.
    Uses all remaining attempts (budget − attempts_used_in_phase_1) to search
    for a correct solution with lower cascade complexity.  Each attempt receives
    the current best correct answer plus a simplification-focused prompt.  If a
    simpler correct answer is found it becomes the new best; otherwise the
    current best is retained.  At the end the lowest-complexity correct answer
    found across both phases is returned.

  If Phase 1 exhausts the budget without finding a correct solution, Phase 2 is
  skipped and the best partial answer is returned (same as direct_feedback).

Budget semantics
----------------
The total budget is controlled by a single parameter: `max_reward_iters` as
passed by the harness (--max-reward-iters flag, default 3).  There are no
separate per-phase caps — correctness gets the full budget first, and
simplification gets whatever is left.

Flags (via main.py):
  --max-reward-iters N   total attempt budget (default 3; use 32 for serious runs)
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from symbolic_agent.baselines.direct_feedback.controller import DirectFeedbackController

logger = logging.getLogger(__name__)

_PROGRAM_SEQUENCE_MARKER = "### Program Sequence"

_REPLACE_RE = re.compile(
    r"""replace\(\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\)""",
    re.IGNORECASE,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_complexity(answer: Any) -> Optional[int]:
    """Return cascade complexity of answer, or None if no replace() calls found."""
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
            raw = "\n".join(str(x) for x in parsed) if isinstance(parsed, list) else str(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        return None
    programs = _REPLACE_RE.findall(raw)
    if not programs:
        return None
    return sum(len(pred) + len(transform) for pred, transform in programs)


def _build_history_block(history: List[Dict]) -> str:
    lines = ["### Previous attempts"]
    for h in history:
        lines.append(f"\nAttempt {h['attempt']} (reward: {h['reward']:.3f})")
        if h.get("answer_text"):
            lines.append(f"Your answer: {h['answer_text']}")
        lines.append(f"Verifier feedback: {h['feedback']}")
    lines.append("\nUsing the feedback above, produce a corrected solution.")
    return "\n".join(lines)


def _inject_history(task_text: str, history: List[Dict]) -> str:
    block = _build_history_block(history)
    idx = task_text.rfind(_PROGRAM_SEQUENCE_MARKER)
    if idx != -1:
        before = task_text[:idx].rstrip()
        after = task_text[idx:]
        return f"{before}\n\n{block}\n\n{after}"
    return f"{task_text}\n\n{block}"


def _build_simplify_block(best_answer: str, best_complexity: Optional[int], simplify_history: List[Dict]) -> str:
    lines = [
        "### Simplification mode",
        "",
        "You have already found a correct solution. Your goal now is to find an "
        "equally correct but simpler solution — one with fewer or shorter replace() "
        "programs (lower total cascade complexity).",
        "",
        f"Current best correct solution (complexity={best_complexity}):",
        best_answer,
    ]
    if simplify_history:
        lines.append("\n### Previous simplification attempts")
        for h in simplify_history:
            lines.append(f"\nSimplify attempt {h['attempt']} (complexity={h['complexity']})")
            if h.get("answer_text"):
                lines.append(f"Your answer: {h['answer_text']}")
            lines.append(f"Verifier feedback: {h['feedback']}")
    lines.append(
        "\nProduce a correct solution that is simpler than the current best: "
        "use shorter predicates/transforms and/or a shorter cascade (fewer programs)."
    )
    return "\n".join(lines)


def _inject_simplify(task_text: str, best_answer: str, best_complexity: Optional[int], simplify_history: List[Dict]) -> str:
    block = _build_simplify_block(best_answer, best_complexity, simplify_history)
    idx = task_text.rfind(_PROGRAM_SEQUENCE_MARKER)
    if idx != -1:
        before = task_text[:idx].rstrip()
        after = task_text[idx:]
        return f"{before}\n\n{block}\n\n{after}"
    return f"{task_text}\n\n{block}"


# ── controller ────────────────────────────────────────────────────────────────

class DirectFeedbackSimplifyController(DirectFeedbackController):
    """
    Two-phase direct-feedback controller with a single shared attempt budget.

    All attempts go to Phase 1 (correctness) first.  The moment a correct
    solution is found, any remaining attempts are used for Phase 2
    (simplification).  If the budget is exhausted during Phase 1 without
    finding a correct solution, Phase 2 is skipped.

    The total budget is `max_reward_iters` passed by the harness — there are
    no separate per-phase caps.  Pass a large --max-reward-iters (e.g. 32) to
    give the model a meaningful simplification budget.

    Inherits all LLM machinery from DirectFeedbackController.
    """

    def solve_with_reward(
        self,
        task_input: Any,
        task_type: str,
        budget: float,
        reward_fn: Callable,
        entry: Dict,
        max_reward_iters: int = 3,
    ) -> Dict:
        self.reset_task_log()
        task_text = self._task_text(task_input)
        total_budget = self.k

        best_reward = 0.0
        best_answer: Optional[str] = None
        best_complexity: Optional[int] = None
        reward_history: List[Dict] = []
        history: List[Dict] = []
        iter_idx = 0

        # ── Phase 1: Correctness ──────────────────────────────────────────
        # Use the full budget; exit early the moment reward == 1.0.
        for i in range(total_budget):
            logger.info("dfs correctness attempt %d/%d", i + 1, total_budget)
            prompt = task_text if i == 0 else _inject_history(task_text, history)
            raw = self._call_llm(prompt, tag=f"dfs_correct{i}")
            exec_ok = bool(raw)

            reward_result = reward_fn(raw if raw else None, exec_ok, entry)
            reward_value = float(reward_result.get("value", 0.0))
            feedback_msg = reward_result.get("message", "")

            if reward_value > best_reward:
                best_reward = reward_value
                best_answer = raw
                best_complexity = _parse_complexity(raw)

            answer_summary = raw[:300] if raw else ""
            history.append({
                "attempt": i + 1,
                "answer_text": answer_summary,
                "feedback": feedback_msg or "No feedback available.",
                "reward": reward_value,
            })
            reward_history.append({
                "iteration": iter_idx,
                "reward": reward_value,
                "message": feedback_msg,
                "blame": "partial" if reward_value < 1.0 else "none",
                "solution_summary": answer_summary,
                "phase": "correct",
            })
            iter_idx += 1

            logger.info("dfs correctness %d: reward=%.3f", i + 1, reward_value)

            if reward_value >= 1.0:
                logger.info(
                    "dfs: correct solution found on attempt %d, %d attempts remaining for simplification",
                    i + 1, total_budget - iter_idx,
                )
                break

        # ── Phase 2: Simplification (only if Phase 1 found a correct solution) ──
        if best_reward >= 1.0:
            simplify_history: List[Dict] = []
            simplify_budget = total_budget - iter_idx

            for j in range(simplify_budget):
                logger.info(
                    "dfs simplify attempt %d/%d  (best complexity=%s)",
                    j + 1, simplify_budget, best_complexity,
                )
                prompt = _inject_simplify(task_text, best_answer, best_complexity, simplify_history)
                raw = self._call_llm(prompt, tag=f"dfs_simplify{j}")
                exec_ok = bool(raw)

                reward_result = reward_fn(raw if raw else None, exec_ok, entry)
                reward_value = float(reward_result.get("value", 0.0))
                feedback_msg = reward_result.get("message", "")
                complexity = _parse_complexity(raw)

                answer_summary = raw[:300] if raw else ""
                simplify_history.append({
                    "attempt": j + 1,
                    "answer_text": answer_summary,
                    "complexity": complexity if complexity is not None else -1,
                    "feedback": ("Correct!" if reward_value >= 1.0 else feedback_msg) or "No feedback.",
                })

                improved = (
                    reward_value >= 1.0
                    and complexity is not None
                    and (best_complexity is None or complexity < best_complexity)
                )
                if improved:
                    logger.info(
                        "dfs simplify %d: simpler correct solution found (complexity %s → %s)",
                        j + 1, best_complexity, complexity,
                    )
                    best_answer = raw
                    best_complexity = complexity

                reward_history.append({
                    "iteration": iter_idx,
                    "reward": reward_value,
                    "message": feedback_msg,
                    "blame": (
                        "simplify" if improved
                        else "simplify_fail" if reward_value < 1.0
                        else "simplify_no_gain"
                    ),
                    "solution_summary": answer_summary,
                    "phase": "simplify",
                    "complexity": complexity,
                })
                iter_idx += 1

                logger.info(
                    "dfs simplify %d: reward=%.3f  complexity=%s",
                    j + 1, reward_value, complexity,
                )

        solved = best_reward >= 1.0
        return {
            "solved": solved,
            "task_type": task_type,
            "original_prompt": task_text[:200],
            "answer": best_answer,
            "best_reward": best_reward,
            "final_reward": reward_history[-1] if reward_history else {},
            "reward_history": reward_history,
            "trace": [
                {
                    "agent": "direct_feedback_simplify",
                    "attempt": h["iteration"],
                    "answer": h["solution_summary"],
                    "phase": h.get("phase", "correct"),
                }
                for h in reward_history
            ],
            "final_output": {
                "answer": str(best_answer) if best_answer is not None else "",
                "explanation": (
                    f"DirectFeedbackSimplify (budget={total_budget}, max_tokens={self.max_tokens})"
                ),
                "confidence": "high" if solved else "low",
                "execution_result": best_answer,
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {
                "total_budget": total_budget,
                "actual_attempts": len(reward_history),
                "max_tokens_per_attempt": self.max_tokens,
            },
            "library_snapshot": [],
        }

    def projected_budget(self, max_reward_iters: int = 3) -> Dict:
        return {
            "total_budget": self.k,
            "max_tokens_per_attempt": self.max_tokens,
            "projected_max_tokens": self.k * self.max_tokens,
            "formula": f"{self.k} attempts × {self.max_tokens} tokens = {self.k * self.max_tokens}",
            "note": "All attempts go to correctness first; remaining go to simplification.",
        }
