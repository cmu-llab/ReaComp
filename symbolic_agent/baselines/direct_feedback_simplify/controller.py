"""
Direct-feedback + simplification baseline.

Two-phase variant of the direct_feedback baseline:

  Phase 1 — Correctness: identical to DirectFeedbackController; runs until
    reward == 1.0 or the correctness budget (k_correct attempts) is exhausted.

  Phase 2 — Simplification: once a perfect solution is found, uses the
    remaining budget (up to k_simplify more attempts) to search for a correct
    solution with lower cascade complexity.  Each attempt receives the current
    best correct answer plus a simplification-focused prompt injection.  If a
    simpler correct answer is found it becomes the new best.  At the end the
    lowest-complexity correct answer is returned regardless of when it was found.

  If Phase 1 never finds a correct solution the result is identical to plain
  DirectFeedbackController (best partial answer, best_reward < 1.0).

Flags:
  --dfs-k-correct K   correctness-phase budget (default 3)
  --dfs-k-simplify K  simplification-phase budget (default 3)
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
    """
    Build the simplification-mode context block.

    Each simplify_history entry: {"attempt": int, "answer_text": str,
                                   "complexity": int, "feedback": str}
    """
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
    Two-phase direct-feedback controller.

    Inherits all LLM machinery from DirectFeedbackController.

    Parameters
    ----------
    k_correct : int
        Max attempts in Phase 1 (correctness).  (default: 3)
    k_simplify : int
        Max attempts in Phase 2 (simplification).  (default: 3)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
        k_correct: int = 3,
        k_simplify: int = 3,
        max_tokens: int = 4096,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=debug_dir,
            k=k_correct + k_simplify,
            max_tokens=max_tokens,
        )
        self.k_correct = k_correct
        self.k_simplify = k_simplify

    def solve_with_reward(
        self,
        task_input: Any,
        task_type: str,
        budget: float,
        reward_fn: Callable,
        entry: Dict,
        max_reward_iters: int = 6,
    ) -> Dict:
        self.reset_task_log()
        task_text = self._task_text(task_input)

        best_reward = 0.0
        best_answer: Optional[str] = None
        best_complexity: Optional[int] = None
        reward_history: List[Dict] = []
        history: List[Dict] = []
        iter_idx = 0

        # ── Phase 1: Correctness ──────────────────────────────────────────
        correct_attempts = min(self.k_correct, max_reward_iters)
        for i in range(correct_attempts):
            logger.info("dfs correctness attempt %d/%d", i + 1, correct_attempts)
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
                logger.info("dfs: perfect reward on correctness attempt %d", i + 1)
                break

        # ── Phase 2: Simplification (only if a correct solution exists) ───
        if best_reward >= 1.0:
            remaining = max_reward_iters - iter_idx
            simplify_attempts = min(self.k_simplify, remaining)
            simplify_history: List[Dict] = []

            for j in range(simplify_attempts):
                logger.info(
                    "dfs simplify attempt %d/%d  (current best complexity=%s)",
                    j + 1, simplify_attempts, best_complexity,
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
                    f"DirectFeedbackSimplify (k_correct={self.k_correct}, "
                    f"k_simplify={self.k_simplify}, max_tokens={self.max_tokens})"
                ),
                "confidence": "high" if solved else "low",
                "execution_result": best_answer,
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {
                "k_correct": self.k_correct,
                "k_simplify": self.k_simplify,
                "actual_attempts": len(reward_history),
                "max_tokens_per_attempt": self.max_tokens,
            },
            "library_snapshot": [],
        }

    def projected_budget(self, max_reward_iters: int = 6) -> Dict:
        correct = min(self.k_correct, max_reward_iters)
        simplify = min(self.k_simplify, max(0, max_reward_iters - correct))
        total = (correct + simplify) * self.max_tokens
        return {
            "k_correct": correct,
            "k_simplify": simplify,
            "max_tokens_per_attempt": self.max_tokens,
            "projected_max_tokens": total,
            "formula": f"({correct}+{simplify}) attempts × {self.max_tokens} tokens = {total}",
        }
