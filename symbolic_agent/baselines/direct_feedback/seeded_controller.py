"""
Solver-seeded Direct Feedback (additive).

New file. Does not modify DirectFeedbackController or the DF baseline; it
subclasses the controller and adds one method, `solve_seeded`, that starts the
DF refinement loop from the symbolic solver's near-miss program instead of from
scratch. This is the "better use of the solver's output in the fallback" ablation
requested by reviewer orXJ (Concern 3).

Plain DF attempt 0 = raw task prompt. Seeded DF attempt 0 = raw task prompt with a
one-entry history block injected, where that entry is the solver's failed program
plus the verifier feedback for it. Every later attempt behaves exactly like plain
DF (history of prior answers + feedback), so the only change is the starting point.

We reuse the parent's `_call_llm`, `reset_task_log`, token accounting, and the
module-level `_inject_history` / `_build_history_block`, so behavior for existing
code paths is unchanged.
"""

import logging
from typing import Any, Callable, Dict, List

from .controller import DirectFeedbackController, _inject_history

logger = logging.getLogger(__name__)


class SeededDirectFeedbackController(DirectFeedbackController):
    """DirectFeedbackController plus a solver-seeded solve loop."""

    def solve_seeded(
        self,
        task_input: Any,
        reward_fn: Callable,
        entry: Dict,
        seed_answer: Any,
        task_type: str = "symbolic",
    ) -> Dict:
        """
        Run DF starting from the symbolic solver's near-miss.

        Parameters
        ----------
        seed_answer : the solver's failed program (raw string or list of program
            strings). Its verifier reward/feedback is recomputed here so the seed
            entry carries the exact message the model will see.
        """
        self.reset_task_log()
        task_text = self._task_text(task_input)
        max_attempts = self.k

        # Score the solver seed with the verifier so the injected feedback is exact.
        seed_text = self._answer_to_text(seed_answer)
        seed_result = reward_fn(seed_answer if seed_answer not in (None, "") else None,
                                bool(seed_answer), entry)
        seed_reward = float(seed_result.get("value", 0.0))
        seed_feedback = seed_result.get("message", "") or "No feedback available."

        best_reward = seed_reward
        best_answer = seed_answer if seed_answer else None

        # History begins with the solver's attempt, labeled so the model knows it
        # is a strong-but-imperfect candidate to refine rather than one of its own.
        history: List[Dict] = [{
            "attempt": 0,
            "answer_text": f"(symbolic solver candidate) {seed_text[:280]}",
            "feedback": seed_feedback,
            "reward": seed_reward,
        }]
        reward_history: List[Dict] = [{
            "iteration": -1,
            "reward": seed_reward,
            "message": seed_feedback,
            "blame": "seed",
            "solution_summary": seed_text[:300],
            "source": "symbolic_solver_seed",
        }]

        # If the seed already solves it, nothing to do (should not happen: we only
        # call this on solver-failed tasks, but guard anyway).
        if seed_reward < 1.0:
            for i in range(max_attempts):
                prompt = _inject_history(task_text, history)
                raw = self._call_llm(prompt, tag=f"seeded_df_attempt{i}")
                exec_ok = bool(raw)
                res = reward_fn(raw if raw else None, exec_ok, entry)
                rv = float(res.get("value", 0.0))
                fb = res.get("message", "")

                if rv > best_reward:
                    best_reward = rv
                    best_answer = raw

                summary = raw[:300] if raw else ""
                history.append({
                    "attempt": i + 1,
                    "answer_text": summary,
                    "feedback": fb or "No feedback available.",
                    "reward": rv,
                })
                reward_history.append({
                    "iteration": i,
                    "reward": rv,
                    "message": fb,
                    "blame": "partial" if rv < 1.0 else "none",
                    "solution_summary": summary,
                })
                logger.info("seeded_df attempt %d: reward=%.3f", i + 1, rv)
                if rv >= 1.0:
                    break

        solved = best_reward >= 1.0
        return {
            "solved": solved,
            "task_type": task_type,
            "answer": best_answer,
            "best_reward": best_reward,
            "seed_reward": seed_reward,
            "reward_history": reward_history,
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {
                "k": max_attempts,
                # attempts excludes the seed entry (iteration -1)
                "actual_attempts": len([r for r in reward_history if r["iteration"] >= 0]),
                "max_tokens_per_attempt": self.max_tokens,
                "seeded": True,
            },
        }

    @staticmethod
    def _answer_to_text(ans: Any) -> str:
        if ans is None:
            return ""
        if isinstance(ans, list):
            return str(ans)
        return str(ans)
