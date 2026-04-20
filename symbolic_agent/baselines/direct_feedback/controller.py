"""
Direct-feedback baseline.

Each task attempt is a single LLM call (no multi-turn conversation).
On the first attempt the prompt is just the raw task.
On subsequent attempts the prompt is the original task + a history block showing
each prior attempt's solution and the verifier feedback for it.

This sidesteps all Harmony / multi-turn issues because every call is stateless:
system + one user message in, one assistant message out.

Flags:
  --df-k N   maximum number of attempts (default 3)
"""

import io
import json
import logging
import os
import re
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_JSON_INSTRUCTION = (
    "\n\nRespond with a single valid JSON object and nothing else. "
    "No markdown fences, no prose before or after the JSON."
)

_SYSTEM = """\
You are a Python programming assistant. Solve the given task.

Respond with a JSON object:
{
  "reasoning": "brief explanation of your approach",
  "code": "def solve(...):\\n    ...\\n\\nresult = solve(...)\\nprint(result)",
  "answer": "direct answer if no code needed (leave empty string if code is provided)"
}

Rules:
- For algorithmic / transformation tasks write a Python function, call it, print the result.
- For pure Q&A tasks that need no code set "answer" to the exact value and "code" to "".
- Print the result so it can be captured; do not only return it.
- The "answer" field (if set) must be the exact answer, not a sentence.
"""


def _run_code(code: str) -> tuple:
    """Execute code in an isolated namespace. Returns (ok, result_str, error_str)."""
    namespace: dict = {}
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        exec(compile(code, "<direct_feedback>", "exec"), namespace)
        printed = buf.getvalue().strip()
        return True, printed if printed else None, ""
    except Exception:
        return False, None, traceback.format_exc()
    finally:
        sys.stdout = old_stdout


def _build_history_block(history: List[Dict]) -> str:
    """
    Format prior attempts into a readable history block.

    Each entry in history is:
      {"attempt": int, "solution": str, "feedback": str, "reward": float}
    """
    lines = ["--- Previous attempts ---"]
    for h in history:
        lines.append(f"\nAttempt {h['attempt']}:")
        lines.append(f"  Solution: {h['solution']}")
        lines.append(f"  Verifier feedback: {h['feedback']}")
        lines.append(f"  Reward: {h['reward']:.3f}")
    lines.append("--- End of history ---")
    lines.append(
        "\nUsing the feedback above, produce a corrected solution."
    )
    return "\n".join(lines)


class DirectFeedbackController:
    """
    Direct-feedback baseline controller.

    Parameters
    ----------
    api_key : str, optional
    model : str
    base_url : str, optional
        OpenAI-compatible endpoint (e.g. vLLM).
    debug_dir : str, optional
    k : int
        Maximum attempts per task (default 3).
    max_tokens : int
        Token budget per call.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
        k: int = 3,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.k = k
        self.max_tokens = max_tokens

        self._session_tokens: Dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}
        self._task_tokens: Dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}
        self._task_log: List[Dict] = []
        self._debug_dir: Optional[str] = None
        self._call_counter = 0

        if debug_dir:
            from datetime import datetime, timezone
            run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            self._debug_dir = os.path.join(debug_dir, f"run_{run_ts}")
            os.makedirs(self._debug_dir, exist_ok=True)

        if base_url:
            import openai
            self._backend = "gpt_oss" if "gpt-oss" in model else "openai"
            self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        else:
            import anthropic
            self._backend = "anthropic"
            self._client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # LLM call — always single-turn
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str, tag: str = "") -> Dict:
        """One single-turn call. Returns parsed JSON dict (or {} on failure)."""
        system = _SYSTEM + _JSON_INSTRUCTION

        for attempt in range(3):
            try:
                if self._backend == "anthropic":
                    resp = self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": user_message}],
                    )
                    raw = next(
                        (b.text for b in resp.content if getattr(b, "type", None) == "text"), ""
                    )
                    usage = {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "reasoning_tokens": 0,
                    }
                    cot = ""
                elif self._backend == "gpt_oss":
                    # gpt-oss-120b: use developer role, no response_format, no temperature.
                    # Each call is a fresh single-turn — history is encoded in the user message.
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "developer", "content": system},
                            {"role": "user", "content": user_message},
                        ],
                    )
                    msg = resp.choices[0].message
                    raw = msg.content or ""
                    cot = getattr(msg, "reasoning_content", "") or ""
                    u = getattr(resp, "usage", None)
                    details = getattr(u, "completion_tokens_details", None)
                    usage = {
                        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                        "reasoning_tokens": (
                            getattr(details, "reasoning_tokens", 0) or 0
                            if details else 0
                        ),
                    }
                else:
                    # Generic OpenAI-compatible (not gpt-oss), use json_object mode
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_message},
                        ],
                        response_format={"type": "json_object"},
                    )
                    msg = resp.choices[0].message
                    raw = msg.content or ""
                    cot = getattr(msg, "reasoning_content", "") or ""
                    u = getattr(resp, "usage", None)
                    details = getattr(u, "completion_tokens_details", None)
                    usage = {
                        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                        "reasoning_tokens": (
                            getattr(details, "reasoning_tokens", 0) or 0
                            if details else 0
                        ),
                    }

                for key, sess_key in [
                    ("input_tokens", "input"),
                    ("output_tokens", "output"),
                    ("reasoning_tokens", "reasoning"),
                ]:
                    v = usage.get(key, 0) or 0
                    self._task_tokens[sess_key] += v
                    self._session_tokens[sess_key] += v

                # Strip markdown fences if the model wrapped its JSON anyway
                text = raw.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)
                    text = text.strip()

                try:
                    result = json.loads(text)
                    if not isinstance(result, dict):
                        result = {}
                except json.JSONDecodeError:
                    logger.warning("direct_feedback: JSON parse error (tag=%s): %s", tag, raw[:200])
                    result = {}

                log_entry = {
                    "tag": tag,
                    "model": self.model,
                    "request": {"user_message": user_message, "max_tokens": self.max_tokens},
                    "response": {"content": raw, "reasoning_content": cot, "usage": usage},
                    "parsed_result": result,
                    "token_usage": usage,
                }
                self._task_log.append(log_entry)
                if self._debug_dir:
                    self._write_debug(tag, user_message, raw, cot, usage, result)
                return result

            except Exception as exc:
                if getattr(exc, "status_code", None) == 400:
                    raise
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        "direct_feedback LLM call failed (attempt %d/3): %s. Retry in %ds.",
                        attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("direct_feedback LLM call failed: %s", exc)
                    return {}

    def _write_debug(self, tag, user_message, raw, cot, usage, parsed):
        self._call_counter += 1
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        path = os.path.join(self._debug_dir, f"{self._call_counter:04d}_{tag}_{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "tag": tag,
                        "user_message": user_message,
                        "raw": raw,
                        "reasoning_content": cot,
                        "usage": usage,
                        "parsed": parsed,
                    },
                    f, indent=2, default=str,
                )
        except Exception as exc:
            logger.warning("Could not write debug log %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Task log helpers (standard interface)
    # ------------------------------------------------------------------

    def reset_task_log(self) -> None:
        self._task_log = []
        self._task_tokens = {"input": 0, "output": 0, "reasoning": 0}

    def get_task_log(self) -> List[Dict]:
        return list(self._task_log)

    def get_task_token_usage(self) -> Dict[str, int]:
        return dict(self._task_tokens)

    def get_session_token_usage(self) -> Dict[str, int]:
        return dict(self._session_tokens)

    def restore_session_tokens(self, d: Dict[str, int]) -> None:
        for k in ("input", "output", "reasoning"):
            self._session_tokens[k] = int(d.get(k, 0))

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _task_text(self, task_input: Any) -> str:
        if isinstance(task_input, dict):
            return (
                task_input.get("question")
                or task_input.get("prompt")
                or task_input.get("task")
                or str(task_input)
            )
        return str(task_input)

    def _build_first_prompt(self, task_text: str) -> str:
        return f"Solve this task:\n\n{task_text}"

    def _build_retry_prompt(self, task_text: str, history: List[Dict]) -> str:
        return (
            f"Solve this task:\n\n{task_text}\n\n"
            + _build_history_block(history)  # module-level helper
        )

    # ------------------------------------------------------------------
    # Core solve logic
    # ------------------------------------------------------------------

    def _run_attempt(self, prompt: str, attempt_idx: int) -> Dict:
        """Run one LLM call and execute any returned code."""
        result = self._call_llm(prompt, tag=f"df_attempt{attempt_idx}")
        code = result.get("code", "") or ""
        direct_answer = result.get("answer") or None

        if code:
            ok, exec_result, err = _run_code(code)
            if ok and exec_result is not None:
                return {"answer": exec_result, "code": code, "exec_ok": True, "error": ""}
            elif ok and direct_answer is not None:
                return {"answer": direct_answer, "code": code, "exec_ok": True, "error": ""}
            else:
                return {"answer": direct_answer, "code": code, "exec_ok": False, "error": err}

        return {
            "answer": direct_answer,
            "code": "",
            "exec_ok": direct_answer is not None,
            "error": "",
        }

    def solve_with_reward(
        self,
        task_input: Any,
        task_type: str,
        budget: float,
        reward_fn: Callable,
        entry: Dict,
        max_reward_iters: int = 3,
    ) -> Dict:
        """
        Up to k=min(self.k, max_reward_iters) sequential attempts.

        Attempt 1: raw task prompt only.
        Attempt 2+: task prompt + history of prior attempts and their verifier feedback.
        Stop early if reward == 1.0.
        """
        self.reset_task_log()
        task_text = self._task_text(task_input)
        max_attempts = min(self.k, max_reward_iters)

        best_reward = 0.0
        best_answer = None
        best_code = None
        reward_history: List[Dict] = []
        history: List[Dict] = []  # grows across attempts, fed into retry prompts

        for i in range(max_attempts):
            logger.info("direct_feedback: attempt %d/%d", i + 1, max_attempts)

            if i == 0:
                prompt = self._build_first_prompt(task_text)
            else:
                prompt = self._build_retry_prompt(task_text, history)

            att = self._run_attempt(prompt, i)

            reward_result = reward_fn(att["answer"], att["exec_ok"], entry)
            reward_value = float(reward_result.get("value", 0.0))
            feedback_msg = reward_result.get("message", "")

            if reward_value > best_reward:
                best_reward = reward_value
                best_answer = att["answer"]
                best_code = att["code"]

            solution_summary = str(att["answer"])[:300] if att["answer"] is not None else ""
            if att["error"]:
                # Include execution error in feedback so retry prompt is informative
                feedback_msg = (att["error"][:300] + "\n" + feedback_msg).strip()

            history.append({
                "attempt": i + 1,
                "solution": solution_summary,
                "feedback": feedback_msg or "No feedback available.",
                "reward": reward_value,
            })
            reward_history.append({
                "iteration": i,
                "reward": reward_value,
                "message": feedback_msg,
                "blame": (
                    "execution" if not att["exec_ok"]
                    else ("partial" if reward_value < 1.0 else "none")
                ),
                "solution_summary": solution_summary,
            })

            logger.info("direct_feedback attempt %d: reward=%.3f  %s", i + 1, reward_value, feedback_msg[:80])

            if reward_value >= 1.0:
                logger.info("direct_feedback: perfect reward on attempt %d, stopping.", i + 1)
                break

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
                {"agent": "direct_feedback", "attempt": h["iteration"], "answer": h["solution_summary"]}
                for h in reward_history
            ],
            "final_output": {
                "answer": str(best_answer) if best_answer is not None else "",
                "explanation": f"Direct-feedback (k={max_attempts}, max_tokens={self.max_tokens})",
                "confidence": "high" if solved else "low",
                "execution_result": best_answer,
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {
                "k": max_attempts,
                "actual_attempts": len(reward_history),
                "max_tokens_per_attempt": self.max_tokens,
            },
            "library_snapshot": [],
        }

    def solve(self, task_input: Any, task_type: str = "symbolic", budget: float = 15.0) -> Dict:
        """Single attempt, no reward signal."""
        self.reset_task_log()
        task_text = self._task_text(task_input)
        att = self._run_attempt(self._build_first_prompt(task_text), 0)
        solved = att["exec_ok"] and att["answer"] is not None
        return {
            "solved": solved,
            "task_type": task_type,
            "original_prompt": task_text[:200],
            "answer": att["answer"],
            "best_reward": 1.0 if solved else 0.0,
            "final_output": {
                "answer": str(att["answer"]) if att["answer"] is not None else "",
                "confidence": "medium" if solved else "low",
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {"k": 1, "max_tokens_per_attempt": self.max_tokens},
            "library_snapshot": [],
            "reward_history": [],
        }

    def projected_budget(self, max_reward_iters: int = 3) -> Dict:
        max_attempts = min(self.k, max_reward_iters)
        total = max_attempts * self.max_tokens
        return {
            "k": max_attempts,
            "max_tokens_per_attempt": self.max_tokens,
            "projected_max_tokens": total,
            "formula": f"{max_attempts} attempts × {self.max_tokens} tokens = {total}",
        }
