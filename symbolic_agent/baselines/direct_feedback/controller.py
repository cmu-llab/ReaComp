"""
Direct-feedback baseline.

Each task attempt is a single LLM call (no multi-turn conversation).
On the first attempt the prompt is the raw task exactly as-is.
On subsequent attempts the history of prior solutions + verifier feedback is
injected into the prompt, and the model completes it fresh.

No JSON wrapping. No code execution. The reward function receives the raw model
output and parses it (fenced block, plain list, etc.) itself.

Flags:
  --df-k N   maximum number of attempts (default 3)
"""

import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Marker used by PBEBench prompts — we insert feedback history just before it
# on retry attempts so the model sees: task → feedback → open completion slot.
_PROGRAM_SEQUENCE_MARKER = "### Program Sequence"


def _build_history_block(history: List[Dict]) -> str:
    """
    Format prior attempts into a text block.

    Each entry: {"attempt": int, "answer_text": str, "feedback": str, "reward": float}
    """
    lines = ["### Previous attempts"]
    for h in history:
        lines.append(f"\nAttempt {h['attempt']} (reward: {h['reward']:.3f})")
        if h.get("answer_text"):
            lines.append(f"Your answer: {h['answer_text']}")
        lines.append(f"Verifier feedback: {h['feedback']}")
    lines.append("\nUsing the feedback above, produce a corrected solution.")
    return "\n".join(lines)


def _inject_history(task_text: str, history: List[Dict]) -> str:
    """
    Build a retry prompt.

    If the task ends with a PBEBench-style open completion slot
    (### Program Sequence), insert the history block just before it so the
    model sees: [task] → [feedback] → [open slot to complete].

    Otherwise append the history at the end.
    """
    history_block = _build_history_block(history)
    # Find the last occurrence of the PBEBench completion marker
    idx = task_text.rfind(_PROGRAM_SEQUENCE_MARKER)
    if idx != -1:
        before = task_text[:idx].rstrip()
        after = task_text[idx:]
        return f"{before}\n\n{history_block}\n\n{after}"
    return f"{task_text}\n\n{history_block}"


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
        self._session_tokens_lock = threading.Lock()
        # Per-thread state: each worker gets its own task log and token counters.
        self._local = threading.local()
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
    # LLM call — always single-turn, returns raw text
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str, tag: str = "") -> str:
        """
        Single-turn call. Returns raw response text (empty string on failure).

        No system prompt is injected — the task prompt is passed as-is.
        For gpt-oss-120b: only a user message is sent (no developer message)
        to avoid any Harmony template parsing issues from injected content.
        """
        for attempt in range(3):
            try:
                if self._backend == "anthropic":
                    resp = self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": user_message}],
                    )
                    raw = next(
                        (b.text for b in resp.content if getattr(b, "type", None) == "text"), ""
                    )
                    cot = ""
                    usage = {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "reasoning_tokens": 0,
                    }
                elif self._backend == "gpt_oss":
                    # No developer/system message — task prompt is the full input.
                    # This avoids any injected content triggering Harmony segment
                    # splitting ("Expected 2 output messages, got N").
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": user_message}],
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
                    # Generic OpenAI-compatible endpoint
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[{"role": "user", "content": user_message}],
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

                task_tokens = getattr(self._local, "task_tokens", {"input": 0, "output": 0, "reasoning": 0})
                for key, sess_key in [
                    ("input_tokens", "input"),
                    ("output_tokens", "output"),
                    ("reasoning_tokens", "reasoning"),
                ]:
                    v = usage.get(key, 0) or 0
                    task_tokens[sess_key] += v
                    with self._session_tokens_lock:
                        self._session_tokens[sess_key] += v

                log_entry = {
                    "tag": tag,
                    "model": self.model,
                    "request": {"user_message": user_message, "max_tokens": self.max_tokens},
                    "response": {"content": raw, "reasoning_content": cot, "usage": usage},
                    "token_usage": usage,
                }
                task_log = getattr(self._local, "task_log", [])
                task_log.append(log_entry)
                if self._debug_dir:
                    self._write_debug(tag, user_message, raw, cot, usage)
                return raw

            except Exception as exc:
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        "direct_feedback LLM call failed (attempt %d/3, tag=%s): %s. Retry in %ds.",
                        attempt + 1, tag, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("direct_feedback LLM call failed (tag=%s): %s", tag, exc)
                    return ""

    def _write_debug(self, tag, user_message, raw, cot, usage):
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
                    },
                    f, indent=2, default=str,
                )
        except Exception as exc:
            logger.warning("Could not write debug log %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Task log helpers (standard interface)
    # ------------------------------------------------------------------

    def reset_task_log(self) -> None:
        self._local.task_log = []
        self._local.task_tokens = {"input": 0, "output": 0, "reasoning": 0}

    def get_task_log(self) -> List[Dict]:
        return list(getattr(self._local, "task_log", []))

    def get_task_token_usage(self) -> Dict[str, int]:
        return dict(getattr(self._local, "task_tokens", {"input": 0, "output": 0, "reasoning": 0}))

    def get_session_token_usage(self) -> Dict[str, int]:
        return dict(self._session_tokens)

    def restore_session_tokens(self, d: Dict[str, int]) -> None:
        for k in ("input", "output", "reasoning"):
            self._session_tokens[k] = int(d.get(k, 0))

    # ------------------------------------------------------------------
    # Task text extraction
    # ------------------------------------------------------------------

    def _task_text(self, task_input: Any) -> str:
        if isinstance(task_input, dict):
            return (
                task_input.get("prompt")
                or task_input.get("question")
                or task_input.get("task")
                or str(task_input)
            )
        return str(task_input)

    # ------------------------------------------------------------------
    # Core solve logic
    # ------------------------------------------------------------------

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
        Up to k sequential single-turn attempts (max_reward_iters is ignored).

        Attempt 1: raw task prompt.
        Attempt 2+: history of prior answers + verifier feedback injected into prompt.
        Stops early on reward == 1.0.
        """
        self.reset_task_log()
        task_text = self._task_text(task_input)
        max_attempts = self.k

        best_reward = 0.0
        best_answer = None
        reward_history: List[Dict] = []
        history: List[Dict] = []

        for i in range(max_attempts):
            logger.info("direct_feedback: attempt %d/%d", i + 1, max_attempts)

            prompt = task_text if i == 0 else _inject_history(task_text, history)
            raw = self._call_llm(prompt, tag=f"df_attempt{i}")
            exec_ok = bool(raw)

            reward_result = reward_fn(raw if raw else None, exec_ok, entry)
            reward_value = float(reward_result.get("value", 0.0))
            feedback_msg = reward_result.get("message", "")

            if reward_value > best_reward:
                best_reward = reward_value
                best_answer = raw

            answer_summary = raw[:300] if raw else ""
            history.append({
                "attempt": i + 1,
                "answer_text": answer_summary,
                "feedback": feedback_msg or "No feedback available.",
                "reward": reward_value,
            })
            reward_history.append({
                "iteration": i,
                "reward": reward_value,
                "message": feedback_msg,
                "blame": "partial" if reward_value < 1.0 else "none",
                "solution_summary": answer_summary,
            })

            logger.info(
                "direct_feedback attempt %d: reward=%.3f  %s",
                i + 1, reward_value, feedback_msg[:80],
            )

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
        raw = self._call_llm(task_text, tag="df_attempt0")
        solved = bool(raw)
        return {
            "solved": solved,
            "task_type": task_type,
            "original_prompt": task_text[:200],
            "answer": raw if raw else None,
            "best_reward": 1.0 if solved else 0.0,
            "final_output": {
                "answer": raw or "",
                "confidence": "medium" if solved else "low",
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {"k": 1, "max_tokens_per_attempt": self.max_tokens},
            "library_snapshot": [],
            "reward_history": [],
        }

    def projected_budget(self, max_reward_iters: int = 3) -> Dict:
        total = self.k * self.max_tokens
        return {
            "k": self.k,
            "max_tokens_per_attempt": self.max_tokens,
            "projected_max_tokens": total,
            "formula": f"{self.k} attempts × {self.max_tokens} tokens = {total}",
        }

