"""
Best-of-K sampling baseline — direct-feedback style.

Makes K independent single-shot attempts per task.  Each attempt sends the
raw task prompt verbatim with no system message and no JSON wrapping,
identical to the direct_feedback baseline.  The reward function receives the
raw model output and parses it (fenced block, plain list, etc.) itself.

The moment any attempt achieves reward == 1.0 the remaining attempts for that
task are skipped — the pool is exhausted early.

Use --bok-k to control K; --max-tokens controls the per-attempt token budget.
"""

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BestOfKController:
    """
    Best-of-K sampling baseline.

    Parameters
    ----------
    api_key : str, optional
    model : str
    base_url : str, optional
        OpenAI-compatible base URL for vLLM.
    debug_dir : str, optional
    k : int
        Number of independent attempts per task.
    max_tokens : int
        Max tokens per attempt.
    temperature : float
        Sampling temperature (>0 for diversity across K samples).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
        k: int = 5,
        max_tokens: int = 4096,
        temperature: float = 0.8,
    ):
        self.model = model
        self.k = k
        self.max_tokens = max_tokens
        self.temperature = temperature

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
    # LLM call — single-turn, no system prompt, returns raw text
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str, tag: str = "") -> str:
        """
        Single-turn call.  No system prompt — task prompt sent as-is.
        For gpt-oss-120b: only a user message (no developer message) to avoid
        Harmony segment splitting ("Expected 2 output messages, got N").
        Returns raw response text (empty string on failure).
        """
        kwargs: Dict = {"max_tokens": self.max_tokens}
        # Only pass temperature for non-gpt-oss backends (gpt-oss rejects it)
        if self._backend != "gpt_oss":
            kwargs["temperature"] = self.temperature

        for attempt in range(3):
            try:
                if self._backend == "anthropic":
                    resp = self._client.messages.create(
                        model=self.model,
                        messages=[{"role": "user", "content": user_message}],
                        **kwargs,
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
                else:
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": user_message}],
                        **kwargs,
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

                log_entry = {
                    "tag": tag,
                    "model": self.model,
                    "request": {"user_message": user_message, "max_tokens": self.max_tokens},
                    "response": {"content": raw, "reasoning_content": cot, "usage": usage},
                    "token_usage": usage,
                }
                self._task_log.append(log_entry)
                if self._debug_dir:
                    self._write_debug(tag, user_message, raw, cot, usage)
                return raw

            except Exception as exc:
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        "best_of_k LLM call failed (attempt %d/3, tag=%s): %s. Retry in %ds.",
                        attempt + 1, tag, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("best_of_k LLM call failed (tag=%s): %s", tag, exc)
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
    # Core solve
    # ------------------------------------------------------------------

    def solve_with_reward(
        self,
        task_input: Any,
        task_type: str,
        budget: float,
        reward_fn: Callable,
        entry: Dict,
        max_reward_iters: int = 3,  # ignored — Best-of-K uses K attempts instead
    ) -> Dict:
        """
        Make up to K independent attempts, returning the best by reward.
        Stops immediately when any attempt scores reward >= 1.0.
        """
        self.reset_task_log()
        task_text = self._task_text(task_input)

        best_reward = 0.0
        best_answer = None
        reward_history: List[Dict] = []

        for i in range(self.k):
            logger.info("best_of_k: attempt %d/%d", i + 1, self.k)
            raw = self._call_llm(task_text, tag=f"bok_attempt{i}")
            exec_ok = bool(raw)

            reward_result = reward_fn(raw if raw else None, exec_ok, entry)
            reward_value = float(reward_result.get("value", 0.0))

            if reward_value > best_reward:
                best_reward = reward_value
                best_answer = raw

            reward_history.append({
                "iteration": i,
                "reward": reward_value,
                "message": reward_result.get("message", ""),
                "blame": "partial" if reward_value < 1.0 else "none",
                "solution_summary": raw[:200] if raw else "",
            })
            logger.info("best_of_k attempt %d: reward=%.3f", i + 1, reward_value)

            if reward_value >= 1.0:
                logger.info("best_of_k: perfect reward on attempt %d, stopping.", i + 1)
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
                {"agent": "best_of_k", "attempt": h["iteration"], "answer": h["solution_summary"]}
                for h in reward_history
            ],
            "final_output": {
                "answer": str(best_answer) if best_answer is not None else "",
                "explanation": f"Best-of-{self.k} sampling (K={self.k}, max_tokens={self.max_tokens})",
                "confidence": "high" if solved else "low",
                "execution_result": best_answer,
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {
                "k": self.k,
                "actual_attempts": len(reward_history),
                "max_tokens_per_attempt": self.max_tokens,
            },
            "library_snapshot": [],
        }

    def solve(self, task_input: Any, task_type: str = "symbolic", budget: float = 15.0) -> Dict:
        """K independent attempts with no reward signal, return first non-empty response."""
        self.reset_task_log()
        task_text = self._task_text(task_input)
        best_raw = None
        for i in range(self.k):
            raw = self._call_llm(task_text, tag=f"bok_attempt{i}")
            if raw and best_raw is None:
                best_raw = raw
                break
        solved = bool(best_raw)
        return {
            "solved": solved,
            "task_type": task_type,
            "original_prompt": task_text[:200],
            "answer": best_raw,
            "best_reward": 1.0 if solved else 0.0,
            "final_output": {"answer": best_raw or "", "confidence": "medium" if solved else "low"},
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {"k": self.k, "max_tokens_per_attempt": self.max_tokens},
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
