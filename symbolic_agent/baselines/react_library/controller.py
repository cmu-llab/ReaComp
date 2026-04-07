"""
ReAct + Library Controller.

Stateless per-step design: each LLM call is a single-turn message
containing only the current task, top-K retrieved library functions,
the most recent execution output, and the most recent verifier feedback.
No prior trajectory is accumulated — the model's own chain-of-thought
(reasoning_content) serves as implicit working memory.

Three actions the agent may take per step:
  execute      — run Python; library functions are pre-loaded in scope.
  add_function — register a new reusable helper in the shared library.
  submit       — propose a final answer; triggers immediate reward eval.

The reward-iteration loop is folded into the step loop: each 'submit'
action calls the reward function immediately and feeds the score +
message back as verifier feedback in the next step's prompt.  The loop
exits when reward >= 1.0 or max_steps is exhausted.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .executor import run_code_with_library
from .library import ReactLibrary
from .prompts import _SYSTEM, build_prompt

logger = logging.getLogger(__name__)


class ReActLibraryController:
    """
    ReAct agent with a shared, growing Python function library.

    Parameters
    ----------
    api_key : str, optional
    model : str
    base_url : str, optional
        OpenAI-compatible base URL for vLLM.
    debug_dir : str, optional
    lib_k : int
        Number of library functions to retrieve per step (default: 5).
    max_steps : int
        Maximum total actions (execute + add_function + submit) per task (default: 10).
    max_tokens : int
        Max tokens per LLM call (default: 4096).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
        lib_k: int = 5,
        max_steps: int = 10,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.lib_k = lib_k
        self.max_steps = max_steps
        self.max_tokens = max_tokens

        # Token usage tracking
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
            logger.info("Debug logs → %s", self._debug_dir)

        if base_url:
            import openai
            self._backend = "openai"
            self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        else:
            import anthropic
            self._backend = "anthropic"
            self._client = anthropic.Anthropic(api_key=api_key)

        self.library = ReactLibrary()

    # ------------------------------------------------------------------
    # LLM call — single-turn, fresh context each step
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, tag: str = "") -> Dict:
        """
        Single-turn LLM call.  For vLLM the system prompt is prepended to
        the user message (gpt-oss-120b's chat template rejects a separate
        system role and returns 400).  Returns a parsed JSON dict.
        """
        import json, re, time

        for attempt in range(3):
            try:
                if self._backend == "anthropic":
                    resp = self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=_SYSTEM,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = next(
                        (b.text for b in resp.content if getattr(b, "type", None) == "text"), ""
                    )
                    usage = {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "reasoning_tokens": 0,
                    }
                else:
                    # gpt-oss-120b uses the Harmony format via vLLM.
                    # - Use "developer" role for the system prompt (the chat
                    #   template maps "developer"/"system" to the developer block).
                    #   Injecting system content into the user message caused the
                    #   "Expected 2 output messages, got N" error.
                    # - No response_format — model outputs 2-part Harmony response
                    #   (analysis CoT + final); json_object mode conflicts.
                    # - No temperature/top_p — rejected by the vLLM endpoint.
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "developer", "content": _SYSTEM},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    msg = resp.choices[0].message
                    raw = msg.content or ""
                    u = getattr(resp, "usage", None)
                    details = getattr(u, "completion_tokens_details", None)
                    usage = {
                        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                        "reasoning_tokens": (
                            getattr(details, "reasoning_tokens", 0) or 0 if details else 0
                        ),
                    }

                # Accumulate tokens
                for key, sess_key in [
                    ("input_tokens", "input"),
                    ("output_tokens", "output"),
                    ("reasoning_tokens", "reasoning"),
                ]:
                    v = usage.get(key, 0) or 0
                    self._task_tokens[sess_key] += v
                    self._session_tokens[sess_key] += v

                # Parse JSON (strip markdown fences if the model added them)
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
                    logger.warning(
                        "react_library: JSON parse error (tag=%s): %s", tag, raw[:200]
                    )
                    result = {}

                self._task_log.append({
                    "tag": tag,
                    "model": self.model,
                    "request": {"prompt": prompt, "max_tokens": self.max_tokens},
                    "response": {"content": raw, "usage": usage},
                    "parsed_result": result,
                    "token_usage": usage,
                })
                if self._debug_dir:
                    self._write_debug(tag, prompt, raw, usage, result)
                return result

            except Exception as exc:
                if getattr(exc, "status_code", None) == 400:
                    raise
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        "react_library LLM call failed (attempt %d/3): %s. Retry in %ds.",
                        attempt + 1, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error("react_library LLM call failed after 3 attempts: %s", exc)
                    return {}

    def _write_debug(self, tag: str, prompt: str, raw: str, usage: Dict, parsed: Dict) -> None:
        import json
        from datetime import datetime, timezone
        self._call_counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        path = os.path.join(self._debug_dir, f"{self._call_counter:04d}_{tag}_{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"tag": tag, "prompt": prompt, "raw": raw, "usage": usage, "parsed": parsed},
                    f, indent=2, default=str,
                )
        except Exception as exc:
            logger.warning("Could not write debug log %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Task log helpers
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
    # Core solve loop
    # ------------------------------------------------------------------

    def _run_react_loop(
        self,
        task_description: str,
        reward_fn: Callable,
        entry: Dict,
        max_steps: int,
        max_submits: int,
    ) -> Dict:
        """
        Run the ReAct loop for one task.

        Each step is a fresh single-turn LLM call.  The last execution
        output and latest verifier feedback are passed forward as plain
        text — no conversation history is accumulated in the prompt.

        Returns a dict: {answer, best_reward, reward_history, library_additions, steps_taken}.
        """
        last_result: Optional[str] = None      # stdout/stderr from last execute
        verifier_feedback: Optional[str] = None  # score + message from last submit
        best_reward = 0.0
        best_answer = None
        submit_count = 0
        reward_history: List[Dict] = []
        library_additions: List[str] = []

        for step in range(max_steps):
            lib_fns = self.library.retrieve(task_description, k=self.lib_k)
            prompt = build_prompt(task_description, lib_fns, last_result, verifier_feedback)
            result = self._call_llm(prompt, tag=f"step{step}")

            action = result.get("action", "")

            # ---- execute ----
            if action == "execute":
                code = result.get("code", "")
                ok, out, err = run_code_with_library(code, self.library.namespace())
                last_result = (out[:800] if ok else f"ERROR: {err[:400]}") or "(no output)"
                verifier_feedback = None
                logger.info("react_library step %d: execute → %s", step, last_result[:80])

            # ---- add_function ----
            elif action == "add_function":
                name = result.get("name", "").strip()
                desc = result.get("description", "").strip()
                code = result.get("code", "").strip()
                if name and code:
                    added = self.library.add(name, desc, code)
                    last_result = f"Function '{name}' {'added' if added else 'updated'} in library."
                    library_additions.append(name)
                    logger.info("react_library step %d: add_function '%s'", step, name)
                else:
                    last_result = "ERROR: add_function requires 'name' and 'code' fields."
                    logger.warning("react_library step %d: add_function missing fields", step)
                verifier_feedback = None

            # ---- submit ----
            elif action == "submit":
                answer = result.get("answer")
                submit_count += 1
                rr = reward_fn(answer, answer is not None, entry)
                rv = float(rr.get("value", 0.0))
                rm = rr.get("message", "")

                if rv > best_reward:
                    best_reward = rv
                    best_answer = answer

                reward_history.append({
                    "step": step,
                    "reward": rv,
                    "message": rm,
                    "blame": "react_library",
                    "solution_summary": str(answer)[:200] if answer is not None else "",
                })
                verifier_feedback = f"score={rv:.2f}\n{rm}"
                last_result = None
                logger.info(
                    "react_library step %d: submit → reward=%.3f  %s", step, rv, rm[:80]
                )

                if rv >= 1.0 or submit_count >= max_submits:
                    break

            # ---- unknown ----
            else:
                if result.get("answer") is not None:
                    answer = result.get("answer")
                    rr = reward_fn(answer, True, entry)
                    rv = float(rr.get("value", 0.0))
                    if rv > best_reward:
                        best_reward = rv
                        best_answer = answer
                    reward_history.append({
                        "step": step,
                        "reward": rv,
                        "message": rr.get("message", ""),
                        "blame": "react_library",
                        "solution_summary": str(answer)[:200],
                    })
                logger.warning("react_library step %d: unknown action %r", step, action)
                break

        return {
            "answer": best_answer,
            "best_reward": best_reward,
            "reward_history": reward_history,
            "library_additions": library_additions,
            "steps_taken": step + 1,
        }

    # ------------------------------------------------------------------
    # Public API
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
        Solve a task with verifier feedback.

        max_reward_iters caps the number of 'submit' attempts within the
        step loop.  Library functions added during this task persist and
        are available to all subsequent tasks.
        """
        self.reset_task_log()

        if isinstance(task_input, dict):
            task_description = (
                task_input.get("question")
                or task_input.get("prompt")
                or task_input.get("task")
                or str(task_input)
            )
        else:
            task_description = str(task_input)

        loop_result = self._run_react_loop(
            task_description,
            reward_fn=reward_fn,
            entry=entry,
            max_steps=self.max_steps,
            max_submits=max_reward_iters,
        )

        best_reward = loop_result["best_reward"]
        best_answer = loop_result["answer"]
        reward_history = loop_result["reward_history"]
        library_additions = loop_result["library_additions"]
        solved = best_reward >= 1.0

        return {
            "solved": solved,
            "task_type": task_type,
            "original_prompt": task_description,
            "answer": best_answer,
            "best_reward": best_reward,
            "final_reward": reward_history[-1] if reward_history else {},
            "reward_history": reward_history,
            "trace": [{"agent": "react_library", "steps_taken": loop_result["steps_taken"]}],
            "final_output": {
                "answer": str(best_answer) if best_answer is not None else "",
                "explanation": (
                    f"ReAct+Library: {len(self.library)} fns in library, "
                    f"added {library_additions} this task"
                ),
                "confidence": "high" if solved else "low",
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {
                "library_size": len(self.library),
                "library_additions_this_task": library_additions,
            },
            "library_snapshot": self.library.to_list(),
        }

    def solve(self, task_input: Any, task_type: str = "symbolic", budget: float = 15.0) -> Dict:
        """Single-attempt solve (no reward feedback)."""
        self.reset_task_log()
        if isinstance(task_input, dict):
            task_description = (
                task_input.get("question")
                or task_input.get("prompt")
                or task_input.get("task")
                or str(task_input)
            )
        else:
            task_description = str(task_input)

        def _dummy_reward(answer, has_answer, _entry):
            return {"value": 1.0 if has_answer else 0.0, "message": ""}

        loop_result = self._run_react_loop(
            task_description,
            reward_fn=_dummy_reward,
            entry={},
            max_steps=self.max_steps,
            max_submits=1,
        )
        answer = loop_result["answer"]
        solved = answer is not None

        return {
            "solved": solved,
            "task_type": task_type,
            "original_prompt": task_description,
            "answer": answer,
            "best_reward": 1.0 if solved else 0.0,
            "final_reward": {},
            "reward_history": [],
            "trace": [],
            "final_output": {
                "answer": str(answer) if answer is not None else "",
                "explanation": f"ReAct+Library (no reward), library={len(self.library)} fns",
                "confidence": "medium",
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {"library_size": len(self.library)},
            "library_snapshot": self.library.to_list(),
        }

    def projected_budget(self, max_reward_iters: int = 3) -> Dict:
        """Estimate the maximum token spend for one task."""
        total = self.max_steps * self.max_tokens
        return {
            "max_steps": self.max_steps,
            "max_submits": max_reward_iters,
            "max_tokens": self.max_tokens,
            "projected_max_tokens": total,
            "formula": f"{self.max_steps} steps × {self.max_tokens} tokens = {total}",
        }

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {"library": self.library.to_list()}

    def from_dict(self, data: Dict) -> None:
        self.library = ReactLibrary.from_list(data.get("library", []))
