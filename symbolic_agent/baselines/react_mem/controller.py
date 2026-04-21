"""
ReAct + Memory Controller.

Implements a ReAct agent (Yao et al. 2022) with an episodic memory store
that retrieves similar past solutions as in-context examples for new tasks.
Inspired by the memory module in Hypothetical Minds (Cross et al. 2024).

Architecture:
  For each task:
    1. Retrieve top-K similar solved tasks from memory.
    2. Run ReAct loop (Thought → Code → Observation) for up to max_steps.
    3. Evaluate with reward_fn.
    4. Store the best solution in memory for future tasks.

The agent never maintains a shared function library — all generalisation
happens through few-shot retrieval from memory.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .executor import run_code
from .memory import ReActMemory
from .prompts import _SYSTEM, build_followup_prompt, build_initial_prompt

logger = logging.getLogger(__name__)

_JSON_INSTRUCTION = (
    "\n\nRespond with a single valid JSON object and nothing else. "
    "No markdown fences, no prose before or after the JSON."
)


class ReActMemController:
    """
    ReAct agent with episodic memory.

    Parameters
    ----------
    api_key : str, optional
    model : str
    base_url : str, optional
        OpenAI-compatible base URL for vLLM.
    debug_dir : str, optional
    memory_k : int
        Number of memory examples to retrieve per task (default: 3).
    max_steps : int
        Maximum ReAct steps per task (default: 5).
    max_tokens : int
        Max tokens per LLM call (default: 4096).
    temperature : float
        Sampling temperature (default: 0.0 for greedy).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
        memory_k: int = 3,
        max_steps: int = 5,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.model = model
        self.memory_k = memory_k
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature

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

        self.memory = ReActMemory(k=memory_k)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict], tag: str = "") -> Dict:
        """Call the LLM and return parsed JSON dict."""
        import json, re, time
        system = _SYSTEM + _JSON_INSTRUCTION

        for attempt in range(3):
            try:
                if self._backend == "anthropic":
                    resp = self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=system,
                        messages=messages,
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
                    oai_msgs = [{"role": "system", "content": system}] + messages
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=oai_msgs,
                        response_format={"type": "json_object"},
                    )
                    raw = resp.choices[0].message.content or ""
                    u = getattr(resp, "usage", None)
                    details = getattr(u, "completion_tokens_details", None)
                    usage = {
                        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0 if details else 0,
                    }

                # Accumulate tokens
                for key, sess_key in [("input_tokens", "input"), ("output_tokens", "output"), ("reasoning_tokens", "reasoning")]:
                    v = usage.get(key, 0) or 0
                    self._task_tokens[sess_key] += v
                    self._session_tokens[sess_key] += v

                # Parse JSON
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
                    logger.warning("react_mem: JSON parse error (tag=%s): %s", tag, raw[:200])
                    result = {}

                # Log
                self._task_log.append({
                    "tag": tag,
                    "model": self.model,
                    "request": {"messages": messages, "max_tokens": self.max_tokens},
                    "response": {"content": raw, "usage": usage},
                    "parsed_result": result,
                    "token_usage": usage,
                })
                if self._debug_dir:
                    self._write_debug(tag, messages, raw, usage, result)
                return result

            except Exception as exc:
                if getattr(exc, "status_code", None) == 400:
                    raise
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning("react_mem LLM call failed (attempt %d/3): %s. Retry in %ds.", attempt + 1, exc, wait)
                    time.sleep(wait)
                else:
                    logger.error("react_mem LLM call failed after 3 attempts: %s", exc)
                    return {}

    def _write_debug(self, tag, messages, raw, usage, parsed):
        import json
        from datetime import datetime, timezone
        self._call_counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        path = os.path.join(self._debug_dir, f"{self._call_counter:04d}_{tag}_{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"tag": tag, "messages": messages, "raw": raw, "usage": usage, "parsed": parsed}, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not write debug log %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Task log helpers (compatible with main.py session token logging)
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
        max_steps: int,
        reward_feedback: Optional[str] = None,
    ) -> Dict:
        """
        Run the ReAct loop for one attempt.

        Returns a dict with keys: answer, code, history, steps_taken.
        """
        examples = self.memory.retrieve(task_description)
        history: List[Dict] = []
        messages: List[Dict] = []
        last_answer = None
        last_code = None

        for step in range(max_steps):
            if step == 0:
                user_content = build_initial_prompt(task_description, examples, step=step)
            else:
                user_content = build_followup_prompt(
                    task_description, history, step, reward_feedback=reward_feedback
                )

            messages.append({"role": "user", "content": user_content})
            result = self._call_llm(messages, tag=f"react_step{step}")

            thought = result.get("thought", "")
            action_type = result.get("action_type", "")
            messages.append({"role": "assistant", "content": str(result)})

            if action_type == "finish":
                last_answer = result.get("answer")
                history.append({
                    "thought": thought,
                    "action_type": "finish",
                    "answer": last_answer,
                    "observation": "",
                })
                logger.info("react_mem step %d: finish → %s", step, str(last_answer)[:80])
                break

            elif action_type == "execute_code":
                code = result.get("code", "")
                last_code = code
                ok, exec_result, err = run_code(code)
                if ok:
                    observation = str(exec_result) if exec_result is not None else "(no output)"
                else:
                    observation = f"ERROR: {err[:400]}"
                logger.info("react_mem step %d: execute_code → %s", step, observation[:80])
                history.append({
                    "thought": thought,
                    "action_type": "execute_code",
                    "code": code,
                    "observation": observation,
                })
                # Inject observation back into messages
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

            else:
                # Unknown action — treat as finish if answer present, else break
                if result.get("answer") is not None:
                    last_answer = result.get("answer")
                history.append({
                    "thought": thought,
                    "action_type": action_type or "unknown",
                    "observation": "",
                })
                break

        return {
            "answer": last_answer,
            "code": last_code,
            "history": history,
            "steps_taken": len(history),
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
        ReAct solve with reward-feedback iterations.

        On each iteration the agent gets the reward feedback and retries.
        The best result is stored in memory after the task.
        """
        self.reset_task_log()

        # Extract the text description for the task
        if isinstance(task_input, dict):
            task_description = (
                task_input.get("question") or task_input.get("prompt") or task_input.get("task") or str(task_input)
            )
        else:
            task_description = str(task_input)

        best_reward = 0.0
        best_answer = None
        best_code = None
        reward_history = []
        reward_feedback: Optional[str] = None

        for iteration in range(max_reward_iters):
            logger.info("react_mem: task iteration %d/%d", iteration + 1, max_reward_iters)
            loop_result = self._run_react_loop(
                task_description,
                max_steps=self.max_steps,
                reward_feedback=reward_feedback,
            )
            answer = loop_result["answer"]
            code = loop_result.get("code")

            reward_result = reward_fn(answer, answer is not None, entry)
            reward_value = float(reward_result.get("value", 0.0))
            reward_message = reward_result.get("message", "")

            if reward_value > best_reward:
                best_reward = reward_value
                best_answer = answer
                best_code = code

            reward_history.append({
                "iteration": iteration,
                "reward": reward_value,
                "message": reward_message,
                "blame": "react",
                "solution_summary": str(answer)[:200] if answer is not None else "",
            })
            logger.info("react_mem iter %d: reward=%.3f  %s", iteration + 1, reward_value, reward_message[:80])

            if reward_value >= 1.0:
                break
            reward_feedback = reward_message

        # Store best solution in memory for future tasks
        self.memory.store(task_description, best_code, best_answer, best_reward)

        solved = best_reward >= 1.0
        return {
            "solved": solved,
            "task_type": task_type,
            "original_prompt": task_description,
            "answer": best_answer,
            "best_reward": best_reward,
            "final_reward": reward_history[-1] if reward_history else {},
            "reward_history": reward_history,
            "trace": [{"agent": "react_mem", "step": h["steps_taken"]} for h in [loop_result]],
            "final_output": {
                "answer": str(best_answer) if best_answer is not None else "",
                "explanation": f"ReAct+Memory: {len(self.memory)} stored examples",
                "confidence": "high" if solved else "low",
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {"memory_size": len(self.memory)},
            "library_snapshot": [],
        }

    def solve(self, task_input: Any, task_type: str = "symbolic", budget: float = 15.0) -> Dict:
        """Single-attempt solve (no reward feedback)."""
        self.reset_task_log()
        if isinstance(task_input, dict):
            task_description = (
                task_input.get("question") or task_input.get("prompt") or task_input.get("task") or str(task_input)
            )
        else:
            task_description = str(task_input)

        loop_result = self._run_react_loop(task_description, max_steps=self.max_steps)
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
                "explanation": "ReAct+Memory (no reward)",
                "confidence": "medium",
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {"memory_size": len(self.memory)},
            "library_snapshot": [],
        }

    def projected_budget(self, max_reward_iters: int = 3) -> Dict:
        """Estimate the maximum token spend for one task."""
        calls_per_iter = self.max_steps * 2  # user + observation per step
        total = max_reward_iters * calls_per_iter * self.max_tokens
        return {
            "max_reward_iters": max_reward_iters,
            "max_steps": self.max_steps,
            "calls_per_iter": calls_per_iter,
            "max_tokens": self.max_tokens,
            "projected_max_tokens": total,
            "formula": f"{max_reward_iters} iters × {calls_per_iter} calls × {self.max_tokens} tokens = {total}",
        }

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {"memory": self.memory.to_dict()}

    def from_dict(self, data: Dict) -> None:
        self.memory = ReActMemory.from_dict(data.get("memory", []), k=self.memory_k)
