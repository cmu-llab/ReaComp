"""
ReAct + Library Controller.

Implements a ReAct agent (Yao et al. 2022) that grows a shared library of
reusable Python helper functions across tasks, rather than storing episodic
memories of past solutions.

Architecture:
  Shared state:
    - ReactLibrary: named helper functions that persist across all tasks.

  For each task:
    1. Retrieve top-K relevant library functions (BM25 on name+description).
    2. Run ReAct loop (Thought → Action → Observation) for up to max_steps.
       Actions: execute_code (library in namespace), add_to_library, finish.
    3. Evaluate with reward_fn.
    4. Up to max_reward_iters iterations with reward feedback.

Differences from ReAct+Memory (react_mem):
  - Memory stores (task, code, answer) pairs for few-shot retrieval.
  - Library stores named helper *functions* loaded into the execution
    namespace so the agent calls them directly, not by example.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .executor import run_code_with_library
from .library import ReactLibrary
from .prompts import _SYSTEM, build_followup_prompt, build_initial_prompt

logger = logging.getLogger(__name__)

_JSON_INSTRUCTION = (
    "\n\nRespond with a single valid JSON object and nothing else. "
    "No markdown fences, no prose before or after the JSON."
)


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
    library_k : int
        Number of library functions to retrieve per task (default: 5).
    max_steps : int
        Maximum ReAct steps per task (default: 6).
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
        library_k: int = 5,
        max_steps: int = 6,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        self.model = model
        self.library_k = library_k
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

        self.library = ReactLibrary()

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, messages: List[Dict], tag: str = "") -> Dict:
        """Call the LLM and return a parsed JSON dict."""
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
                    reasoning_raw = ""
                    usage = {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "reasoning_tokens": 0,
                    }
                else:
                    oai_msgs = [{"role": "system", "content": system}] + messages
                    # No response_format — gpt-oss-120b (and other reasoning models on vLLM)
                    # produce [reasoning, final] as 2 internal output segments; json_object
                    # mode conflicts with this 2-part structure and returns 400.
                    # We instruct via the system prompt instead (_JSON_INSTRUCTION).
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=oai_msgs,
                    )
                    msg = resp.choices[0].message
                    raw = msg.content or ""
                    reasoning_raw = getattr(msg, "reasoning_content", "") or ""
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

                # Accumulate tokens
                for key, sess_key in [
                    ("input_tokens", "input"),
                    ("output_tokens", "output"),
                    ("reasoning_tokens", "reasoning"),
                ]:
                    v = usage.get(key, 0) or 0
                    self._task_tokens[sess_key] += v
                    self._session_tokens[sess_key] += v

                # Strip markdown fences if present
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
                    "request": {"messages": messages, "max_tokens": self.max_tokens},
                    "response": {"content": raw, "reasoning_content": reasoning_raw, "usage": usage},
                    "parsed_result": result,
                    "token_usage": usage,
                })
                if self._debug_dir:
                    self._write_debug(tag, messages, raw, usage, result)
                # Embed raw response text so _run_react_loop can use it as the
                # assistant message (not str(result)).  Callers must pop this key.
                result["_raw"] = raw
                result["_reasoning"] = reasoning_raw
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

    def _write_debug(self, tag, messages, raw, usage, parsed):
        import json
        from datetime import datetime, timezone
        self._call_counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        path = os.path.join(
            self._debug_dir, f"{self._call_counter:04d}_{tag}_{ts}.json"
        )
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"tag": tag, "messages": messages, "raw": raw, "usage": usage, "parsed": parsed},
                    f, indent=2, default=str,
                )
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
        Run one ReAct loop for the task.

        Returns a dict: {answer, code, history, steps_taken, library_additions}.
        """
        # Retrieve relevant library functions for this task
        lib_functions = self.library.retrieve(task_description, k=self.library_k)

        history: List[Dict] = []
        messages: List[Dict] = []
        last_answer = None
        last_code = None
        library_additions: List[str] = []

        for step in range(max_steps):
            if step == 0:
                user_content = build_initial_prompt(task_description, lib_functions, step=step)
            else:
                # Refresh relevant functions in case new ones were added
                lib_functions = self.library.retrieve(task_description, k=self.library_k)
                user_content = build_followup_prompt(
                    task_description,
                    history,
                    step,
                    library_functions=lib_functions,
                    reward_feedback=reward_feedback,
                )

            messages.append({"role": "user", "content": user_content})
            result = self._call_llm(messages, tag=f"react_lib_step{step}")

            # Pop internal transport keys before using result as the parsed action.
            raw_response = result.pop("_raw", "")
            reasoning_response = result.pop("_reasoning", "")

            thought = result.get("thought", "")
            action_type = result.get("action_type", "")

            # Build the assistant turn using the original response text.
            # For gpt-oss-120b multi-turn, include reasoning_content if present so
            # vLLM can validate the 2-part structure of historical assistant turns.
            asst_msg: Dict = {"role": "assistant", "content": raw_response or "{}"}
            if reasoning_response:
                asst_msg["reasoning_content"] = reasoning_response
            messages.append(asst_msg)

            # ---- finish ----
            if action_type == "finish":
                last_answer = result.get("answer")
                history.append({
                    "thought": thought,
                    "action_type": "finish",
                    "answer": last_answer,
                    "observation": "",
                })
                logger.info(
                    "react_library step %d: finish → %s", step, str(last_answer)[:80]
                )
                break

            # ---- execute_code ----
            elif action_type == "execute_code":
                code = result.get("code", "")
                last_code = code
                lib_ns = self.library.namespace()
                ok, exec_result, err = run_code_with_library(code, library_namespace=lib_ns)
                if ok:
                    observation = (
                        str(exec_result) if exec_result is not None else "(no output)"
                    )
                else:
                    observation = f"ERROR: {err[:400]}"
                logger.info(
                    "react_library step %d: execute_code → %s", step, observation[:80]
                )
                history.append({
                    "thought": thought,
                    "action_type": "execute_code",
                    "code": code,
                    "observation": observation,
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

            # ---- add_to_library ----
            elif action_type == "add_to_library":
                name = result.get("name", "").strip()
                description = result.get("description", "").strip()
                code = result.get("code", "").strip()

                if not name or not code:
                    observation = "ERROR: add_to_library requires 'name' and 'code' fields."
                else:
                    # Validate the function compiles and executes cleanly
                    lib_ns = self.library.namespace()
                    ok, _, err = run_code_with_library(code, library_namespace=lib_ns)
                    if not ok:
                        observation = f"ERROR: function code failed validation: {err[:300]}"
                    else:
                        self.library.update(name, description, code)
                        library_additions.append(name)
                        observation = f"Added '{name}' to library. ({len(self.library)} functions total)"
                        logger.info(
                            "react_library step %d: add_to_library '%s'", step, name
                        )

                history.append({
                    "thought": thought,
                    "action_type": "add_to_library",
                    "name": name,
                    "description": description,
                    "code": code,
                    "observation": observation,
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

            # ---- unknown ----
            else:
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
            "library_additions": library_additions,
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
        ReAct+Library solve with reward-feedback iterations.

        Any functions added to the library during this task persist and are
        available to future tasks.
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

        best_reward = 0.0
        best_answer = None
        best_code = None
        reward_history = []
        reward_feedback: Optional[str] = None
        all_library_additions: List[str] = []

        for iteration in range(max_reward_iters):
            logger.info(
                "react_library: task iteration %d/%d (library=%d fns)",
                iteration + 1, max_reward_iters, len(self.library),
            )
            loop_result = self._run_react_loop(
                task_description,
                max_steps=self.max_steps,
                reward_feedback=reward_feedback,
            )
            answer = loop_result["answer"]
            code = loop_result.get("code")
            all_library_additions.extend(loop_result.get("library_additions", []))

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
                "blame": "react_library",
                "solution_summary": str(answer)[:200] if answer is not None else "",
            })
            logger.info(
                "react_library iter %d: reward=%.3f  %s",
                iteration + 1, reward_value, reward_message[:80],
            )

            if reward_value >= 1.0:
                break
            reward_feedback = reward_message

        solved = best_reward >= 1.0
        return {
            "solved": solved,
            "task_type": task_type,
            "original_prompt": task_description,
            "answer": best_answer,
            "best_reward": best_reward,
            "final_reward": reward_history[-1] if reward_history else {},
            "reward_history": reward_history,
            "trace": [{"agent": "react_library", "step": loop_result["steps_taken"]}],
            "final_output": {
                "answer": str(best_answer) if best_answer is not None else "",
                "explanation": (
                    f"ReAct+Library: {len(self.library)} library fns, "
                    f"added {all_library_additions} this task"
                ),
                "confidence": "high" if solved else "low",
            },
            "agent_messages": self.get_task_log(),
            "token_usage": self.get_task_token_usage(),
            "cost_summary": {
                "library_size": len(self.library),
                "library_additions_this_task": all_library_additions,
            },
            "library_snapshot": self.library.to_list(),
        }

    def solve(
        self, task_input: Any, task_type: str = "symbolic", budget: float = 15.0
    ) -> Dict:
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
        calls_per_iter = self.max_steps * 2
        total = max_reward_iters * calls_per_iter * self.max_tokens
        return {
            "max_reward_iters": max_reward_iters,
            "max_steps": self.max_steps,
            "calls_per_iter": calls_per_iter,
            "max_tokens": self.max_tokens,
            "projected_max_tokens": total,
            "formula": (
                f"{max_reward_iters} iters × {calls_per_iter} calls "
                f"× {self.max_tokens} tokens = {total}"
            ),
        }

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {"library": self.library.to_list()}

    def from_dict(self, data: Dict) -> None:
        self.library = ReactLibrary.from_list(data.get("library", []))
