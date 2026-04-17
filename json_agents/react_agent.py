"""
JSON-mode ReAct Agent (no library).

Each iteration the model receives the task and the history of previous
attempts, reasons, optionally executes code, and submits a candidate answer.
No pre-built library — the model relies purely on its own reasoning.

Mirrors the openhands react_library baseline but without the library so it
serves as a clean zero-library baseline for gpt-oss-120b.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbolic_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert Python programmer solving program-synthesis tasks.

On each turn you will receive:
  - The task description
  - Optionally: the output from running your previous code snippet
  - Optionally: reward feedback from a previous answer attempt

You must respond with a JSON object with these fields:
  "reasoning" : string  — your step-by-step reasoning (think aloud)
  "code"       : string  — a short Python snippet to explore or verify
                           (leave empty string "" if not needed)
  "answer"     : string  — your best candidate answer to submit

Rules:
- After seeing reward feedback, address the specific issue described.
- Keep code short and exploratory — do not write the full solution in code.
- Submit your best answer every round even if unsure.
"""


def _task_text(task_input) -> str:
    if isinstance(task_input, dict):
        return (task_input.get("question") or task_input.get("prompt")
                or task_input.get("task") or str(task_input))
    return str(task_input)


class ReActJsonAgent:
    """
    JSON-mode ReAct agent (no library).

    Parameters
    ----------
    llm : LLMClient
    model : str
    execute_fn : callable(code, lib_dir="") -> (ok, stdout, stderr)  or None
    max_iters : int
    max_tokens : int
    """

    def __init__(
        self,
        llm: LLMClient,
        model: str,
        execute_fn=None,
        max_iters: int = 8,
        max_tokens: int = 4096,
    ):
        self.llm = llm
        self.model = model
        self.execute_fn = execute_fn
        self.max_iters = max_iters
        self.max_tokens = max_tokens

    def solve(self, task_input, reward_fn, entry: dict) -> dict:
        self.llm.reset_task_log()

        task_text = _task_text(task_input)
        messages = []
        reward_history = []
        best_reward = 0.0
        best_answer = None

        messages.append({"role": "user", "content": f"Task:\n{task_text}"})

        for i in range(self.max_iters):
            resp = self.llm.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=_SYSTEM,
                messages=messages,
                tag=f"react_json_iter{i}",
            )

            reasoning = resp.get("reasoning", "")
            code = resp.get("code", "").strip()
            answer = str(resp.get("answer", "")).strip()

            assistant_text = f"reasoning: {reasoning}\ncode: {code}\nanswer: {answer}"
            messages.append({"role": "assistant", "content": assistant_text})

            exec_out = ""
            if code and self.execute_fn:
                ok, stdout, stderr = self.execute_fn(code)
                exec_out = stdout.strip() if ok else f"EXECUTION ERROR:\n{stderr[:500]}"

            if not answer:
                feedback = "You did not provide an answer. Please reason through the task and provide an answer."
                if exec_out:
                    feedback = f"Code output:\n{exec_out}\n\n{feedback}"
                messages.append({"role": "user", "content": feedback})
                continue

            result = reward_fn(answer, True, entry)
            reward = float(result.get("value", 0.0))
            message = result.get("message", "")
            reward_history.append({"iteration": i, "reward": reward, "message": message, "answer": answer})

            if reward > best_reward:
                best_reward = reward
                best_answer = answer

            logger.info("react_json iter=%d reward=%.3f", i, reward)

            if reward >= 1.0:
                break

            feedback_parts = []
            if exec_out:
                feedback_parts.append(f"Code output:\n{exec_out}")
            feedback_parts.append(
                f"Reward: {reward:.3f}\nFeedback: {message}\n\n"
                "Fix the specific issue and try again."
            )
            messages.append({"role": "user", "content": "\n\n".join(feedback_parts)})

        token_usage = self.llm.get_task_token_usage()
        return {
            "solved": best_reward >= 1.0,
            "answer": best_answer,
            "best_reward": best_reward,
            "reward_history": reward_history,
            "agent_messages": self.llm.get_task_log(),
            "token_usage": {
                "prompt_tokens": token_usage.get("input", 0),
                "completion_tokens": token_usage.get("output", 0),
                "reasoning_tokens": token_usage.get("reasoning", 0),
            },
        }
