"""Reporting Agent.

Translates an existing solution into a clean final output.
Receives the original NL prompt so it can honour any output-format
instructions or in-context examples embedded in it.

Responds with a plain JSON object — no tool calling.
"""

import logging
from typing import Any, Dict

from .executor import execute_with_library
from .library import FunctionLibrary

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are the Reporting agent in a symbolic reasoning system.
Translate an existing solution into the output format requested by the original prompt.
The original prompt may contain format instructions or in-context examples — follow them exactly.
Do NOT do any new reasoning; only reformat the provided solution.

Respond with exactly this JSON:
{
  "answer": "<final answer, formatted exactly as the original prompt requests>",
  "explanation": "<one-sentence plain-English explanation>",
  "confidence": <float 0.0–1.0>
}
"""


class ReportingAgent:
    def __init__(self, client, model: str = "claude-sonnet-4-5"):
        self.client = client
        self.model = model

    def run(self, state: Dict, library: FunctionLibrary) -> Dict:
        if not state.get("solved") or not state.get("solution"):
            state["final_output"] = {"error": "Task was not solved."}
            return state

        solution = state["solution"]
        original_prompt = state.get("original_prompt") or str(state.get("task_input", ""))

        execution_result: Any = None
        execution_error: str = ""

        code = solution.get("code", "")
        func_name = solution.get("function", "")

        if code and func_name:
            call_args = self._infer_args(state.get("task_input"))
            ok, result, err = execute_with_library(
                solution_code=code,
                function_name=func_name,
                args=call_args,
                library_functions=library.functions,
            )
            if ok:
                execution_result = result
                logger.info("Reporting: executed successfully, result=%s", result)
            else:
                execution_error = err
                logger.warning("Reporting: execution error: %s", err)

        exec_info = (
            f"Execution result: {execution_result}"
            if execution_result is not None
            else (f"Execution error: {execution_error}" if execution_error else "Not executed.")
        )

        user_msg = (
            f"Original task prompt:\n{original_prompt}\n\n"
            f"Solution code:\n```python\n{code}\n```\n\n"
            f"Reasoning: {solution.get('reasoning', 'N/A')}\n\n"
            f"{exec_info}\n\n"
            "Format the final answer according to the output format specified in the original prompt."
        )

        result = self.client.create(
            model=self.model,
            max_tokens=512,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tag="reporting",
        )

        if result.get("answer") is not None:
            state["final_output"] = {
                "answer": result["answer"],
                "explanation": result.get("explanation", ""),
                "confidence": result.get("confidence", 0.5),
                "execution_result": execution_result,
            }
            return state

        # Fallback if JSON parsing failed or answer was absent
        state["final_output"] = {
            "answer": str(execution_result if execution_result is not None else solution),
            "explanation": solution.get("reasoning", ""),
            "confidence": 0.5,
            "execution_result": execution_result,
        }
        return state

    def _infer_args(self, task) -> list:
        if isinstance(task, dict):
            if "input" in task:
                inp = task["input"]
                return [inp] if not isinstance(inp, list) else [inp]
            if "examples" in task:
                ex = task["examples"][0] if task["examples"] else {}
                return [ex.get("input")] if "input" in ex else []
        if isinstance(task, list):
            return [task]
        return [task]
