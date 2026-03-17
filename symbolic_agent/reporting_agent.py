"""Reporting Agent.

Translates an existing solution into a clean final output.
No new reasoning is permitted here.
"""

import logging
from typing import Any, Dict

from .executor import execute_with_library
from .library import FunctionLibrary

logger = logging.getLogger(__name__)

_TOOLS = [
    {
        "name": "format_solution",
        "description": "Format the solution into a clean final answer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer, formatted for the task type.",
                },
                "explanation": {
                    "type": "string",
                    "description": "One-sentence plain-English explanation.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0 and 1.",
                },
            },
            "required": ["answer", "explanation", "confidence"],
        },
    }
]

_SYSTEM = (
    "You are the Reporting agent.  Your job is ONLY to translate an existing "
    "solution into the required output format.  Do NOT do any new reasoning."
)


class ReportingAgent:
    def __init__(self, client, model: str = "claude-haiku-4-5-20251001"):
        self.client = client
        self.model = model

    def run(self, state: Dict, library: FunctionLibrary) -> Dict:
        if not state.get("solved") or not state.get("solution"):
            state["final_output"] = {"error": "Task was not solved."}
            return state

        solution = state["solution"]
        task = state["task_input"]

        # Try to execute the solution to get a concrete result
        execution_result: Any = None
        execution_error: str = ""

        code = solution.get("code", "")
        func_name = solution.get("function", "")

        if code and func_name:
            call_args = self._infer_args(task)
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
            f"Task: {task}\n\n"
            f"Solution code:\n```python\n{code}\n```\n\n"
            f"Reasoning: {solution.get('reasoning', 'N/A')}\n\n"
            f"{exec_info}\n\n"
            "Format the final answer."
        )

        response = self.client.create(
            model=self.model,
            max_tokens=512,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=_TOOLS,
            tool_choice={"type": "any"},
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "format_solution":
                state["final_output"] = {
                    "answer": block.input["answer"],
                    "explanation": block.input["explanation"],
                    "confidence": block.input["confidence"],
                    "execution_result": execution_result,
                }
                return state

        # Fallback if tool was not called
        state["final_output"] = {
            "answer": str(execution_result if execution_result is not None else solution),
            "explanation": solution.get("reasoning", ""),
            "confidence": 0.5,
            "execution_result": execution_result,
        }
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_args(self, task) -> list:
        """Best-effort: extract call arguments from the task description."""
        if isinstance(task, dict):
            if "input" in task:
                inp = task["input"]
                return [inp] if not isinstance(inp, list) else [inp]
            if "examples" in task:
                # Return the input of the first example for spot-checking
                ex = task["examples"][0] if task["examples"] else {}
                return [ex.get("input")] if "input" in ex else []
        if isinstance(task, list):
            return [task]
        return [task]
