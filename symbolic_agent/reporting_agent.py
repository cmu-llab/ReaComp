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

_LONG_OUTPUT_HINTS = {
    "sequence", "steps", "moves", "path", "list", "enumerate",
    "simulate", "grid", "matrix", "multiple",
}


def _reporting_max_tokens(task_spec) -> int:
    """Scale reporting budget up for tasks whose answers can be long sequences or grids.

    task_spec may be a TaskSpec dataclass (from agents) or a plain dict (from state).
    """
    if task_spec is None:
        return 1024
    hints = (
        task_spec.get("operation_hints", [])
        if isinstance(task_spec, dict)
        else task_spec.operation_hints
    )
    words = {w for hint in hints for w in hint.lower().split()}
    return 2048 if words & _LONG_OUTPUT_HINTS else 1024

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

        task_spec = state.get("task_spec")
        result = self.client.create(
            model=self.model,
            max_tokens=_reporting_max_tokens(task_spec),
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
        """
        Best-effort extraction of call arguments from the task description.
        Returns [] when no concrete input data can be found — execution is
        optional and the reporting agent handles None results gracefully.
        """
        if isinstance(task, dict):
            # Built-in task format: {"description": "...", "examples": [...]}
            if "examples" in task:
                ex = task["examples"][0] if task["examples"] else {}
                return [ex["input"]] if "input" in ex else []
            # Some task formats carry a direct "input" value
            if "input" in task:
                inp = task["input"]
                return [inp] if not isinstance(inp, list) else [inp]
            # Prompt-only dict (e.g. JSONL tasks) — no concrete args available
            return []
        if isinstance(task, list):
            return [task]
        # Plain string prompt — no concrete args
        return []
