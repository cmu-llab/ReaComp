"""
TroVE prompt builders. Three modes: IMPORT, CREATE, SKIP.

Following the paper:
- IMPORT: use functions from the toolbox in the solution.
- CREATE: write a NEW standalone helper function, then use it in the solution.
- SKIP:   solve directly with primitive Python (no toolbox).

Key constraints baked into all prompts:
- Solutions must print the final answer to stdout.
- CREATE functions must be completely standalone (no imports from the toolbox).
- The response format is JSON: {"function": ..., "code": ...}
  where "function" is null for SKIP/IMPORT modes (no new function created).
"""

from typing import Any


_BASE_SYSTEM = """\
You are an expert Python programmer solving tasks by writing executable programs.
Your solutions must print the final answer to stdout.
Respond with a single JSON object — no markdown fences, no prose outside the JSON."""

_TOOLBOX_HEADER = "Available toolbox functions (import from toolbox):"


def _task_text(task_input: Any) -> str:
    if isinstance(task_input, dict):
        return (
            task_input.get("question")
            or task_input.get("prompt")
            or task_input.get("task")
            or str(task_input)
        )
    return str(task_input)


def build_skip_prompt(task_input: Any) -> tuple[str, str]:
    """Return (system, user) for SKIP mode."""
    system = _BASE_SYSTEM + """

Your task: solve the problem using only Python primitives and standard libraries.

Response format:
{"function": null, "code": "<complete Python program that prints the answer>"}"""

    user = f"Task:\n{_task_text(task_input)}"
    return system, user


def build_import_prompt(task_input: Any, toolbox_listing: str) -> tuple[str, str]:
    """Return (system, user) for IMPORT mode."""
    system = _BASE_SYSTEM + f"""

Your task: solve the problem by importing and using functions from the toolbox below.

{_TOOLBOX_HEADER}
{toolbox_listing}

Import exactly what you use: `from toolbox import fn_name`

Response format:
{{"function": null, "code": "<complete Python program that imports from toolbox and prints the answer>"}}"""

    user = f"Task:\n{_task_text(task_input)}"
    return system, user


def build_create_prompt(task_input: Any, toolbox_listing: str) -> tuple[str, str]:
    """Return (system, user) for CREATE mode."""
    system = _BASE_SYSTEM + f"""

Your task: (1) write a NEW reusable Python helper function, then (2) use it to solve the problem.

Rules for the new function:
- Must be completely standalone: import only from stdlib or installed packages (numpy, scipy, sympy).
- Must NOT import from toolbox or call other toolbox functions.
- Should be generic enough to reuse on similar tasks.
- Include a concise docstring.

Existing toolbox (for reference — do not duplicate these):
{toolbox_listing}

Response format:
{{
  "function": {{
    "name": "<snake_case_name>",
    "description": "<one-line description>",
    "code": "<complete function definition including any needed imports>"
  }},
  "code": "<complete solution program — defines the function above, then calls it and prints the answer>"
}}"""

    user = f"Task:\n{_task_text(task_input)}"
    return system, user
