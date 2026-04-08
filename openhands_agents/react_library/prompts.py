"""
Prompt builders for the ReAct+Library OpenHands agent.

The agent has three tools:
  execute_code    — run Python in the sandbox; library available via `from library import fn`
  add_to_library  — write a new standalone helper to the shared library
  finish          — submit the final answer

The system prompt explains the tools, the library constraint (standalone functions),
and the reward-feedback loop.
"""

from typing import Any


def _task_text(task_input: Any) -> str:
    if isinstance(task_input, dict):
        return (
            task_input.get("question")
            or task_input.get("prompt")
            or task_input.get("task")
            or str(task_input)
        )
    return str(task_input)


def build_system_prompt() -> str:
    return """\
You are an expert Python programmer with access to a shared function library.

Available tools:
  execute_code(code)
      Run Python code in a sandbox. The shared library is available:
          from library import fn_name
      The code must print the final answer to stdout if you want to observe it.

  add_to_library(name, description, code)
      Add a reusable helper function to the shared library.
      Rules:
        - The function must be completely standalone.
        - Import only from stdlib or installed packages (numpy, scipy, sympy).
        - Do NOT import from `library` or call other library functions inside the function body.
        - Include a concise docstring.
        - Use snake_case names.

  finish(answer)
      Submit the final answer. Call this when you are confident in the result.

Strategy:
  1. Check the library listing for relevant functions.
  2. If a useful function exists, use execute_code to call it.
  3. If you need a new reusable helper, use add_to_library first, then execute_code.
  4. Once you have the answer, call finish.
  5. If you receive reward feedback, refine your approach accordingly."""


def build_task_prompt(task_input: Any, library_listing: str, reward_feedback: str = "") -> str:
    parts = [f"Task:\n{_task_text(task_input)}"]

    if library_listing and library_listing != "(empty)":
        parts.append(f"\nLibrary functions available:\n{library_listing}")
    else:
        parts.append("\nLibrary: (empty — no functions yet)")

    if reward_feedback:
        parts.append(f"\nFeedback from previous attempt:\n{reward_feedback}")

    return "\n".join(parts)
