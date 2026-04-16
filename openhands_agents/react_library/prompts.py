"""
Prompt builders for the ReAct+Library OpenHands agent.
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


def _library_section(relevant: list[dict]) -> str:
    """
    Format retrieved library functions for the task prompt.
    Shows the full function code so the agent can decide whether to reuse.
    """
    if not relevant:
        return "Library: (empty — no functions yet)"

    lines = ["Library functions available (reuse these if applicable):"]
    for fn in relevant:
        lines.append(f"\n  # {fn['name']}: {fn['description']}")
        # Indent the code block so it reads cleanly in the prompt
        for line in fn["code"].splitlines():
            lines.append(f"  {line}")
    return "\n".join(lines)


def build_task_prompt(task_input: Any, relevant: list[dict]) -> str:
    parts = [
        f"Task:\n{_task_text(task_input)}",
        "",
        _library_section(relevant),
    ]
    return "\n".join(parts)


def build_system_prompt() -> str:
    """Kept for reference; the live system prompt is system_prompt.j2."""
    return """\
You are an expert Python programmer with access to a shared function library.

Available tools:
  execute_code(code)
      Run Python code in a sandbox. The shared library is available:
          from library import fn_name
      Print the answer to stdout to observe it.

  add_to_library(name, description, code)
      Add a reusable helper to the shared library.
      Rules:
        - Standalone: import only from stdlib or installed packages (numpy, scipy, sympy).
        - Do NOT import from `library` inside the function body.
        - Include a concise docstring. Use snake_case names.

  check_reward(answer)
      Verify your candidate answer against the task verifier.
      Returns a reward in [0, 1] and feedback. Use this to iterate
      without restarting — call it as often as needed.

  finish(answer)
      Submit the final answer. Call once you are satisfied.

Strategy:
  1. Read the library listing — if a function fits, import and call it directly.
  2. If you need a new helper, add_to_library first, then execute_code.
  3. Use check_reward to verify your answer and fix issues.
  4. Call finish with your best answer."""
