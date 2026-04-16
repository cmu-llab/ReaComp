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
    # Keep in sync with prompts/system_prompt.j2
    return open(
        __import__("os").path.join(__import__("os").path.dirname(__file__), "prompts", "system_prompt.j2")
    ).read()
