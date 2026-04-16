"""
Prompt builders for the StaticLibrary agent.

The system prompt embeds the PROMPTING_GUIDE from the pre-built library so the
weaker agent follows the stronger model's prescribed workflow exactly.
The task prompt shows the task and the list of available library functions.
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


def build_system_prompt(prompting_guide: str, function_names: list[str]) -> str:
    """
    Build the system prompt by embedding the PROMPTING_GUIDE verbatim.
    The guide already describes the recommended workflow, tools, and constraints —
    we add only the tool interface on top.
    """
    fn_list = "\n".join(f"  - {name}" for name in function_names)

    tool_section = f"""\
You are an expert Python programmer. You have access to a pre-built library of \
helper functions and three tools:

  execute_code(code)
      Run Python code in a sandbox. Import library functions with:
          from library import fn_name
      Print results to stdout to observe them.

  check_reward(answer)
      Verify your candidate answer against the task verifier.
      Returns a reward in [0, 1] and actionable feedback.
      Call this as many times as needed.

  finish(answer)
      Submit the final answer. Call once check_reward returns 1.0 or you \
are satisfied with your best answer.

Available library functions:
{fn_list}

You CANNOT add new functions to the library. Use only the functions listed above.

---

"""

    if prompting_guide:
        return tool_section + prompting_guide
    return tool_section


def build_task_prompt(task_input: Any) -> str:
    return f"Task:\n{_task_text(task_input)}"
