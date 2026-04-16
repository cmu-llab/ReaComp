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
      Use this ONLY for helper computations (e.g. inspect data, test a single
      string). Do NOT use it to manually verify your full answer — use check_reward.

  check_reward(answer)
      Score your candidate answer against the ground-truth verifier.
      Returns a reward in [0, 1] and SPECIFIC feedback on what is wrong.
      This is your primary feedback signal. Call it after EVERY candidate answer.

  finish(answer)
      Submit the final answer. Call once check_reward returns 1.0 or you are
      satisfied with your best answer.

Available library functions:
{fn_list}

You CANNOT add new functions to the library. Use only the functions listed above.

Tight iteration loop — follow this exactly:
  1. Use the library functions to reason about the task (execute_code for small
     helper computations if needed — max 1-2 calls before checking reward).
  2. Form a candidate answer and call check_reward(candidate) IMMEDIATELY.
  3. Read the feedback. If reward < 1.0, fix the specific issue and repeat.
  4. Call finish once reward = 1.0 or you have your best answer.

Do NOT spend many execute_code steps manually comparing outputs to expected values.
check_reward scores all examples at once and tells you exactly what failed.

---

"""

    if prompting_guide:
        return tool_section + prompting_guide
    return tool_section


def build_task_prompt(task_input: Any) -> str:
    return f"Task:\n{_task_text(task_input)}"
