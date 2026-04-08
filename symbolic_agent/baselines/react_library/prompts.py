"""
Prompts for the ReAct + Library baseline.

The agent interleaves Thought and Action in a ReAct loop.  Unlike the
plain ReAct+Memory baseline, it maintains a shared library of reusable
Python helper functions.  Library functions are always in the execution
namespace so the agent calls them by name directly.

Three action types:
  execute_code    — run Python; library functions are available by name.
  add_to_library  — define and register a new reusable helper function.
  finish          — emit the final answer.
"""

from typing import Dict, List, Optional


_SYSTEM = """\
You are a ReAct agent that solves programming and reasoning tasks.
You maintain a shared library of reusable Python helper functions that grows as you work.

At each step respond with a single valid JSON object in one of these three formats:

Format A — execute Python code (library functions are already in scope):
{
  "thought": "My reasoning about what to try next",
  "action_type": "execute_code",
  "code": "# call library helpers by name; use print() to see output\\nresult = my_helper(x)\\nprint(result)"
}

Format B — add a reusable helper to the library:
{
  "thought": "I need a reusable helper for this pattern",
  "action_type": "add_to_library",
  "name": "snake_case_function_name",
  "description": "One-line description of what this function does",
  "code": "def snake_case_function_name(...):\\n    ..."
}

Format C — provide the final answer:
{
  "thought": "I now have the correct answer",
  "action_type": "finish",
  "answer": "<exact final answer>"
}

Rules:
- execute_code: library functions are pre-loaded into the namespace — call them by name directly.
  Never use 'from __main__ import fn' or re-implement a library function inline.
  Use print() to emit output; that output becomes your Observation.
- add_to_library: only add *genuinely reusable* functions (parameterised, not task-specific).
  The function must be self-contained Python (no os/sys/subprocess imports).
  A function with that name overwrites any prior definition.
- finish: use ONLY when you are confident in the answer.
  The answer must be the exact value requested (not a sentence).
- Do not output prose outside the JSON object.
"""


def _format_library_block(functions: List[Dict]) -> str:
    if not functions:
        return ""
    lines = ["--- Available library functions ---"]
    for fn in functions:
        lines.append(f"\n{fn['name']}: {fn['description']}")
        lines.append(f"```python\n{fn['code']}\n```")
    lines.append("--- End of library ---\n")
    return "\n".join(lines)


def build_initial_prompt(
    task: str,
    library_functions: List[Dict],
    step: int = 0,
) -> str:
    """Build the user message for the first ReAct step."""
    lib_block = _format_library_block(library_functions)
    return (
        f"{lib_block}"
        f"Task:\n{task}\n\n"
        "Step 1: Think about how to solve this task.\n"
        "If a library function covers part of the work, call it directly in execute_code.\n"
        "If you need a reusable helper that is not in the library, add it first with add_to_library.\n"
        "Remember: print() your result in execute_code so you can see the output."
    )


def build_followup_prompt(
    task: str,
    history: List[Dict],
    step: int,
    library_functions: Optional[List[Dict]] = None,
    reward_feedback: Optional[str] = None,
) -> str:
    """
    Build the user message for subsequent ReAct steps.

    library_functions: current top-K relevant library functions (may have grown since step 0).
    history: list of step dicts produced during the loop.
    """
    lines = [f"Task:\n{task}\n"]

    for i, h in enumerate(history, 1):
        lines.append(f"Step {i}:")
        lines.append(f"  Thought: {h['thought']}")
        atype = h["action_type"]
        if atype == "execute_code":
            code_preview = h.get("code", "")[:400]
            lines.append(f"  Action: execute_code\n  Code:\n```python\n{code_preview}\n```")
            obs = h.get("observation", "")
            lines.append(f"  Observation: {obs[:500] if obs else '(no output)'}")
        elif atype == "add_to_library":
            lines.append(
                f"  Action: add_to_library → {h.get('name', '?')} "
                f"({h.get('description', '')[:80]})"
            )
            lines.append(f"  Observation: {h.get('observation', '')}")
        elif atype == "finish":
            lines.append(f"  Action: finish → {h.get('answer', '')}")

    if library_functions:
        lines.append("")
        lines.append(_format_library_block(library_functions))

    if reward_feedback:
        lines.append(f"\nFeedback on previous answer: {reward_feedback}")

    lines.append(
        f"\nStep {step + 1}: Continue reasoning. "
        "Execute code, add a library helper, or provide the final answer."
    )
    return "\n".join(lines)
