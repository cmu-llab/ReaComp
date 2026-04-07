"""
Prompts for the ReAct + Library baseline.

Stateless per-step context design: each LLM call receives a single
self-contained message with the current task, top-K retrieved library
functions, the last execution output, and the latest verifier feedback.
No prior trajectory is included — the model's own chain-of-thought
(reasoning_content) serves as implicit state.

Three flat actions:
  execute      — run Python; library functions are in the namespace.
  add_function — register a new reusable helper in the shared library.
  submit       — propose a final answer; triggers immediate reward eval.
"""

from typing import Dict, List, Optional


_SYSTEM = """\
You are a programming agent that solves tasks by writing and testing Python code.
You have access to a shared library of reusable helper functions (always pre-loaded in scope).

Output exactly one JSON action and nothing else:

Run code (library functions are in scope; use print() for output):
{"action": "execute", "code": "<python code>"}

Add a reusable helper to the shared library:
{"action": "add_function", "name": "<snake_case>", "description": "<one-line>", "code": "<def snake_case(...): ...>"}

Submit your final answer:
{"action": "submit", "answer": "<exact answer>"}

Rules:
- execute: library functions are pre-loaded by name — call them directly, no imports.
  Use print() to emit output; that becomes your observation.
- add_function: parameterised helpers only, no task-specific hardcoding.
  Self-contained Python; no os/sys/subprocess.
  Same name overwrites any prior definition.
- submit: only when confident. Exact value requested, not prose.
- Always wrap your response in a ```json fence — nothing outside the fence.\
"""


def _format_library_block(functions: List[Dict]) -> str:
    if not functions:
        return ""
    lines = ["Available library functions:"]
    for fn in functions:
        lines.append(f"  {fn['name']}: {fn['description']}")
        lines.append(f"  ```python\n  {fn['code']}\n  ```")
    return "\n".join(lines)


def build_prompt(
    task: str,
    library_functions: List[Dict],
    last_result: Optional[str] = None,
    verifier_feedback: Optional[str] = None,
) -> str:
    """
    Build a self-contained single-turn prompt for one ReAct step.

    Parameters
    ----------
    task : str
        The task description shown every step.
    library_functions : List[Dict]
        Top-K retrieved library functions to show.
    last_result : str, optional
        Stdout / stderr from the last execute action (None on first step).
    verifier_feedback : str, optional
        Reward score + message from the last submit action (None if no submit yet).
    """
    parts = []
    lib_block = _format_library_block(library_functions)
    if lib_block:
        parts.append(lib_block)
    parts.append(f"Task:\n{task}")
    if last_result is not None:
        parts.append(f"\nExecution output:\n{last_result}")
    if verifier_feedback is not None:
        parts.append(f"\nVerifier:\n{verifier_feedback}")
    parts.append("\nNext action:")
    return "\n\n".join(parts)
