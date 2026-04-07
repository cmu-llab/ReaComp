"""
Prompts for the ReAct + Memory baseline.

Stateless per-step context design: each LLM call receives a single
self-contained message with the current task, top-K retrieved memory
examples (similar past solutions), the last execution output, and the
latest verifier feedback.  No prior trajectory is passed; the model's
chain-of-thought (reasoning_content) serves as implicit state.

Two flat actions:
  execute — run Python code and observe output.
  submit  — propose a final answer; triggers immediate reward eval.
"""

from typing import Dict, List, Optional


_SYSTEM = """\
You are a programming agent that solves tasks by writing and testing Python code.

Output exactly one JSON action and nothing else:

Run code (use print() for output):
{"action": "execute", "code": "<python code>"}

Submit your final answer:
{"action": "submit", "answer": "<exact answer>"}

Rules:
- execute: use print() to emit output; that becomes your observation.
- submit: only when confident. Exact value requested, not prose.
- Output a single JSON object and nothing else.\
"""


def _format_memory_examples(examples: List[Dict]) -> str:
    if not examples:
        return ""
    lines = ["Similar past solutions (for reference):"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"  Example {i} (score={ex['reward']:.2f}): {ex['task'][:200]}")
        if ex.get("code"):
            lines.append(f"  ```python\n  {ex['code'][:400]}\n  ```")
        if ex.get("answer") is not None:
            lines.append(f"  Answer: {ex['answer']}")
    return "\n".join(lines)


def build_prompt(
    task: str,
    memory_examples: List[Dict],
    last_result: Optional[str] = None,
    verifier_feedback: Optional[str] = None,
) -> str:
    """
    Build a self-contained single-turn prompt for one ReAct step.

    Parameters
    ----------
    task : str
        The task description shown every step.
    memory_examples : List[Dict]
        Top-K retrieved similar past solutions (shown every step).
    last_result : str, optional
        Stdout / stderr from the last execute action (None on first step).
    verifier_feedback : str, optional
        Reward score + message from the last submit action.
    """
    parts = []
    mem_block = _format_memory_examples(memory_examples)
    if mem_block:
        parts.append(mem_block)
    parts.append(f"Task:\n{task}")
    if last_result is not None:
        parts.append(f"\nExecution output:\n{last_result}")
    if verifier_feedback is not None:
        parts.append(f"\nVerifier:\n{verifier_feedback}")
    parts.append("\nNext action:")
    return "\n\n".join(parts)
