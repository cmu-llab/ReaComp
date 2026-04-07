"""
Prompts for the ReAct + Memory baseline.

Implements the ReAct framework (Yao et al. 2022) adapted for programming tasks:
  Thought → Action (code | finish) → Observation → Thought → ...

The memory component injects retrieved similar past solutions as few-shot
context, inspired by the memory module in Hypothetical Minds (Cross et al. 2024).
"""

from typing import Any, Dict, List, Optional


_SYSTEM = """\
You are a ReAct agent that solves programming and reasoning tasks.

You interleave Thought (reasoning) and Action (code execution or final answer).
At each step, respond with a JSON object in one of these two formats:

Format A — write and execute code:
{
  "thought": "My reasoning about what to try next",
  "action_type": "execute_code",
  "code": "# Python code to run\\nresult = ...\\nprint(result)"
}

Format B — provide the final answer:
{
  "thought": "I now have the correct answer",
  "action_type": "finish",
  "answer": "<exact final answer>"
}

Rules:
- Use action_type="execute_code" to write Python code and see its output.
- Your code runs in a clean Python namespace each step (no state persists between steps).
- To output a result, use print().
- Use action_type="finish" ONLY when you are confident in the answer.
- The answer must be the exact value requested, not a sentence.
- Do not output prose outside the JSON.
"""


def _format_memory_examples(examples: List[Dict]) -> str:
    if not examples:
        return ""
    lines = ["--- Similar past solutions (for reference) ---"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"\nExample {i} (reward={ex['reward']:.2f}):")
        lines.append(f"Task: {ex['task'][:300]}")
        if ex.get("code"):
            lines.append(f"Code:\n```python\n{ex['code'][:600]}\n```")
        if ex.get("answer") is not None:
            lines.append(f"Answer: {ex['answer']}")
    lines.append("--- End of examples ---\n")
    return "\n".join(lines)


def build_initial_prompt(
    task: str,
    memory_examples: List[Dict],
    step: int = 0,
) -> str:
    """Build the user message for the first ReAct step."""
    mem_block = _format_memory_examples(memory_examples)
    return (
        f"{mem_block}"
        f"Task:\n{task}\n\n"
        "Step 1: Think about how to solve this task and write code to start.\n"
        "Remember: print() your result so you can see it."
    )


def build_followup_prompt(
    task: str,
    history: List[Dict],
    step: int,
    reward_feedback: Optional[str] = None,
) -> str:
    """
    Build the user message for subsequent ReAct steps.

    history: list of {"thought": ..., "action_type": ..., "code": ..., "observation": ...}
    """
    lines = [f"Task:\n{task}\n"]

    for i, h in enumerate(history, 1):
        lines.append(f"Step {i}:")
        lines.append(f"  Thought: {h['thought']}")
        if h["action_type"] == "execute_code":
            code_preview = h.get("code", "")[:400]
            lines.append(f"  Action: execute_code\n  Code:\n```python\n{code_preview}\n```")
            obs = h.get("observation", "")
            lines.append(f"  Observation: {obs[:500] if obs else '(no output)'}")
        elif h["action_type"] == "finish":
            lines.append(f"  Action: finish → {h.get('answer', '')}")

    if reward_feedback:
        lines.append(f"\nFeedback on previous answer: {reward_feedback}")

    lines.append(f"\nStep {step + 1}: Continue reasoning. "
                 "Execute more code or provide the final answer.")
    return "\n".join(lines)
