"""BCR (Bottom-up Conceptual Reasoning) Agent.

Attempts to solve the current task using available library functions.
If that is not possible, decomposes the task into simpler sub-problems.

Responds with a plain JSON object — no tool calling.
"""

import logging
import re
from typing import Dict, Optional

from .costs import CostTracker
from .library import FunctionLibrary
from .task_parser import TaskSpec

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are the BCR (Bottom-up Conceptual Reasoning) agent in a symbolic reasoning system.
Your job is to solve symbolic reasoning tasks using the shared function library.

Rules:
1. PREFER solving directly with existing library functions.
2. Your code must call at least one library function when the library is non-empty.
3. Use the symbolic input representation to understand the exact data structures involved.
4. Only decompose if the task is genuinely too complex for a direct solution.
5. Keep solutions concise. Avoid reimplementing what is already in the library.
6. All code must be pure Python (no external imports).

For action=solve, respond with:
{
  "action": "solve",
  "code": "<complete Python function definition that solves the task>",
  "reasoning": "<step-by-step explanation>",
  "functions_used": ["<library function names called in code>"]
}

For action=decompose, respond with:
{
  "action": "decompose",
  "subtasks": [{"description": "<sub-task>", "input": "<input description>"}],
  "composition_plan": "<how to combine sub-task results into the final answer>"
}
"""


class BCRAgent:
    def __init__(self, client, model: str = "claude-sonnet-4-5"):
        self.client = client
        self.model = model

    def run(
        self,
        state: Dict,
        library: FunctionLibrary,
        cost_tracker: CostTracker,
        task_spec: Optional[TaskSpec] = None,
    ) -> Dict:
        task = state["task_input"]
        task_type = state["task_type"]
        working_memory = state.get("working_memory") or {}
        active_funcs = working_memory.get("active_functions", [])
        active_str = ", ".join(active_funcs) if active_funcs else "none suggested"

        spec_block = ""
        if task_spec:
            spec_block = (
                f"Task domain: {task_spec.domain}\n"
                f"Input types: {task_spec.input_types}  →  output: {task_spec.output_type}\n"
                f"Symbolic input example:\n  {task_spec.symbolic_inputs}\n"
                f"Operation hints: {task_spec.operation_hints}\n\n"
            )

        relevant = library.retrieve_relevant(str(task), task_spec=task_spec, top_k=5)
        relevant_str = (
            "\n".join(f"- {f.name} [{f.domain}]: {f.description}" for f in relevant)
            if relevant else "none"
        )

        user_msg = (
            f"Task type: {task_type}\n"
            f"Task: {task}\n\n"
            f"{spec_block}"
            f"{library.format_for_prompt()}\n\n"
            f"Domain/type-matched suggestions: {relevant_str}\n"
            f"Suggested active functions: {active_str}\n\n"
            "Solve the task directly using library functions, or decompose if necessary."
        )

        result = self.client.create(
            model=self.model,
            max_tokens=2048,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tag="bcr",
        )

        action = result.get("action", "")

        if action == "solve":
            # Accept both 'code' (current schema) and 'solution_code' (model habit)
            code = result.get("code") or result.get("solution_code", "")
            # Infer entry-point name from the def statement — never require model to repeat it
            func_name = result.get("solution_function", "")
            if not func_name and code:
                m = re.search(r"\bdef\s+(\w+)\s*\(", code)
                if m:
                    func_name = m.group(1)

            if not code or not func_name:
                logger.warning("BCR: solve missing code, skipping. keys: %s", list(result.keys()))
                return state

            for fname in result.get("functions_used", []):
                func = library.get(fname)
                if func:
                    cost_tracker.record_reuse(func)

            reasoning = result.get("reasoning", "")
            state["solution"] = {
                "code": code,
                "function": func_name,
                "reasoning": reasoning,
                "functions_used": result.get("functions_used", []),
            }
            state["solved"] = True
            state["trace"].append({
                "step": state["steps"],
                "agent": "BCR",
                "action": "solve",
                "reasoning": reasoning,
            })
            logger.info("BCR: solved using %s", result.get("functions_used", []))

        elif action == "decompose":
            subtasks = result.get("subtasks", [])
            if not subtasks:
                logger.warning("BCR: decompose missing subtasks, skipping. keys: %s", list(result.keys()))
                return state

            state["working_memory"] = {
                "subtasks": subtasks,
                "composition_plan": result.get("composition_plan", ""),
                "active_functions": active_funcs,
            }
            state["trace"].append({
                "step": state["steps"],
                "agent": "BCR",
                "action": "decompose",
                "subtasks": [s["description"] for s in subtasks],
            })
            logger.info("BCR: decomposed into %d subtasks", len(subtasks))

        else:
            logger.warning("BCR: unrecognised action %r, skipping. keys: %s", action, list(result.keys()))

        return state
