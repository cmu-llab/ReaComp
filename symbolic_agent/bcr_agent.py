"""BCR (Bottom-up Conceptual Reasoning) Agent.

Attempts to solve the current task using available library functions.
If that is not possible, decomposes the task into simpler sub-problems.
"""

import logging
from typing import Dict

from .costs import CostTracker
from .library import FunctionLibrary

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "solve_task",
        "description": (
            "Provide a complete Python solution for the task, using one or more "
            "library functions.  The solution_code must define a callable function."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "solution_code": {
                    "type": "string",
                    "description": (
                        "Complete Python code that defines a function solving the task. "
                        "May call library functions by name (they will be loaded first). "
                        "No external imports."
                    ),
                },
                "solution_function": {
                    "type": "string",
                    "description": "Name of the entry-point function in solution_code.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step explanation of the solution.",
                },
                "functions_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of library functions called by the solution.",
                },
            },
            "required": ["solution_code", "solution_function", "reasoning", "functions_used"],
        },
    },
    {
        "name": "decompose_task",
        "description": (
            "Break the task into simpler sub-tasks when a direct solution is not possible. "
            "Use this only if the task genuinely requires intermediate steps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "input": {"type": "string"},
                        },
                        "required": ["description", "input"],
                    },
                    "description": "Ordered list of sub-tasks.",
                },
                "composition_plan": {
                    "type": "string",
                    "description": "How results from sub-tasks combine to solve the main task.",
                },
            },
            "required": ["subtasks", "composition_plan"],
        },
    },
]

_SYSTEM = """\
You are the BCR (Bottom-up Conceptual Reasoning) agent in a symbolic reasoning system.

Your job is to solve symbolic reasoning tasks using the shared function library.

Rules:
1. PREFER solving directly with existing library functions.
2. Your solution_code must call at least one library function.
3. Only call decompose_task if the task is genuinely too complex for a direct solution.
4. Keep solutions concise.  Avoid reimplementing what's already in the library.
5. All code must be pure Python (no external imports).
"""


class BCRAgent:
    def __init__(self, client, model: str = "claude-sonnet-4-5"):
        self.client = client
        self.model = model

    def run(self, state: Dict, library: FunctionLibrary, cost_tracker: CostTracker) -> Dict:
        task = state["task_input"]
        task_type = state["task_type"]
        working_memory = state.get("working_memory") or {}
        active_funcs = working_memory.get("active_functions", [])
        active_str = ", ".join(active_funcs) if active_funcs else "none suggested"

        user_msg = (
            f"Task type: {task_type}\n"
            f"Task: {task}\n\n"
            f"{library.format_for_prompt()}\n\n"
            f"Suggested active functions: {active_str}\n\n"
            "Solve the task directly using library functions, or decompose if necessary."
        )

        response = self.client.create(
            model=self.model,
            max_tokens=2048,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=_TOOLS,
            tool_choice={"type": "any"},
        )

        for block in response.content:
            if block.type != "tool_use":
                continue

            name = block.name
            inp = block.input

            if name == "solve_task":
                for fname in inp.get("functions_used", []):
                    func = library.get(fname)
                    if func:
                        cost_tracker.record_reuse(func)

                state["solution"] = {
                    "code": inp["solution_code"],
                    "function": inp["solution_function"],
                    "reasoning": inp["reasoning"],
                    "functions_used": inp["functions_used"],
                }
                state["solved"] = True
                state["trace"].append({
                    "step": state["steps"],
                    "agent": "BCR",
                    "action": "solve",
                    "reasoning": inp["reasoning"],
                })
                logger.info("BCR: solved using %s", inp["functions_used"])

            elif name == "decompose_task":
                state["working_memory"] = {
                    "subtasks": inp["subtasks"],
                    "composition_plan": inp["composition_plan"],
                    "active_functions": active_funcs,
                }
                state["trace"].append({
                    "step": state["steps"],
                    "agent": "BCR",
                    "action": "decompose",
                    "subtasks": [s["description"] for s in inp["subtasks"]],
                })
                logger.info("BCR: decomposed into %d subtasks", len(inp["subtasks"]))

        return state
