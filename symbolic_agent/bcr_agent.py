"""BCR (Bottom-up Conceptual Reasoning) Agent.

Attempts to solve the current task using available library functions.
If that is not possible, decomposes the task into simpler sub-problems.

Receives a TaskSpec so it can present the symbolic input representation
to the LLM alongside the NL description.
"""

import logging
from typing import Dict, Optional

from .costs import CostTracker
from .library import FunctionLibrary
from .task_parser import TaskSpec

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "bcr_action",
        "description": (
            "Either solve the task directly with library functions (action='solve'), "
            "or break it into simpler sub-tasks (action='decompose'). "
            "Prefer solve over decompose."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["solve", "decompose"],
                    "description": (
                        "solve: provide a complete Python solution using library functions. "
                        "decompose: split into ordered sub-tasks when a direct solution is not possible."
                    ),
                },
                # --- solve fields ---
                "solution_code": {
                    "type": "string",
                    "description": (
                        "(solve) Complete Python code defining a function that solves the task. "
                        "May call library functions by name. No external imports."
                    ),
                },
                "solution_function": {
                    "type": "string",
                    "description": "(solve) Name of the entry-point function in solution_code.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "(solve) Step-by-step explanation of the solution.",
                },
                "functions_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "(solve) Names of library functions called by the solution.",
                },
                # --- decompose fields ---
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
                    "description": "(decompose) Ordered list of sub-tasks.",
                },
                "composition_plan": {
                    "type": "string",
                    "description": "(decompose) How sub-task results combine to solve the main task.",
                },
            },
            "required": ["action"],
        },
    },
]

_SYSTEM = """\
You are the BCR (Bottom-up Conceptual Reasoning) agent in a symbolic reasoning system.

Your job is to solve symbolic reasoning tasks using the shared function library.

Rules:
1. PREFER solving directly with existing library functions.
2. Your solution_code must call at least one library function.
3. Use the symbolic input representation to understand the exact data structures involved.
4. Only call decompose_task if the task is genuinely too complex for a direct solution.
5. Keep solutions concise.  Avoid reimplementing what's already in the library.
6. All code must be pure Python (no external imports).
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

        # Use domain-aware retrieval for the suggested functions section
        relevant = library.retrieve_relevant(str(task), task_spec=task_spec, top_k=5)
        relevant_str = (
            "\n".join(f"- {f.name} [{f.domain}]: {f.description}" for f in relevant)
            if relevant
            else "none"
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

            inp = block.input
            action = inp.get("action", "solve")

            if action == "solve":
                for fname in inp.get("functions_used", []):
                    func = library.get(fname)
                    if func:
                        cost_tracker.record_reuse(func)

                state["solution"] = {
                    "code": inp["solution_code"],
                    "function": inp["solution_function"],
                    "reasoning": inp["reasoning"],
                    "functions_used": inp.get("functions_used", []),
                }
                state["solved"] = True
                state["trace"].append({
                    "step": state["steps"],
                    "agent": "BCR",
                    "action": "solve",
                    "reasoning": inp["reasoning"],
                })
                logger.info("BCR: solved using %s", inp.get("functions_used", []))

            elif action == "decompose":
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
