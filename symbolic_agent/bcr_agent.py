"""BCR (Bottom-up Conceptual Reasoning) Agent.

Attempts to solve the current task using available library functions.
If that is not possible, decomposes the task into simpler sub-problems.

Receives a TaskSpec so it can present the symbolic input representation
to the LLM alongside the NL description.
"""

import logging
import re
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
        "description": "Solve the task with library functions (action='solve'), or decompose into sub-tasks (action='decompose'). Prefer solve.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["solve", "decompose"],
                    "description": "solve: write Python code that solves the task. decompose: split into simpler sub-tasks.",
                },
                # --- solve fields ---
                "code": {
                    "type": "string",
                    "description": "Complete Python function definition that solves the task. May call library functions by name. No external imports.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step explanation of the solution approach.",
                },
                "functions_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of library functions called inside code.",
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
                    "description": "Ordered list of sub-tasks to solve independently.",
                },
                "composition_plan": {
                    "type": "string",
                    "description": "How to combine the sub-task results into the final answer.",
                },
            },
            "required": ["action", "code"],
        },
    },
]

_SYSTEM = """\
You are the BCR (Bottom-up Conceptual Reasoning) agent in a symbolic reasoning system.

Your job is to solve symbolic reasoning tasks using the shared function library.

Rules:
1. PREFER solving directly with existing library functions.
2. Your code must call at least one library function when the library is non-empty.
3. Use the symbolic input representation to understand the exact data structures involved.
4. Only decompose if the task is genuinely too complex for a direct solution.
5. Keep solutions concise. Avoid reimplementing what's already in the library.
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
            tag="bcr",
        )

        for block in response.content:
            if block.type != "tool_use":
                continue

            inp = block.input
            action = inp.get("action", "solve")

            if action == "solve":
                # Accept 'code' (new schema) or 'solution_code' (old schema / model habit)
                solution_code = inp.get("code") or inp.get("solution_code", "")
                # Extract entry-point name from the def statement — never require the model to repeat it
                solution_function = inp.get("solution_function", "")
                if not solution_function and solution_code:
                    m = re.search(r'\bdef\s+(\w+)\s*\(', solution_code)
                    if m:
                        solution_function = m.group(1)

                if not solution_code or not solution_function:
                    logger.warning(
                        "BCR: solve action missing code, skipping. inp keys: %s", list(inp.keys()),
                    )
                    continue

                for fname in inp.get("functions_used", []):
                    func = library.get(fname)
                    if func:
                        cost_tracker.record_reuse(func)

                reasoning = inp.get("reasoning", "")
                state["solution"] = {
                    "code": solution_code,
                    "function": solution_function,
                    "reasoning": reasoning,
                    "functions_used": inp.get("functions_used", []),
                }
                state["solved"] = True
                state["trace"].append({
                    "step": state["steps"],
                    "agent": "BCR",
                    "action": "solve",
                    "reasoning": reasoning,
                })
                logger.info("BCR: solved using %s", inp.get("functions_used", []))

            elif action == "decompose":
                if not inp.get("subtasks"):
                    logger.warning(
                        "BCR: decompose action missing subtasks, skipping. inp keys: %s",
                        list(inp.keys()),
                    )
                    continue

                state["working_memory"] = {
                    "subtasks": inp["subtasks"],
                    "composition_plan": inp.get("composition_plan", ""),
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
