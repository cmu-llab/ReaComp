"""SSL (Symbolic Search and Library) Agent.

Decides whether to reuse existing functions, compose them,
or create a new one.  Prefers reuse over invention.
"""

import logging
from typing import Dict

from .costs import CostTracker
from .executor import safe_exec
from .library import FunctionLibrary
from .models import Function

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "reuse_function",
        "description": (
            "Indicate that an existing library function is sufficient for the current task. "
            "Use this whenever a function already exists that can help."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function_name": {
                    "type": "string",
                    "description": "Name of the existing function to reuse.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this function is relevant to the task.",
                },
            },
            "required": ["function_name", "reasoning"],
        },
    },
    {
        "name": "create_function",
        "description": (
            "Create a brand-new library function that does not yet exist. "
            "Only use this when no existing function (or composition) can help."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "snake_case function name."},
                "description": {"type": "string", "description": "One-line docstring."},
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Python function definition. "
                        "Must be self-contained (no external imports)."
                    ),
                },
            },
            "required": ["name", "description", "code"],
        },
    },
    {
        "name": "compose_functions",
        "description": (
            "Build a new function by composing two or more existing library functions. "
            "Prefer this over create_function when existing pieces can be combined."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "code": {
                    "type": "string",
                    "description": "Python function that calls the listed existing functions.",
                },
                "uses": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of existing library functions this composition calls.",
                },
            },
            "required": ["name", "description", "code", "uses"],
        },
    },
]

_SYSTEM = """\
You are the SSL (Symbolic Search and Library) agent in a symbolic reasoning system.

Your ONLY job is to maintain a shared function library used across tasks.

Rules:
1. FIRST check whether an existing function already covers the need → call reuse_function.
2. If two or more existing functions can be combined → call compose_functions.
3. Create a brand-new function ONLY when nothing else works → call create_function.
4. Keep functions short, general, and reusable across many tasks.
5. Functions must be pure Python with no external imports.
6. Never create a function that duplicates an existing one.
"""


class SSLAgent:
    def __init__(self, client, model: str = "claude-sonnet-4-5"):
        self.client = client
        self.model = model

    def run(self, state: Dict, library: FunctionLibrary, cost_tracker: CostTracker) -> Dict:
        task = state["task_input"]
        task_type = state["task_type"]

        relevant = library.retrieve_relevant(str(task), top_k=5)
        relevant_str = (
            "\n".join(f"- {f.name}: {f.description}" for f in relevant)
            if relevant
            else "None found."
        )

        user_msg = (
            f"Task type: {task_type}\n"
            f"Task: {task}\n\n"
            f"{library.format_for_prompt()}\n\n"
            f"Most relevant existing functions:\n{relevant_str}\n\n"
            "Decide: reuse an existing function, compose existing functions, "
            "or create a new one.  Prefer reuse > compose > create."
        )

        response = self.client.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=_TOOLS,
            tool_choice={"type": "any"},
        )

        actions: list = []
        active_functions: list = []

        for block in response.content:
            if block.type != "tool_use":
                continue

            name = block.name
            inp = block.input

            if name == "reuse_function":
                func = library.get(inp["function_name"])
                if func:
                    cost_tracker.record_reuse(func)
                    active_functions.append(func.name)
                    actions.append({"action": "reuse", "function": func.name, "reasoning": inp["reasoning"]})
                    logger.info("SSL: reusing '%s'", func.name)
                else:
                    logger.warning("SSL: reuse requested for unknown function '%s'", inp["function_name"])

            elif name in ("create_function", "compose_functions"):
                new_func = Function(
                    name=inp["name"],
                    description=inp["description"],
                    code=inp["code"],
                )

                # Validate syntax before adding
                ok, _, err = safe_exec(new_func.code)
                if not ok:
                    logger.warning("SSL: generated function '%s' has errors: %s", new_func.name, err)

                library.add(new_func)
                cost_tracker.record_new_function(new_func)
                active_functions.append(new_func.name)
                actions.append({"action": name, "function": new_func.name})

                if name == "compose_functions":
                    for used_name in inp.get("uses", []):
                        used = library.get(used_name)
                        if used:
                            cost_tracker.record_reuse(used)

                logger.info("SSL: %s '%s'", name, new_func.name)

        state["working_memory"] = {"active_functions": active_functions}
        state["trace"].append({"step": state["steps"], "agent": "SSL", "actions": actions})
        return state
