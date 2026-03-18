"""SSL (Symbolic Search and Library) Agent.

Decides whether to reuse existing functions, compose them,
or create a new one.  Prefers reuse over invention.

Receives a TaskSpec so it can:
  - retrieve functions by domain affinity and type matching
  - tag newly created functions with domain and I/O types
"""

import logging
from typing import Dict, Optional

from .costs import CostTracker
from .executor import safe_exec
from .library import FunctionLibrary
from .models import Function
from .task_parser import TaskSpec

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
                        "Complete Python function definition with type annotations. "
                        "Must be self-contained (no external imports)."
                    ),
                },
                "domain": {
                    "type": "string",
                    "description": "Task domain this function belongs to (e.g. list_manipulation).",
                },
                "input_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Python types of the parameters, e.g. ['list[int]'].",
                },
                "output_type": {
                    "type": "string",
                    "description": "Python return type, e.g. 'list[int]'.",
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
                "domain": {"type": "string"},
                "input_types": {"type": "array", "items": {"type": "string"}},
                "output_type": {"type": "string"},
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
4. Keep functions short, general, and reusable across many tasks in the same domain.
5. Functions must be pure Python with type annotations and no external imports.
6. Never create a function that duplicates an existing one.
7. When creating or composing, include domain, input_types, and output_type.
"""


class SSLAgent:
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

        relevant = library.retrieve_relevant(str(task), task_spec=task_spec, top_k=5)
        relevant_str = (
            "\n".join(f"- {f.name} [{f.domain}]: {f.description}" for f in relevant)
            if relevant
            else "None found."
        )

        spec_block = ""
        if task_spec:
            spec_block = (
                f"Task domain: {task_spec.domain}\n"
                f"Input types: {task_spec.input_types}\n"
                f"Output type: {task_spec.output_type}\n"
                f"Operation hints: {task_spec.operation_hints}\n"
                f"Symbolic input example: {task_spec.symbolic_inputs}\n\n"
            )

        user_msg = (
            f"Task type: {task_type}\n"
            f"Task: {task}\n\n"
            f"{spec_block}"
            f"{library.format_for_prompt()}\n\n"
            f"Most relevant existing functions (domain + type matched):\n{relevant_str}\n\n"
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
                # Fall back to task_spec for domain/types if the model didn't fill them
                new_func = Function(
                    name=inp["name"],
                    description=inp["description"],
                    code=inp["code"],
                    domain=inp.get("domain") or (task_spec.domain if task_spec else "general"),
                    input_types=inp.get("input_types") or (task_spec.input_types if task_spec else []),
                    output_type=inp.get("output_type") or (task_spec.output_type if task_spec else ""),
                )

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

                logger.info("SSL: %s '%s' [%s]", name, new_func.name, new_func.domain)

        state["working_memory"] = {"active_functions": active_functions}
        state["trace"].append({"step": state["steps"], "agent": "SSL", "actions": actions})
        return state
