"""SSL (Symbolic Search and Library) Agent.

Decides whether to reuse existing functions, compose them,
or create a new one.  Prefers reuse over invention.

Responds with a plain JSON object — no tool calling.
"""

import logging
import re
from typing import Dict, Optional

from .costs import CostTracker
from .executor import safe_exec
from .library import FunctionLibrary
from .models import Function
from .task_parser import TaskSpec

logger = logging.getLogger(__name__)

_COMPLEX_HINTS = {
    "bfs", "dfs", "backtrack", "backtracking", "recursion", "recursive",
    "search", "enumerate", "simulate", "sequence", "steps", "moves", "path",
}


def _ssl_max_tokens(task_spec) -> int:
    """Scale SSL output budget up for tasks that require complex function bodies."""
    if task_spec is None:
        return 2048
    words = {w for hint in task_spec.operation_hints for w in hint.lower().split()}
    return 4096 if words & _COMPLEX_HINTS else 2048

_SYSTEM = """\
You are the SSL (Symbolic Search and Library) agent in a symbolic reasoning system.
Your ONLY job is to maintain a shared function library used across tasks.

Rules:
1. FIRST check whether an existing function already covers the need → action=reuse.
2. If two or more existing functions can be combined → action=compose.
3. Create a brand-new function ONLY when nothing else works → action=create.
4. Keep functions short, general, and reusable across many tasks in the same domain.
5. Functions must be pure Python with type annotations and no external imports.
6. Never create a function that duplicates an existing one.
7. For create and compose always include code, domain, input_types, and output_type.

Respond with exactly this JSON structure:
{
  "action": "reuse" | "compose" | "create",
  "name": "<snake_case function name — existing name for reuse, new name for create/compose>",
  "code": "<complete Python def with type annotations — create/compose only>",
  "description": "<one-line docstring>",
  "uses": ["<existing_fn>"],
  "domain": "<e.g. list_manipulation>",
  "input_types": ["<e.g. list[int]>"],
  "output_type": "<e.g. list[int]>"
}
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
        relevant_names = [f.name for f in relevant]

        spec_block = ""
        if task_spec:
            spec_block = (
                f"Task domain: {task_spec.domain}\n"
                f"Input types: {task_spec.input_types}\n"
                f"Output type: {task_spec.output_type}\n"
                f"Operation hints: {task_spec.operation_hints}\n"
                f"Symbolic input example: {task_spec.symbolic_inputs}\n\n"
            )

        # Relevant functions are shown with full code; the rest appear as compact one-liners.
        relevant_note = (
            f"Functions shown in full ({', '.join(relevant_names)}) are the closest matches "
            "to this task — check these first before creating anything new."
            if relevant_names else "Library is empty — create a new function."
        )

        user_msg = (
            f"Task type: {task_type}\n"
            f"Task: {task}\n\n"
            f"{spec_block}"
            f"{library.format_for_prompt(full_code_for=relevant_names)}\n\n"
            f"{relevant_note}\n\n"
            "Decide: reuse an existing function, compose existing functions, "
            "or create a new one. Prefer reuse > compose > create."
        )

        result = self.client.create(
            model=self.model,
            max_tokens=_ssl_max_tokens(task_spec),
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tag="ssl",
        )

        actions: list = []
        active_functions: list = []

        action = result.get("action", "")
        # Accept function_name as a fallback for name (model habit from old schema)
        name = result.get("name") or result.get("function_name", "")

        if action == "reuse":
            func = library.get(name)
            if func:
                cost_tracker.record_reuse(func)
                active_functions.append(func.name)
                actions.append({"action": "reuse", "function": func.name, "reasoning": result.get("reasoning", "")})
                logger.info("SSL: reusing '%s'", func.name)
            else:
                logger.warning("SSL: reuse requested for unknown function '%s'", name)

        elif action in ("create", "compose"):
            code = result.get("code", "")
            # Fallback: scan all string values for a Python def if code field is empty
            if not code:
                for v in result.values():
                    if isinstance(v, str) and re.search(r"\bdef\s+\w+\s*\(", v):
                        code = v
                        logger.info("SSL: extracted code from non-standard field")
                        break
            # Infer name from code if still missing
            if not name and code:
                m = re.search(r"\bdef\s+(\w+)\s*\(", code)
                if m:
                    name = m.group(1)
                    logger.info("SSL: inferred name=%r from code", name)

            if not name or not code:
                logger.warning("SSL: %s action missing name/code, skipping. keys: %s", action, list(result.keys()))
            else:
                new_func = Function(
                    name=name,
                    description=result.get("description", ""),
                    code=code,
                    domain=result.get("domain") or (task_spec.domain if task_spec else "general"),
                    input_types=result.get("input_types") or (task_spec.input_types if task_spec else []),
                    output_type=result.get("output_type") or (task_spec.output_type if task_spec else ""),
                )
                ok, _, err = safe_exec(new_func.code)
                if not ok:
                    logger.warning("SSL: generated function '%s' has errors: %s", new_func.name, err)

                library.add(new_func)
                cost_tracker.record_new_function(new_func)
                active_functions.append(new_func.name)
                actions.append({"action": action, "function": new_func.name})

                if action == "compose":
                    for used_name in result.get("uses", []):
                        used = library.get(used_name)
                        if used:
                            cost_tracker.record_reuse(used)

                logger.info("SSL: %s '%s' [%s]", action, new_func.name, new_func.domain)

        else:
            logger.warning("SSL: unrecognised action %r, skipping. keys: %s", action, list(result.keys()))

        state["working_memory"] = {"active_functions": active_functions}
        state["trace"].append({"step": state["steps"], "agent": "SSL", "actions": actions})
        return state
