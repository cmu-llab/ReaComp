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

_COMPLEX_HINTS = {
    "bfs", "dfs", "backtrack", "backtracking", "recursion", "recursive",
    "search", "enumerate", "simulate", "sequence", "steps", "moves", "path",
}


def _bcr_max_tokens(task_spec) -> int:
    """Scale BCR output budget up for tasks requiring complex solution code."""
    if task_spec is None:
        return 2048
    words = {w for hint in task_spec.operation_hints for w in hint.lower().split()}
    return 4096 if words & _COMPLEX_HINTS else 2048

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
7. Return the answer value directly (e.g. "42", not "The answer is 42").
   Minimal, clean answer strings avoid partial-credit penalties from scoring functions.
8. For question/prompt-based tasks where a library function directly applies: use
   action=direct. Read the question, extract the concrete input value with your
   understanding, and return the answer. E.g. for a Caesar cipher question you read
   the cipher text from the question and apply caesar_decrypt mentally — no code needed.
   Use action=solve only when a reusable algorithmic function is genuinely needed
   (e.g. the task involves structured input like a list, grid, or graph).

For action=direct (question/prompt tasks — answer derived by applying library function):
{
  "action": "direct",
  "answer": "<the answer value as a clean minimal string>",
  "reasoning": "<how you derived it, which library function you applied and to what input>",
  "functions_used": ["<library function names conceptually applied>"]
}

For action=solve (algorithmic tasks — reusable function over structured input):
{
  "action": "solve",
  "code": "<complete Python function definition that solves the task>",
  "reasoning": "<step-by-step explanation>",
  "functions_used": ["<library function names called in code>"]
}

For action=decompose (task too complex for a direct solution):
{
  "action": "decompose",
  "subtasks": [{"description": "<sub-task>", "input": "<input description>"}],
  "composition_plan": "<how to combine sub-task results into the final answer>"
}
"""


def _format_reward_history(history: list) -> str:
    lines = ["Previous attempts (most recent last):"]
    for h in history[-3:]:
        lines.append(
            f"  iter={h['iteration']}  reward={h.get('reward', 0.0):.3f}  blame={h.get('blame', '?')}\n"
            f"  feedback: {h.get('message', '')}\n"
            f"  approach: {h.get('solution_summary', '')}"
        )
    return "\n".join(lines)


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

        # Show full code for functions SSL flagged as active + top retrieved matches.
        # Everything else is a compact one-liner to keep the prompt short.
        full_code_names = list({f.name for f in relevant} | set(active_funcs))

        user_msg = (
            f"Task type: {task_type}\n"
            f"Task: {task}\n\n"
            f"{spec_block}"
            f"{library.format_for_prompt(full_code_for=full_code_names)}\n\n"
            f"Suggested active functions: {active_str}\n\n"
            "Solve the task directly using library functions, or decompose if necessary."
        )

        reward_history = state.get("reward_history", [])
        if reward_history:
            history_block = _format_reward_history(reward_history)
            user_msg += (
                f"\n\n--- Fix mode (attempt {len(reward_history) + 1}) ---\n"
                f"{history_block}\n\n"
                "Previous solutions scored below 1.0. Try a DIFFERENT approach: "
                "fix the logic error, use a different algorithm, or correct the output format. "
                "Return the answer as a minimal clean value (e.g. '42', not 'The answer is 42')."
            )

        result = self.client.create(
            model=self.model,
            max_tokens=_bcr_max_tokens(task_spec),
            system=_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tag="bcr",
        )

        action = result.get("action", "")

        if action == "direct":
            answer = result.get("answer")
            if answer is None:
                logger.warning("BCR: direct missing answer, skipping. keys: %s", list(result.keys()))
                return state

            for fname in result.get("functions_used", []):
                func = library.get(fname)
                if func:
                    cost_tracker.record_reuse(func)

            reasoning = result.get("reasoning", "")
            state["solution"] = {
                "action": "direct",
                "answer": str(answer),
                "reasoning": reasoning,
                "functions_used": result.get("functions_used", []),
            }
            state["solved"] = True
            state["trace"].append({
                "step": state["steps"],
                "agent": "BCR",
                "action": "direct",
                "reasoning": reasoning,
            })
            logger.info("BCR: direct answer=%s using %s", answer, result.get("functions_used", []))

        elif action == "solve":
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
