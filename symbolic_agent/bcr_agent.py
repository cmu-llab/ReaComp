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
    """Scale BCR output budget up for tasks requiring complex solution code.

    Reasoning models (e.g. gpt-oss-120b) consume their chain-of-thought tokens
    from the same max_tokens budget as the final output.  Long CoTs on cipher /
    arithmetic tasks can easily exceed 4k tokens, leaving nothing for the JSON
    response.  The higher baselines here give the model headroom to write its
    response after reasoning.
    """
    if task_spec is None:
        return 4096
    words = {w for hint in task_spec.operation_hints for w in hint.lower().split()}
    return 8192 if words & _COMPLEX_HINTS else 4096

_PATCH_SYSTEM = """\
You are a reasoning assistant. A symbolic solver attempted a task but only reached a
partial solution after exhausting all retries.

Your job:
1. Study the task carefully.
2. Review the best partial answer the symbolic solver produced and the full reward
   feedback explaining what is wrong or incomplete.
3. Starting from that partial answer, reason through the task and produce the CORRECT
   final answer. Correct or complete it — do not start from scratch unless the
   symbolic answer is entirely wrong.

You are NOT writing code. You are NOT using library functions.
The "answer" field must contain ONLY the exact final answer value — no preamble,
no trailing punctuation, no explanation text.

Return exactly this JSON:
{"answer": "<exact final answer>"}
"""

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
   NAMESPACE RULE: All library functions are available directly by name in your solve
   function — they share the same execution namespace. NEVER import them: do NOT write
   `from __main__ import fn`, `from fn import fn`, or re-implement a function that is
   already in the library. Just call it: `result = fn(args)`.
7. Return the answer value directly (e.g. "42", not "The answer is 42").
   Minimal, clean answer strings avoid partial-credit penalties from scoring functions.
8. For question/prompt-based tasks (any task presented as a "Question:" or natural-
   language prompt), ALWAYS use action=direct — even when the answer requires tracing
   through an algorithm. Use action=direct, work through the computation in your
   reasoning, and return the final answer. NEVER use action=solve for Q&A tasks.
   action=solve is ONLY for tasks where the input is structured data (list, grid,
   dict, graph) that a reusable function should operate over. A tell-tale sign you
   should use action=direct instead of action=solve: if you would write a solve
   function that hardcodes the starting value from the task, that is not a reusable
   function — use action=direct instead.
   Two strict sub-rules for direct mode:
   a. EXTRACT ONLY THE FUNCTION INPUT: isolate exactly the data value the library
      function operates on — strip away instructions, preamble, and trailing directives.
      Pass only the raw input argument the function needs, nothing more.
   b. RETURN EXACTLY WHAT THE FUNCTION PRODUCES: no trailing punctuation, no added
      words, no capitalisation changes, no preamble. Return the function's output
      verbatim.
9. When the active function accepts task-specific parameters (e.g. a list of rules,
   patterns, or mappings), extract those values from the task description and pass
   them as arguments in your solve code. Do NOT write a helper function that hardcodes
   those values and delegates to the generic function — that is a one-use wrapper that
   does not belong in the library. Keep task-specific constants local to your solve
   function only.

For action=direct (question/prompt tasks — answer derived by applying library function):
{
  "action": "direct",
  "answer": "<exact output the library function produces — no added punctuation or words>",
  "reasoning": "<which library function applied, what exact input was extracted and why>",
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
    """Format the last 2 reward iterations as a compact block.

    Kept deliberately short: reward history is injected into every BCR retry
    prompt and grows with each iteration.  Long feedback messages compound the
    token budget problem on reasoning models where CoT tokens count against
    max_tokens.
    """
    lines = ["Previous attempts (most recent last):"]
    for h in history[-2:]:
        msg = h.get("message", "")[:120]
        approach = h.get("solution_summary", "")[:80]
        lines.append(
            f"  iter={h['iteration']}  reward={h.get('reward', 0.0):.3f}  blame={h.get('blame', '?')}\n"
            f"  feedback: {msg}\n"
            f"  approach: {approach}"
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

        # reasoning_gym task_input carries both `question` and `prompt`, where
        # `prompt` repeats the question plus a few-shot example (~600 extra tokens
        # the model doesn't need for a direct-answer task).  Show only `question`
        # when present; PBEBench tasks only have `prompt` so they fall through.
        if isinstance(task, dict) and "question" in task:
            task_display = task["question"]
        else:
            task_display = task

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
            f"Task: {task_display}\n\n"
            f"{spec_block}"
            f"{library.format_for_prompt(full_code_for=full_code_names)}\n\n"
            f"Suggested active functions: {active_str}\n"
            f"[Library budget] {cost_tracker.budget_summary()}\n\n"
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

    def patch_solve(
        self,
        task_input,
        best_answer,
        reward_history: list,
        task_spec=None,
    ):
        """Neural patch: one final stripped-down LLM call after all symbolic iterations fail.

        No library context, no active functions — pure reasoning starting from the best
        partial symbolic answer.  Uses a large token budget so the model can reason at
        length in its chain-of-thought without being cut off.

        Returns {"answer": str, "reasoning": str} or None on parse failure.
        The "reasoning" field is populated from the model's chain-of-thought
        (reasoning_content) captured in the task log rather than from the JSON response,
        so no extra output tokens are spent on it.
        """
        # Show only the question for reasoning_gym tasks (same logic as run())
        if isinstance(task_input, dict) and "question" in task_input:
            task_display = task_input["question"]
        else:
            task_display = task_input

        best_reward = max((h["reward"] for h in reward_history), default=0.0)

        history_lines = ["Reward history from symbolic iterations:"]
        for h in reward_history:
            history_lines.append(
                f"  iter={h['iteration']}  reward={h['reward']:.3f}  blame={h.get('blame', '?')}\n"
                f"  feedback: {h.get('message', '')}\n"
                f"  approach: {h.get('solution_summary', '')[:120]}"
            )

        user_msg = (
            f"Task:\n{task_display}\n\n"
            f"Best symbolic answer (reward={best_reward:.3f}):\n{best_answer}\n\n"
            f"{chr(10).join(history_lines)}\n\n"
            "Starting from the symbolic answer above, produce the correct final answer.\n"
            "Return ONLY the answer value in the 'answer' field."
        )

        result = self.client.create(
            model=self.model,
            max_tokens=16384,
            system=_PATCH_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tag="bcr_patch",
        )

        answer = result.get("answer")
        if answer is None:
            logger.warning("BCR patch: missing 'answer' in response. keys: %s", list(result.keys()))
            return None

        # Pull CoT from task log (reasoning_content for openai/vLLM reasoning models;
        # falls back to empty string on Anthropic where CoT is internal).
        reasoning = ""
        if self.client._task_log:
            last = self.client._task_log[-1]
            reasoning = last.get("response", {}).get("reasoning_content", "") or ""

        logger.info("BCR patch: answer=%s  cot_len=%d", str(answer)[:80], len(reasoning))
        return {"action": "patch", "answer": str(answer), "reasoning": reasoning}
