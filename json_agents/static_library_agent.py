"""
JSON-mode Static Library Agent.

Mirrors openhands_agents/static_library but uses the existing LLMClient
(JSON output, no tool-calling) so it works with gpt-oss-120b.

Workflow per task (up to max_iters rounds):
  1. Show task + library function signatures + prompting guide.
  2. Model returns JSON: {"reasoning": "...", "code": "...", "answer": "..."}
     - code   : Python snippet using library functions (optional helper computation)
     - answer : candidate answer to submit
  3. Execute code in sandbox if provided.
  4. Score answer with reward_fn.
  5. If reward < 1.0 and iters remain, feed back the error trace + reward message
     and repeat.
"""

import ast
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbolic_agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

# No markdown fences — gpt-oss-120b's chat template treats triple backticks as
# structural delimiters and will cause "Expected 2 output messages, got N" errors.
_SYSTEM = """\
You are an expert Python programmer solving program-synthesis tasks.
You have access to a pre-built library of helper functions. All functions
are already available in the execution namespace — do NOT write import
statements for them.

On each turn you will receive:
  - The task description
  - The available library functions (names + signatures)
  - Optionally: feedback from a previous attempt (reward score + message)

You must respond with a JSON object with these fields:
  "reasoning" : string  — your step-by-step reasoning (think aloud)
  "code"       : string  — Python snippet using library functions to explore
                           the task (leave empty string "" if not needed)
  "answer"     : string  — your best candidate answer to submit

The answer format depends on the task. Read the task description carefully.

Rules:
- Use library functions by name directly (they are pre-imported).
- Keep code short and focused on understanding the task.
- After seeing reward feedback, fix the specific issue described.
- Do NOT re-implement library functions inline.

---

"""


def _parse_library(library_path: str) -> tuple[list[str], dict[str, str], str, str]:
    """Return (names, sources, guide, system_suffix)."""
    lib_file = os.path.join(library_path, "LIBRARY.py")
    guide_file = os.path.join(library_path, "PROMPTING_GUIDE.md")

    source = open(lib_file).read()
    lines = source.splitlines()
    tree = ast.parse(source)
    names, sources = [], {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            names.append(node.name)
            sources[node.name] = "\n".join(lines[node.lineno - 1: node.end_lineno])

    guide = open(guide_file).read().strip() if os.path.exists(guide_file) else ""
    return names, sources, source, guide


def _build_fn_listing(names: list[str], sources: dict[str, str]) -> str:
    """One-line signature per function."""
    lines = []
    for name in names:
        src = sources.get(name, "")
        sig = src.splitlines()[0] if src else f"def {name}(...):"
        lines.append(f"  {sig}")
    return "\n".join(lines)


def _task_text(task_input) -> str:
    if isinstance(task_input, dict):
        return (task_input.get("question") or task_input.get("prompt")
                or task_input.get("task") or str(task_input))
    return str(task_input)


class StaticLibraryJsonAgent:
    """
    JSON-mode static library agent.

    Parameters
    ----------
    llm : LLMClient
    model : str
    library_path : str   Path to directory containing LIBRARY.py + PROMPTING_GUIDE.md
    execute_fn   : callable(code, lib_dir) -> (ok, stdout, stderr)  or None
    lib_dir      : str   Path to stage for sandbox bind-mount (pkg_dir/library/)
    max_iters    : int
    max_tokens   : int
    """

    def __init__(
        self,
        llm: LLMClient,
        model: str,
        library_path: str,
        execute_fn=None,
        lib_dir: str = "",
        max_iters: int = 8,
        max_tokens: int = 4096,
    ):
        self.llm = llm
        self.model = model
        self.execute_fn = execute_fn
        self.lib_dir = lib_dir
        self.max_iters = max_iters
        self.max_tokens = max_tokens

        self._names, self._sources, self._lib_source, self._guide = _parse_library(library_path)
        self._fn_listing = _build_fn_listing(self._names, self._sources)

        system_suffix = f"Available library functions:\n{self._fn_listing}\n"
        if self._guide:
            system_suffix += f"\n---\n\n{self._guide}"
        self._system = _SYSTEM + system_suffix

    def solve(self, task_input, reward_fn, entry: dict) -> dict:
        self.llm.reset_task_log()

        task_text = _task_text(task_input)
        messages = []
        reward_history = []
        best_reward = 0.0
        best_answer = None

        first_user = (
            f"Task:\n{task_text}\n\n"
            "Use the library functions to reason about the task, then provide your answer."
        )
        messages.append({"role": "user", "content": first_user})

        for i in range(self.max_iters):
            resp = self.llm.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self._system,
                messages=messages,
                tag=f"sl_json_iter{i}",
            )

            reasoning = resp.get("reasoning", "")
            code = resp.get("code", "").strip()
            answer = str(resp.get("answer", "")).strip()

            # Build assistant reply text for history
            assistant_text = f"reasoning: {reasoning}\ncode: {code}\nanswer: {answer}"
            messages.append({"role": "assistant", "content": assistant_text})

            # Execute code if provided
            exec_out = ""
            if code and self.execute_fn:
                preamble = f"from library import {', '.join(self._names)}\n" if self._names else ""
                ok, stdout, stderr = self.execute_fn(preamble + code, self.lib_dir)
                if ok:
                    exec_out = stdout.strip() or "(no output)"
                else:
                    # Append source of called functions to help model debug
                    import re
                    called = [n for n in self._sources if re.search(r'\b' + n + r'\s*\(', code)]
                    src_hint = ""
                    if called:
                        snippets = "\n\n".join(f"# {n}:\n{self._sources[n]}" for n in called)
                        src_hint = f"\n\nSource of library functions used:\n{snippets}"
                    exec_out = f"EXECUTION ERROR:\n{stderr[:500]}{src_hint}"

            if not answer:
                feedback = "You did not provide an answer. Please reason through the task and provide an answer."
                if exec_out:
                    feedback = f"Code output:\n{exec_out}\n\n{feedback}"
                messages.append({"role": "user", "content": feedback})
                continue

            # Score the answer
            result = reward_fn(answer, True, entry)
            reward = float(result.get("value", 0.0))
            message = result.get("message", "")
            reward_history.append({"iteration": i, "reward": reward, "message": message, "answer": answer})

            if reward > best_reward:
                best_reward = reward
                best_answer = answer

            logger.info("sl_json iter=%d reward=%.3f", i, reward)

            if reward >= 1.0:
                break

            # Build feedback for next iteration
            feedback_parts = []
            if exec_out:
                feedback_parts.append(f"Code output:\n{exec_out}")
            feedback_parts.append(
                f"Reward: {reward:.3f}\nFeedback: {message}\n\n"
                "Fix the specific issue described above and try again."
            )
            messages.append({"role": "user", "content": "\n\n".join(feedback_parts)})

        token_usage = self.llm.get_task_token_usage()
        return {
            "solved": best_reward >= 1.0,
            "answer": best_answer,
            "best_reward": best_reward,
            "reward_history": reward_history,
            "agent_messages": self.llm.get_task_log(),
            "token_usage": {
                "prompt_tokens": token_usage.get("input", 0),
                "completion_tokens": token_usage.get("output", 0),
                "reasoning_tokens": token_usage.get("reasoning", 0),
            },
        }
