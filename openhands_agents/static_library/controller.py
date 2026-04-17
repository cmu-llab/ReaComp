"""
StaticLibrary controller — weaker agent using a pre-built fixed library.

The library (LIBRARY.py) and workflow guide (PROMPTING_GUIDE.md) are produced
offline by a strong coding agent. This controller:

  1. Loads the library into a pkg_dir so the sandbox can bind-mount it.
  2. Embeds the PROMPTING_GUIDE into the system prompt so the weak agent
     follows the prescribed workflow.
  3. Runs a single conversation per task with three tools:
       execute_code  — run code; library importable as `from library import fn`
       check_reward  — inline verifier; agent iterates until satisfied
       finish        — submit answer

No add_to_library tool — the library is fixed for the entire run.
"""

import logging
import os
import tempfile
from typing import Any, Callable

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation

from .library_loader import StaticLibrary
from .prompts import build_system_prompt, build_task_prompt
from .tools import CheckRewardTool, ExecuteCodeTool, FinishTool

logger = logging.getLogger(__name__)


class StaticLibraryController:
    """
    Weaker agent using a pre-built fixed library from a strong coding agent.

    Parameters
    ----------
    base_url : str
    model : str
    static_library : StaticLibrary
        Loaded pre-built library (LIBRARY.py + guide).
    sandbox : ApptainerSandbox
    api_key : str
    max_steps : int
        Max agent steps per conversation (default 100).
    max_tokens : int
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        static_library: StaticLibrary,
        sandbox,
        api_key: str = "EMPTY",
        max_steps: int = 100,
        max_tokens: int = 4096,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.static_library = static_library
        self.sandbox = sandbox
        self.max_steps = max_steps
        self.max_tokens = max_tokens

        self._llm = LLM(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            max_tokens=max_tokens,
        )

        # Write the system prompt to a temp .j2 file once — reused for all tasks.
        # The SDK requires system_prompt_filename= (an absolute path to a .j2 file).
        system_prompt_text = build_system_prompt(
            static_library.prompting_guide,
            static_library.function_names,
        )
        self._system_prompt_file = self._write_system_prompt(system_prompt_text)
        logger.info(
            "StaticLibraryController ready: %d library functions, guide=%d chars",
            len(static_library.function_names),
            len(static_library.prompting_guide),
        )

    def _write_system_prompt(self, text: str) -> str:
        """Write system prompt to a named temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".j2", prefix="sl_system_prompt_")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        return path

    def _make_agent(self, answer_path: str, reward_fn: Callable, entry: Any) -> "tuple[Agent, CheckRewardTool.__class__]":
        lib_pkg_dir = os.path.join(self.static_library.pkg_dir, "library")

        exec_tools = ExecuteCodeTool.create(self.sandbox, lib_pkg_dir, self.static_library.function_names, self.static_library.function_sources)
        check_reward_tools, check_reward_executor = CheckRewardTool.create(reward_fn, entry)
        finish_tools = FinishTool.create(answer_path)

        tool_instances = [*exec_tools, *check_reward_tools, *finish_tools]
        agent = Agent(
            llm=self._llm,
            tools=[],
            include_default_tools=[],
            system_prompt_filename=self._system_prompt_file,
        )
        agent.__pydantic_private__["_tools"] = {t.name: t for t in tool_instances}
        agent.__pydantic_private__["_initialized"] = True
        logger.debug("agent tools: %s", list(agent.tools_map.keys()))
        return agent, check_reward_executor

    def solve(
        self,
        task_input: Any,
        reward_fn: Callable,
        entry: dict,
    ) -> dict:
        """Solve one task using the fixed pre-built library."""
        usage_before = self._token_usage_snapshot()

        task_dir = tempfile.mkdtemp(prefix="oh_sl_task_")
        answer_path = os.path.join(task_dir, "answer.txt")

        prompt = build_task_prompt(task_input)
        agent, check_reward_executor = self._make_agent(answer_path, reward_fn, entry)
        conversation = Conversation(
            agent=agent,
            workspace=task_dir,
            max_iteration_per_run=self.max_steps,
        )
        conversation.send_message(prompt)

        try:
            conversation.run()
        except Exception as exc:
            import traceback
            logger.warning("static_library conversation error: %s\n%s",
                           exc, traceback.format_exc())

        trajectory = []
        try:
            trajectory = [e.model_dump() for e in conversation.state.events]
        except Exception:
            pass

        answer = None
        if os.path.exists(answer_path):
            answer = open(answer_path).read().strip() or None

        # Build reward_history from inline check_reward calls during conversation
        reward_history = check_reward_executor.call_log
        if reward_history:
            best_reward = max(e["reward"] for e in reward_history)
        else:
            # Agent never called check_reward — score the final answer post-hoc
            reward_result = reward_fn(answer, answer is not None, entry)
            best_reward = float(reward_result.get("value", 0.0))
            reward_history = [{"reward": best_reward, "message": reward_result.get("message", "")}]

        usage_after = self._token_usage_snapshot()
        token_usage = {
            "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
            "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
        }

        logger.info(
            "static_library: reward=%.3f  check_reward_calls=%d  tokens=%d+%d",
            best_reward,
            len(check_reward_executor.call_log),
            token_usage["prompt_tokens"],
            token_usage["completion_tokens"],
        )
        return {
            "solved": best_reward >= 1.0,
            "answer": answer,
            "best_reward": best_reward,
            "reward_history": reward_history,
            "token_usage": token_usage,
            "_trajectory": trajectory,
        }

    def _token_usage_snapshot(self) -> dict:
        usage = self._llm.metrics.accumulated_token_usage
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        return {
            "prompt_tokens": usage.prompt_tokens or 0,
            "completion_tokens": usage.completion_tokens or 0,
        }

    # ------------------------------------------------------------------
    # Checkpoint (no mutable state — library is fixed)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"library_path": self.static_library.library_path}

    def from_dict(self, data: dict) -> None:
        pass  # nothing to restore — library is always loaded fresh from disk
