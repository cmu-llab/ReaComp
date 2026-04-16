"""
ReAct + Library controller using the OpenHands SDK.

For each task a single Conversation is created with four tools:
  execute_code    — run Python in the sandbox; library available
  add_to_library  — validate + add a function to the shared library
  check_reward    — call the verifier inline; agent uses this to iterate
  finish          — submit the final answer and stop

The agent calls check_reward as many times as it needs within one
conversation (no outer restart loop), retaining full context across
all attempts. The max_steps budget governs how long it may explore.

Functions added during a task persist for future tasks (shared library).
BM25 retrieval surfaces the top-K relevant functions in each task prompt,
including their full code so the agent can judge whether to reuse them.
"""

import logging
import os
import tempfile
from typing import Any, Callable, Optional

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation

from ..pkg_library import PkgLibrary
from .prompts import build_task_prompt
from .tools import AddToLibraryTool, CheckRewardTool, ExecuteCodeTool, FinishTool

logger = logging.getLogger(__name__)


class ReActLibraryController:
    """
    ReAct + shared library via OpenHands SDK.

    Parameters
    ----------
    base_url : str
    model : str
    library : PkgLibrary
    sandbox : ApptainerSandbox
    api_key : str
    library_k : int
        Number of functions to retrieve for each task prompt.
    max_steps : int
        Max agent steps per conversation.
    max_tokens : int
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        library: PkgLibrary,
        sandbox,
        api_key: str = "EMPTY",
        library_k: int = 5,
        max_steps: int = 100,
        max_tokens: int = 4096,
        # kept for API compatibility; no longer used
        max_reward_iters: int = 3,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.library = library
        self.sandbox = sandbox
        self.library_k = library_k
        self.max_steps = max_steps
        self.max_tokens = max_tokens

        self._llm = LLM(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            max_tokens=max_tokens,
        )

    def _make_agent(self, answer_path: str, reward_fn: Callable, entry: Any) -> Agent:
        tool_instances = [
            *ExecuteCodeTool.create(self.sandbox, self.library),
            *AddToLibraryTool.create(self.sandbox, self.library),
            *CheckRewardTool.create(reward_fn, entry),
            *FinishTool.create(answer_path),
        ]
        _prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        agent = Agent(
            llm=self._llm,
            tools=[],
            include_default_tools=[],
            system_prompt_filename=os.path.join(_prompts_dir, "system_prompt.j2"),
        )
        agent.__pydantic_private__["_tools"] = {t.name: t for t in tool_instances}
        agent.__pydantic_private__["_initialized"] = True
        logger.info("agent tools_map: %s", list(agent.tools_map.keys()))
        return agent

    def solve(
        self,
        task_input: Any,
        reward_fn: Callable,
        entry: dict,
    ) -> dict:
        """
        Solve one task in a single conversation.
        The agent calls check_reward inline to verify and iterate.
        Library additions from this task persist for future tasks.
        """
        library_size_before = len(self.library)

        # Retrieve relevant library functions (with full code for reuse decisions)
        task_text = _task_text(task_input)
        relevant = self.library.retrieve(task_text, k=self.library_k)

        task_dir = tempfile.mkdtemp(prefix="oh_rl_task_")
        answer_path = os.path.join(task_dir, "answer.txt")

        prompt = build_task_prompt(task_input, relevant)
        agent = self._make_agent(answer_path, reward_fn, entry)
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
            logger.warning("react_library conversation error: %s\n%s",
                           exc, traceback.format_exc())

        # Extract answer
        answer = None
        if os.path.exists(answer_path):
            answer = open(answer_path).read().strip() or None

        # Final score
        reward_result = reward_fn(answer, answer is not None, entry)
        best_reward = float(reward_result.get("value", 0.0))

        library_additions = len(self.library) - library_size_before
        logger.info(
            "react_library: reward=%.3f  library=%d (+%d)",
            best_reward, len(self.library), library_additions,
        )
        return {
            "solved": best_reward >= 1.0,
            "answer": answer,
            "best_reward": best_reward,
            "reward_history": [{"reward": best_reward, "message": reward_result.get("message", "")}],
            "library_size": len(self.library),
            "library_additions_this_task": library_additions,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"library": self.library.to_dict()}

    def from_dict(self, data: dict) -> None:
        self.library._load_meta()


def _task_text(task_input: Any) -> str:
    if isinstance(task_input, dict):
        return (
            task_input.get("question")
            or task_input.get("prompt")
            or task_input.get("task")
            or str(task_input)
        )
    return str(task_input)
