"""
ReAct + Library controller using the OpenHands SDK.

For each task:
  1. Build a Conversation with three custom tools (execute_code, add_to_library, finish).
  2. Run the conversation (SDK handles the ReAct loop internally).
  3. Extract answer from answer.txt written by FinishTool.
  4. Score with reward_fn. If reward < 1.0 and iterations remain, restart
     the conversation with reward feedback injected into the task prompt.
  5. Functions added to the library during the task persist for future tasks.

The library lives as a PkgLibrary at pkg_dir/library/ (per-function .py files).
BM25 retrieval surfaces the top-K relevant functions in each task prompt.
"""

import logging
import os
import tempfile
from typing import Any, Callable, Optional

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation
from openhands.tools.preset.default import get_default_agent

from ..pkg_library import PkgLibrary
from .prompts import build_system_prompt, build_task_prompt
from .tools import AddToLibraryTool, ExecuteCodeTool, FinishTool

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
        Max agent steps per conversation (passed to SDK as max_iterations).
    max_tokens : int
    max_reward_iters : int
        Outer reward-feedback loop iterations.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        library: PkgLibrary,
        sandbox,
        api_key: str = "EMPTY",
        library_k: int = 5,
        max_steps: int = 8,
        max_tokens: int = 4096,
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
        self.max_reward_iters = max_reward_iters

        self._llm = LLM(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            max_tokens=max_tokens,
        )

    def _make_agent(self, answer_path: str) -> Agent:
        return Agent(
            llm=self._llm,
            system_prompt=build_system_prompt(),
            tools=[
                ExecuteCodeTool.create(self.sandbox, self.library),
                AddToLibraryTool.create(self.sandbox, self.library),
                FinishTool.create(answer_path),
            ],
            max_iterations=self.max_steps,
        )

    def solve(
        self,
        task_input: Any,
        reward_fn: Callable,
        entry: dict,
    ) -> dict:
        """
        Solve one task with reward-feedback iterations.
        Library additions from this task persist for future tasks.
        """
        best_reward = 0.0
        best_answer = None
        reward_history = []
        reward_feedback = ""
        library_size_before = len(self.library)

        for iteration in range(self.max_reward_iters):
            logger.info(
                "react_library: task iter %d/%d (library=%d fns)",
                iteration + 1, self.max_reward_iters, len(self.library),
            )

            # Retrieve relevant library functions for this task
            task_text = _task_text(task_input)
            relevant = self.library.retrieve(task_text, k=self.library_k)
            listing = "\n".join(
                f"  {fn['name']}: {fn['description']}" for fn in relevant
            ) if relevant else "(empty)"

            # Each iteration gets a fresh workspace + answer.txt path
            task_dir = tempfile.mkdtemp(prefix="oh_rl_task_")
            answer_path = os.path.join(task_dir, "answer.txt")

            prompt = build_task_prompt(task_input, listing, reward_feedback)
            agent = self._make_agent(answer_path)
            conversation = Conversation(agent=agent, workspace=task_dir)
            conversation.send_message(prompt)

            try:
                conversation.run()
            except Exception as exc:
                logger.warning("react_library conversation error (iter %d): %s", iteration + 1, exc)

            # Extract answer
            answer = None
            if os.path.exists(answer_path):
                answer = open(answer_path).read().strip() or None

            # Score
            reward_result = reward_fn(answer, answer is not None, entry)
            reward_value = float(reward_result.get("value", 0.0))
            reward_message = reward_result.get("message", "")

            if reward_value > best_reward:
                best_reward = reward_value
                best_answer = answer

            reward_history.append({
                "iteration": iteration,
                "reward": reward_value,
                "message": reward_message,
                "answer": str(answer)[:200] if answer else "",
            })
            logger.info(
                "react_library iter %d: reward=%.3f  %s",
                iteration + 1, reward_value, reward_message[:80],
            )

            if best_reward >= 1.0:
                break
            reward_feedback = reward_message

        library_additions = len(self.library) - library_size_before
        return {
            "solved": best_reward >= 1.0,
            "answer": best_answer,
            "best_reward": best_reward,
            "reward_history": reward_history,
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
