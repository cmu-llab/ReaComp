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
from .tools import AddToLibraryTool, CheckRewardTool, ExecuteCodeTool, FinishTool, RLCheckRewardExecutor

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

    def _make_agent(self, answer_path: str, reward_fn: Callable, entry: Any) -> "tuple[Agent, RLCheckRewardExecutor]":
        exec_tools = ExecuteCodeTool.create(self.sandbox, self.library)
        exec_executor = exec_tools[0].executor  # grab executor to wire into check_reward
        check_reward_tools, check_reward_executor = CheckRewardTool.create(
            reward_fn, entry, exec_executor=exec_executor
        )
        tool_instances = [
            *exec_tools,
            *AddToLibraryTool.create(self.sandbox, self.library),
            *check_reward_tools,
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
        return agent, check_reward_executor

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

        # Snapshot accumulated token usage before this task
        usage_before = self._token_usage_snapshot()

        # Retrieve relevant library functions (with full code for reuse decisions)
        task_text = _task_text(task_input)
        relevant = self.library.retrieve(task_text, k=self.library_k)

        task_dir = tempfile.mkdtemp(prefix="oh_rl_task_")
        answer_path = os.path.join(task_dir, "answer.txt")

        prompt = build_task_prompt(task_input, relevant)
        agent, check_reward_executor = self._make_agent(answer_path, reward_fn, entry)
        conversation = Conversation(
            agent=agent,
            workspace=task_dir,
            max_iteration_per_run=self.max_steps,
        )
        conversation.send_message(prompt)

        trajectory = []
        try:
            conversation.run()
            # Extract full event log for debug logging
            try:
                trajectory = [e.model_dump() for e in conversation.state.events]
            except Exception:
                pass
        except Exception as exc:
            import traceback
            logger.warning("react_library conversation error: %s\n%s",
                           exc, traceback.format_exc())

        # Extract answer
        answer = None
        if os.path.exists(answer_path):
            answer = open(answer_path).read().strip() or None

        # Build reward_history from all check_reward calls made during the conversation.
        # Fall back to a single post-hoc score if the agent never called check_reward.
        reward_history = check_reward_executor.call_log
        if reward_history:
            best_reward = max(e["reward"] for e in reward_history)
        else:
            reward_result = reward_fn(answer, answer is not None, entry)
            best_reward = float(reward_result.get("value", 0.0))
            reward_history = [{"reward": best_reward, "message": reward_result.get("message", "")}]

        # Token usage delta for this task
        usage_after = self._token_usage_snapshot()
        token_usage = {
            "prompt_tokens": usage_after["prompt_tokens"] - usage_before["prompt_tokens"],
            "completion_tokens": usage_after["completion_tokens"] - usage_before["completion_tokens"],
        }

        library_additions = len(self.library) - library_size_before
        logger.info(
            "react_library: reward=%.3f  check_reward_calls=%d  library=%d (+%d)  tokens=%d+%d",
            best_reward, len(reward_history), len(self.library), library_additions,
            token_usage["prompt_tokens"], token_usage["completion_tokens"],
        )
        return {
            "solved": best_reward >= 1.0,
            "answer": answer,
            "best_reward": best_reward,
            "reward_history": reward_history,
            "library_size": len(self.library),
            "library_additions_this_task": library_additions,
            "token_usage": token_usage,
            "_trajectory": trajectory,  # full event log; stripped before JSONL write
        }

    def _token_usage_snapshot(self) -> dict:
        """Return current accumulated prompt/completion token counts from the LLM metrics."""
        usage = self._llm.metrics.accumulated_token_usage
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        return {
            "prompt_tokens": usage.prompt_tokens or 0,
            "completion_tokens": usage.completion_tokens or 0,
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
