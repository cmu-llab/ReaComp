"""
OpenHands SDK tools for the StaticLibrary agent.

Three tools only — no add_to_library (library is fixed):
  ExecuteCodeTool   — run code in Apptainer sandbox; pre-built library available
  CheckRewardTool   — call the task verifier inline
  FinishTool        — write answer.txt and stop

Action/Observation classes are prefixed SL (StaticLibrary) to avoid SDK
discriminated-union collisions with other baselines' RL-prefixed classes.
"""

import os
from collections.abc import Sequence
from typing import Any, Callable

from pydantic import Field

from openhands.sdk import (
    Action,
    Observation,
    TextContent,
    ImageContent,
    ToolDefinition,
)
from openhands.sdk.tool import ToolExecutor


# ──────────────────────────────────────────────────────────────────────────────
# ExecuteCode
# ──────────────────────────────────────────────────────────────────────────────

class SLExecuteCodeAction(Action):
    code: str = Field(
        description=(
            "Complete Python program to execute. "
            "Import library functions with `from library import fn_name`. "
            "Print the answer to stdout."
        )
    )


class SLExecuteCodeObservation(Observation):
    ok: bool = False
    stdout: str = ""
    stderr: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.ok:
            out = self.stdout.strip() or "(no output)"
            return [TextContent(text=f"stdout:\n{out}")]
        err = self.stderr.strip()[:500] or "(no stderr)"
        return [TextContent(text=f"Execution failed.\nstderr:\n{err}")]


class SLExecuteCodeExecutor(ToolExecutor[SLExecuteCodeAction, SLExecuteCodeObservation]):
    def __init__(self, sandbox, lib_pkg_dir: str):
        self._sandbox = sandbox
        # lib_pkg_dir = pkg_dir/library/  (basename = "library")
        # sandbox.run_code binds lib_dir → /exec/<basename(lib_dir)>
        # so passing lib_pkg_dir directly makes it importable as `from library import fn`
        self._lib_pkg_dir = lib_pkg_dir

    def __call__(self, action: SLExecuteCodeAction, conversation=None) -> SLExecuteCodeObservation:
        ok, stdout, stderr = self._sandbox.run_code(action.code, lib_dir=self._lib_pkg_dir)
        return SLExecuteCodeObservation(ok=ok, stdout=stdout, stderr=stderr)


class ExecuteCodeTool(ToolDefinition[SLExecuteCodeAction, SLExecuteCodeObservation]):
    @classmethod
    def create(cls, sandbox, lib_pkg_dir: str) -> "list[ExecuteCodeTool]":
        return [cls(
            description=(
                "Execute Python code in a sandboxed environment. "
                "The pre-built library is available: `from library import fn_name`. "
                "Print results to stdout to observe them."
            ),
            action_type=SLExecuteCodeAction,
            observation_type=SLExecuteCodeObservation,
            executor=SLExecuteCodeExecutor(sandbox, lib_pkg_dir),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# CheckReward
# ──────────────────────────────────────────────────────────────────────────────

class SLCheckRewardAction(Action):
    answer: str = Field(description="Candidate answer to verify against the task verifier.")


class SLCheckRewardObservation(Observation):
    reward: float = 0.0
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=f"Reward: {self.reward:.3f}\n{self.message}")]


class SLCheckRewardExecutor(ToolExecutor[SLCheckRewardAction, SLCheckRewardObservation]):
    def __init__(self, reward_fn: Callable, entry: Any):
        self._reward_fn = reward_fn
        self._entry = entry

    def __call__(self, action: SLCheckRewardAction, conversation=None) -> SLCheckRewardObservation:
        result = self._reward_fn(action.answer, True, self._entry)
        return SLCheckRewardObservation(
            reward=float(result.get("value", 0.0)),
            message=result.get("message", ""),
        )


class CheckRewardTool(ToolDefinition[SLCheckRewardAction, SLCheckRewardObservation]):
    @classmethod
    def create(cls, reward_fn: Callable, entry: Any) -> "list[CheckRewardTool]":
        return [cls(
            description=(
                "Check your candidate answer against the task verifier. "
                "Returns reward in [0, 1] and feedback. "
                "Call as many times as needed before calling finish."
            ),
            action_type=SLCheckRewardAction,
            observation_type=SLCheckRewardObservation,
            executor=SLCheckRewardExecutor(reward_fn, entry),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# Finish
# ──────────────────────────────────────────────────────────────────────────────

class SLFinishAction(Action):
    answer: str = Field(description="The final answer to submit.")


class SLFinishObservation(Observation):
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class SLFinishExecutor(ToolExecutor[SLFinishAction, SLFinishObservation]):
    def __init__(self, answer_path: str):
        self._answer_path = answer_path

    def __call__(self, action: SLFinishAction, conversation=None) -> SLFinishObservation:
        os.makedirs(os.path.dirname(self._answer_path), exist_ok=True)
        with open(self._answer_path, "w") as f:
            f.write(action.answer)
        if conversation is not None:
            state = conversation.state
            state.execution_status = type(state.execution_status).FINISHED
        return SLFinishObservation(message=f"Answer submitted: {action.answer[:100]}")


class FinishTool(ToolDefinition[SLFinishAction, SLFinishObservation]):
    @classmethod
    def create(cls, answer_path: str) -> "list[FinishTool]":
        return [cls(
            description="Submit the final answer. Call once check_reward returns 1.0 or you are satisfied.",
            action_type=SLFinishAction,
            observation_type=SLFinishObservation,
            executor=SLFinishExecutor(answer_path),
        )]
