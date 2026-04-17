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
            "All library functions are already imported — call them directly "
            "(e.g. find_changed_pairs(...), score_rules(...), apply_rules(...)). "
            "Do NOT write import statements for library functions. "
            "Print results to stdout to observe them."
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
    def __init__(self, sandbox, lib_pkg_dir: str, function_names: list[str], fn_sources: "dict[str, str]"):
        self._sandbox = sandbox
        self._lib_pkg_dir = lib_pkg_dir
        self._fn_sources = fn_sources
        if function_names:
            self._preamble = f"from library import {', '.join(function_names)}\n"
        else:
            self._preamble = ""

    def __call__(self, action: SLExecuteCodeAction, conversation=None) -> SLExecuteCodeObservation:
        import re
        code = self._preamble + action.code
        ok, stdout, stderr = self._sandbox.run_code(code, lib_dir=self._lib_pkg_dir)
        if not ok and self._fn_sources:
            # Find library functions called in the failing code and append their
            # full source so the agent can see the exact return schema.
            called = [n for n in self._fn_sources if re.search(r'\b' + n + r'\s*\(', action.code)]
            if called:
                snippets = "\n\n".join(f"# {n}:\n{self._fn_sources[n]}" for n in called)
                stderr = (
                    stderr
                    + "\n\n--- Source of library functions used in this call ---\n"
                    + snippets
                )
        return SLExecuteCodeObservation(ok=ok, stdout=stdout, stderr=stderr)


class ExecuteCodeTool(ToolDefinition[SLExecuteCodeAction, SLExecuteCodeObservation]):
    @classmethod
    def create(cls, sandbox, lib_pkg_dir: str, function_names: list[str], fn_sources: "dict[str, str]") -> "list[ExecuteCodeTool]":
        return [cls(
            description=(
                "Execute Python code in a sandboxed environment. "
                "All library functions are pre-imported — call them directly without import statements. "
                "Print results to stdout to observe them."
            ),
            action_type=SLExecuteCodeAction,
            observation_type=SLExecuteCodeObservation,
            executor=SLExecuteCodeExecutor(sandbox, lib_pkg_dir, function_names, fn_sources),
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
        self.call_log: list[dict] = []

    def __call__(self, action: SLCheckRewardAction, conversation=None) -> SLCheckRewardObservation:
        result = self._reward_fn(action.answer, True, self._entry)
        reward = float(result.get("value", 0.0))
        message = result.get("message", "")
        self.call_log.append({"reward": reward, "message": message, "answer": action.answer})
        return SLCheckRewardObservation(reward=reward, message=message)


class CheckRewardTool(ToolDefinition[SLCheckRewardAction, SLCheckRewardObservation]):
    @classmethod
    def create(cls, reward_fn: Callable, entry: Any) -> "tuple[list[CheckRewardTool], SLCheckRewardExecutor]":
        executor = SLCheckRewardExecutor(reward_fn, entry)
        return [cls(
            description=(
                "Check your candidate answer against the task verifier. "
                "Returns reward in [0, 1] and feedback. "
                "Call as many times as needed before calling finish."
            ),
            action_type=SLCheckRewardAction,
            observation_type=SLCheckRewardObservation,
            executor=executor,
        )], executor


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
