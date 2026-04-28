"""
Tools for the DirectSolve agent.

Two tools:
  DSExecuteCodeTool  — run Python in the sandbox (rewards/pbebench.py mounted at /workspace)
  DSSubmitAnswerTool — submit the final answer (list of replace(A,B) strings) and terminate

Action/Observation class names prefixed with "DS" (DirectSolve) to avoid global registry
collisions with the SolverBuilder "SB" prefix.
"""

import json
import os
from collections.abc import Sequence
from typing import Any

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

class DSExecuteCodeAction(Action):
    code: str = Field(
        description=(
            "Python code to execute in the sandbox. "
            "The reward function is importable: "
            "    from rewards.pbebench import reward\n"
            "Call it as: reward(answer, True, task_record) where answer is a list "
            "of replace(A,B) strings and task_record has 'inputs'/'outputs' keys. "
            "Print results to stdout."
        )
    )


class DSExecuteCodeObservation(Observation):
    ok: bool = False
    stdout: str = ""
    stderr: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.ok:
            out = self.stdout.strip() or "(no output)"
            return [TextContent(text=f"stdout:\n{out}")]
        err = self.stderr.strip()[:1000] or "(no stderr)"
        return [TextContent(text=f"Execution failed.\nstderr:\n{err}")]


class DSExecuteCodeExecutor(ToolExecutor[DSExecuteCodeAction, DSExecuteCodeObservation]):
    def __init__(self, sandbox, extra_binds: list):
        self._sandbox = sandbox
        self._extra_binds = extra_binds

    def __call__(self, action: DSExecuteCodeAction, conversation=None) -> DSExecuteCodeObservation:
        preamble = "import sys; sys.path.insert(0, '/workspace')\n"
        ok, stdout, stderr = self._sandbox.run_code(
            preamble + action.code,
            lib_dir=None,
            timeout=60,
            extra_binds=self._extra_binds,
        )
        return DSExecuteCodeObservation(ok=ok, stdout=stdout, stderr=stderr)


class DSExecuteCodeTool(ToolDefinition[DSExecuteCodeAction, DSExecuteCodeObservation]):
    @classmethod
    def create(cls, sandbox, extra_binds: list) -> "list[DSExecuteCodeTool]":
        return [cls(
            description=(
                "Execute Python code in the sandbox. "
                "Use this to test candidate programs, call the reward function, "
                "or write and debug any logic. "
                "rewards/pbebench.py is importable from /workspace."
            ),
            action_type=DSExecuteCodeAction,
            observation_type=DSExecuteCodeObservation,
            executor=DSExecuteCodeExecutor(sandbox, extra_binds),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# SubmitAnswer
# ──────────────────────────────────────────────────────────────────────────────

class DSSubmitAnswerAction(Action):
    programs: list[str] = Field(
        description=(
            "Your final answer: an ordered list of replace(A,B) strings, e.g. "
            '["replace(\'the\', \'a\')", "replace(\'cat\', \'dog\')"] . '
            "Submit as soon as you have a correct or best-effort solution."
        )
    )


class DSSubmitAnswerObservation(Observation):
    message: str = ""
    reward: float = 0.0

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class DSSubmitAnswerExecutor(ToolExecutor[DSSubmitAnswerAction, DSSubmitAnswerObservation]):
    def __init__(self, task_record: dict, done_path: str, reward_fn):
        self._task_record = task_record
        self._done_path = done_path
        self._reward_fn = reward_fn

    def __call__(self, action: DSSubmitAnswerAction, conversation=None) -> DSSubmitAnswerObservation:
        programs = action.programs
        result = self._reward_fn(programs, True, self._task_record)
        reward_value = float(result.get("value", 0.0))
        feedback = result.get("feedback", "")

        # Write answer to done file
        os.makedirs(os.path.dirname(self._done_path), exist_ok=True)
        with open(self._done_path, "w") as f:
            json.dump({"programs": programs, "reward": reward_value}, f)

        # Terminate the conversation
        if conversation is not None:
            state = conversation.state
            state.execution_status = type(state.execution_status).FINISHED

        if reward_value >= 1.0:
            msg = f"Correct! Reward = {reward_value:.3f}. Answer accepted."
        else:
            msg = f"Submitted. Reward = {reward_value:.3f}. Feedback: {feedback}"
        return DSSubmitAnswerObservation(message=msg, reward=reward_value)


class DSSubmitAnswerTool(ToolDefinition[DSSubmitAnswerAction, DSSubmitAnswerObservation]):
    @classmethod
    def create(cls, task_record: dict, done_path: str, reward_fn) -> "list[DSSubmitAnswerTool]":
        return [cls(
            description=(
                "Submit your final answer. Provide programs as a list of replace(A,B) strings. "
                "You will receive the reward score. Submit immediately when you have reward=1.0; "
                "also submit your best attempt if you are running out of steps."
            ),
            action_type=DSSubmitAnswerAction,
            observation_type=DSSubmitAnswerObservation,
            executor=DSSubmitAnswerExecutor(task_record, done_path, reward_fn),
        )]
