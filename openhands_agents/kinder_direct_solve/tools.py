"""
Tools for the KinDER DirectSolve agent.

Two tools:
  KDSPythonTool — run Python in the task's sandbox (kinder_client talks to a simulator of
                  this task's instance; numpy/scipy/standard library are importable)
  KDSSubmitTool — submit the final GeneratedApproach policy source and terminate

Action/Observation class names are prefixed with "KDS" (KinDER DirectSolve) to avoid
global registry collisions with the "DS" and "SB" tools.
"""

import ast
import os
import tempfile
from collections.abc import Sequence
from typing import ClassVar

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
# Python
# ──────────────────────────────────────────────────────────────────────────────

class KDSPythonAction(Action):
    code: str = Field(description="A complete Python script. Print what you need to see.")
    reason: str = Field(description="One sentence: what this run checks, computes or tests.")


class KDSPythonObservation(Observation):
    ok: bool = False
    output: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.output)]


class KDSPythonExecutor(ToolExecutor[KDSPythonAction, KDSPythonObservation]):
    def __init__(self, sandbox):
        self._sandbox = sandbox

    def __call__(self, action: KDSPythonAction, conversation=None) -> KDSPythonObservation:
        result = self._sandbox.run(action.code)
        return KDSPythonObservation(ok=result.ok, output=result.text())


class KDSPythonTool(ToolDefinition[KDSPythonAction, KDSPythonObservation]):
    name: ClassVar[str] = "python"

    @classmethod
    def create(cls, sandbox) -> "list[KDSPythonTool]":
        return [cls(
            description=(
                "Run a Python script in a fresh process in your working directory (files persist "
                "between calls). The simulator of your instance is reachable through kinder_client."
            ),
            action_type=KDSPythonAction,
            observation_type=KDSPythonObservation,
            executor=KDSPythonExecutor(sandbox),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# Submit
# ──────────────────────────────────────────────────────────────────────────────

class KDSSubmitAction(Action):
    code: str = Field(description="Complete source of a Python module defining class GeneratedApproach.")
    notes: str = Field(default="", description="One or two sentences on the strategy and how you tested it.")


class KDSSubmitObservation(Observation):
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


def policy_problems(code: str) -> list:
    """Reasons a submitted policy cannot be evaluated (empty if it can)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    if not any(isinstance(n, ast.ClassDef) and n.name == "GeneratedApproach" for n in tree.body):
        return ["the module does not define class GeneratedApproach at top level"]
    from domains.kinder.strict import check_solver
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "policy.py")
        with open(path, "w") as fh:
            fh.write(code)
        return check_solver(path)


class KDSSubmitExecutor(ToolExecutor[KDSSubmitAction, KDSSubmitObservation]):
    def __init__(self, done: dict):
        self._done = done

    def __call__(self, action: KDSSubmitAction, conversation=None) -> KDSSubmitObservation:
        problems = policy_problems(action.code)
        if problems:
            # rejected before evaluation: the agent can fix it and submit again
            return KDSSubmitObservation(message="Submission rejected; fix these and submit again:\n  "
                                        + "\n  ".join(problems))
        self._done.update({"code": action.code, "notes": action.notes})
        if conversation is not None:
            state = conversation.state
            state.execution_status = type(state.execution_status).FINISHED
        return KDSSubmitObservation(message="Policy submitted. It will be evaluated on this instance.")


class KDSSubmitTool(ToolDefinition[KDSSubmitAction, KDSSubmitObservation]):
    name: ClassVar[str] = "submit"

    @classmethod
    def create(cls, done: dict) -> "list[KDSSubmitTool]":
        return [cls(
            description=(
                "Submit the final policy (complete module source defining GeneratedApproach) and end the "
                "session. Submit as soon as your own rollout solves the instance; also submit your best "
                "attempt before you run out of turns."
            ),
            action_type=KDSSubmitAction,
            observation_type=KDSSubmitObservation,
            executor=KDSSubmitExecutor(done),
        )]
