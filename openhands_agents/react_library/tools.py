"""
Custom OpenHands SDK tools for the ReAct+Library agent.

Three tools:
  ExecuteCodeTool   — run code in Apptainer sandbox; library available
  AddToLibraryTool  — validate + add function to PkgLibrary
  FinishTool        — write answer.txt, signals agent completion

Each tool follows the OpenHands Action / Observation / Executor pattern.
The Executors hold references to the sandbox and library so they can be
injected at controller init time.
"""

import os
from collections.abc import Sequence
from typing import Any, Optional

from pydantic import Field

from openhands.sdk import (
    Action,
    Observation,
    TextContent,
    ImageContent,
    ToolDefinition,
)
from openhands.sdk.tool import ToolExecutor

from ..pkg_library import PkgLibrary


# ──────────────────────────────────────────────────────────────────────────────
# ExecuteCode
# ──────────────────────────────────────────────────────────────────────────────

class ExecuteCodeAction(Action):
    code: str = Field(description="Complete Python program to execute. Use `from library import fn` to call library functions. Print the answer to stdout.")


class ExecuteCodeObservation(Observation):
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


class ExecuteCodeExecutor(ToolExecutor[ExecuteCodeAction, ExecuteCodeObservation]):
    def __init__(self, sandbox, library: PkgLibrary):
        self._sandbox = sandbox
        self._library = library

    def __call__(self, action: ExecuteCodeAction, conversation=None) -> ExecuteCodeObservation:
        ok, stdout, stderr = self._sandbox.run_code(
            action.code,
            lib_dir=self._library.pkg_dir if len(self._library) > 0 else None,
        )
        return ExecuteCodeObservation(ok=ok, stdout=stdout, stderr=stderr)


class ExecuteCodeTool(ToolDefinition[ExecuteCodeAction, ExecuteCodeObservation]):
    @classmethod
    def create(cls, sandbox, library: PkgLibrary) -> "list[ExecuteCodeTool]":
        return [cls(
            description=(
                "Execute Python code in a sandboxed environment. "
                "Import shared library functions with `from library import fn_name`. "
                "Print the answer to stdout so you can observe it."
            ),
            action_type=ExecuteCodeAction,
            observation_type=ExecuteCodeObservation,
            executor=ExecuteCodeExecutor(sandbox, library),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# AddToLibrary
# ──────────────────────────────────────────────────────────────────────────────

class AddToLibraryAction(Action):
    name: str = Field(description="snake_case function name")
    description: str = Field(description="One-line description of what the function does")
    code: str = Field(description="Complete function definition. Must be standalone — no imports from `library`.")


class AddToLibraryObservation(Observation):
    ok: bool = False
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class AddToLibraryExecutor(ToolExecutor[AddToLibraryAction, AddToLibraryObservation]):
    def __init__(self, sandbox, library: PkgLibrary):
        self._sandbox = sandbox
        self._library = library

    def __call__(self, action: AddToLibraryAction, conversation=None) -> AddToLibraryObservation:
        name = action.name.strip()
        code = action.code.strip()
        if not name or not code:
            return AddToLibraryObservation(ok=False, message="name and code are required.")

        # Validate: must compile and execute without error
        ok, _, err = self._sandbox.run_code(code)
        if not ok:
            return AddToLibraryObservation(
                ok=False,
                message=f"Function validation failed:\n{err[:300]}",
            )

        self._library.add(name, action.description.strip(), code)
        return AddToLibraryObservation(
            ok=True,
            message=f"Added '{name}' to library. ({len(self._library)} functions total)",
        )


class AddToLibraryTool(ToolDefinition[AddToLibraryAction, AddToLibraryObservation]):
    @classmethod
    def create(cls, sandbox, library: PkgLibrary) -> "list[AddToLibraryTool]":
        return [cls(
            description=(
                "Add a reusable Python helper function to the shared library. "
                "The function must be standalone (no imports from `library`). "
                "Once added, it's immediately available via `from library import fn_name`."
            ),
            action_type=AddToLibraryAction,
            observation_type=AddToLibraryObservation,
            executor=AddToLibraryExecutor(sandbox, library),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# Finish
# ──────────────────────────────────────────────────────────────────────────────

class FinishAction(Action):
    answer: str = Field(description="The final answer to submit.")


class FinishObservation(Observation):
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class FinishExecutor(ToolExecutor[FinishAction, FinishObservation]):
    def __init__(self, answer_path: str):
        self._answer_path = answer_path

    def __call__(self, action: FinishAction, conversation=None) -> FinishObservation:
        os.makedirs(os.path.dirname(self._answer_path), exist_ok=True)
        with open(self._answer_path, "w") as f:
            f.write(action.answer)
        # Signal the SDK to stop the agent loop
        if conversation is not None:
            conversation.stop()
        return FinishObservation(message=f"Answer submitted: {action.answer[:100]}")


class FinishTool(ToolDefinition[FinishAction, FinishObservation]):
    @classmethod
    def create(cls, answer_path: str) -> "list[FinishTool]":
        return [cls(
            description="Submit the final answer. Call this when you are confident in the result.",
            action_type=FinishAction,
            observation_type=FinishObservation,
            executor=FinishExecutor(answer_path),
        )]
