"""
Custom OpenHands SDK tools for the ReAct+Library agent.

Three tools:
  ExecuteCodeTool   — run code in Apptainer sandbox; library available
  AddToLibraryTool  — validate + add function to PkgLibrary
  FinishTool        — write answer.txt, signals agent completion

Each tool follows the OpenHands Action / Observation / Executor pattern.
The Executors hold references to the sandbox and library so they can be
injected at controller init time.

IMPORTANT: Action/Observation class names must be globally unique across all
imported SDK modules. The SDK registers them in a discriminated union keyed by
class name. Prefixing with "RL" (ReAct+Library) avoids collisions with SDK
built-ins (e.g. openhands.sdk.tool.builtins.finish.FinishAction).
"""

import os
from collections.abc import Sequence

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

class RLExecuteCodeAction(Action):
    code: str = Field(description="Complete Python program to execute. Use `from library import fn` to call library functions. Print the answer to stdout.")


class RLExecuteCodeObservation(Observation):
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


class RLExecuteCodeExecutor(ToolExecutor[RLExecuteCodeAction, RLExecuteCodeObservation]):
    def __init__(self, sandbox, library: PkgLibrary):
        self._sandbox = sandbox
        self._library = library

    def __call__(self, action: RLExecuteCodeAction, conversation=None) -> RLExecuteCodeObservation:
        ok, stdout, stderr = self._sandbox.run_code(
            action.code,
            lib_dir=self._library.pkg_dir if len(self._library) > 0 else None,
        )
        return RLExecuteCodeObservation(ok=ok, stdout=stdout, stderr=stderr)


class ExecuteCodeTool(ToolDefinition[RLExecuteCodeAction, RLExecuteCodeObservation]):
    @classmethod
    def create(cls, sandbox, library: PkgLibrary) -> "list[ExecuteCodeTool]":
        return [cls(
            description=(
                "Execute Python code in a sandboxed environment. "
                "Import shared library functions with `from library import fn_name`. "
                "Print the answer to stdout so you can observe it."
            ),
            action_type=RLExecuteCodeAction,
            observation_type=RLExecuteCodeObservation,
            executor=RLExecuteCodeExecutor(sandbox, library),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# AddToLibrary
# ──────────────────────────────────────────────────────────────────────────────

class RLAddToLibraryAction(Action):
    name: str = Field(description="snake_case function name")
    description: str = Field(description="One-line description of what the function does")
    code: str = Field(description="Complete function definition. Must be standalone — no imports from `library`.")


class RLAddToLibraryObservation(Observation):
    ok: bool = False
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class RLAddToLibraryExecutor(ToolExecutor[RLAddToLibraryAction, RLAddToLibraryObservation]):
    def __init__(self, sandbox, library: PkgLibrary):
        self._sandbox = sandbox
        self._library = library

    def __call__(self, action: RLAddToLibraryAction, conversation=None) -> RLAddToLibraryObservation:
        name = action.name.strip()
        code = action.code.strip()
        if not name or not code:
            return RLAddToLibraryObservation(ok=False, message="name and code are required.")

        # Validate: must compile and execute without error
        ok, _, err = self._sandbox.run_code(code)
        if not ok:
            return RLAddToLibraryObservation(
                ok=False,
                message=f"Function validation failed:\n{err[:300]}",
            )

        self._library.add(name, action.description.strip(), code)
        return RLAddToLibraryObservation(
            ok=True,
            message=f"Added '{name}' to library. ({len(self._library)} functions total)",
        )


class AddToLibraryTool(ToolDefinition[RLAddToLibraryAction, RLAddToLibraryObservation]):
    @classmethod
    def create(cls, sandbox, library: PkgLibrary) -> "list[AddToLibraryTool]":
        return [cls(
            description=(
                "Add a reusable Python helper function to the shared library. "
                "The function must be standalone (no imports from `library`). "
                "Once added, it's immediately available via `from library import fn_name`."
            ),
            action_type=RLAddToLibraryAction,
            observation_type=RLAddToLibraryObservation,
            executor=RLAddToLibraryExecutor(sandbox, library),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# Finish
# ──────────────────────────────────────────────────────────────────────────────

class RLFinishAction(Action):
    answer: str = Field(description="The final answer to submit.")


class RLFinishObservation(Observation):
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class RLFinishExecutor(ToolExecutor[RLFinishAction, RLFinishObservation]):
    def __init__(self, answer_path: str):
        self._answer_path = answer_path

    def __call__(self, action: RLFinishAction, conversation=None) -> RLFinishObservation:
        os.makedirs(os.path.dirname(self._answer_path), exist_ok=True)
        with open(self._answer_path, "w") as f:
            f.write(action.answer)
        # Signal the SDK to stop the agent loop
        if conversation is not None:
            conversation.stop()
        return RLFinishObservation(message=f"Answer submitted: {action.answer[:100]}")


class FinishTool(ToolDefinition[RLFinishAction, RLFinishObservation]):
    @classmethod
    def create(cls, answer_path: str) -> "list[FinishTool]":
        return [cls(
            description="Submit the final answer. Call this when you are confident in the result.",
            action_type=RLFinishAction,
            observation_type=RLFinishObservation,
            executor=RLFinishExecutor(answer_path),
        )]
