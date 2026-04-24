"""
Custom OpenHands SDK tools for the SolverBuilder agent.

Three tools:
  SBExecuteCodeTool  — run Python in the sandbox (read-only access to DEMOS.json
                       and rewards/pbebench.py mounted at /workspace)
  SBWriteFileTool    — write a file to the output directory
  SBFinishTool       — signal completion

The agent reads DEMOS.json to understand task structure, writes SOLVER.py and
SOLVER_ALGORITHM.md to the output directory, and calls finish when done.

IMPORTANT: Action/Observation class names must be globally unique — prefixed
with "SB" (SolverBuilder) to avoid collisions with other tools in the registry.
"""

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

class SBExecuteCodeAction(Action):
    code: str = Field(
        description=(
            "Python code to run in the sandbox. "
            "DEMOS.json is at /workspace/DEMOS.json. "
            "The verifier is at /workspace/rewards/pbebench.py (import via sys.path). "
            "Print results to stdout."
        )
    )


class SBExecuteCodeObservation(Observation):
    ok: bool = False
    stdout: str = ""
    stderr: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.ok:
            out = self.stdout.strip() or "(no output)"
            return [TextContent(text=f"stdout:\n{out}")]
        err = self.stderr.strip()[:800] or "(no stderr)"
        return [TextContent(text=f"Execution failed.\nstderr:\n{err}")]


class SBExecuteCodeExecutor(ToolExecutor[SBExecuteCodeAction, SBExecuteCodeObservation]):
    def __init__(self, sandbox, workspace_dir: str, extra_binds: list):
        self._sandbox = sandbox
        self._workspace_dir = workspace_dir
        self._extra_binds = extra_binds

    def __call__(self, action: SBExecuteCodeAction, conversation=None) -> SBExecuteCodeObservation:
        # Inject sys.path so code can do: import sys; sys.path.insert(0,'/workspace')
        preamble = "import sys; sys.path.insert(0, '/workspace')\n"
        ok, stdout, stderr = self._sandbox.run_code(
            preamble + action.code,
            lib_dir=None,
            timeout=60,
            extra_binds=self._extra_binds,
        )
        return SBExecuteCodeObservation(ok=ok, stdout=stdout, stderr=stderr)


class SBExecuteCodeTool(ToolDefinition[SBExecuteCodeAction, SBExecuteCodeObservation]):
    @classmethod
    def create(cls, sandbox, workspace_dir: str, extra_binds: list) -> "list[SBExecuteCodeTool]":
        return [cls(
            description=(
                "Execute Python code in the sandbox. "
                "Use this to inspect DEMOS.json, test logic snippets, or "
                "validate a candidate program against the verifier. "
                "DEMOS.json is at /workspace/DEMOS.json; "
                "verifier at /workspace/rewards/pbebench.py."
            ),
            action_type=SBExecuteCodeAction,
            observation_type=SBExecuteCodeObservation,
            executor=SBExecuteCodeExecutor(sandbox, workspace_dir, extra_binds),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# WriteFile
# ──────────────────────────────────────────────────────────────────────────────

class SBWriteFileAction(Action):
    filename: str = Field(
        description=(
            "Filename to write (basename only, no path). "
            "Use 'SOLVER.py' or 'SOLVER_ALGORITHM.md'."
        )
    )
    content: str = Field(description="Complete file content to write.")


class SBWriteFileObservation(Observation):
    ok: bool = False
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class SBWriteFileExecutor(ToolExecutor[SBWriteFileAction, SBWriteFileObservation]):
    def __init__(self, output_dir: str):
        self._output_dir = output_dir

    def __call__(self, action: SBWriteFileAction, conversation=None) -> SBWriteFileObservation:
        filename = os.path.basename(action.filename.strip())
        if not filename:
            return SBWriteFileObservation(ok=False, message="filename must not be empty.")
        allowed = {"SOLVER.py", "SOLVER_ALGORITHM.md"}
        if filename not in allowed:
            return SBWriteFileObservation(
                ok=False,
                message=f"Unexpected filename '{filename}'. Only {sorted(allowed)} are expected.",
            )
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, filename)
        with open(path, "w") as f:
            f.write(action.content)
        return SBWriteFileObservation(
            ok=True,
            message=f"Written {filename} ({len(action.content)} chars) to {path}",
        )


class SBWriteFileTool(ToolDefinition[SBWriteFileAction, SBWriteFileObservation]):
    @classmethod
    def create(cls, output_dir: str) -> "list[SBWriteFileTool]":
        return [cls(
            description=(
                "Write a file to the output directory. "
                "Use filename='SOLVER.py' for the solver implementation and "
                "'SOLVER_ALGORITHM.md' for the algorithm description."
            ),
            action_type=SBWriteFileAction,
            observation_type=SBWriteFileObservation,
            executor=SBWriteFileExecutor(output_dir),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# Finish
# ──────────────────────────────────────────────────────────────────────────────

class SBFinishAction(Action):
    summary: str = Field(
        description=(
            "Brief summary of what was written: "
            "which files were produced and the key design decisions."
        )
    )


class SBFinishObservation(Observation):
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class SBFinishExecutor(ToolExecutor[SBFinishAction, SBFinishObservation]):
    def __init__(self, done_path: str):
        self._done_path = done_path

    def __call__(self, action: SBFinishAction, conversation=None) -> SBFinishObservation:
        os.makedirs(os.path.dirname(self._done_path), exist_ok=True)
        with open(self._done_path, "w") as f:
            f.write(action.summary)
        if conversation is not None:
            state = conversation.state
            cur = state.execution_status
            finished = type(cur).FINISHED
            state.execution_status = finished
        return SBFinishObservation(
            message=f"Done. Summary: {action.summary[:200]}"
        )


class SBFinishTool(ToolDefinition[SBFinishAction, SBFinishObservation]):
    @classmethod
    def create(cls, done_path: str) -> "list[SBFinishTool]":
        return [cls(
            description=(
                "Signal that you have finished writing all required files. "
                "Call this after both SOLVER.py and SOLVER_ALGORITHM.md have been written. "
                "Include a brief summary of the design."
            ),
            action_type=SBFinishAction,
            observation_type=SBFinishObservation,
            executor=SBFinishExecutor(done_path),
        )]
