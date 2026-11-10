"""Tools for the autoresearch HPO SolverBuilder.

The shared tools in ``tools.py`` are written for the example-based domains: the write
tool only accepts ``SOLVER.py`` / ``SOLVER_ALGORITHM.md`` and the execute tool's
descriptions point the agent at the PBEBench verifier. The HPO builder writes
``SOLVER_HPO.py`` / ``SOLVER_HPO_ALGORITHM.md`` and tests them against
``rewards/autoresearch_hpo.py``, so it gets its own execute and write tools; the finish
tool is reused unchanged.

Tool names are derived by the SDK from the class names: ``hpo_execute_code`` and
``hpo_write_file`` (plus ``sb_finish``). Action/Observation class names must be
globally unique in the SDK's registry, hence the "HPO" prefix.
"""
import os
from collections.abc import Sequence

from pydantic import Field

from openhands.sdk import Action, ImageContent, Observation, TextContent, ToolDefinition
from openhands.sdk.tool import ToolExecutor

SOLVER_FILENAME = "SOLVER_HPO.py"
ALGORITHM_FILENAME = "SOLVER_HPO_ALGORITHM.md"
ALLOWED_FILENAMES = (SOLVER_FILENAME, ALGORITHM_FILENAME)

# Where things are inside the sandbox.
WORKSPACE_DIR = "/workspace"
SANDBOX_DEMOS = f"{WORKSPACE_DIR}/DEMOS.json"
SANDBOX_TASKS = f"{WORKSPACE_DIR}/SYNTHETIC_TASKS.jsonl"
SANDBOX_SOLVER_DIR = f"{WORKSPACE_DIR}/solver"

MAX_STDOUT_CHARS = 20000
MAX_STDERR_CHARS = 4000


def _clip_head(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more characters not shown]"


def _clip_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"[{len(text) - limit} earlier characters not shown] ...\n" + text[-limit:]


def run_in_sandbox(sandbox, code: str, extra_binds: list, timeout=None):
    """Run ``code`` with the HPO import paths set up; returns (ok, stdout, stderr).

    ``from rewards.autoresearch_hpo import ...`` and ``from SOLVER_HPO import ...``
    both work inside. ``timeout=None`` uses the sandbox's own per-call limit.
    """
    preamble = (f"import sys; sys.path.insert(0, {SANDBOX_SOLVER_DIR!r}); "
                f"sys.path.insert(0, {WORKSPACE_DIR!r})\n")
    return sandbox.run_code(preamble + code, lib_dir=None, timeout=timeout,
                            extra_binds=extra_binds)


# ──────────────────────────────────────────────────────────────────────────────
# ExecuteCode
# ──────────────────────────────────────────────────────────────────────────────

class HPOExecuteCodeAction(Action):
    code: str = Field(
        description=(
            "Python code to run in the sandbox (standard library only, no GPU, no "
            "network). Print results to stdout."
        )
    )


class HPOExecuteCodeObservation(Observation):
    ok: bool = False
    stdout: str = ""
    stderr: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        out = _clip_head(self.stdout.strip(), MAX_STDOUT_CHARS) or "(no output)"
        if self.ok:
            return [TextContent(text=f"stdout:\n{out}")]
        # The end of a traceback names the error, so keep the tail of stderr.
        err = _clip_tail(self.stderr.strip(), MAX_STDERR_CHARS) or "(no stderr)"
        return [TextContent(text=f"Execution failed.\nstderr:\n{err}\nstdout:\n{out}")]


class HPOExecuteCodeExecutor(ToolExecutor[HPOExecuteCodeAction, HPOExecuteCodeObservation]):
    def __init__(self, sandbox, extra_binds: list):
        self._sandbox = sandbox
        self._extra_binds = extra_binds

    def __call__(self, action: HPOExecuteCodeAction, conversation=None) -> HPOExecuteCodeObservation:
        # `from rewards.autoresearch_hpo import ...` and `from SOLVER_HPO import ...`
        ok, stdout, stderr = run_in_sandbox(self._sandbox, action.code, self._extra_binds)
        return HPOExecuteCodeObservation(ok=ok, stdout=stdout, stderr=stderr)


class HPOExecuteCodeTool(ToolDefinition[HPOExecuteCodeAction, HPOExecuteCodeObservation]):
    @classmethod
    def create(cls, sandbox, extra_binds: list) -> "list[HPOExecuteCodeTool]":
        return [cls(
            description=(
                "Execute Python code in the sandbox. Use it to inspect the demos at "
                f"{SANDBOX_DEMOS}, to try out logic, and to run the written solver "
                f"(importable as `from SOLVER_HPO import solve_hpo`) through "
                f"`rewards.autoresearch_hpo.run_episode` on the synthetic practice "
                f"tasks at {SANDBOX_TASKS}. Each call starts a fresh interpreter and "
                f"is limited to {sandbox.timeout} s."
            ),
            action_type=HPOExecuteCodeAction,
            observation_type=HPOExecuteCodeObservation,
            executor=HPOExecuteCodeExecutor(sandbox, extra_binds),
        )]


# ──────────────────────────────────────────────────────────────────────────────
# WriteFile
# ──────────────────────────────────────────────────────────────────────────────

class HPOWriteFileAction(Action):
    filename: str = Field(
        description=(
            "Filename to write (basename only, no path). "
            f"Use '{SOLVER_FILENAME}' or '{ALGORITHM_FILENAME}'."
        )
    )
    content: str = Field(description="Complete file content to write.")


class HPOWriteFileObservation(Observation):
    ok: bool = False
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class HPOWriteFileExecutor(ToolExecutor[HPOWriteFileAction, HPOWriteFileObservation]):
    def __init__(self, output_dir: str):
        self._output_dir = output_dir

    def __call__(self, action: HPOWriteFileAction, conversation=None) -> HPOWriteFileObservation:
        filename = action.filename.strip()
        if filename not in ALLOWED_FILENAMES:
            return HPOWriteFileObservation(
                ok=False,
                message=f"Unexpected filename '{filename}'. Only {list(ALLOWED_FILENAMES)} "
                        f"can be written (basename only, no directories).",
            )
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(action.content)
        return HPOWriteFileObservation(
            ok=True,
            message=f"Written {filename} ({len(action.content)} chars). It is visible in "
                    f"the sandbox at {SANDBOX_SOLVER_DIR}/{filename}.",
        )


class HPOWriteFileTool(ToolDefinition[HPOWriteFileAction, HPOWriteFileObservation]):
    @classmethod
    def create(cls, output_dir: str) -> "list[HPOWriteFileTool]":
        return [cls(
            description=(
                "Write a file to the output directory. "
                f"Use filename='{SOLVER_FILENAME}' for the solver implementation and "
                f"'{ALGORITHM_FILENAME}' for the algorithm description. Writing a file "
                "again replaces it."
            ),
            action_type=HPOWriteFileAction,
            observation_type=HPOWriteFileObservation,
            executor=HPOWriteFileExecutor(output_dir),
        )]
