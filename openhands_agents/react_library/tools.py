"""
Custom OpenHands SDK tools for the ReAct+Library agent.

Four tools:
  ExecuteCodeTool   — run code in Apptainer sandbox; library available
  AddToLibraryTool  — validate + add function to PkgLibrary
  CheckRewardTool   — call the reward function on a candidate answer inline
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


_EXECUTE_CODE_WARN = (
    "\n\n⚠ WARNING: You have called execute_code multiple times without calling "
    "check_reward. Stop manual testing — call check_reward NOW with your best "
    "candidate answer. It scores all examples at once and tells you exactly what "
    "is wrong. Do NOT call execute_code again until you have called check_reward."
)

_EXECUTE_CODE_HARD_STOP = (
    "\n\n🛑 HARD STOP: You have called execute_code too many times in a row. "
    "You MUST call check_reward next. Do NOT call execute_code again."
)


class RLExecuteCodeExecutor(ToolExecutor[RLExecuteCodeAction, RLExecuteCodeObservation]):
    def __init__(self, sandbox, library: PkgLibrary):
        self._sandbox = sandbox
        self._library = library
        self._calls_since_check_reward = 0  # reset by RLCheckRewardExecutor

    def __call__(self, action: RLExecuteCodeAction, conversation=None) -> RLExecuteCodeObservation:
        self._calls_since_check_reward += 1
        ok, stdout, stderr = self._sandbox.run_code(
            action.code,
            lib_dir=self._library.pkg_dir if len(self._library) > 0 else None,
        )
        if ok and len(self._library) > 0:
            import re
            imported = re.findall(r"from\s+library\s+import\s+(\w+)", action.code)
            imported += re.findall(r"from\s+library\s+import\s+\(([^)]+)\)", action.code)
            names = [n.strip() for seg in imported for n in seg.split(",") if n.strip()]
            used = [n for n in names if n in self._library]
            if used:
                self._library.increment_usage(used)
        # Inject escalating warnings to force check_reward after 2+ consecutive execute_code calls
        suffix = ""
        if self._calls_since_check_reward == 2:
            suffix = _EXECUTE_CODE_WARN
        elif self._calls_since_check_reward >= 3:
            suffix = _EXECUTE_CODE_HARD_STOP
        obs = RLExecuteCodeObservation(ok=ok, stdout=stdout + suffix, stderr=stderr)
        return obs


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
# CheckReward
# ──────────────────────────────────────────────────────────────────────────────

class RLCheckRewardAction(Action):
    answer: str = Field(description="Candidate answer string to evaluate against the task verifier.")


class RLCheckRewardObservation(Observation):
    reward: float = 0.0
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=f"Reward: {self.reward:.3f}\n{self.message}")]


class RLCheckRewardExecutor(ToolExecutor[RLCheckRewardAction, RLCheckRewardObservation]):
    def __init__(self, reward_fn: Callable, entry: Any, exec_executor: "RLExecuteCodeExecutor | None" = None):
        self._reward_fn = reward_fn
        self._entry = entry
        self._exec_executor = exec_executor
        self.call_log: list[dict] = []  # accumulates every check_reward invocation

    def __call__(self, action: RLCheckRewardAction, conversation=None) -> RLCheckRewardObservation:
        # Reset the execute_code consecutive-call counter
        if self._exec_executor is not None:
            self._exec_executor._calls_since_check_reward = 0
        result = self._reward_fn(action.answer, True, self._entry)
        reward = float(result.get("value", 0.0))
        message = result.get("message", "")
        self.call_log.append({"reward": reward, "message": message, "answer": action.answer})
        return RLCheckRewardObservation(reward=reward, message=message)


class CheckRewardTool(ToolDefinition[RLCheckRewardAction, RLCheckRewardObservation]):
    @classmethod
    def create(
        cls,
        reward_fn: Callable,
        entry: Any,
        exec_executor: "RLExecuteCodeExecutor | None" = None,
    ) -> "tuple[list[CheckRewardTool], RLCheckRewardExecutor]":
        """Returns (tools_list, executor) so caller can read executor.call_log after the run."""
        executor = RLCheckRewardExecutor(reward_fn, entry, exec_executor=exec_executor)
        tool = cls(
            description=(
                "Check your candidate answer against the task verifier. "
                "Returns a reward in [0, 1] and feedback explaining what is wrong. "
                "Use this to verify your answer before calling finish, and to guide fixes."
            ),
            action_type=RLCheckRewardAction,
            observation_type=RLCheckRewardObservation,
            executor=executor,
        )
        return [tool], executor


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
        # Signal the SDK to stop the agent loop.
        if conversation is not None:
            state = conversation.state
            cur = state.execution_status
            finished = type(cur).FINISHED
            state.execution_status = finished
        return RLFinishObservation(message=f"Answer submitted: {action.answer[:100]}")


class FinishTool(ToolDefinition[RLFinishAction, RLFinishObservation]):
    @classmethod
    def create(cls, answer_path: str) -> "list[FinishTool]":
        return [cls(
            description="Submit the final answer. Call this when check_reward returns 1.0 or you are satisfied with your best answer.",
            action_type=RLFinishAction,
            observation_type=RLFinishObservation,
            executor=RLFinishExecutor(answer_path),
        )]
