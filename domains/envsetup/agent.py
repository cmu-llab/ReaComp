"""
Bash agent for environment-setup tasks, built on the OpenHands SDK.

One conversation per task with two tools:

    bash(command, reason)  - run a command in the task container's persistent shell
    finish(report, success)

It serves three roles in the ReaComp pipeline:
  * trace generation: every run records (reason, command, exit_code, output) steps,
    which ``demos.py`` turns into solver-building demos;
  * LLM baseline: same container, same tools, same budget as the induced solver;
  * hybrid fallback: with ``--warm-start-solver`` the induced solver runs first in the
    container and, if it does not claim success, the agent continues from that state
    with the solver's command log in context.

    python -m domains.envsetup.agent --split setupbench_dev \
        --base-url http://localhost:8000/v1 --model openai/Qwen/Qwen3.8-27B \
        --output outputs/envsetup/agent_setupbench_dev.jsonl --workers 2 --max-steps 60

Rows are quick_eval-compatible (task_index, solved, answer=<script>, best_reward,
token_usage, ...) plus ``trajectory`` (the step list) and verifier details.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import ClassVar, List, Optional

from pydantic import Field, SecretStr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from openhands.sdk import LLM, Action, Agent, Conversation, ImageContent, Observation, TextContent, ToolDefinition  # noqa: E402
from openhands.sdk.tool import ToolExecutor  # noqa: E402

from domains.envsetup import tasks as T  # noqa: E402
from domains import get_domain  # noqa: E402

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an autonomous software engineer working inside a Linux container where you are root.
The task is described in the user message. Complete it end to end without asking questions.

Tools
  bash(command, reason)
      Runs a command in a persistent bash session: the working directory and exported
      variables persist between calls and background processes keep running.
      stdin is /dev/null, so every command must be non-interactive
      (apt-get install -y, pip install --no-input, yes | ..., --non-interactive flags).
      Each call has a timeout of {command_timeout} seconds; long builds should be split or
      run with nohup in the background and polled.
      Output is truncated to the last {max_output_chars} characters.
  finish(report, success)
      Ends the session. Call it as soon as the task is complete (success=true) or when
      you cannot make further progress (success=false). Never stop without calling it.

Guidelines
  - Start by inspecting the repository (ls, README, dependency manifests, CI configs).
  - Install system toolchains before language-level packages.
  - Prefer the project's own dependency manager and lock files.
  - Verify your work before finishing: import the package, run the build or the test
    entry point that the task implies.
  - You have at most {max_steps} tool calls. Do not waste them on repeated inspection.
"""

WARM_START_NOTE = """\

A deterministic setup program already ran in this container before you. It executed the
commands below (exit codes in brackets) and its own self-check reported: {claim}.
Continue from the current container state; do not redo work that already succeeded.

{log}
"""


# --------------------------------------------------------------------------- tools
class EnvBashAction(Action):
    command: str = Field(description="Bash command(s) to run in the persistent session.")
    reason: str = Field(description="One sentence: why you run this command / what you expect to learn or achieve.")


class EnvBashObservation(Observation):
    exit_code: int = 0
    output: str = ""
    timed_out: bool = False

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        head = f"exit_code: {self.exit_code}" + (" (timed out; process group killed, shell reset)" if self.timed_out else "")
        return [TextContent(text=f"{head}\n{self.output.strip() or '(no output)'}")]


class EnvBashExecutor(ToolExecutor[EnvBashAction, EnvBashObservation]):
    def __init__(self, shell, steps: List[dict], max_output_chars: int):
        self._shell = shell
        self._steps = steps
        self._max = max_output_chars

    def __call__(self, action: EnvBashAction, conversation=None) -> EnvBashObservation:
        r = self._shell.run(action.command)
        out = r.output[-self._max:]
        self._steps.append({"reason": action.reason, "command": action.command, "exit_code": r.exit_code,
                            "timed_out": r.timed_out, "elapsed": round(r.elapsed, 1), "output": out[-4000:]})
        return EnvBashObservation(exit_code=r.exit_code, output=out, timed_out=r.timed_out)


class EnvBashTool(ToolDefinition[EnvBashAction, EnvBashObservation]):
    name: ClassVar[str] = "bash"

    @classmethod
    def create(cls, shell, steps: List[dict], max_output_chars: int) -> "list[EnvBashTool]":
        return [cls(description="Run a bash command in the task container's persistent shell.",
                    action_type=EnvBashAction, observation_type=EnvBashObservation,
                    executor=EnvBashExecutor(shell, steps, max_output_chars))]


class EnvFinishAction(Action):
    report: str = Field(description="What was done and how you verified it.")
    success: bool = Field(description="true if you believe the task is complete.")


class EnvFinishObservation(Observation):
    message: str = ""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.message)]


class EnvFinishExecutor(ToolExecutor[EnvFinishAction, EnvFinishObservation]):
    def __init__(self, done: dict):
        self._done = done

    def __call__(self, action: EnvFinishAction, conversation=None) -> EnvFinishObservation:
        self._done.update({"summary": action.report, "success": bool(action.success)})
        if conversation is not None:
            state = conversation.state
            state.execution_status = type(state.execution_status).FINISHED
        return EnvFinishObservation(message="Session finished.")


class EnvFinishTool(ToolDefinition[EnvFinishAction, EnvFinishObservation]):
    name: ClassVar[str] = "finish"

    @classmethod
    def create(cls, done: dict) -> "list[EnvFinishTool]":
        return [cls(description="Finish the session once the task is complete or no further progress is possible.",
                    action_type=EnvFinishAction, observation_type=EnvFinishObservation,
                    executor=EnvFinishExecutor(done))]


# -------------------------------------------------------------------------- runner
class EnvSetupAgentRunner:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", max_steps: int = 60,
                 max_tokens: int = 8192, memory: str = "6g", cpus: float = 2.0, command_timeout: int = 600,
                 max_output_chars: int = 12000, warm_start_solver: Optional[str] = None,
                 solver_budget_s: float = 1800, log_dir: Optional[str] = None, temperature: Optional[float] = None):
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self.max_steps, self.max_tokens = max_steps, max_tokens
        self.memory, self.cpus, self.command_timeout = memory, cpus, command_timeout
        self.max_output_chars = max_output_chars
        self.warm_start_solver = warm_start_solver
        self.solver_budget_s = solver_budget_s
        self.temperature = temperature
        self.log_dir = log_dir or os.path.join(REPO_ROOT, "logs", "envsetup")
        self.domain = get_domain("envsetup")
        os.makedirs(self.log_dir, exist_ok=True)
        fd, self._system_prompt_file = tempfile.mkstemp(suffix=".j2", prefix="envsetup_system_")
        with os.fdopen(fd, "w") as f:
            f.write(SYSTEM_PROMPT.format(command_timeout=command_timeout, max_output_chars=max_output_chars,
                                         max_steps=max_steps))

    def _llm(self) -> LLM:
        kw = dict(model=self.model, base_url=self.base_url, api_key=SecretStr(self.api_key), max_tokens=self.max_tokens)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        return LLM(**kw)

    def _warm_start(self, record: dict, shell) -> dict:
        dom = self.domain
        solve_fn = dom.load_solver(self.warm_start_solver)
        task_view = dict(dom.task_input(record)); task_view["budget_s"] = self.solver_budget_s
        shell.deadline = time.time() + self.solver_budget_s
        t0 = time.time()
        try:
            out = solve_fn(task_view, shell)
            res = out if isinstance(out, dict) else {"success": bool(out)}
            err = None
        except Exception as exc:  # noqa: BLE001
            res, err = {"success": False}, f"{type(exc).__name__}: {exc}"
        shell.deadline = None
        return {"claimed_success": bool(res.get("success")) and err is None, "error": err,
                "n_commands": len(shell.log), "elapsed": round(time.time() - t0, 1),
                "log": [{"command": s.command, "exit_code": s.exit_code} for s in shell.log]}

    def run_task(self, record: dict) -> dict:
        t0 = time.time()
        log_path = os.path.join(self.log_dir, f"agent_{record['task_id']}.log")
        if os.path.exists(log_path):
            os.remove(log_path)
        container = T.prepare_container(record, run_label=f"agent-{os.getpid()}", memory=self.memory, cpus=self.cpus)
        steps: List[dict] = []
        done: dict = {}
        warm: Optional[dict] = None
        agent_ran = False
        token_usage = {"input": 0, "output": 0, "reasoning": 0}
        try:
            shell = container.shell(default_timeout=self.command_timeout, max_output_chars=self.max_output_chars,
                                    log_path=log_path)
            for cmd in record.get("initial_commands", []):
                shell.run(cmd)
            if self.warm_start_solver:
                warm = self._warm_start(record, shell)
            if warm is None or not warm["claimed_success"]:
                agent_ran = True
                llm = self._llm()
                tools = [*EnvBashTool.create(shell, steps, self.max_output_chars), *EnvFinishTool.create(done)]
                agent = Agent(llm=llm, tools=[], include_default_tools=[], system_prompt_filename=self._system_prompt_file)
                agent.__pydantic_private__["_tools"] = {t.name: t for t in tools}
                agent.__pydantic_private__["_initialized"] = True
                task_dir = tempfile.mkdtemp(prefix="envsetup_conv_")
                conversation = Conversation(agent=agent, workspace=task_dir, max_iteration_per_run=self.max_steps,
                                            visualizer=None)
                message = self.domain.agent_task_text(record)
                if warm is not None:
                    log_txt = "\n".join(f"[{s['exit_code']}] {s['command']}" for s in warm["log"][-60:])
                    message += WARM_START_NOTE.format(claim="success" if warm["claimed_success"] else "failure",
                                                      log=log_txt)
                conversation.send_message(message)
                try:
                    conversation.run()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("conversation error on %s: %s", record["task_id"], exc)
                    done.setdefault("error", f"{type(exc).__name__}: {exc}")
                try:
                    u = llm.metrics.accumulated_token_usage
                    token_usage = {"input": u.prompt_tokens, "output": u.completion_tokens,
                                   "reasoning": getattr(u, "reasoning_tokens", 0) or 0}
                except Exception:  # noqa: BLE001
                    pass
            script = shell.script()
            if record["verify"]["mode"] == "setupbench_live":
                ver = self.domain.reward({"container": container}, True, record)
            else:
                ver = self.domain.reward({"script": script}, True, record, memory=self.memory, cpus=self.cpus,
                                         run_label=f"verify-{os.getpid()}")
        finally:
            container.stop()
        value = float(ver.get("value", 0.0))
        return {
            "task_id": record["task_id"],
            "task_index": record.get("task_index"),
            "solved": value >= 1.0,
            "answer": script,
            "best_reward": value,
            "reward_history": [{"iteration": 0, "reward": value}],
            "token_usage": token_usage,
            "cost_summary": {"elapsed_s": round(time.time() - t0, 1)},
            "dataset": record["source"],
            "domain": "envsetup",
            "category": record["category"],
            "language": record.get("language"),
            "base_image": record["base_image"],
            "agent_ran": agent_ran,
            "steps_used": len(steps),
            "claimed_success": done.get("success"),
            "finish_summary": done.get("summary", "")[:2000],
            "agent_error": done.get("error"),
            "warm_start": warm,
            "verify_mode": ver.get("mode"),
            "verify_exit_code": ver.get("exit_code"),
            "verify_message": (ver.get("message") or "")[-1500:],
            "issues_count": ver.get("issues_count"),
            "trajectory": steps,
        }


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="openai/Qwen/Qwen3.8-27B")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--memory", default="6g")
    ap.add_argument("--cpus", type=float, default=2.0)
    ap.add_argument("--command-timeout", type=int, default=600)
    ap.add_argument("--warm-start-solver", default=None, help="run this SOLVER_ENVSETUP.py first (hybrid, warm handoff)")
    ap.add_argument("--solver-budget", type=float, default=1800)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--task-ids", default=None, help="comma-separated subset")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dom = get_domain("envsetup")
    records = dom.load_tasks(args.split, limit=args.limit)
    if args.task_ids:
        keep = set(args.task_ids.split(","))
        records = [r for r in records if r["task_id"] in keep]
    done_ids = set()
    if args.skip_existing and os.path.exists(args.output):
        with open(args.output) as fh:
            done_ids = {json.loads(l)["task_id"] for l in fh if l.strip()}
    records = [r for r in records if r["task_id"] not in done_ids]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    runner = EnvSetupAgentRunner(args.base_url, args.model, args.api_key, args.max_steps, args.max_tokens,
                                 args.memory, args.cpus, args.command_timeout, warm_start_solver=args.warm_start_solver,
                                 solver_budget_s=args.solver_budget, temperature=args.temperature)
    lock = threading.Lock()

    def one(rec):
        try:
            row = runner.run_task(rec)
        except Exception as exc:  # noqa: BLE001
            logger.exception("task %s failed", rec["task_id"])
            row = {"task_id": rec["task_id"], "task_index": rec.get("task_index"), "solved": False, "answer": "",
                   "best_reward": 0.0, "reward_history": [], "token_usage": {"input": 0, "output": 0, "reasoning": 0},
                   "cost_summary": {}, "dataset": rec["source"], "domain": "envsetup", "category": rec["category"],
                   "harness_error": f"{type(exc).__name__}: {exc}", "trajectory": []}
        with lock:
            with open(args.output, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        logger.info("%s solved=%s steps=%s tokens=%s", rec["task_id"], row.get("solved"), row.get("steps_used"),
                    row.get("token_usage"))

    if args.workers <= 1:
        for rec in records:
            one(rec)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(one, r) for r in records]
            for f in as_completed(futs):
                f.result()


if __name__ == "__main__":
    main()
