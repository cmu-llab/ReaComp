"""Autoresearch HPO SolverBuilder controller.

Subclasses ``SolverBuilderController`` the way ``listfn_controller.py`` does (the base
``solver_type in {pbe, slr}`` branches stay untouched) and overrides the
domain-specific pieces: system prompt, @-reference file note, tools, sandbox binds and
output filenames.

Induces ``SOLVER_HPO.py`` + ``SOLVER_HPO_ALGORITHM.md`` implementing
``solve_hpo(task, history)``: a symbolic hyperparameter-optimisation policy for the
nanochat search space of ``rewards/autoresearch_hpo.py``.

Three things differ from the example-based domains:

  * The real objective (300 s of GPU training per trial) cannot run in the sandbox, so
    a file of synthetic practice tasks is mounted next to the demos. The agent can run
    whole episodes of its solver on them in milliseconds to check behaviour (valid
    configs, reaction to failures, budget use). The synthetic landscape is not
    nanochat, and the prompts say so.
  * The output directory is mounted read-only into the sandbox, so the agent tests the
    file it actually wrote rather than a copy pasted into a code snippet.
  * After the conversation the written solver is run once more on the practice tasks,
    inside the sandbox, and the outcome is returned as ``smoke_test``. This is a sanity
    check of the artefact (does it import, does it return usable configurations), not
    an evaluation; ``scripts/eval_hpo_solver.py`` does that.
"""
import json
import logging
import os
import tempfile
import traceback
import uuid

from openhands.sdk import Agent, Conversation

from .controller import SolverBuilderController
from .hpo_tools import (
    ALGORITHM_FILENAME, SANDBOX_DEMOS, SANDBOX_SOLVER_DIR, SANDBOX_TASKS, SOLVER_FILENAME,
    WORKSPACE_DIR, HPOExecuteCodeTool, HPOWriteFileTool, run_in_sandbox,
)
from .tools import SBFinishTool

logger = logging.getLogger(__name__)

SMOKE_TEST_TIMEOUT_S = 900


def system_prompt(execute_timeout_s: int) -> str:
    """The builder agent's system prompt.

    The SDK loads this text as a Jinja template, so it must stay free of Jinja
    delimiters (doubled braces, brace-percent, brace-hash).
    """
    return f"""\
You are an expert in hyperparameter optimization, neural network training dynamics and Python.

Your task is to write a symbolic hyperparameter-optimization policy for the autoresearch
HPO benchmark: tuning 14 hyperparameters of a GPT-style language model (nanochat) to
minimize val_bpb under a fixed training-time budget. You will be given a detailed
specification in the user message.

You have three tools:

  hpo_execute_code(code)
      Run Python code in a sandbox (standard library only, {execute_timeout_s} s per call,
      no GPU, no network, a fresh interpreter each call).
      The demos are mounted at {SANDBOX_DEMOS}.
      The benchmark module is importable (sys.path already includes {WORKSPACE_DIR}):
          from rewards.autoresearch_hpo import run_episode, normalize_config, SEARCH_SPACE
      Synthetic practice tasks are at {SANDBOX_TASKS} (one JSON record per line).
      Files you have written are importable:
          from SOLVER_HPO import solve_hpo
      (they are also readable at {SANDBOX_SOLVER_DIR}/).

  hpo_write_file(filename, content)
      Write a file to the output directory. Only '{SOLVER_FILENAME}' and
      '{ALGORITHM_FILENAME}' are accepted. Writing a file again replaces it.
      You MUST write both files before finishing.

  sb_finish(summary)
      Signal that you are done. Call after both files have been written and tested.

Workflow:
  1. Read the specification in the user message carefully.
  2. Use hpo_execute_code to inspect {SANDBOX_DEMOS}: what the model looked at, how it
     reasoned, which proposals improved val_bpb and which failed (and why).
  3. Design a search strategy from what you observe: how to explore, how to exploit the
     best configuration so far, how to stay inside the memory limit, how to react to
     failures, and how to use the remaining budget.
  4. Write {SOLVER_FILENAME} with the complete implementation.
  5. Run full episodes of it on the synthetic practice tasks:
         import json
         from rewards.autoresearch_hpo import run_episode
         from SOLVER_HPO import solve_hpo
         for line in open("{SANDBOX_TASKS}"):
             entry = json.loads(line)
             episode = run_episode(solve_hpo, entry)
             score = episode["score"]
             print(entry["task_id"], episode["stop_reason"], score["n_trials"],
                   score["status_counts"], score["baseline_val_bpb"], score["best_val_bpb"])
     Check that no trial is 'invalid', that failures become rarer as the episode goes
     on, and that the best val_bpb improves on the baseline. Fix and rewrite as needed.
  6. Write {ALGORITHM_FILENAME} with a clear description of the algorithm.
  7. Call sb_finish with a brief summary.

Requirements:
  - {SOLVER_FILENAME} must define solve_hpo(task, history) and return a dict with a
    "config" key that maps hyperparameter names to values.
  - Use only the Python standard library. No LLM calls, no network, no files.
  - The decision must be a function of (task, history) alone: no global state carried
    between calls, and any randomness seeded from task["seed"] and the history, so a
    resumed episode continues exactly as an uninterrupted one would.
  - Never return a configuration outside the search space. The harness clamps numbers
    and snaps powers of two, but a non-numeric value or an unknown WINDOW_PATTERN
    costs a trial slot.
  - The synthetic practice tasks only share the rough shape of the real problem
    (a memory limit, divergence at aggressive learning rates, a smooth optimum). They
    are NOT nanochat. Use them to test behaviour, never to pick constants: do not
    hard-code values that happen to score well on them, and do not read their hidden
    fields.
  - Do not replay configurations from the demos as a fixed list either. Encode the
    strategy, so the solver adapts to whatever the history shows.

Do NOT produce placeholder code. Write a complete, working implementation.
"""


_FILE_NOTE = (
    "File path reference guide (for @-references in the spec below):\n"
    f"  @demos/DEMOS_AUTORESEARCH_HPO_*.json → {SANDBOX_DEMOS}   (use hpo_execute_code to read)\n"
    "  @rewards/autoresearch_hpo.py → importable as: "
    "from rewards.autoresearch_hpo import run_episode, normalize_config, SEARCH_SPACE\n"
    f"  @data/autoresearch_hpo/synthetic_tasks.jsonl → {SANDBOX_TASKS}\n"
    f"  @{SOLVER_FILENAME} → write via hpo_write_file(filename='{SOLVER_FILENAME}', content=...)\n"
    f"  @{ALGORITHM_FILENAME} → write via "
    f"hpo_write_file(filename='{ALGORITHM_FILENAME}', content=...)\n"
    "\n"
    "---\n\n"
)

# Runs inside the sandbox. __MARKER__ is replaced by a per-call nonce so the line with
# the report can be told apart from anything the solver prints.
_SMOKE_TEST_CODE = f"""\
import json, traceback
from rewards.autoresearch_hpo import STATUS_INVALID, run_episode

report = {{"ok": False, "error": None, "tasks": []}}
try:
    from SOLVER_HPO import solve_hpo
    with open({SANDBOX_TASKS!r}) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    for entry in entries:
        episode = run_episode(solve_hpo, entry)
        score = episode["score"]
        invalid = [t["error"] for t in episode["trials"] if t["status"] == STATUS_INVALID]
        report["tasks"].append({{
            "task_id": entry["task_id"], "stop_reason": episode["stop_reason"],
            "n_trials": score["n_trials"], "status_counts": score["status_counts"],
            "baseline_val_bpb": score["baseline_val_bpb"],
            "best_val_bpb": score["best_val_bpb"],
            "first_invalid_error": invalid[0] if invalid else None,
        }})
    report["ok"] = bool(entries) and all(
        task["status_counts"][STATUS_INVALID] == 0 for task in report["tasks"])
except BaseException:
    report["error"] = traceback.format_exc()[-2000:]
print("__MARKER__" + json.dumps(report))
"""


class HPOSolverBuilderController(SolverBuilderController):
    """Runs a single conversation that writes SOLVER_HPO.py + SOLVER_HPO_ALGORITHM.md.

    Takes the base controller's parameters plus ``smoke_tasks_path``: host path of the
    synthetic task file mounted into the sandbox for practice episodes.
    """

    def __init__(self, *args, smoke_tasks_path: str, **kwargs):
        # A sentinel the base class does not special-case; the prompt file it wrote
        # in its __init__ (the PBE one) is replaced below.
        kwargs["solver_type"] = "hpo"
        super().__init__(*args, **kwargs)
        self.smoke_tasks_path = os.path.abspath(smoke_tasks_path)
        with open(self._system_prompt_file, "w", encoding="utf-8") as f:
            f.write(system_prompt(self.sandbox.timeout))

    def _extra_binds(self) -> list:
        """Base binds (demos, rewards/) plus the practice tasks and the output dir."""
        return super()._extra_binds() + [
            (self.smoke_tasks_path, SANDBOX_TASKS, "ro"),
            (self.output_dir, SANDBOX_SOLVER_DIR, "ro"),
        ]

    def _make_agent(self, task_dir: str) -> Agent:
        done_path = os.path.join(task_dir, "done.txt")
        tool_instances = [
            *HPOExecuteCodeTool.create(self.sandbox, extra_binds=self._extra_binds()),
            *HPOWriteFileTool.create(self.output_dir),
            *SBFinishTool.create(done_path),
        ]
        agent = Agent(
            llm=self._llm,
            tools=[],
            include_default_tools=[],
            system_prompt_filename=self._system_prompt_file,
        )
        agent.__pydantic_private__["_tools"] = {t.name: t for t in tool_instances}
        agent.__pydantic_private__["_initialized"] = True
        logger.info("HPO SolverBuilder agent tools: %s", list(agent.tools_map.keys()))
        return agent

    def smoke_test(self) -> dict:
        """Run the written solver on the practice tasks inside the sandbox.

        Returns ``{"ok", "error", "tasks"}``; ``ok`` means every practice episode ran
        to its end without a single invalid suggestion.
        """
        marker = f"SMOKE_TEST_{uuid.uuid4().hex}:"
        ok, stdout, stderr = run_in_sandbox(
            self.sandbox, _SMOKE_TEST_CODE.replace("__MARKER__", marker),
            self._extra_binds(), timeout=SMOKE_TEST_TIMEOUT_S)
        lines = [line for line in stdout.splitlines() if line.startswith(marker)]
        if not lines:
            return {"ok": False, "tasks": [],
                    "error": (stderr.strip()[-2000:] or "no smoke-test report in the output")}
        try:
            report = json.loads(lines[-1][len(marker):])
        except ValueError as exc:
            return {"ok": False, "tasks": [], "error": f"unreadable smoke-test report: {exc}"}
        report["ok"] = bool(report.get("ok")) and ok
        return report

    def build(self, building_prompt: str) -> dict:
        # The output dir is bind-mounted into the sandbox, so it has to exist before
        # the first execute call.
        os.makedirs(self.output_dir, exist_ok=True)
        task_dir = tempfile.mkdtemp(prefix="oh_sb_hpo_task_")
        done_path = os.path.join(task_dir, "done.txt")

        agent = self._make_agent(task_dir)
        conversation = Conversation(
            agent=agent,
            workspace=task_dir,
            max_iteration_per_run=self.max_steps,
        )
        conversation.send_message(_FILE_NOTE + building_prompt)

        trajectory = []
        try:
            conversation.run()
        except Exception as exc:
            logger.warning("hpo solver_builder conversation error: %s\n%s",
                           exc, traceback.format_exc())
        try:
            trajectory = [e.model_dump() for e in conversation.state.events]
        except Exception as exc:
            logger.warning("hpo solver_builder: could not dump the trajectory: %s", exc)

        solver_path = os.path.join(self.output_dir, SOLVER_FILENAME)
        algorithm_path = os.path.join(self.output_dir, ALGORITHM_FILENAME)
        summary = ""
        if os.path.exists(done_path):
            with open(done_path, encoding="utf-8") as f:
                summary = f.read().strip()

        success = os.path.exists(solver_path) and os.path.exists(algorithm_path)
        smoke = self.smoke_test() if os.path.exists(solver_path) else None
        logger.info("hpo solver_builder: success=%s solver=%s algorithm=%s smoke_ok=%s",
                    success, os.path.exists(solver_path), os.path.exists(algorithm_path),
                    smoke["ok"] if smoke else None)
        return {
            "success": success,
            "solver_path": solver_path if os.path.exists(solver_path) else None,
            "algorithm_path": algorithm_path if os.path.exists(algorithm_path) else None,
            "summary": summary,
            "smoke_test": smoke,
            "_trajectory": trajectory,
        }
