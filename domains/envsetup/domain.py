"""EnvSetupDomain: environment bootstrapping tasks (SetupBench, EnvBench-Python).

The induced solver drives a real container through a persistent bash session:

    def solve_envsetup(task: dict, shell) -> dict:
        # task  : agent-visible record (problem_statement, category, language, workdir, budget_s, ...)
        # shell : PersistentShell; shell.run(cmd, timeout=None) -> ShellResult(exit_code, output, timed_out)
        # return {"success": <bool, the solver's own claim>, "notes": <str, optional>}

The domain provisions the container, enforces the time budget, records every
command, verifies with the hidden check afterwards and tears the container down.
See ``solver_interface.md`` next to this file for the full contract.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, Optional

from ..base import Domain, SolveOutcome, SplitSpec, REPO_ROOT
from . import tasks as T
from .container import remove_containers_by_label

LOG_DIR = os.path.join(REPO_ROOT, "logs", "envsetup")


class EnvSetupDomain(Domain):
    name = "envsetup"
    description = "Environment setup from a bare container: SetupBench (93) and EnvBench-Python (329)"
    reward_module = "envsetup"
    solver_entrypoints = ("solve_envsetup",)
    agent_visible_keys = frozenset({"task_id", "source", "category", "language", "ecosystem", "base_image",
                                    "workdir", "problem_statement", "initial_commands"})
    building_prompt = "building_prompts/SOLVER_BUILDING_PROMPT_ENVSETUP.md"
    demos = {"default": "demos/DEMOS_ENVSETUP.json"}
    trace_runner = "domains.envsetup.agent"
    solver_filename = "SOLVER_ENVSETUP.py"
    algorithm_filename = "SOLVER_ENVSETUP_ALGORITHM.md"

    DEFAULT_PARAMS = {"solver_budget_s": 1800, "memory": "6g", "cpus": 2.0, "command_timeout": 600}

    def splits(self) -> Dict[str, SplitSpec]:
        d = T.TASKS_DIR
        mk = lambda name, label: SplitSpec(name, os.path.join(d, f"{name}.jsonl"), label, dict(self.DEFAULT_PARAMS))
        return {
            "setupbench": mk("setupbench", "SetupBench"),
            "setupbench_dev": mk("setupbench_dev", "SetupBench-dev"),
            "envbench_python_train": mk("envbench_python_train", "EnvBench-Python-train"),
            "envbench_python_test": mk("envbench_python_test", "EnvBench-Python-test"),
        }

    # -- solver ---------------------------------------------------------------
    def solve(self, solve_fn: Callable, record: dict, split: SplitSpec) -> SolveOutcome:
        p = split.params
        budget = float(p.get("solver_budget_s", 1800))
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, f"solver_{record['task_id']}.log")
        if os.path.exists(log_path):
            os.remove(log_path)
        t0 = time.time()
        container = T.prepare_container(record, run_label=f"solve-{os.getpid()}",
                                        memory=p.get("memory", "6g"), cpus=p.get("cpus", 2.0))
        result: dict = {}
        error: Optional[str] = None
        try:
            sh = container.shell(default_timeout=p.get("command_timeout", 600), log_path=log_path)
            sh.deadline = time.time() + budget
            for cmd in record.get("initial_commands", []):
                sh.run(cmd)
            task_view = dict(self.task_input(record))
            task_view["budget_s"] = budget
            try:
                out = solve_fn(task_view, sh)
                result = out if isinstance(out, dict) else {"success": bool(out)}
            except Exception as exc:  # noqa: BLE001 - a crashing solver is a failed attempt, not a harness failure
                error = f"{type(exc).__name__}: {exc}"
            solve_elapsed = round(time.time() - t0, 1)
            script = sh.script()
            n_failed = sum(1 for s in sh.log if s.exit_code != 0)
            if record["verify"]["mode"] == "setupbench_live":
                ver = self.reward({"container": container}, True, record)
            else:
                ver = self.reward({"script": script}, True, record, memory=p.get("memory", "6g"),
                                  cpus=p.get("cpus", 2.0), run_label=f"verify-{os.getpid()}")
        finally:
            container.stop()
        value = float(ver.get("value", 0.0))
        extras = {
            "claimed_success": (None if error else bool(result.get("success"))),
            "solver_notes": (result.get("notes") or "")[:2000],
            "solver_error": error,
            "n_commands": len(sh.log),
            "n_failed_commands": n_failed,
            "solve_elapsed_s": solve_elapsed,
            "verify_mode": ver.get("mode"),
            "verify_exit_code": ver.get("exit_code"),
            "verify_message": (ver.get("message") or "")[-1500:],
            "issues_count": ver.get("issues_count"),
        }
        return SolveOutcome(success=value >= 1.0, score=value, program=script, extras=extras)

    def cleanup_after_timeout(self, record: dict, split: SplitSpec) -> None:
        remove_containers_by_label(f"reacomp.task_id={record['task_id']}")

    # -- rows / agents --------------------------------------------------------
    def row_extras(self, record: dict) -> dict:
        return {"task_id": record["task_id"], "source": record["source"], "category": record["category"],
                "language": record.get("language"), "ecosystem": record.get("ecosystem"),
                "base_image": record["base_image"], "image_substituted": record.get("image_substituted", False)}

    def agent_task_text(self, record: dict, split: Optional[SplitSpec] = None) -> str:
        return record["problem_statement"]

    # -- demos ----------------------------------------------------------------
    def demo_group_key(self, demo: dict):
        return (demo.get("category") or "all", bool(demo.get("success")))

    def demo_prompt(self, record: dict) -> str:
        """Problem statement plus a snapshot of the repository / fixtures the trace's author saw."""
        text = record["problem_statement"]
        try:
            host = T.prepare_repo(record)
        except Exception as exc:  # noqa: BLE001
            host = None
            text += f"\n\n(repository snapshot unavailable: {exc})"
        text += "\n\n" + T.repo_snapshot(host)
        if record.get("fixture_dir"):
            fx = os.path.join(T.SETUPBENCH_DIR, record["fixture_dir"])
            text += "\n\nFixture files placed in the working directory:\n" + T.repo_snapshot(fx, max_chars=2500)
        return text

    def demo_from_trace(self, record: dict, trace: dict, split: Optional[SplitSpec] = None) -> Optional[dict]:
        """Traces are rows of domains.envsetup.agent (``trajectory`` = bash steps) or any row
        with a ``cot`` list of {reason, command, exit_code, output} steps."""
        steps = trace.get("trajectory") or trace.get("cot")
        if not steps:
            return None
        cot = [{"reason": s.get("reason", ""), "command": s["command"], "exit_code": s.get("exit_code"),
                "output": (s.get("output") or "")[-1500:]} for s in steps]
        script = trace.get("answer") or trace.get("final_response") or \
            "\n".join(s["command"] for s in cot if s.get("exit_code") == 0)
        return {
            "task_id": record["task_id"], "source": record["source"], "category": record["category"],
            "language": record.get("language"), "trace_source": trace.get("trace_source", "agent"),
            "prompt": self.demo_prompt(record), "cot": cot, "final_response": script,
            "success": self.trace_success(trace),
        }

    # -- coding agents --------------------------------------------------------
    def solver_builder_spec(self) -> dict:
        spec = super().solver_builder_spec()
        spec["system_prompt"] = spec["system_prompt"].replace(
            "  - Use only the Python standard library.\n",
            "  - Use only the Python standard library; the solver drives a container through the\n"
            "    shell object described in the specification and never sees the hidden verifier.\n"
            "  - The sandbox available to execute_code has no Docker: use it to study DEMOS.json and\n"
            "    to unit-test parsing/decision logic against a fake shell; end-to-end evaluation is\n"
            "    done outside with scripts/eval_domain_solver.py --domain envsetup.\n")
        return spec
