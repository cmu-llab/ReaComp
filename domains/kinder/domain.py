"""KinderDomain: one KinDER environment family as a ReaComp domain.

An induced solver is a policy program in AgenticGenTAMP's interface:

    class GeneratedApproach:
        def __init__(self, action_space, observation_space, primitives): ...
        def reset(self, state, info): ...
        def get_action(self, state): ...

``solve`` rolls the policy out on one instance (strict black box: the policy sees
pure-Python state mirrors and may import only the standard library, NumPy and SciPy),
records the executed actions, and verifies them by exact replay through
``rewards/kinder.py``.  See ``README.md`` next to this file.

Each family is registered as its own domain, ``kinder_<family>``, because ReaComp
induces one solver per family (``domains/__init__.py``).
"""
from __future__ import annotations

import functools
import multiprocessing
import os
import shutil
import tempfile
from typing import Any, Callable, Dict, Optional

from ..base import Domain, SolveOutcome, SplitSpec, REPO_ROOT, wait_for_child_result
from . import prompts as P
from .families import FamilySpec, get_family

#: task files; $KINDER_DATA_DIR points the domains at another build (e.g. a private test suite)
DATA_DIR = os.environ.get("KINDER_DATA_DIR") or os.path.join(REPO_ROOT, "data", "kinder")


@functools.lru_cache(maxsize=4096)
def _initial_state_text(family: str, count: int, seed: int) -> str:
    from . import client as C
    from .sim import initial_state_payload
    return C.format_state(C.state_from_payload(initial_state_payload(family, count, seed)))


class KinderDomain(Domain):
    reward_module = "kinder"
    solver_entrypoints = ("GeneratedApproach",)
    agent_visible_keys = frozenset({"task_id", "family", "count", "max_steps"})
    building_prompt = "building_prompts/SOLVER_BUILDING_PROMPT_KINDER.md"
    trace_runner = "openhands_agents.run_kinder_direct_solve"
    solver_filename = "SOLVER_KINDER.py"
    algorithm_filename = "SOLVER_KINDER_ALGORITHM.md"

    #: policy_budget_s: policy computation per instance (AgenticGenTAMP: 60 s);
    #: verify_by_replay: re-check every rollout by replaying its actions in a fresh env
    DEFAULT_PARAMS = {"policy_budget_s": 60.0, "verify_by_replay": True}
    #: execute_code limit for the solver builder (rollouts over many instances take minutes)
    BUILDER_TIMEOUT_S = 600

    def __init__(self, family: str) -> None:
        self.family: FamilySpec = get_family(family)
        self.name = f"kinder_{self.family.name}"
        self.description = (f"KinDER {self.family.title}: 2D robot manipulation with a variable number of "
                            f"{self.family.count_noun}")
        self.demos = {"default": f"demos/DEMOS_KINDER_{self.family.name.upper()}_seed_42_100_examples_with_CoT.json"}

    # -- splits (data/kinder/<family>_{train,test}.jsonl, built by data_generation/kinder_generator.py) --
    def card_path(self) -> str:
        return os.path.join(DATA_DIR, f"{self.family.name}_env.md")

    def env_card(self) -> str:
        with open(self.card_path()) as fh:
            return fh.read()

    def splits(self) -> Dict[str, SplitSpec]:
        mk = lambda s: SplitSpec(s, os.path.join(DATA_DIR, f"{self.family.name}_{s}.jsonl"),  # noqa: E731
                                 f"KinDER-{self.family.title}-{s}", dict(self.DEFAULT_PARAMS))
        return {"train": mk("train"), "test": mk("test")}

    # -- solver ---------------------------------------------------------------
    def load_solver(self, path: str) -> Callable:
        """The ``GeneratedApproach`` class of a solver file, after the import allowlist check."""
        from .client import load_approach
        from .strict import assert_solver_allowed
        assert_solver_allowed(path)
        return load_approach(path)

    def solve(self, solve_fn: Callable, record: dict, split: SplitSpec) -> SolveOutcome:
        from . import sim
        params = {**self.DEFAULT_PARAMS, **(split.params or {})}
        roll = sim.run_policy(solve_fn, self.family, record["count"], record["seed"],
                              policy_budget_s=float(params["policy_budget_s"]), max_steps=record.get("max_steps"))
        answer = sim.encode_actions(roll.actions)
        extras = {**{f"rollout_{k}": v for k, v in roll.summary().items()}}
        if params.get("verify_by_replay", True):
            ver = self.reward(answer, True, record)
            value = float(ver["value"])
            extras["verify_message"] = ver["message"]
            extras["replay_agrees"] = (value >= 1.0) == roll.solved
        else:
            value = 1.0 if roll.solved else 0.0
        return SolveOutcome(success=value >= 1.0, score=value, program=answer, extras=extras)

    # -- rows -----------------------------------------------------------------
    def complexity(self, answer: Any) -> Optional[float]:
        """Episode length: among several solvers that solve a task, prefer the shorter rollout."""
        if isinstance(answer, dict) and "shape" in answer:
            return float(answer["shape"][0])
        return None

    def row_extras(self, record: dict) -> dict:
        return {"task_id": record["task_id"], "family": record["family"], "count": record["count"],
                "regime": record["regime"], "max_steps": record["max_steps"]}

    # -- agents ---------------------------------------------------------------
    def instance_text(self, record: dict) -> str:
        """The instance part of the per-task agent's prompt (also the prompt of a demo)."""
        return (f"Task {record['task_id']}: {record['count']} {self.family.count_noun}; step budget "
                f"{record['max_steps']} environment steps.\n\nInitial state:\n"
                + _initial_state_text(self.family.name, int(record["count"]), int(record["seed"])))

    def agent_task_text(self, record: dict, split: Optional[SplitSpec] = None) -> str:
        return P.AGENT_TASK_TEMPLATE.format(card=self.env_card(), instance=self.instance_text(record))

    # -- demos ----------------------------------------------------------------
    def demo_group_key(self, demo: dict):
        return (demo.get("count"), bool(demo.get("success")))

    def demo_from_trace(self, record: dict, trace: dict, split: Optional[SplitSpec] = None) -> Optional[dict]:
        """Rows of ``openhands_agents.run_kinder_direct_solve``: the step list becomes ``cot`` and the submitted
        per-instance policy ``final_response``.  Rows without any step are skipped."""
        steps = trace.get("trajectory") or []
        if not steps:
            return None
        # reasoning: the model's reasoning_content (reasoning parser on); thought: its message text, which
        # holds the thinking when the server does not separate it
        cot = [{"reasoning": s.get("reasoning") or "", "thought": s.get("thought") or "", "tool": s.get("tool"),
                "reason": s.get("reason") or "", "code": s.get("code") or "",
                "output": (s.get("output") or "")[-2000:]} for s in steps]
        rollout = trace.get("rollout") or {}
        return {
            "task_id": record["task_id"], "family": record["family"], "count": record["count"],
            "regime": record["regime"], "max_steps": record["max_steps"],
            "prompt": self.instance_text(record), "cot": cot,
            "final_response": trace.get("policy_code") or "",
            "success": self.trace_success(trace),
            "num_steps": rollout.get("rollout_num_steps"),
            "failure": None if self.trace_success(trace) else (rollout.get("rollout_error") or
                                                                 ("no policy submitted" if not trace.get("submitted")
                                                                  else "goal not reached")),
        }

    # -- solver builder -------------------------------------------------------
    def solver_builder_spec(self) -> dict:
        sf, af = self.solver_filename, self.algorithm_filename
        usage = P.client_usage(pinned_only=False)
        timeout = int(getattr(self, "_builder_timeout", None) or self.BUILDER_TIMEOUT_S)
        return {
            "system_prompt": P.BUILDER_SYSTEM_PROMPT.format(solver_file=sf, algorithm_file=af, client_usage=usage,
                                                            timeout=timeout),
            "file_note": P.BUILDER_FILE_NOTE.format(solver_file=sf, algorithm_file=af),
            "prompt_suffix": P.ENV_APPENDIX.format(card=self.env_card()),
            "execute_code_description": (
                "Run a Python script in your working directory (DEMOS.json, ENV.md, kinder_client.py and the "
                "files you wrote are there). Use it to study the demos, probe the simulator and evaluate your "
                "solver on training and fresh instances. Print results to stdout."),
            "solver_filename": sf, "algorithm_filename": af,
        }

    def solver_builder_sandbox(self, demos_path: str, output_dir: str, timeout: Optional[float] = None):
        from .sandbox import BuilderSandbox
        train = [(r["count"], r["seed"]) for r in self.load_tasks("train")]
        self._builder_timeout = timeout or self.BUILDER_TIMEOUT_S   # stated in solver_builder_spec's prompt
        return BuilderSandbox(self.family, train, demos_path=demos_path, card_path=self.card_path(),
                              output_dir=output_dir, timeout=self._builder_timeout)

    def render_building_prompt(self) -> str:
        """The full builder specification for a coding agent working outside the OpenHands builder."""
        with open(self.abspath(self.building_prompt)) as fh:
            return fh.read() + P.ENV_APPENDIX.format(card=self.env_card())


# ------------------------------------------------------------------ isolated evaluation
def _evaluate_child(domain_name: str, record: dict, policy_path: str, queue) -> None:
    try:
        from domains import get_domain
        dom = get_domain(domain_name)
        out = dom.solve(dom.load_solver(policy_path), record, dom.split(record.get("split", "train")))
        queue.put({"success": out.success, "score": out.score, "program": out.program, "extras": out.extras})
    except BaseException as exc:  # noqa: BLE001 - a policy that cannot be loaded is a failed attempt
        queue.put({"success": False, "score": 0.0, "program": None,
                   "extras": {"rollout_error": f"{type(exc).__name__}: {exc}", "rollout_error_kind": "load_error"}})


def evaluate_policy_source(domain_name: str, record: dict, code: str, timeout: float) -> dict:
    """Evaluate policy source on one instance (``KinderDomain.solve``) in a fresh process.

    Safe to call from threads (spawn context); the process is killed after ``timeout``.
    Returns {"success", "score", "program", "extras"}.
    """
    tmp = tempfile.mkdtemp(prefix="kinder_policy_")
    path = os.path.join(tmp, "policy.py")
    with open(path, "w") as fh:
        fh.write(code)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_evaluate_child, args=(domain_name, record, path, queue), daemon=True)
    proc.start()
    try:
        result, status = wait_for_child_result(proc, queue, timeout)
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=10)
        shutil.rmtree(tmp, ignore_errors=True)
    if status == "timeout":
        return {"success": False, "score": 0.0, "program": None,
                "extras": {"rollout_error": f"evaluation exceeded {timeout:.0f}s", "rollout_error_kind": "timeout"}}
    if status == "died":
        return {"success": False, "score": 0.0, "program": None,
                "extras": {"rollout_error": f"evaluation process exited with code {proc.exitcode}",
                           "rollout_error_kind": "crash"}}
    return result
