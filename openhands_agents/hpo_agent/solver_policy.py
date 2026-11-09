"""An induced ``SOLVER_HPO.py`` as an episode policy.

Every decision runs ``solve_hpo(task, history)`` in a fresh Python process:

  * a decision that hangs is killed after ``decision_timeout_s`` and costs the solver
    an ``invalid`` trial instead of stalling an episode that holds a GPU;
  * a solver that keeps to the building prompt's contract (a pure function of
    ``task`` and ``history``) has no in-process state to lose, so an episode that is
    resumed from its log continues exactly like one that ran through;
  * a solver that crashes the interpreter does not take the episode down with it.

Starting an interpreter costs a fraction of a second per decision, against 300 s of
training per trial.

Whatever the solver returns besides the configuration is stored under ``solver_meta``
in the episode log, apart from the fields the harness writes itself (``source``,
``token_usage``), so a solver cannot be mistaken for one of the harness's own
policies in a results table.

The process is not a security sandbox: the solver runs with the caller's permissions,
like every solver evaluated by the other ``scripts/eval_*solver.py`` scripts. Code
that writes to the output directory or the training checkout can falsify its own
results; the per-process isolation guards against accidents (leaked state, hangs,
crashes), not against a solver written to cheat. Read an induced solver before
running it, as with any generated code.

Worker mode (used internally):
    python -m openhands_agents.hpo_agent.solver_policy <SOLVER_HPO.py> <marker>
reads ``{"task", "history"}`` as JSON on stdin and prints ``<marker><json>`` on stdout.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
import uuid
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOLVER_FUNCTION = "solve_hpo"
DEFAULT_DECISION_TIMEOUT_S = 60.0
SOURCE = "solver"            # ``policy_meta["source"]`` of every solver suggestion
_MAX_ERROR_CHARS = 2000
_MAX_META_CHARS = 20000      # per decision, of what a solver returns besides the config


class SolverError(RuntimeError):
    """The solver produced no usable suggestion (crash, timeout, unserialisable value)."""


def load_solver(path: str):
    """Import a solver file and return its ``solve_hpo`` function."""
    solver_dir = os.path.dirname(os.path.abspath(path))
    for directory in (solver_dir, PROJECT_ROOT):     # project root first: `import rewards`
        if directory in sys.path:
            sys.path.remove(directory)
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location("SOLVER_HPO", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["SOLVER_HPO"] = module
    spec.loader.exec_module(module)
    solve = getattr(module, SOLVER_FUNCTION, None)
    if not callable(solve):
        raise ImportError(f"{path} does not define {SOLVER_FUNCTION}(task, history)")
    return solve


def solver_fingerprint(path: str) -> str:
    """Short content hash of the solver file (part of the policy name)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


class SolverPolicy:
    """``policy(task, history)`` backed by a solver file, one process per decision."""

    def __init__(self, solver_path: str,
                 decision_timeout_s: float = DEFAULT_DECISION_TIMEOUT_S):
        self.solver_path = os.path.abspath(solver_path)
        if not os.path.isfile(self.solver_path):
            raise FileNotFoundError(f"solver file not found: {solver_path}")
        self.decision_timeout_s = decision_timeout_s
        # The hash makes a changed solver file a different policy, so an episode log
        # started by one version is never continued by another.
        self.fingerprint = solver_fingerprint(self.solver_path)
        self.name = f"solver:{os.path.basename(self.solver_path)}@{self.fingerprint}"

    def __call__(self, task: Dict[str, Any], history: List[Dict[str, Any]]) -> Any:
        if solver_fingerprint(self.solver_path) != self.fingerprint:
            raise SolverError(f"{self.solver_path} changed during the episode")
        marker = f"SOLVER_DECISION_{uuid.uuid4().hex}:"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        try:
            proc = subprocess.run(
                [sys.executable, "-m", __name__, self.solver_path, marker],
                input=json.dumps({"task": task, "history": history}),
                capture_output=True, text=True, cwd=PROJECT_ROOT, env=env,
                timeout=self.decision_timeout_s)
        except subprocess.TimeoutExpired:
            raise SolverError(
                f"solver decision timed out after {self.decision_timeout_s:g} s") from None
        lines = [line for line in proc.stdout.splitlines() if line.startswith(marker)]
        if not lines:
            tail = proc.stderr.strip()[-_MAX_ERROR_CHARS:]
            raise SolverError(f"solver process exited with status {proc.returncode} "
                              f"and no answer: {tail}")
        answer = json.loads(lines[-1][len(marker):])
        if answer["error"] is not None:
            raise SolverError(answer["error"])
        return self._as_suggestion(answer["suggestion"])

    @staticmethod
    def _as_suggestion(returned: Any) -> Dict[str, Any]:
        """The solver's return value in the form ``run_episode`` expects of a policy."""
        if not (isinstance(returned, dict) and "config" in returned):
            return {"config": returned, "source": SOURCE}        # a bare configuration
        suggestion = {"config": returned["config"], "source": SOURCE}
        extras = {key: value for key, value in returned.items() if key != "config"}
        if extras:
            text = json.dumps(extras)
            suggestion["solver_meta"] = (extras if len(text) <= _MAX_META_CHARS else
                                         {"truncated": text[:_MAX_META_CHARS]})
        return suggestion


def _worker(solver_path: str, marker: str) -> int:
    request = json.load(sys.stdin)
    # Whatever the solver prints goes to stderr; stdout carries only the answer line.
    answer_stream, sys.stdout = sys.stdout, sys.stderr
    answer: Dict[str, Any] = {"suggestion": None, "error": None}
    try:
        suggestion = load_solver(solver_path)(request["task"], request["history"])
        json.dumps(suggestion)      # must survive the trip back to the episode loop
        answer["suggestion"] = suggestion
    except BaseException as exc:  # includes SystemExit raised by the solver
        detail = traceback.format_exc()[-_MAX_ERROR_CHARS:]
        answer["error"] = f"{type(exc).__name__}: {exc}\n{detail}"
    answer_stream.write(marker + json.dumps(answer) + "\n")
    answer_stream.flush()
    return 0


if __name__ == "__main__":
    sys.exit(_worker(sys.argv[1], sys.argv[2]))
