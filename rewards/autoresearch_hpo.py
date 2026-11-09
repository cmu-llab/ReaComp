"""Verifier and episode harness for the autoresearch HPO domain.

Benchmark: "Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on
autoresearch" (arXiv:2603.24647). A policy tunes 14 hyperparameters of the nanochat
``train.py`` from Karpathy's autoresearch to minimise ``val_bpb`` (validation bits per
byte). Every trial trains for a fixed 300 s on one GPU, a run has a cumulative
training-time budget (24 h in the paper), failed trials are recorded with a penalty,
and every method starts from the same default configuration as trial 0.

Unlike PBEBench / SLR-Bench / List Functions, a task here is a sequential decision
problem, so this module holds three things:

  * the search space and the config normalisation every method goes through
    (bounds, integer casting, power-of-two snapping, the batch-size constraint);
  * the objectives: ``NanochatObjective`` runs the pinned upstream ``train.py`` (GPU
    required) and ``SyntheticObjective`` is a cheap stand-in landscape for CPU tests
    and dry runs. Synthetic numbers are NOT benchmark results;
  * the episode loop ``run_episode`` (trial 0 = starting config, budget accounting,
    JSONL log with resume) and its scoring.

``reward(result, execution_ok, entry)`` keeps the rewards/ registry contract
(``{"value": float in [0, 1], "message": str}``): it scores one proposed configuration
by evaluating it with the task's objective. Numbers reported by the candidate itself
are never read.

Policy interface (what an induced ``SOLVER_HPO.py`` implements as ``solve_hpo``):

    policy(task: dict, history: list[dict]) -> {"config": {HP name: value}, ...}

``task`` is ``public_task_view(entry)``; ``history`` holds one dict per finished trial
(see ``HISTORY_KEYS``), trial 0 being the starting configuration. Extra keys in the
returned dict are stored in the episode log as policy metadata.

Standard library only: this file is bind-mounted into the solver-builder sandbox.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import random
import re
import shlex
import signal
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark definition
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK = "autoresearch_hpo"
METRIC = "val_bpb"
FAILURE_PENALTY = 100.0        # val_bpb the paper records for a failed trial
MAX_SEQ_LEN = 2048             # prepare.py constant, used by the batch-size constraint
TRIAL_TIME_BUDGET_S = 300.0    # prepare.py TIME_BUDGET: training seconds per trial

# Pinned upstream sources. The sha256 values are those of train.py / prepare.py at
# TRAIN_COMMIT; data/autoresearch_hpo/check_upstream_fidelity.py re-derives them.
UPSTREAM = {
    "paper": "arXiv:2603.24647",
    "automl_repo": "ferreirafabio/autoresearch-automl",
    "automl_commit": "150cd1418b7bd65c3821a844a91598110d3b36b8",
    "train_repo": "ferreirafabio/autoresearch",
    "train_commit": "b4180f8c5ce4b42871b1eaefb6c68bbb2319fbfa",
    "train_py_sha256": "7711431f5fa15c34c9e3210751ee87640709b429243b3525584bb14248e0bf89",
    "prepare_py_sha256": "4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150",
}

# The 14-hyperparameter search space (upstream KNOWN_HP_METADATA, same order).
# "log" marks log-scale sampling; "power_of_2" marks values snapped after clamping.
SEARCH_SPACE: Tuple[Dict[str, Any], ...] = (
    {"name": "DEPTH", "type": "int", "low": 4, "high": 24, "log": False, "power_of_2": False},
    {"name": "ASPECT_RATIO", "type": "int", "low": 32, "high": 128, "log": False, "power_of_2": False},
    {"name": "HEAD_DIM", "type": "int", "low": 64, "high": 256, "log": True, "power_of_2": True},
    {"name": "DEVICE_BATCH_SIZE", "type": "int", "low": 32, "high": 256, "log": True, "power_of_2": True},
    {"name": "TOTAL_BATCH_SIZE", "type": "int", "low": 65536, "high": 2097152, "log": True, "power_of_2": True},
    {"name": "EMBEDDING_LR", "type": "float", "low": 0.01, "high": 2.0, "log": True, "power_of_2": False},
    {"name": "UNEMBEDDING_LR", "type": "float", "low": 0.0005, "high": 0.05, "log": True, "power_of_2": False},
    {"name": "MATRIX_LR", "type": "float", "low": 0.005, "high": 0.2, "log": True, "power_of_2": False},
    {"name": "SCALAR_LR", "type": "float", "low": 0.05, "high": 2.0, "log": True, "power_of_2": False},
    {"name": "WEIGHT_DECAY", "type": "float", "low": 0.0, "high": 0.5, "log": False, "power_of_2": False},
    {"name": "WARMUP_RATIO", "type": "float", "low": 0.0, "high": 0.3, "log": False, "power_of_2": False},
    {"name": "WARMDOWN_RATIO", "type": "float", "low": 0.1, "high": 0.8, "log": False, "power_of_2": False},
    {"name": "FINAL_LR_FRAC", "type": "float", "low": 0.0, "high": 0.2, "log": False, "power_of_2": False},
    {"name": "WINDOW_PATTERN", "type": "cat",
     "choices": ["SSSL", "SSLL", "SLSL", "LLLL", "SSSS", "LSSL"]},
)
HP_NAMES: Tuple[str, ...] = tuple(hp["name"] for hp in SEARCH_SPACE)
POWER_OF_2_HPS = frozenset(hp["name"] for hp in SEARCH_SPACE if hp.get("power_of_2"))

# Trial 0 of every run (upstream KARPATHY_STARTING_CONFIG == the train.py defaults).
STARTING_CONFIG: Dict[str, Any] = {
    "DEPTH": 8,
    "ASPECT_RATIO": 64,
    "HEAD_DIM": 128,
    "DEVICE_BATCH_SIZE": 128,
    "TOTAL_BATCH_SIZE": 524288,
    "EMBEDDING_LR": 0.6,
    "UNEMBEDDING_LR": 0.004,
    "MATRIX_LR": 0.04,
    "SCALAR_LR": 0.5,
    "WEIGHT_DECAY": 0.2,
    "WARMUP_RATIO": 0.0,
    "WARMDOWN_RATIO": 0.5,
    "FINAL_LR_FRAC": 0.0,
    "WINDOW_PATTERN": "SSSL",
}

# Trial outcomes. Only "ok" trials have a val_bpb.
STATUS_OK = "ok"
STATUS_OOM = "oom"              # GPU out of memory
STATUS_DIVERGED = "diverged"    # train.py aborted on a NaN / exploding loss
STATUS_ERROR = "error"          # any other non-zero exit, or no val_bpb in the output
STATUS_TIMEOUT = "timeout"      # the training process had to be killed
STATUS_INVALID = "invalid"      # the policy returned no usable config (never trained)
STATUSES = (STATUS_OK, STATUS_OOM, STATUS_DIVERGED, STATUS_ERROR, STATUS_TIMEOUT,
            STATUS_INVALID)

# Keys of a history entry, i.e. everything a policy learns about a finished trial.
HISTORY_KEYS = ("trial_id", "config", "status", "val_bpb", "error", "wall_time_s",
                "peak_vram_mb", "metrics")

# Keys of a task record a policy may see. Everything else (objective internals,
# reference values) stays with the verifier.
PUBLIC_TASK_KEYS = ("task_id", "task", "prompt", "metric", "seed", "search_space",
                    "starting_config", "budget", "hardware", "failure_penalty")

DEFAULT_MAX_CONSECUTIVE_INVALID = 10


class InvalidConfig(ValueError):
    """A proposed configuration cannot be turned into a runnable one."""


class ObjectiveUnavailable(RuntimeError):
    """The objective cannot run here (missing checkout, wrong sources, no launcher)."""


class BaselineFailed(RuntimeError):
    """The starting configuration failed: the environment cannot run the benchmark."""


class EpisodeLogMismatch(ValueError):
    """An existing episode log cannot be resumed (other task, objective or policy, or
    rows missing), so continuing would mix two runs in one file."""


class EpisodeInterrupted(RuntimeError):
    """Raised by a policy to stop an episode without recording a trial.

    For failures that say nothing about the policy's decisions (e.g. the LLM server
    is unreachable). The episode log stays resumable.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Configurations
# ──────────────────────────────────────────────────────────────────────────────

def snap_to_power_of_2(config: Dict[str, Any]) -> Dict[str, Any]:
    """Port of upstream ``snap_to_power_of_2``.

    HEAD_DIM / DEVICE_BATCH_SIZE / TOTAL_BATCH_SIZE go to the nearest power of two
    (ties go down), then TOTAL_BATCH_SIZE is raised to DEVICE_BATCH_SIZE * 2048 when
    below it, which keeps train.py's divisibility assertion satisfied.
    """
    result = dict(config)
    for name in HP_NAMES:
        if name not in POWER_OF_2_HPS or name not in result:
            continue
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        v = max(1, int(value))
        low = 1 << (v.bit_length() - 1)
        high = low << 1
        result[name] = low if (v - low) <= (high - v) else high
    dbs = result.get("DEVICE_BATCH_SIZE")
    tbs = result.get("TOTAL_BATCH_SIZE")
    if isinstance(dbs, int) and isinstance(tbs, int):
        result["TOTAL_BATCH_SIZE"] = max(tbs, dbs * MAX_SEQ_LEN)
    return result


def normalize_config(raw: Any) -> Dict[str, Any]:
    """Turn a proposed configuration into the one that is actually trained.

    Same treatment the upstream benchmark gives a suggestion: a missing
    hyperparameter takes its starting value, unknown keys are dropped, numeric values
    are cast to the hyperparameter's type (integers truncate) and clamped to the
    bounds, then ``snap_to_power_of_2`` is applied. Idempotent.

    Raises ``InvalidConfig`` for anything that has no sensible reading: a non-mapping,
    a non-numeric / non-finite / boolean value for a numeric hyperparameter, or a
    WINDOW_PATTERN outside the listed choices.
    """
    if not isinstance(raw, dict):
        raise InvalidConfig(f"config must be a JSON object, got {type(raw).__name__}")
    config: Dict[str, Any] = {}
    for hp in SEARCH_SPACE:
        name = hp["name"]
        if name not in raw:
            config[name] = STARTING_CONFIG[name]
            continue
        value = raw[name]
        if hp["type"] == "cat":
            if not isinstance(value, str) or value not in hp["choices"]:
                raise InvalidConfig(f"{name}={value!r} is not one of {hp['choices']}")
            config[name] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidConfig(f"{name}={value!r} is not a number")
        if not math.isfinite(value):
            raise InvalidConfig(f"{name}={value!r} is not finite")
        value = int(value) if hp["type"] == "int" else float(value)
        config[name] = max(hp["low"], min(hp["high"], value))
    return snap_to_power_of_2(config)


def sample_config(rng: random.Random) -> Dict[str, Any]:
    """Draw a configuration from the search space (log-uniform where marked)."""
    raw: Dict[str, Any] = {}
    for hp in SEARCH_SPACE:
        if hp["type"] == "cat":
            raw[hp["name"]] = rng.choice(hp["choices"])
        elif hp["log"]:
            value = math.exp(rng.uniform(math.log(hp["low"]), math.log(hp["high"])))
            raw[hp["name"]] = int(round(value)) if hp["type"] == "int" else value
        elif hp["type"] == "int":
            raw[hp["name"]] = rng.randint(hp["low"], hp["high"])
        else:
            raw[hp["name"]] = rng.uniform(hp["low"], hp["high"])
    return normalize_config(raw)


def derive_architecture(config: Dict[str, Any]) -> Dict[str, int]:
    """Model shape train.py builds from a normalised config (same integer formulas)."""
    base_dim = config["DEPTH"] * config["ASPECT_RATIO"]
    head_dim = config["HEAD_DIM"]
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    return {
        "model_dim": model_dim,
        "num_heads": model_dim // head_dim,
        "grad_accum_steps": config["TOTAL_BATCH_SIZE"]
        // (config["DEVICE_BATCH_SIZE"] * MAX_SEQ_LEN),
    }


def search_space_spec() -> List[Dict[str, Any]]:
    """The search space with defaults, as stored in task records and shown to policies."""
    return [dict(hp, default=STARTING_CONFIG[hp["name"]]) for hp in SEARCH_SPACE]


def describe_search_space() -> str:
    """One line per hyperparameter, for prompts."""
    lines = []
    for hp in search_space_spec():
        if hp["type"] == "cat":
            text = f"one of {hp['choices']}, default {hp['default']!r}"
        else:
            text = f"{hp['type']} in [{hp['low']}, {hp['high']}], default {hp['default']}"
            if hp["log"]:
                text += ", log scale"
            if hp["power_of_2"]:
                text += ", snapped to a power of 2"
        lines.append(f"  {hp['name']}: {text}")
    return "\n".join(lines)


def _unit(hp: Dict[str, Any], value: float) -> float:
    """Position of a numeric value inside its bounds, in [0, 1] (log scale if marked)."""
    if hp["log"]:
        return (math.log(value) - math.log(hp["low"])) / (math.log(hp["high"]) - math.log(hp["low"]))
    return (value - hp["low"]) / (hp["high"] - hp["low"])


# ──────────────────────────────────────────────────────────────────────────────
# Objectives
# ──────────────────────────────────────────────────────────────────────────────

def _trial_result(status: str, wall_time_s: float, val_bpb: Optional[float] = None,
                  error: Optional[str] = None, peak_vram_mb: Optional[float] = None,
                  metrics: Optional[Dict[str, float]] = None,
                  detail: Optional[str] = None) -> Dict[str, Any]:
    return {
        "status": status,
        "val_bpb": val_bpb,
        "error": error,
        "wall_time_s": float(wall_time_s),
        "peak_vram_mb": peak_vram_mb,
        "metrics": dict(metrics or {}),
        "detail": detail,          # debugging text for the episode log, not shown to policies
    }


class SyntheticObjective:
    """A cheap, deterministic stand-in for nanochat training. NOT the benchmark.

    Exists so the episode loop, policies and induced solvers can be exercised on a CPU
    in milliseconds. ``landscape_seed`` selects one member of a family of landscapes
    that share the qualitative structure of the real problem and nothing else:

      * val_bpb is a bowl around a hidden optimum (random per landscape, always a
        valid config that trains);
      * memory grows with DEVICE_BATCH_SIZE * DEPTH * model_dim, calibrated to the
        ~45 GB the default config uses upstream, and exceeding the task's training
        VRAM gives an out-of-memory failure;
      * aggressive learning rates without warmup diverge;
      * a finished trial costs the full trial budget, a failure a fraction of it.
    """

    kind = "synthetic"
    hardware_warning = None       # no hardware is involved
    OOM_WALL_TIME_S = 37.0
    DIVERGED_WALL_TIME_S = 45.0

    def __init__(self, landscape_seed: int, training_vram_gb: float,
                 trial_time_budget_s: float = TRIAL_TIME_BUDGET_S):
        self.landscape_seed = int(landscape_seed)
        self.training_vram_gb = float(training_vram_gb)
        self.trial_time_budget_s = float(trial_time_budget_s)

        rng = random.Random(f"{BENCHMARK}/synthetic/{self.landscape_seed}")
        default_load = (STARTING_CONFIG["DEVICE_BATCH_SIZE"] * STARTING_CONFIG["DEPTH"]
                        * derive_architecture(STARTING_CONFIG)["model_dim"])
        self._vram_base_gb = 2.0
        self._vram_gb_per_load = 43.0 / default_load * rng.uniform(0.9, 1.1)
        self._divergence_threshold = rng.uniform(1.15, 1.35)
        self._floor = rng.uniform(0.955, 0.972)
        self._weights = {hp["name"]: rng.uniform(0.005, 0.08)
                         for hp in SEARCH_SPACE if hp["type"] != "cat"}
        self._choice_penalty = {hp["name"]: {c: rng.uniform(0.002, 0.02) for c in hp["choices"]}
                                for hp in SEARCH_SPACE if hp["type"] == "cat"}
        if self._failure(STARTING_CONFIG) is not None:
            raise ValueError(
                f"training_vram_gb={self.training_vram_gb} is too small for the starting "
                f"config of synthetic landscape {self.landscape_seed}")
        for _ in range(100000):
            candidate = sample_config(rng)
            if self._failure(candidate) is None:
                break
        else:  # pragma: no cover - needs a VRAM limit almost nothing fits into
            candidate = dict(STARTING_CONFIG)
        self._optimum = candidate
        for name, penalties in self._choice_penalty.items():
            penalties[self._optimum[name]] = 0.0

    @property
    def optimum_val_bpb(self) -> float:
        """Lowest reachable val_bpb of this landscape (verifier-side reference)."""
        return round(self._floor, 6)

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "landscape_seed": self.landscape_seed,
                "training_vram_gb": self.training_vram_gb,
                "trial_time_budget_s": self.trial_time_budget_s}

    def _peak_vram_gb(self, config: Dict[str, Any]) -> float:
        load = config["DEVICE_BATCH_SIZE"] * config["DEPTH"] * derive_architecture(config)["model_dim"]
        return self._vram_base_gb + self._vram_gb_per_load * load

    def _failure(self, config: Dict[str, Any]) -> Optional[str]:
        if self._peak_vram_gb(config) > self.training_vram_gb:
            return STATUS_OOM
        by_name = {hp["name"]: hp for hp in SEARCH_SPACE}
        heat = (_unit(by_name["MATRIX_LR"], config["MATRIX_LR"])
                + 0.5 * _unit(by_name["EMBEDDING_LR"], config["EMBEDDING_LR"])
                - 2.0 * config["WARMUP_RATIO"])
        if heat > self._divergence_threshold:
            return STATUS_DIVERGED
        return None

    def evaluate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config = normalize_config(config)
        failure = self._failure(config)
        if failure == STATUS_OOM:
            return _trial_result(STATUS_OOM, self.OOM_WALL_TIME_S, error="GPU out-of-memory")
        if failure == STATUS_DIVERGED:
            return _trial_result(STATUS_DIVERGED, self.DIVERGED_WALL_TIME_S,
                                 error="Training diverged (loss NaN or > 100)")
        value = self._floor
        for hp in SEARCH_SPACE:
            name = hp["name"]
            if hp["type"] == "cat":
                value += self._choice_penalty[name][config[name]]
            else:
                value += self._weights[name] * (_unit(hp, config[name]) - _unit(hp, self._optimum[name])) ** 2
        return _trial_result(STATUS_OK, self.trial_time_budget_s, val_bpb=round(value, 6),
                             peak_vram_mb=round(self._peak_vram_gb(config) * 1024, 1))


class InjectionError(ObjectiveUnavailable):
    """A hyperparameter could not be written into train.py."""


def _module_assignments(source: str, names) -> Dict[str, List[ast.expr]]:
    """Value nodes of module-level ``NAME = <expr>`` statements, for the given names."""
    found: Dict[str, List[ast.expr]] = {name: [] for name in names}
    for node in ast.parse(source).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in found):
            found[node.targets[0].id].append(node.value)
    return found


def inject_config(source: str, config: Dict[str, Any]) -> str:
    """Return ``source`` with each hyperparameter's module-level assignment replaced.

    Only the value expression is rewritten (comments and layout survive) and values
    are written with ``repr`` so floats round-trip exactly. The result is re-parsed
    and checked, and a hyperparameter that is not assigned exactly once at module
    level raises ``InjectionError`` rather than silently training something else.
    """
    found = _module_assignments(source, config)
    wrong = {name: len(nodes) for name, nodes in found.items() if len(nodes) != 1}
    if wrong:
        raise InjectionError(
            f"expected exactly one module-level assignment per hyperparameter, got {wrong}")
    lines = source.splitlines(keepends=True)
    nodes = sorted(((nodes[0], name) for name, nodes in found.items()),
                   key=lambda item: (item[0].lineno, item[0].col_offset), reverse=True)
    for node, name in nodes:
        first, last = node.lineno - 1, node.end_lineno - 1
        head = lines[first].encode("utf-8")[:node.col_offset]
        tail = lines[last].encode("utf-8")[node.end_col_offset:]
        lines[first:last + 1] = [(head + repr(config[name]).encode("utf-8") + tail).decode("utf-8")]
    injected = "".join(lines)
    for name, nodes in _module_assignments(injected, config).items():
        try:
            written = ast.literal_eval(nodes[0]) if len(nodes) == 1 else None
        except ValueError:
            written = None
        if type(written) is not type(config[name]) or written != config[name]:
            raise InjectionError(f"{name} did not round-trip through injection")
    return injected


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _kill_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


_SUMMARY_LINE = re.compile(r"^(\w+):\s+([0-9.]+)\s*$", re.MULTILINE)


def parse_training_summary(stdout: str) -> Dict[str, float]:
    """Metrics from the ``key: value`` block train.py prints after its last ``---`` line."""
    marker = stdout.rfind("\n---\n")
    if marker < 0:
        return {}
    summary: Dict[str, float] = {}
    for key, value in _SUMMARY_LINE.findall(stdout[marker:]):
        try:
            summary[key] = float(value)
        except ValueError:
            continue
    return summary


class NanochatObjective:
    """Runs the pinned autoresearch ``train.py`` with a configuration injected.

    Port of the upstream ObjectiveFunction / ExperimentRunner / ConfigInjector: the
    config is written into a per-job copy of train.py next to the original (so
    ``from prepare import ...`` keeps working), the copy is run from the checkout,
    and the summary block is parsed. A successful trial costs its reported
    ``training_seconds``; a failed one costs the wall time until the process ended.

    ``repo_dir`` must be a checkout of ``UPSTREAM["train_repo"]`` prepared with
    ``scripts/setup_autoresearch_hpo.sh``.

    Sources. The sha256 of train.py and prepare.py must match the pinned ones
    (``UPSTREAM``), and the files must not change afterwards: they are hashed again
    before every trial. With ``verify_pin=False`` other sources do not stop the run;
    ``describe()`` then reports ``pinned`` as false. The same switch lets a run start
    when the GPU cannot be identified (``device`` is then None).

    Hardware. A task record states the GPU its trials run on. Training runs on
    whatever GPU the training environment sees, but that GPU is recorded, and
    ``describe()`` reports ``hardware_matches`` as false when its total memory is not
    the one the task states (``expected_total_vram_gb``); ``hardware_warning`` then
    says what this means for the results. The same task on another GPU is a different
    benchmark: a trial is a fixed 300 s of whatever the device computes, and the
    memory that is available decides which configurations can be trained at all.

    Memory cap. train.py caps its memory to a *fraction* of the device
    (``CUDA_MEM_FRACTION``). On the hardware the task states, the fraction is the
    task's own (``cuda_mem_fraction``; upstream's 0.54 on the H200). On other hardware
    that fraction would stand for another amount of memory, so it is replaced by the
    one that gives the task's ``training_vram_gb`` on the device at hand, and by no
    cap when the device has no more memory than that. ``describe()`` reports the
    fraction that is applied.

    ``describe()`` is part of the episode-log header, so an episode is never resumed
    on other sources or another device.
    """

    kind = "nanochat"

    # Printed by the training environment's own interpreter, so it sees exactly the
    # device train.py will use (CUDA_VISIBLE_DEVICES included).
    _DEVICE_PROBE = (
        "import json, torch; p = torch.cuda.get_device_properties(0); "
        "print('HPO_DEVICE ' + json.dumps({'gpu': p.name, "
        "'total_vram_gb': round(p.total_memory / 2**30, 2)}))")
    _DEVICE_PROBE_TIMEOUT_S = 600.0
    # "141 GB" in the paper is about 140.4 GiB as reported by the driver.
    _VRAM_TOLERANCE = 0.05
    # Upstream's grace period on top of the training budget: startup, torch.compile
    # and the final evaluation are not part of the 300 s.
    _TRIAL_GRACE_S = 600.0
    _MIN_TRIAL_TIMEOUT_S = 900.0

    def __init__(self, repo_dir: str, run_command=("uv", "run", "python"),
                 trial_time_budget_s: float = TRIAL_TIME_BUDGET_S,
                 cuda_mem_fraction: Optional[float] = None,
                 verify_pin: bool = True,
                 extra_env: Optional[Dict[str, str]] = None,
                 job_id: Optional[str] = None,
                 expected_total_vram_gb: Optional[float] = None,
                 training_vram_gb: Optional[float] = None):
        self.repo_dir = os.path.abspath(repo_dir)
        self.run_command = list(run_command)
        self.trial_time_budget_s = float(trial_time_budget_s)
        self.expected_total_vram_gb = expected_total_vram_gb
        self.training_vram_gb = training_vram_gb
        self.extra_env = dict(extra_env or {})
        self.job_id = job_id or f"{os.getpid()}_{uuid.uuid4().hex[:8]}"

        train_py = os.path.join(self.repo_dir, "train.py")
        prepare_py = os.path.join(self.repo_dir, "prepare.py")
        for path in (train_py, prepare_py):
            if not os.path.isfile(path):
                raise ObjectiveUnavailable(
                    f"{path} not found. Run scripts/setup_autoresearch_hpo.sh or point "
                    f"AUTORESEARCH_DIR / --autoresearch-dir at an autoresearch checkout.")
        self.train_py_sha256 = _sha256(train_py)
        self.prepare_py_sha256 = _sha256(prepare_py)
        self.pinned = (self.train_py_sha256 == UPSTREAM["train_py_sha256"]
                       and self.prepare_py_sha256 == UPSTREAM["prepare_py_sha256"])
        if verify_pin and not self.pinned:
            raise ObjectiveUnavailable(
                f"train.py / prepare.py in {self.repo_dir} differ from the pinned upstream "
                f"sources ({UPSTREAM['train_repo']}@{UPSTREAM['train_commit'][:12]}). "
                f"Check out the pinned commit, or pass verify_pin=False / --allow-unpinned "
                f"to run anyway (results are then labelled 'unpinned sources').")

        with open(train_py, encoding="utf-8") as f:
            self._source = f.read()
        with open(prepare_py, encoding="utf-8") as f:
            budget_nodes = _module_assignments(f.read(), ["TIME_BUDGET"])["TIME_BUDGET"]
        try:
            declared = float(ast.literal_eval(budget_nodes[0])) if len(budget_nodes) == 1 else None
        except ValueError:
            declared = None
        if declared != self.trial_time_budget_s:
            raise ObjectiveUnavailable(
                f"prepare.py TIME_BUDGET is {declared}, the task expects "
                f"{self.trial_time_budget_s} training seconds per trial")
        inject_config(self._source, STARTING_CONFIG)   # fail now, not at trial 0

        self.device, probe_error = self._probe_device()
        if verify_pin and self.device is None:
            raise ObjectiveUnavailable(
                f"no CUDA device visible to {' '.join(self.run_command)!r} in "
                f"{self.repo_dir}: {probe_error}")
        if self.expected_total_vram_gb is None:
            self.hardware_matches = None        # the task record makes no claim
        else:
            self.hardware_matches = self.device is not None and (
                abs(self.device["total_vram_gb"] - self.expected_total_vram_gb)
                <= self._VRAM_TOLERANCE * self.expected_total_vram_gb)
        self.cuda_mem_fraction = self._applied_mem_fraction(cuda_mem_fraction)
        self.hardware_warning = self._hardware_warning(probe_error)

    def _applied_mem_fraction(self, task_fraction: Optional[float]) -> Optional[float]:
        """``CUDA_MEM_FRACTION`` for the device training runs on (None: no cap)."""
        if (self.hardware_matches is not False or self.device is None
                or self.training_vram_gb is None):
            return task_fraction
        if self.training_vram_gb >= self.device["total_vram_gb"]:
            return None
        fraction = round(self.training_vram_gb / self.device["total_vram_gb"], 2)
        return fraction if fraction < 1.0 else None

    def _hardware_warning(self, probe_error: Optional[str]) -> Optional[str]:
        """What a reader of the results has to know when the GPU is not the task's."""
        if self.hardware_matches is not False:
            return None
        if self.device is None:
            return (f"the GPU training runs on could not be identified ({probe_error}); "
                    f"the task is defined for one with {self.expected_total_vram_gb:g} GB. "
                    f"Results are labelled 'hardware not verified'.")
        total = self.device["total_vram_gb"]
        text = (f"the task is defined for a GPU with {self.expected_total_vram_gb:g} GB; "
                f"training runs on {self.device['gpu']} ({total:g} GB). Results are "
                f"labelled 'hardware differs from the task': a trial is "
                f"{self.trial_time_budget_s:g} s of whatever this GPU computes, so val_bpb "
                f"is comparable between runs on this GPU only, not with the paper's "
                f"numbers or with runs on other hardware.")
        if self.training_vram_gb is None:
            memory = ""
        elif self.cuda_mem_fraction is not None:
            memory = (f" Training memory is capped to the {self.training_vram_gb:g} GB the "
                      f"task states (CUDA_MEM_FRACTION={self.cuda_mem_fraction:g}).")
        elif self.training_vram_gb <= total:
            memory = (f" Training uses the whole device, which is about the "
                      f"{self.training_vram_gb:g} GB the task states.")
        else:
            memory = (f" This GPU has less than the {self.training_vram_gb:g} GB of "
                      f"training memory the task states: training uses the whole device, "
                      f"and policies, which are told {self.training_vram_gb:g} GB, will "
                      f"propose configurations that run out of memory.")
        return (f"{text}{memory} Task records that state this GPU can be generated with "
                f"data_generation/autoresearch_hpo_generator.py (README, 'Running on "
                f"another GPU').")

    def _env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.update(self.extra_env)
        return env

    def _probe_device(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """(device, None) for the GPU training will use, or (None, why it is unknown)."""
        try:
            proc = subprocess.run(
                self.run_command + ["-c", self._DEVICE_PROBE], cwd=self.repo_dir,
                env=self._env(), capture_output=True, text=True, errors="replace",
                timeout=self._DEVICE_PROBE_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, f"{type(exc).__name__}: {exc}"
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("HPO_DEVICE "):
                try:
                    device = json.loads(line[len("HPO_DEVICE "):])
                    return {"gpu": str(device["gpu"]),
                            "total_vram_gb": float(device["total_vram_gb"])}, None
                except (ValueError, KeyError, TypeError) as exc:
                    return None, f"unreadable device report {line!r}: {exc}"
        return None, (proc.stderr.strip()[-500:] or f"probe exited with code {proc.returncode}")

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "pinned": self.pinned,
                "train_py_sha256": self.train_py_sha256,
                "prepare_py_sha256": self.prepare_py_sha256,
                "cuda_mem_fraction": self.cuda_mem_fraction,
                "trial_time_budget_s": self.trial_time_budget_s,
                "device": self.device, "hardware_matches": self.hardware_matches}

    def _check_sources_unchanged(self) -> None:
        """train.py imports prepare.py from the checkout at every trial, so what
        ``describe()`` says about the sources has to hold for each of them."""
        for name, expected in (("train.py", self.train_py_sha256),
                               ("prepare.py", self.prepare_py_sha256)):
            path = os.path.join(self.repo_dir, name)
            try:
                actual = _sha256(path)
            except OSError as exc:
                raise ObjectiveUnavailable(f"cannot read {path}: {exc}") from exc
            if actual != expected:
                raise ObjectiveUnavailable(
                    f"{path} changed while the run was in progress (sha256 {actual[:12]}, "
                    f"was {expected[:12]}); restore it before resuming")

    def evaluate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config = normalize_config(config)
        self._check_sources_unchanged()
        work_name = f"train_hpo_{self.job_id}.py"
        work_path = os.path.join(self.repo_dir, work_name)
        env = self._env()
        if self.cuda_mem_fraction is not None:
            env["CUDA_MEM_FRACTION"] = repr(float(self.cuda_mem_fraction))
        timeout = max(self.trial_time_budget_s + self._TRIAL_GRACE_S, self._MIN_TRIAL_TIMEOUT_S)
        with open(work_path, "w", encoding="utf-8") as f:
            f.write(inject_config(self._source, config))
        try:
            start = time.monotonic()
            try:
                # Own process group: a timeout must also take down the python process
                # a launcher such as `uv run` starts, or it keeps holding the GPU.
                proc = subprocess.Popen(
                    self.run_command + [work_name], cwd=self.repo_dir, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, errors="replace", start_new_session=True)
            except OSError as exc:
                raise ObjectiveUnavailable(
                    f"cannot start {self.run_command!r}: {exc}") from exc
            timed_out = False
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_process_group(proc)
                stdout, stderr = proc.communicate()
            except BaseException:
                _kill_process_group(proc)
                proc.wait()
                raise
            elapsed = time.monotonic() - start
        finally:
            try:
                os.remove(work_path)
            except FileNotFoundError:
                pass
        return self._classify(proc.returncode, stdout, stderr, elapsed, timed_out, timeout)

    @staticmethod
    def _classify(returncode: int, stdout: str, stderr: str, elapsed: float,
                  timed_out: bool, timeout: float) -> Dict[str, Any]:
        detail = (stderr or "")[-2000:] or None
        if not timed_out and returncode != 0:
            text = (stderr or "") + (stdout or "")
            if "OutOfMemoryError" in text or "out of memory" in text.lower():
                return _trial_result(STATUS_OOM, elapsed, error="GPU out-of-memory", detail=detail)
            if (stdout or "").rstrip().endswith("FAIL"):
                # train.py prints FAIL and exits 1 when the loss is NaN or above 100.
                return _trial_result(STATUS_DIVERGED, elapsed,
                                     error="Training diverged (loss NaN or > 100)", detail=detail)
            error = "AssertionError" if "AssertionError" in text else f"train.py exited with code {returncode}"
            return _trial_result(STATUS_ERROR, elapsed, error=error, detail=detail)

        summary = parse_training_summary(stdout or "")
        val_bpb = summary.get("val_bpb")
        if val_bpb is None or not math.isfinite(val_bpb) or val_bpb <= 0.0:
            if timed_out:
                return _trial_result(STATUS_TIMEOUT, elapsed,
                                     error=f"Timeout after {timeout:.0f}s", detail=detail)
            return _trial_result(STATUS_ERROR, elapsed,
                                 error="train.py finished without reporting val_bpb", detail=detail)
        metrics = {k: v for k, v in summary.items()
                   if k not in ("val_bpb", "peak_vram_mb", "training_seconds")}
        return _trial_result(STATUS_OK, summary.get("training_seconds", elapsed), val_bpb=val_bpb,
                             peak_vram_mb=summary.get("peak_vram_mb"), metrics=metrics)


def default_autoresearch_dir() -> str:
    """Where scripts/setup_autoresearch_hpo.sh puts the upstream checkout."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.environ.get("AUTORESEARCH_DIR") or os.path.join(
        root, "original_baseline_repos", "autoresearch")


def make_objective(entry: Dict[str, Any], autoresearch_dir: Optional[str] = None,
                   run_command=None, verify_pin: bool = True, job_id: Optional[str] = None):
    """Build the objective a task record asks for.

    Machine-specific settings are not part of a task record: the checkout comes from
    ``autoresearch_dir`` / ``$AUTORESEARCH_DIR`` and the launcher from ``run_command``
    / ``$AUTORESEARCH_TRAIN_COMMAND`` (default ``uv run python``).
    """
    check_entry(entry)
    spec = entry["objective"]
    hardware = entry["hardware"]
    trial_budget = entry["budget"]["trial_time_budget_s"]
    if spec["kind"] == SyntheticObjective.kind:
        return SyntheticObjective(spec["landscape_seed"], hardware["training_vram_gb"], trial_budget)
    if spec["kind"] == NanochatObjective.kind:
        if run_command is None:
            run_command = shlex.split(os.environ.get("AUTORESEARCH_TRAIN_COMMAND", "uv run python"))
        return NanochatObjective(
            autoresearch_dir or default_autoresearch_dir(), run_command=run_command,
            trial_time_budget_s=trial_budget,
            cuda_mem_fraction=hardware.get("cuda_mem_fraction"),
            verify_pin=verify_pin, job_id=job_id,
            expected_total_vram_gb=hardware.get("total_vram_gb"),
            training_vram_gb=hardware.get("training_vram_gb"))
    raise ValueError(f"unknown objective kind {spec['kind']!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Tasks
# ──────────────────────────────────────────────────────────────────────────────

def check_entry(entry: Dict[str, Any]) -> None:
    """Reject task records that do not describe this version of the benchmark."""
    if entry.get("reward") != BENCHMARK:
        raise ValueError(f"not an {BENCHMARK} task record: reward={entry.get('reward')!r}")
    if entry.get("search_space") != search_space_spec() or entry.get("starting_config") != STARTING_CONFIG:
        raise ValueError(
            f"task {entry.get('task_id')!r} was generated for a different search space; "
            f"regenerate it with data_generation/autoresearch_hpo_generator.py")
    spec = entry.get("objective") or {}
    if spec.get("kind") == NanochatObjective.kind:
        pin = {"repo": UPSTREAM["train_repo"], "commit": UPSTREAM["train_commit"],
               "train_py_sha256": UPSTREAM["train_py_sha256"],
               "prepare_py_sha256": UPSTREAM["prepare_py_sha256"]}
        if any(spec.get(key) != value for key, value in pin.items()):
            raise ValueError(
                f"task {entry.get('task_id')!r} names other training sources than the ones "
                f"this code pins ({pin['repo']}@{pin['commit'][:12]}); regenerate it with "
                f"data_generation/autoresearch_hpo_generator.py")


def public_task_view(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The part of a task record a policy may see (deep copy)."""
    return copy.deepcopy({key: entry[key] for key in PUBLIC_TASK_KEYS if key in entry})


def normalized_score(val_bpb: Optional[float], entry: Dict[str, Any]) -> float:
    """Map a val_bpb to [0, 1]: 0 at the task's reference start, 1 at its reference target."""
    if val_bpb is None:
        return 0.0
    reference = entry["reference"]
    start, target = reference["start_val_bpb"], reference["target_val_bpb"]
    return min(1.0, max(0.0, (start - val_bpb) / (start - target)))


# ──────────────────────────────────────────────────────────────────────────────
# Episodes
# ──────────────────────────────────────────────────────────────────────────────

def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _task_digest(entry: Dict[str, Any]) -> str:
    """Identifies a task record in an episode log's header."""
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode("utf-8")).hexdigest()


def _split_suggestion(raw: Any) -> Tuple[Any, Dict[str, Any]]:
    """Split a policy's return value into (proposed config, metadata).

    A mapping with a ``config`` key is ``{"config": <config>, <metadata>...}``; anything
    else is taken as the configuration itself.
    """
    if isinstance(raw, dict) and "config" in raw:
        return raw["config"], {k: v for k, v in raw.items() if k != "config"}
    return raw, {}


def history_view(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What a policy is shown of the finished trials (deep copy)."""
    return copy.deepcopy([{key: trial[key] for key in HISTORY_KEYS} for trial in trials])


def _read_log(path: str) -> List[Dict[str, Any]]:
    """Rows of an episode log (none if the file does not exist).

    A final line cut short by a crash is removed from the file.
    """
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = f.read()
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
        with open(path, "wb") as f:
            f.write(data)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _append_log(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_episode(policy: Callable[[Dict[str, Any], List[Dict[str, Any]]], Any],
                entry: Dict[str, Any], objective=None, log_path: Optional[str] = None,
                policy_name: str = "", on_trial: Optional[Callable[[Dict[str, Any]], None]] = None,
                max_consecutive_invalid: int = DEFAULT_MAX_CONSECUTIVE_INVALID) -> Dict[str, Any]:
    """Run one HPO episode and return ``{"task_id", "stop_reason", "trials", "score"}``.

    Protocol (the upstream Runner's): trial 0 evaluates the starting configuration;
    before every later trial the loop stops once the cumulative trial time has
    reached ``budget.time_budget_s`` or ``budget.max_trials`` trials exist; otherwise
    the policy is asked for a configuration, which is normalised and evaluated.

    A policy that raises, or returns something ``normalize_config`` rejects, gets an
    ``invalid`` trial (no training, zero seconds, one trial slot) and the episode ends
    after ``max_consecutive_invalid`` of them in a row. ``EpisodeInterrupted`` from
    the policy propagates without recording anything. A failing starting
    configuration raises ``BaselineFailed``: that config is known to train, so the
    environment is broken and nothing measured in it would be the benchmark.

    With ``log_path`` every trial is appended to a JSONL file as soon as it finishes,
    and an existing log for the same task record, objective and policy name is resumed (any
    other log raises ``EpisodeLogMismatch``, which a policy may also raise when it
    cannot continue from the logged trials). ``on_trial`` is called with each new log row.

    The time a policy takes to decide is recorded (``policy_time_s``) but not charged
    to the budget, which counts training time only, as in the paper.
    """
    check_entry(entry)
    if objective is None:
        objective = make_objective(entry)
    task = public_task_view(entry)
    budget = entry["budget"]
    header = {"type": "header", "benchmark": BENCHMARK, "task_id": entry["task_id"],
              "task_sha256": _task_digest(entry),
              "objective": objective.describe(), "policy": policy_name}

    trials: List[Dict[str, Any]] = []
    rows = _read_log(log_path) if log_path else []
    if rows:
        differing = [key for key, value in header.items() if rows[0].get(key) != value]
        if differing:
            details = "; ".join(f"{key}: log has {json.dumps(rows[0].get(key))}, this run "
                                f"{json.dumps(header[key])}" for key in differing)
            raise EpisodeLogMismatch(
                f"{log_path} was written by a different run ({details}); refusing to "
                f"resume into it. Use another output directory.")
        trials = rows[1:]
        if [t.get("trial_id") for t in trials] != list(range(len(trials))):
            raise EpisodeLogMismatch(f"{log_path} has missing or out-of-order trials")
    elif log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        _append_log(log_path, dict(header, created=time.strftime("%Y-%m-%dT%H:%M:%S%z")))

    def record(row: Dict[str, Any]) -> None:
        row["cumulative_time_s"] = (trials[-1]["cumulative_time_s"] if trials else 0.0) + row["wall_time_s"]
        best = trials[-1]["best_val_bpb"] if trials else None
        if row["status"] == STATUS_OK and (best is None or row["val_bpb"] < best):
            best = row["val_bpb"]
        row["best_val_bpb"] = best      # incumbent after this trial
        trials.append(row)
        if log_path:
            _append_log(log_path, row)
        if on_trial:
            on_trial(row)

    def trial_row(config, result, raw=None, meta=None, policy_time_s=0.0) -> Dict[str, Any]:
        return {"type": "trial", "trial_id": len(trials), "config": config,
                "status": result["status"], "val_bpb": result["val_bpb"], "error": result["error"],
                "wall_time_s": result["wall_time_s"], "peak_vram_mb": result["peak_vram_mb"],
                "metrics": result["metrics"], "detail": result["detail"],
                "raw_suggestion": _json_safe(raw), "policy_meta": _json_safe(meta or {}),
                "policy_time_s": round(policy_time_s, 3)}

    if not trials:
        baseline = normalize_config(entry["starting_config"])
        result = objective.evaluate(baseline)
        if result["status"] != STATUS_OK:
            raise BaselineFailed(
                f"starting configuration failed ({result['status']}: {result['error']}); "
                f"this environment cannot run the task (on a GPU with less memory than "
                f"the default configuration needs, no run is possible). "
                f"{result['detail'] or ''}".strip())
        record(trial_row(baseline, result, meta={"source": "starting_config"}))

    while True:
        if trials[-1]["cumulative_time_s"] >= budget["time_budget_s"]:
            stop_reason = "time_budget"
            break
        if len(trials) >= budget["max_trials"]:
            stop_reason = "max_trials"
            break
        tail = 0
        for trial in reversed(trials):
            if trial["status"] != STATUS_INVALID:
                break
            tail += 1
        if tail >= max_consecutive_invalid:
            stop_reason = "policy_invalid"
            break

        started = time.monotonic()
        raw, meta, config = None, {}, None
        try:
            raw, meta = _split_suggestion(policy(copy.deepcopy(task), history_view(trials)))
            config = normalize_config(raw)
            error = None
        except (EpisodeInterrupted, EpisodeLogMismatch):
            raise
        except InvalidConfig as exc:
            error = f"invalid config: {exc}"
        except Exception as exc:  # a policy bug is the policy's failure, not the harness's
            error = f"policy raised {type(exc).__name__}: {exc}"
        policy_time_s = time.monotonic() - started
        if config is None:
            result = _trial_result(STATUS_INVALID, 0.0, error=error)
        else:
            result = objective.evaluate(config)
        record(trial_row(config, result, raw=raw, meta=meta, policy_time_s=policy_time_s))

    return {"task_id": entry["task_id"], "stop_reason": stop_reason, "trials": trials,
            "score": score_episode(trials, entry)}


def score_episode(trials: List[Dict[str, Any]], entry: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise an episode from its trial rows (never from anything a policy reported)."""
    best = None
    curve = []
    counts = {status: 0 for status in STATUSES}
    for trial in trials:
        counts[trial["status"]] += 1
        if trial["status"] == STATUS_OK and (best is None or trial["val_bpb"] < best["val_bpb"]):
            best = trial
            curve.append({"trial_id": trial["trial_id"],
                          "cumulative_time_s": trial["cumulative_time_s"],
                          "best_val_bpb": trial["val_bpb"]})
    baseline = trials[0]["val_bpb"] if trials and trials[0]["status"] == STATUS_OK else None
    best_val = best["val_bpb"] if best else None
    n_trials = len(trials)
    return {
        "n_trials": n_trials,
        "status_counts": counts,
        "failure_rate": (n_trials - counts[STATUS_OK]) / n_trials if n_trials else 0.0,
        "train_time_s": trials[-1]["cumulative_time_s"] if trials else 0.0,
        "baseline_val_bpb": baseline,
        "best_val_bpb": best_val,
        "best_trial_id": best["trial_id"] if best else None,
        "best_config": best["config"] if best else None,
        "improvement": (baseline - best_val) if best and baseline is not None else None,
        "value": normalized_score(best_val, entry),
        "curve": curve,
    }


# ──────────────────────────────────────────────────────────────────────────────
# rewards/ registry contract
# ──────────────────────────────────────────────────────────────────────────────

def _proposed_config(result: Any) -> Any:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except ValueError:
            return None
    return _split_suggestion(result)[0]


def reward(result: Any, execution_ok: bool, entry: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Score one proposed configuration by evaluating it with the task's objective.

    ``result`` may be a config dict, ``{"config": {...}}`` or a JSON string of either.
    Any val_bpb or success flag the candidate reports about itself is ignored: the
    value comes from running the objective here. Pass ``objective=`` to reuse one;
    for a nanochat task this trains for the full trial budget on the GPU.

    value = clip((start - val_bpb) / (start - target), 0, 1) with the task's
    reference start / target val_bpb; a failed or invalid configuration scores 0.
    """
    if not execution_ok:
        return {"value": 0.0, "message": "execution failed"}
    try:
        config = normalize_config(_proposed_config(result))
    except InvalidConfig as exc:
        return {"value": 0.0, "message": f"invalid config: {exc}"}
    objective = kwargs.get("objective") or make_objective(entry)
    outcome = objective.evaluate(config)
    if outcome["status"] != STATUS_OK:
        return {"value": 0.0, "message": f"{outcome['status']}: {outcome['error']}"}
    value = normalized_score(outcome["val_bpb"], entry)
    return {"value": value,
            "message": f"val_bpb={outcome['val_bpb']:.6f} (normalised {value:.4f}) "
                       f"config={json.dumps(config)}"}
