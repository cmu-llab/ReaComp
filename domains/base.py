"""
Domain abstraction for ReaComp.

A *domain* bundles everything the ReaComp pipeline needs to know about a task
family so that trace generation, solver induction, solver evaluation and hybrid
inference can be driven generically:

  * where the task files live (``splits``) and how to load them
  * which record keys an agent/solver may see (``agent_visible_keys``);
    everything else (oracle answers, hidden verifiers) is stripped
  * the verifier, exposed through the existing ``rewards/`` contract
        reward(result, execution_ok, entry, **kw) -> {"value": float, "message": str}
  * how to call an induced ``SOLVER.py`` on one record (``solve``) and how to
    turn its output into a quick_eval-compatible row (``row_extras``,
    ``normalise_answer``, ``complexity``)
  * how a raw LLM trace becomes a solver-induction demo (``demo_from_trace``,
    ``build_demos``) and how demos are balanced (``demo_group_key``)
  * what the coding agents need: the per-task agent's prompts
    (``direct_solve_spec``) and the solver builder's prompts and output file
    names (``solver_builder_spec``)

The base class is deliberately small.  Existing domains (PBEBench, SLR-Bench,
List Functions) are wrapped without changing any of the code they delegate to,
so the historical scripts (``scripts/eval_solver.py`` etc.) keep working
unchanged; the generic entry points (``scripts/eval_domain_solver.py``,
``scripts/hybrid_gate.py``, ``scripts/build_demos.py`` and the ``--domain``
flags of the OpenHands runners) are additive.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: keys every solver-induction demo carries (domains add task-specific fields)
DEMO_KEYS = ("prompt", "cot", "final_response", "success")


@dataclass(frozen=True)
class SplitSpec:
    """One evaluation split of a domain."""
    name: str
    path: str                       # absolute path to a JSONL task file
    label: str                      # human label written into result rows
    params: Dict[str, Any] = field(default_factory=dict)   # domain-specific knobs (e.g. max_programs)


@dataclass
class SolveOutcome:
    """Normalised result of running a solver on one task."""
    success: bool
    score: float
    program: Any = None             # the solver's answer in the domain's native form
    extras: Dict[str, Any] = field(default_factory=dict)   # extra columns for the output row


def load_traces(path: str) -> List[dict]:
    """Load trace rows from JSONL (one row per line) or from a JSON list."""
    with open(path) as fh:
        text = fh.read()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def balanced_sample(demos: List[dict], n: int, key_fn: Callable[[dict], Tuple], seed: int = 42) -> List[dict]:
    """Round-robin over the groups induced by ``key_fn`` (e.g. difficulty x success)
    until ``n`` demos are chosen; groups are shuffled with ``seed``."""
    rng = random.Random(seed)
    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for d in demos:
        groups[key_fn(d)].append(d)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups, key=repr)
    picked: List[dict] = []
    while len(picked) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    rng.shuffle(picked)
    return picked


class Domain:
    """Base class.  Subclasses set the class attributes and override the hooks they need."""

    #: registry name, e.g. "pbebench"
    name: str = ""
    #: one-line description used by ``list_domains`` and generic prompts
    description: str = ""
    #: name of the module under ``rewards/`` implementing the verifier contract
    reward_module: str = ""
    #: candidate entry-point function names in an induced solver file, in priority order
    solver_entrypoints: tuple = ()
    #: record keys that may be shown to an LLM agent or to a solver
    agent_visible_keys: frozenset = frozenset({"question", "prompt", "task"})
    #: solver-building prompt (relative to repo root) and demo trace files by name
    building_prompt: Optional[str] = None
    demos: Dict[str, str] = {}
    #: module path of a bespoke trace generator, for domains whose agent cannot be
    #: expressed as the generic direct-solve loop (e.g. "domains.envsetup.agent")
    trace_runner: Optional[str] = None
    #: default output filenames of the solver builder
    solver_filename: str = "SOLVER.py"
    algorithm_filename: str = "SOLVER_ALGORITHM.md"

    # ------------------------------------------------------------------ splits
    def splits(self) -> Dict[str, SplitSpec]:
        raise NotImplementedError

    def split(self, name: str) -> SplitSpec:
        specs = self.splits()
        if name not in specs:
            raise KeyError(f"{self.name}: unknown split {name!r}; available: {sorted(specs)}")
        return specs[name]

    def load_tasks(self, split: str, limit: Optional[int] = None) -> List[dict]:
        spec = self.split(split)
        records = []
        with open(spec.path) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec.setdefault("task_index", i)
                records.append(rec)
                if limit is not None and len(records) >= limit:
                    break
        return records

    # ---------------------------------------------------------------- privacy
    def task_input(self, record: dict) -> dict:
        """The agent-visible view of a record (oracle fields removed)."""
        return {k: v for k, v in record.items() if k in self.agent_visible_keys}

    # --------------------------------------------------------------- verifier
    def reward_fn(self) -> Callable:
        mod = importlib.import_module(f"rewards.{self.reward_module}")
        return mod.reward

    def reward(self, result: Any, execution_ok: bool, entry: dict, **kw) -> dict:
        return self.reward_fn()(result, execution_ok, entry, **kw)

    # ----------------------------------------------------------------- solver
    @staticmethod
    def load_solver_module(path: str):
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"solver not found: {path}")
        spec = importlib.util.spec_from_file_location("REACOMP_SOLVER", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def load_solver(self, path: str) -> Callable:
        mod = self.load_solver_module(path)
        for name in self.solver_entrypoints:
            if hasattr(mod, name):
                return getattr(mod, name)
        raise AttributeError(
            f"{path} defines none of the {self.name} entry points {list(self.solver_entrypoints)}")

    def solve(self, solve_fn: Callable, record: dict, split: SplitSpec) -> SolveOutcome:
        """Run the solver on one record and normalise its result."""
        raise NotImplementedError

    def cleanup_after_timeout(self, record: dict, split: SplitSpec) -> None:
        """Release any external resources a timed-out solve may have leaked (containers etc.)."""

    # ------------------------------------------------------------------- rows
    def normalise_answer(self, program: Any) -> Any:
        return program

    def complexity(self, answer: Any) -> Optional[float]:
        """Program complexity used to break ties when ensembling; None if undefined."""
        return None

    def row_extras(self, record: dict) -> dict:
        """Domain-specific context columns copied into result rows."""
        return {}

    # ---------------------------------------------------------------- agents
    def agent_task_text(self, record: dict, split: Optional[SplitSpec] = None) -> str:
        """Task statement shown to a per-task LLM agent."""
        ti = self.task_input(record)
        return ti.get("question") or ti.get("prompt") or ti.get("task") or json.dumps(ti)

    def direct_solve_spec(self, split: Optional[SplitSpec] = None, **kw) -> dict:
        """Prompts for the generic per-task agent (``openhands_agents.run --framework direct_solve``):
        {"system_prompt": str, "task_message": callable(record) -> str}."""
        entry = self.solver_entrypoints[0] if self.solver_entrypoints else "solve"
        system_prompt = (
            f"You are an expert problem solver working on {self.description or self.name} tasks.\n\n"
            "You have two tools:\n\n"
            "  execute_code(code)\n"
            "      Run Python code in a sandbox. The verifier is importable:\n"
            f"          import sys; sys.path.insert(0, '/workspace')\n"
            f"          from rewards.{self.reward_module} import reward\n"
            "      Call reward(candidate, True, task_record) to score a candidate (value in [0, 1]).\n"
            "      All task information is in this message; no other files exist in the sandbox.\n\n"
            "  submit_answer(programs)\n"
            "      Submit your final answer as a list with your candidate. Submit as soon as you\n"
            "      reach reward=1.0, or your best attempt before you run out of steps.\n\n"
            f"The induced program must follow the domain's answer format (see the task); a\n"
            f"symbolic solver for this domain would expose it as `{entry}`.\n"
        )
        return {"system_prompt": system_prompt,
                "task_message": lambda record: self.agent_task_text(record, split)}

    def solver_builder_spec(self) -> dict:
        """What the solver-building coding agent needs:
        {"system_prompt": str, "file_note": str, "solver_filename": str, "algorithm_filename": str}."""
        entry = self.solver_entrypoints[0] if self.solver_entrypoints else "solve"
        sf, af = self.solver_filename, self.algorithm_filename
        system_prompt = (
            "You are an expert in symbolic program induction and Python.\n\n"
            f"Your task is to write a symbolic solver for {self.description or self.name} tasks.\n"
            "You will be given a detailed specification in the user message.\n\n"
            "You have three tools:\n\n"
            "  execute_code(code)\n"
            "      Run Python code in a sandbox.\n"
            "      The demo traces are mounted at /workspace/DEMOS.json.\n"
            "      The verifier is directly importable (sys.path already includes /workspace):\n"
            f"          from rewards.{self.reward_module} import reward\n"
            "      Use this to explore /workspace/DEMOS.json, understand task structure, and test\n"
            "      small solver logic snippets before writing the final files.\n\n"
            "  write_file(filename, content)\n"
            "      Write a file to the output directory.\n"
            f"      Call once with filename='{sf}' and once with filename='{af}'.\n"
            "      You MUST call this for both files before finishing.\n\n"
            "  finish(summary)\n"
            "      Signal that you are done. Call after both files have been written.\n\n"
            "Workflow:\n"
            "  1. Read the specification in the user message carefully.\n"
            "  2. Use execute_code to inspect DEMOS.json (structure, patterns, constraints).\n"
            "  3. Design your solver algorithm based on what you observe.\n"
            f"  4. Write {sf} with the complete implementation.\n"
            f"  5. Write {af} with a clear algorithm description.\n"
            "  6. Call finish with a brief summary.\n\n"
            "Requirements:\n"
            f"  - The solver must implement {entry} as described in the specification.\n"
            "  - Use only the Python standard library.\n"
            "  - Do NOT produce placeholder code. Write a complete, working implementation.\n"
        )
        file_note = (
            "File path reference guide (for @-references in the spec below):\n"
            "  @DEMOS.json  → /workspace/DEMOS.json   (use execute_code to read)\n"
            f"  @rewards/{self.reward_module}.py → importable as: from rewards.{self.reward_module} import reward\n"
            f"  @{sf} → write via write_file(filename='{sf}', content=...)\n"
            f"  @{af} → write via write_file(filename='{af}', content=...)\n"
            "\n---\n\n"
        )
        return {"system_prompt": system_prompt, "file_note": file_note,
                "solver_filename": sf, "algorithm_filename": af}

    # ----------------------------------------------------------------- demos
    @staticmethod
    def trace_success(trace: dict) -> bool:
        if "success" in trace:
            return bool(trace["success"])
        if trace.get("best_reward") is not None:
            return float(trace["best_reward"]) >= 1.0
        return bool(trace.get("solved"))

    @staticmethod
    def trace_cot(trace: dict) -> str:
        """Reasoning text of a trace row: explicit ``cot``/``reasoning`` fields, or the
        reasoning content of the last LLM response in ``agent_messages`` (BoK / DF rows)."""
        for k in ("cot", "reasoning", "chain_of_thought"):
            if trace.get(k):
                return trace[k] if isinstance(trace[k], str) else json.dumps(trace[k])
        msgs = trace.get("agent_messages") or []
        if msgs:
            resp = msgs[-1].get("response") or {}
            return resp.get("reasoning_content") or resp.get("reasoning") or ""
        return ""

    @staticmethod
    def trace_answer(trace: dict) -> str:
        """Final response text of a trace row (the model's last message, else the answer)."""
        if trace.get("final_response"):
            return trace["final_response"]
        msgs = trace.get("agent_messages") or []
        if msgs:
            content = (msgs[-1].get("response") or {}).get("content")
            if content:
                return content
        ans = trace.get("answer")
        if ans is None:
            return ""
        return ans if isinstance(ans, str) else json.dumps(ans)

    def demo_group_key(self, demo: dict) -> Tuple:
        """Grouping used to balance demos; default: (difficulty-like field, success)."""
        bucket = demo.get("difficulty") or demo.get("category") or demo.get("hardness_bucket") or "all"
        return (bucket, bool(demo.get("success")))

    def demo_from_trace(self, record: dict, trace: dict, split: Optional[SplitSpec] = None) -> Optional[dict]:
        """Convert one trace row for ``record`` into a demo, or None to skip it."""
        return {
            "prompt": self.agent_task_text(record, split),
            "final_response": self.trace_answer(trace),
            "cot": self.trace_cot(trace),
            "success": self.trace_success(trace),
            "task_index": record.get("task_index"),
        }

    def match_trace(self, trace: dict, by_index: Dict[Any, dict], by_id: Dict[Any, dict]) -> Optional[dict]:
        if trace.get("task_index") is not None and trace["task_index"] in by_index:
            return by_index[trace["task_index"]]
        if trace.get("task_id") is not None and trace["task_id"] in by_id:
            return by_id[trace["task_id"]]
        return None

    def build_demos(self, trace_paths: Iterable[str], split: str, n: int = 100, seed: int = 42,
                    out: Optional[str] = None, per_task: Optional[int] = None) -> List[dict]:
        """Generic demo builder: traces -> demos -> balanced sample -> JSON file."""
        spec = self.split(split)
        records = self.load_tasks(split)
        by_index = {r["task_index"]: r for r in records}
        by_id = {r.get("task_id"): r for r in records if r.get("task_id") is not None}
        pool: List[dict] = []
        per_task_count: Dict[Any, int] = defaultdict(int)
        for path in trace_paths:
            for trace in load_traces(path):
                rec = self.match_trace(trace, by_index, by_id)
                if rec is None:
                    continue
                key = rec.get("task_id", rec["task_index"])
                if per_task is not None and per_task_count[key] >= per_task:
                    continue
                demo = self.demo_from_trace(rec, trace, spec)
                if demo is None:
                    continue
                per_task_count[key] += 1
                pool.append(demo)
        chosen = balanced_sample(pool, n, self.demo_group_key, seed)
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
            with open(out, "w") as fh:
                json.dump(chosen, fh, indent=1)
        return chosen

    # ------------------------------------------------------------------ misc
    @staticmethod
    def _call_with_supported_kwargs(fn: Callable, *args, **kwargs):
        """Call ``fn`` passing only the keyword arguments its signature accepts."""
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return fn(*args, **kwargs)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return fn(*args, **kwargs)
        return fn(*args, **{k: v for k, v in kwargs.items() if k in params})

    def abspath(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(REPO_ROOT, rel)
