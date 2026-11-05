"""List Functions domain: integer-list -> integer-list programs scored on held-out pairs.

Mirrors ``scripts/eval_listfn_solver.py``: the solver sees the shown (held-in)
pairs only and a task counts as solved when the induced program reproduces
every held-out pair.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, List, Optional
import json
import os
import random
from collections import defaultdict

from .base import Domain, SolveOutcome, SplitSpec, REPO_ROOT


class ListFunctionsDomain(Domain):
    name = "list_functions"
    description = "List Functions (Rule et al.) with held-out generalisation scoring"
    reward_module = "list_functions"
    solver_entrypoints = ("solve_listfn",)
    agent_visible_keys = frozenset({"question", "prompt", "task", "task_id",
                                    "inputs", "outputs", "held_in_inputs", "held_in_outputs"})
    building_prompt = "building_prompts/SOLVER_BUILDING_PROMPT_LISTFN.md"
    demos = {"balanced": "demos/DEMOS_LISTFN_balanced.json",
             "with_cot": "demos/DEMOS_LISTFN_with_CoT.json"}

    def splits(self) -> Dict[str, SplitSpec]:
        d = os.path.join(REPO_ROOT, "data", "list_functions")
        return {"pilot": SplitSpec("pilot", os.path.join(d, "pilot_tasks.jsonl"), "ListFunctions-pilot", {}),
                "full": SplitSpec("full", os.path.join(d, "full_tasks.jsonl"), "ListFunctions", {})}

    def solve(self, solve_fn: Callable, record: dict, split: SplitSpec) -> SolveOutcome:
        from rewards.list_functions import _extract_callable, score_program
        examples = list(zip([list(x) for x in record["held_in_inputs"]],
                            [list(y) for y in record["held_in_outputs"]]))
        result = solve_fn(examples)
        program = result.get("program") if isinstance(result, dict) else result
        fn = _extract_callable(program) if program is not None else None
        if fn is None:
            return SolveOutcome(False, 0.0, program, {"shown_score": 0.0, "claimed_success": None})
        shown, _ = score_program(fn, record["held_in_inputs"], record["held_in_outputs"])
        held, mismatches = score_program(fn, record["held_out_inputs"], record["held_out_outputs"])
        claimed = bool(result.get("success")) if isinstance(result, dict) else None
        return SolveOutcome(held >= 1.0, float(held), program if isinstance(program, str) else str(program),
                            {"shown_score": round(shown, 4), "claimed_success": claimed,
                             "mismatches": mismatches[:3]})

    def row_extras(self, record: dict) -> dict:
        return {"task_id": record.get("task_id")}

    # -- demos ----------------------------------------------------------------
    solver_filename = "SOLVER_LISTFN.py"
    algorithm_filename = "SOLVER_LISTFN_ALGORITHM.md"

    @staticmethod
    def _trace_prompt(record: dict) -> str:
        from openhands_agents.run_listfn_traces import build_prompt
        return build_prompt(record)

    def demo_from_trace(self, record: dict, trace: dict, split: Optional[SplitSpec] = None) -> Optional[dict]:
        """Schema of demos/DEMOS_LISTFN_*.json; traces are rows of outputs/listfn_*_traces.jsonl."""
        is_succ = bool(trace.get("solved") or trace.get("heldout_score") == 1.0)
        return {
            "prompt": self._trace_prompt(record),
            "input_examples": record["held_in_inputs"],
            "output_examples": record["held_in_outputs"],
            "final_response": trace.get("code") or trace.get("final_response") or "",
            "cot": trace.get("reasoning") or trace.get("cot") or "",
            "success": is_succ,
            "task_id": record["task_id"],
            "shown_score": trace.get("shown_score"),
            "heldout_score": trace.get("heldout_score"),
        }

    @staticmethod
    def task_hardness(record: dict) -> float:
        """Mean normalised edit distance between shown inputs and outputs (model-free)."""
        def lev(a, b):
            if a == b:
                return 0
            la, lb = len(a), len(b)
            if la == 0:
                return lb
            if lb == 0:
                return la
            prev = list(range(lb + 1))
            for i in range(1, la + 1):
                cur = [i] + [0] * lb
                for j in range(1, lb + 1):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
                prev = cur
            return prev[lb]
        scores = [lev(a, b) / max(len(a), len(b), 1)
                  for a, b in zip(record["held_in_inputs"], record["held_in_outputs"])]
        return sum(scores) / len(scores) if scores else 0.0

    def build_demos(self, trace_paths: Iterable[str], split: str, n: int = 100, seed: int = 42,
                    out: Optional[str] = None, per_task: Optional[int] = None) -> List[dict]:
        """Balanced over intrinsic task hardness terciles and success/failure *within* a task,
        keeping ``per_task`` attempts per task (the paper's List Functions protocol; reproduces
        demos/DEMOS_LISTFN_balanced.json with split="full", n=100, per_task=4, seed=42).
        Falls back to the generic builder when per_task is None."""
        if per_task is None:
            return super().build_demos(trace_paths, split, n, seed, out, per_task)
        from domains.base import load_traces
        attempts = per_task
        n_tasks = max(1, n // attempts)
        rng = random.Random(seed)
        tasks = {r["task_id"]: r for r in self.load_tasks(split)}
        cand_by_task: Dict[str, List[dict]] = defaultdict(list)
        for path in trace_paths:
            for c in load_traces(path):
                cand_by_task[c["task_id"]].append(c)
        hard_score = {t: self.task_hardness(tasks[t]) for t in cand_by_task if t in tasks}
        ordered = sorted(hard_score, key=lambda t: hard_score[t])
        third = len(ordered) // 3
        bucket = {t: ("easy" if i < third else ("medium" if i < 2 * third else "hard"))
                  for i, t in enumerate(ordered)}
        buckets: Dict[str, List[str]] = defaultdict(list)
        for t, b in bucket.items():
            buckets[b].append(t)
        per_bucket = max(1, n_tasks // 3)
        picked: List[str] = []
        for b in ("hard", "medium", "easy"):
            ids = buckets.get(b, [])[:]
            rng.shuffle(ids)
            picked.extend(ids[:per_bucket])
        remaining = [t for t in cand_by_task if t not in picked]
        rng.shuffle(remaining)
        picked = (picked + remaining)[:n_tasks]
        demos: List[dict] = []
        for tid in picked:
            cs = cand_by_task[tid]
            succ = [c for c in cs if c.get("solved") or c.get("heldout_score") == 1.0]
            fail = [c for c in cs if not (c.get("solved") or c.get("heldout_score") == 1.0)]
            rng.shuffle(succ)
            rng.shuffle(fail)
            half = attempts // 2
            chosen = succ[:half] + fail[: attempts - half]
            if len(chosen) < attempts:
                for c in succ + fail:
                    if c not in chosen:
                        chosen.append(c)
                    if len(chosen) >= attempts:
                        break
            rec = tasks[tid]
            for c in chosen:
                d = self.demo_from_trace(rec, c)
                d["hardness"] = round(hard_score[tid], 3)
                d["hardness_bucket"] = bucket[tid]
                demos.append(d)
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
            with open(out, "w") as fh:
                json.dump(demos, fh, indent=2)
        return demos

    # -- coding agents (verbatim legacy prompts) ------------------------------
    def solver_builder_spec(self) -> dict:
        from openhands_agents.solver_builder.listfn_controller import _SYSTEM_PROMPT_LISTFN, _FILE_NOTE
        return {"system_prompt": _SYSTEM_PROMPT_LISTFN, "file_note": _FILE_NOTE,
                "solver_filename": "SOLVER_LISTFN.py", "algorithm_filename": "SOLVER_LISTFN_ALGORITHM.md"}
