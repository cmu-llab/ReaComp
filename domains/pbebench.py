"""PBEBench domain: ordered cascades of replace(A, B) programs (Lite / Hard / real forward reconstruction).

Thin wrapper over ``rewards/pbebench.py`` and the split configuration that
``scripts/eval_solver.py`` has always used; behaviour is identical.
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, Optional

from .base import Domain, SolveOutcome, SplitSpec, REPO_ROOT


class PBEBenchDomain(Domain):
    name = "pbebench"
    description = "PBEBench string-rewrite cascades (Lite, Hard) and real forward reconstruction"
    reward_module = "pbebench"
    solver_entrypoints = ("solve_pbe",)
    agent_visible_keys = frozenset({"question", "prompt", "task", "inputs", "outputs"})
    building_prompt = "building_prompts/SOLVER_BUILDING_PROMPT_PBE.md"
    demos = {
        "100_cot": "demos/DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json",
        "100_nocot": "demos/DEMOS_PBEBENCH_seed_42_100_examples.json",
        "48_cot": "demos/DEMOS_PBEBENCH_seed_42_48_examples_with_CoT.json",
        "12_cot": "demos/DEMOS_PBEBENCH_seed_42_12_examples_with_CoT.json",
    }

    def splits(self) -> Dict[str, SplitSpec]:
        d = os.path.join(REPO_ROOT, "data")
        common = {"max_pred_len": 3, "max_transform_len": 3}
        return {
            "lite": SplitSpec("lite", os.path.join(d, "pbebench", "lite_tasks_full_og.jsonl"),
                              "PBEBench-Lite", {"max_programs": 5, **common}),
            "hard": SplitSpec("hard", os.path.join(d, "pbebench", "tasks_full_og.jsonl"),
                              "PBEBench-Hard", {"max_programs": 20, **common}),
            "real": SplitSpec("real", os.path.join(d, "real_forward_reconstruction.jsonl"),
                              "RealFR", {"max_programs": 20, **common}),
        }

    # -- solver ---------------------------------------------------------------
    def solve(self, solve_fn: Callable, record: dict, split: SplitSpec) -> SolveOutcome:
        examples = list(zip(record["inputs"], record["outputs"]))
        p = split.params
        max_programs = p.get("max_programs", 5)
        ratio = p.get("max_programs_ratio")
        if ratio is not None:
            max_programs = max(2, math.ceil(len(examples) / ratio))
        kwargs = {"max_programs": max_programs}
        for k in ("max_pred_len", "max_transform_len"):
            if k in p:
                kwargs[k] = p[k]
        result = self._call_with_supported_kwargs(solve_fn, examples, **kwargs)
        success = bool(result.get("success"))
        if "score" in result:
            score = float(result["score"])
        elif result.get("scores"):
            score = float(max(result["scores"]))
        else:
            score = 1.0 if success else 0.0
        return SolveOutcome(success=success, score=score, program=result.get("program"))

    # -- rows -----------------------------------------------------------------
    def normalise_answer(self, program: Any) -> Any:
        if isinstance(program, list) and program and isinstance(program[0], (list, tuple)):
            return [f"replace('{a}', '{b}')" for a, b in program]
        return program

    def complexity(self, answer: Any) -> Optional[float]:
        from rewards.pbebench import _parse_programs, cascade_complexity
        programs, _ = _parse_programs(self.normalise_answer(answer))
        return None if not programs else float(cascade_complexity(programs))

    def row_extras(self, record: dict) -> dict:
        return {"cascade_length": record.get("cascade_length"),
                "bfcc_string": record.get("bfcc_string")}

    # -- demos ----------------------------------------------------------------
    #: the paper's demo files label Lite tasks "easy" and Hard tasks "hard"
    DIFFICULTY_BY_SPLIT = {"lite": "easy", "hard": "hard"}

    def demo_from_trace(self, record: dict, trace: dict, split: Optional[SplitSpec] = None) -> Optional[dict]:
        difficulty = trace.get("difficulty") or (self.DIFFICULTY_BY_SPLIT.get(split.name) if split else None)
        return {
            "prompt": record["prompt"],
            "input_examples": record["inputs"],
            "output_examples": record["outputs"],
            "cascade_length": record.get("cascade_length"),
            "bfcc_string": record.get("bfcc_string"),
            "difficulty": difficulty,
            "final_response": self.trace_answer(trace),
            "cot": self.trace_cot(trace),
            "success": self.trace_success(trace),
        }

    # -- coding agents (verbatim legacy prompts) ------------------------------
    def direct_solve_spec(self, split: Optional[SplitSpec] = None, **kw) -> dict:
        from openhands_agents.direct_solve.controller import _make_system_prompt_pbe, _build_task_message_pbe
        max_programs = kw.get("max_programs") or (split.params.get("max_programs") if split else None) or 5
        return {"system_prompt": _make_system_prompt_pbe(max_programs), "task_message": _build_task_message_pbe}

    def solver_builder_spec(self) -> dict:
        from openhands_agents.solver_builder.controller import _SYSTEM_PROMPT_PBE, _FILE_NOTE_PBE
        return {"system_prompt": _SYSTEM_PROMPT_PBE, "file_note": _FILE_NOTE_PBE,
                "solver_filename": "SOLVER.py", "algorithm_filename": "SOLVER_ALGORITHM.md"}
