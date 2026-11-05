"""SLR-Bench domain: inductive Prolog rule synthesis (eastbound(T) :- Body.)."""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from .base import Domain, SolveOutcome, SplitSpec, REPO_ROOT


class SLRBenchDomain(Domain):
    name = "slr_bench"
    description = "SLR-Bench inductive logic programming over train descriptions"
    reward_module = "slr_bench"
    solver_entrypoints = ("solve_slr",)
    # The validation program is the judge harness, not the answer; agents need it to call the reward.
    agent_visible_keys = frozenset({"question", "prompt", "task", "validation program"})
    building_prompt = "building_prompts/SOLVER_BUILDING_PROMPT_SLR.md"
    demos = {"92_cot": "demos/DEMOS_SLRBENCH_seed_42_92_examples_with_CoT.json"}

    def splits(self) -> Dict[str, SplitSpec]:
        return {"slr": SplitSpec("slr", os.path.join(REPO_ROOT, "data", "slr_bench", "v1_All_full.jsonl"),
                                 "SLR-Bench", {})}

    def solve(self, solve_fn: Callable, record: dict, split: SplitSpec) -> SolveOutcome:
        from rewards.slr_bench import parse_prompt_examples
        parsed = parse_prompt_examples(record["prompt"])
        examples = list(zip(parsed["inputs"], parsed["outputs"]))
        result = solve_fn(examples)
        success = bool(result.get("success"))
        score = float(result.get("score", 1.0 if success else 0.0))
        return SolveOutcome(success=success, score=score, program=result.get("program"))

    def complexity(self, answer: Any) -> Optional[float]:
        try:
            from rewards.slr_bench import _extract_rule, rule_complexity
            rule, _ = _extract_rule(answer)
            return None if rule is None else float(rule_complexity(rule))
        except Exception:
            return None

    def row_extras(self, record: dict) -> dict:
        return {"rule_complexity": record.get("rule complexity"),
                "curriculum_level": record.get("curriculum level"),
                "curriculum_tier": record.get("curriculum tier"),
                "ground_truth_rule": record.get("ground-truth rule")}

    # -- demos (schema of demos/DEMOS_SLRBENCH_seed_42_92_examples_with_CoT.json) ---
    def demo_from_trace(self, record: dict, trace: dict, split: Optional[SplitSpec] = None) -> Optional[dict]:
        from rewards.slr_bench import parse_prompt_examples
        parsed = parse_prompt_examples(record["prompt"])
        return {
            "prompt": record["prompt"],
            "input_examples": parsed["inputs"],
            "output_examples": parsed["outputs"],
            "difficulty": record.get("curriculum tier"),
            "curriculum_level": record.get("curriculum level"),
            "rule_complexity": record.get("rule complexity"),
            "rule_sampling": record.get("rule sampling"),
            "background_sampling": record.get("background sampling"),
            "problem_size": record.get("problem size"),
            "symbols": record.get("symbols"),
            "vocabulary_predicates": record.get("vocabulary predicates"),
            "vocabulary_car_constants": record.get("vocabulary car constants"),
            "validation_program": record.get("validation program"),
            "final_response": self.trace_answer(trace),
            "cot": self.trace_cot(trace),
            "success": self.trace_success(trace),
        }

    # -- coding agents (verbatim legacy prompts) ------------------------------
    def direct_solve_spec(self, split: Optional[SplitSpec] = None, **kw) -> dict:
        from openhands_agents.direct_solve.controller import _SYSTEM_PROMPT_SLR, _build_task_message_slr
        return {"system_prompt": _SYSTEM_PROMPT_SLR, "task_message": _build_task_message_slr}

    def solver_builder_spec(self) -> dict:
        from openhands_agents.solver_builder.controller import _SYSTEM_PROMPT_SLR, _FILE_NOTE_SLR
        return {"system_prompt": _SYSTEM_PROMPT_SLR, "file_note": _FILE_NOTE_SLR,
                "solver_filename": "SOLVER_SLR.py", "algorithm_filename": "SOLVER_SLR_ALGORITHM.md"}
