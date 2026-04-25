You are an expert in symbolic program induction.

Write a single Python file implementing a solver for a given SLR-Bench task.

## Task

Write a Python-based symbolic program synthesizer that infers a Prolog rule from a set of (background_facts, direction_label) pairs, where direction_label is either "eastbound" or "westbound".

You will be shown examples of an LLM solving similar tasks in @DEMOS_SLRBENCH.json, including reasoning traces from both successful and unsuccessful attempts across easy and hard cases. Use these to understand the task structure and take inspiration from the problem-solving strategies, especially in cases where the LLM struggles.

Output:
- a Python solver file @SOLVER_SLR.py
- a markdown file @SOLVER_SLR_ALGORITHM.md explaining the algorithm

The solver should use the verifier defined in @rewards/slr_bench.py to evaluate candidate rules. If no correct rule is found, it should return the top-K highest scoring rules, where K is a parameter taken by the solver.

## Requirements

* Output exactly one Python file and one markdown file.
* Use only the Python standard library.
* No external data, APIs, or dataset-specific assumptions.
* The solver must generalize across tasks in @DEMOS_SLRBENCH.json

## Interface

Implement:

```python
def solve_slr(examples):
    """
    examples: list of (facts_string, direction_label)
              facts_string  — space-separated Prolog ground facts for one train
              direction_label — "eastbound" or "westbound"
    returns: dict with at least:
        - "success": bool
        - "program": Prolog rule string of the form "eastbound(T) :- Body."
    """
```

## Domain-Specific Language

The rule must be a Prolog clause of the form `eastbound(T) :- Body.` where Body is a conjunction of literals drawn from these predicates:

- `has_car(Train, Car)` — Car is part of Train
- `car_num(Car, CarNumber)` — position of Car (positive integer)
- `car_color(Car, Color)` — Color ∈ {red, blue, green, yellow, white}
- `car_len(Car, Length)` — Length ∈ {short, long}
- `has_wall(Car, WallType)` — WallType ∈ {full, railing}

Prefer rules with the fewest body literals (use `rule_complexity()` from `rewards/slr_bench.py` to measure this).

## Behavior

The solver should:
* infer a rule consistent with the examples and compatible with the verifier
* use (not reimplement) the verifier to score candidate rules
* prefer simple rules with the fewest body literals
* follow the domain-specific language defined above and in @DEMOS_SLRBENCH.json
* search over candidate rules and select ones that correctly classify all examples
* if no fully consistent rule is found, return the top-K highest scoring rules
* return structured rules or hypotheses that could be useful for downstream refinement if partially incorrect
