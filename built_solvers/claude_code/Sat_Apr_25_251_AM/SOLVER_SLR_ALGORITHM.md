# SOLVER_SLR Algorithm

## Overview

A pure-Python symbolic rule inducer for SLR-Bench. Given a set of `(facts_string, direction_label)` pairs it searches the space of Prolog rules of the form `eastbound(T) :- Body.` and returns the simplest rule that perfectly separates eastbound from westbound trains.

No external libraries are required; the solver falls back to the `rewards/slr_bench` verifier for final validation only when it is available.

---

## Algorithm

### 1. Parsing

Each `facts_string` is parsed into a **normalised car model**: a dict `{c1: {pred: val}, c2: {pred: val}, ...}` where cars are sorted by `car_num` and re-indexed as `c1, c2, …`. This makes models train-id agnostic while preserving positional order.

Predicates are discovered dynamically so the solver handles any extensions to the DSL (e.g. `has_roof`).

### 2. Candidate Generation (`generate_candidates`)

Candidates are generated in ascending **rule complexity** order (as defined by `rule_complexity()` in `rewards/slr_bench.py`: number of non-`has_car` body literals).

| Complexity | Rule shape |
|---|---|
| 1 | `has_car(T,C), prop(C,val)` |
| 2 | `has_car(T,C), prop1(C,v1), prop2(C,v2)` or `…car_num(C,n), prop(C,v)` or two-car rules with one property each |
| 3 | Three properties on one car; car_num + two properties; two-car with position on one side |
| 4 | Four properties; car_num + three properties; two-car with position on both sides |

The search only generates rules with **distinct predicates per car** to avoid contradictions. Duplicate rule strings are discarded.

### 3. Local Evaluation

Each candidate spec is evaluated against all examples using a Python emulation of Prolog's existential semantics:

- **Single-car rule**: fires if *any* car in the train satisfies all conditions.
- **Two-car rule**: fires if *any two distinct cars* satisfy their respective conditions.

Accuracy = fraction of examples classified correctly.

### 4. Ranking and Selection

Candidates are ranked by `(-accuracy, complexity)` — perfect accuracy first, then fewest literals. The best rule is returned as `program`.

### 5. Optional Verifier Re-scoring (`solve_slr_with_entry`)

When a `validation program` (from the task entry) is available and SWI-Prolog + the HuggingFace `evaluate` library are installed, the top-K candidate rules are re-scored using the official verifier. The highest-scoring verified rule becomes the final answer.

---

## Interface

```python
result = solve_slr(examples, top_k=5)
# examples: list of (facts_string, direction_label)
# result keys:
#   'success'        : bool — True if a perfect rule was found
#   'program'        : str  — best Prolog rule
#   'top_k_programs' : list[str] — up to top_k rules by score
#   'score'          : float — accuracy on training examples
#   'complexity'     : int   — rule_complexity of best rule
```

---

## Design Choices

- **Ascending-complexity search** ensures the simplest consistent rule is returned, matching the dataset's preference for minimal hypotheses.
- **Local Python evaluator** avoids a SWI-Prolog dependency for the search phase, making the solver fast and portable.
- **Dynamic predicate discovery** means the solver handles any predicate vocabulary (e.g. `has_roof`) without modification.
- **Two-car rules** cover multi-car patterns where no single-car property is discriminative.
- **Top-K fallback** returns the best partial rules when no perfect solution is found, useful for downstream refinement.
