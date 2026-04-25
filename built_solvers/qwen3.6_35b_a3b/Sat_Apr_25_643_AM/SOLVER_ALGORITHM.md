# SLR-Bench Solver Algorithm

## Overview

This solver performs symbolic rule induction for the Symbolic Logic Rule (SLR-Bench) task. Given training examples of trains characterized by Prolog facts, the solver infers a Prolog rule of the form `eastbound(T) :- Body.` that correctly classifies all eastbound and westbound trains.

## Algorithm Design

The solver uses a **layered hypothesis generation with early-exit evaluation** strategy:

### Step 1: Parse Examples into a Validation Program

The input is a list of `(facts_string, direction_label)` pairs, where `facts_string` is a space-separated string of Prolog ground atoms (e.g., `has_car(train0, car0_1). car_num(car0_1, 1). ...`), and `direction_label` is either `"eastbound"` or `"westbound"`.

We convert this into a Prolog validation program that contains:
- All ground facts from each train
- An `eastbound(trainN)` or `westbound(trainN)` declaration for each train

This validation program is used by the judge to evaluate candidate rules.

### Step 2: Build Feature Space

We extract all predicates and their argument value domains from the examples using regex matching. The result is a dictionary:

```python
{
    'car_color': {0: {'car0_1', 'car0_2', ...}, 1: {'red', 'blue', ...}},
    'car_len':   {0: {'car0_1', 'car0_2', ...}, 1: {'short', 'long'}},
    'has_wall':  {0: {'car0_1', 'car0_2', ...}, 1: {'full', 'railing'}},
    ...
}
```

### Step 3: Generate Candidate Body Literals

We classify each predicate based on its argument types:

- **Car-level predicates**: Arg0 is a car reference (e.g., `car0_1`), arg1 is a property value (e.g., `short`, `red`). These generate literals of the form `pred(C, value)`.
- **Train-level predicates**: Arg0 is a train reference (e.g., `train0`), arg1 is a property value. These generate literals of the form `pred(T, value)`.

The `has_car/2` predicate is always included as a structural glue but is not counted as a body literal for complexity measurement.

### Step 4: Layered Rule Generation with Early Exit

We generate candidate rules in layers of increasing complexity:

```
Layer 1: eastbound(T) :- has_car(T, C), literal_1.
Layer 2: eastbound(T) :- has_car(T, C), literal_1, literal_2.
Layer 3: eastbound(T) :- has_car(T, C), literal_1, literal_2, literal_3.
```

Within each layer, all combinations of body literals are enumerated. The key optimization is **early exit**: once a layer contains a perfect rule (score = 1.0), the search stops without evaluating deeper layers. This ensures we find the simplest correct rule efficiently.

### Step 5: Evaluate Candidates with the Judge

Each candidate rule is evaluated against the validation program using the HuggingFace judge (`_get_judge()`). The judge returns:
- `partial_score`: 0.0 to 1.0 (1.0 means perfect separation)
- `syntax_valid`: whether the rule is syntactically valid Prolog
- `error`: any error message

### Step 6: Select Best Rule

Candidates are sorted by:
1. **Score** (descending) — highest scoring rules first
2. **Complexity** (ascending) — fewer body literals first

The best rule is the one with the highest score, breaking ties by lowest complexity. If no rule achieves a perfect score (1.0), the top-K highest-scoring rules are returned as `top_k_programs`.

## Key Design Decisions

1. **Hypothesis space restriction**: Only conjunctions of car-level or train-level predicates are considered. This matches the domain's structure where direction is determined by car properties within a train.

2. **Complexity preference**: Simpler rules are preferred. The `rule_complexity()` function from `rewards/slr_bench.py` measures complexity as the number of body literals excluding `has_car/2`.

3. **Early exit optimization**: The search stops as soon as a layer containing a perfect rule is found. This guarantees finding the simplest correct rule without unnecessary evaluation of deeper layers.

4. **No external dependencies**: The solver uses only the Python standard library and the provided verifier. No SWI-Prolog or external APIs are required.

## Complexity

- **Hypothesis space size**: O(k^n) where k is the number of distinct literals and n is the maximum number of body literals.
- **Evaluation cost**: Each candidate requires one judge call, which involves Prolog compilation and evaluation.
- **Practical limit**: With ~20 literals and early exit after finding a 1 or 2-literal perfect rule, typical search requires evaluating only tens of rules.

## Limitations

- The solver only generates conjunctions (no negation, disjunction, or multi-clause rules).
- For tasks requiring complex rules (2-3+ literals) or non-conjunctive conditions, the solver may not find perfect rules.
- The approach is systematic and exhaustive within the defined hypothesis space, but does not explore alternative structural forms.
