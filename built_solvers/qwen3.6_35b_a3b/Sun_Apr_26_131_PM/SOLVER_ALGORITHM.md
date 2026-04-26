# SOLVER_ALGORITHM.md

## Algorithm Overview

The solver implements a multi-stage property-based rule induction pipeline for SLR-Bench
train-direction classification tasks. It infers a Prolog rule of the form:

```prolog
eastbound(T) :- Body.
```

from labelled `(background_facts, direction_label)` pairs, where `direction_label` is
either `"eastbound"` or `"westbound"`.

## Key Design Principles

1. **In-Python filtering first**: All candidate generation and pruning is done in Python
   to avoid the ~27-second cost of the SWI-Prolog verifier per rule evaluation.
2. **Iterative complexity escalation**: Candidates are generated from simplest to most
   complex forms, and the verifier is only invoked on the top-K simplest candidates.
3. **Budget-limited verification**: At most 5 verifier calls are made per task.

## Algorithm Pipeline

### Stage 0: Parse and Extract Facts

For each training example:
1. Parse the space-separated Prolog ground-atom string into a list of
   `(predicate, (arg1, arg2, ...))` tuples.
2. Identify the train ID and the set of car constants.
3. Group all facts by car constant for efficient lookup.

The examples are then split into:
- `pos_facts`, `pos_cars` — background knowledge for eastbound trains
- `neg_facts`, `neg_cars` — background knowledge for westbound trains

### Stage 1: Single-Car Property Rules

Generate rules of the form `eastbound(T) :- has_car(T, C), pred(C, value).`

Three sub-strategies:

#### 1a. Direct separating properties
Find `(pred, value)` pairs where:
- **Every** eastbound train has at least one car with `pred(Car, value)`
- **No** westbound train has any car with `pred(Car, value)`

#### 1b. Integer predicate rules
For integer predicates (`car_num`, `has_wheel`, `load_num`, `passenger_num`):
- Specific integer values: `pred(Car, N)` where N matches some positive value
- Arithmetic: `pred(Car, N), N > 0` where all positives have positive values
  and no negatives do

#### 1c. Negation-as-failure rules (`_X \\= excluded`)
Find `pred(Car, X), X \\= excluded` rules where:
- Every eastbound train has at least one car with `pred != excluded`
- Every westbound train has ONLY `pred == excluded` (no car with different value)

#### 1d. Universal negation (`\\+ (...)`)
Find `\\+ (has_car(T, C), pred(C, val))` rules where:
- **No** eastbound train has any car satisfying `pred(Car, val)`
- **At least one** westbound train has a car satisfying `pred(Car, val)`

**Key fix**: Gather `(pred, value)` candidates from BOTH positive AND negative
examples, because the distinguishing value might not appear in any positive example
(e.g., `has_wheel(Car, 3)` in some demos).

### Stage 2: Multi-Property Conjunctions

If no single-property rule is found, generate conjunctions of the form:
`eastbound(T) :- has_car(T, C), pred1(C, val1), pred2(C, val2), ...`

- Try 2-property conjunctions first, then 3-property, up to a limit of 50,000
  combinations per property level.
- A conjunction separates if:
  - Every eastbound train has at least one car satisfying ALL predicates
  - No westbound train has any car satisfying ALL predicates

### Stage 3: Mixed Integer + Categorical Conjunctions

Generate rules like `pred_cat(C, val), pred_int(C, N), N > 0` where the same
car satisfies both a categorical condition and an integer condition.

## Verification

After collecting all in-Python candidates, they are sorted by complexity
(using `rule_complexity()` which counts body literals excluding `has_car/2`).
The top-K simplest candidates are evaluated with the SWI-Prolog verifier
(`judge.compute`). If no perfect score (1.0) is found within the budget,
the best in-Python candidate is returned as a fallback.

## Predicate Coverage

The solver handles all predicates observed across DEMOS.json:
- **Core (5)**: `has_car`, `car_num`, `car_color`, `car_len`, `has_wall`
- **Extended**: `has_roof`, `has_payload`, `has_wheel`, `load_num`,
  `passenger_num`, `car_type`, `has_window`

## Complexity Analysis

- Single-property: O(num_properties × num_examples) — very fast
- 2-property conjunctions: O(C(num_properties, 2) × num_examples) — manageable
- 3-property conjunctions: O(C(num_properties, 3) × num_examples) — may need limits
- Verification: O(K × 27s) where K ≤ 5

The solver typically completes within 1-2 seconds for most tasks, with hard
tasks (4-5 property complexity) taking up to ~1 second due to conjunction
search.
