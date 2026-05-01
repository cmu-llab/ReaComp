# SOLVER_SLR Algorithm

## Overview

`SOLVER_SLR.py` implements a symbolic rule-induction solver for SLR-Bench tasks. Given a set of `(background_facts, direction_label)` pairs, it searches for a Prolog rule of the form `eastbound(T) :- Body.` that correctly separates eastbound from westbound trains.

The algorithm is an ILP-style (Inductive Logic Programming) coverage-based search, inspired by the general structure of FOIL/Aleph, adapted to this fixed domain-specific language.

---

## Domain-Specific Language

Rules are single Prolog clauses with head `eastbound(T)`. The body is a conjunction of literals drawn from:

| Predicate | Arity | Notes |
|-----------|-------|-------|
| `has_car(T, C)` | 2 | Structural glue; not counted toward complexity |
| `car_num(C, N)` | 2 | Car position (integer) |
| `car_color(C, Color)` | 2 | `red`, `blue`, `green`, `yellow`, `white` |
| `car_len(C, Length)` | 2 | `short`, `long` |
| `has_wall(C, WallType)` | 2 | `full`, `railing` |
| `has_roof(C, RoofType)` | 2 | various types |
| `has_wheel(C, N)` | 2 | integer |
| `has_payload(C, Payload)` | 2 | various types |
| `has_window(C, WinType)` | 2 | `full`, `half`, `none` |
| `car_type(C, Type)` | 2 | various types |
| `load_num(C, N)` | 2 | integer |
| `passenger_num(C, N)` | 2 | integer |

Negation `\+ pred(C, val)` is also supported (Prolog closed-world negation-as-failure).

Rule complexity = number of top-level body literals excluding `has_car/2` (see `rule_complexity()` in `rewards/slr_bench.py`).

---

## Algorithm

### Step 1 — Parse

Each `facts_string` is parsed into an ordered list of car-property dicts. The `has_car/2` atoms determine car membership and ordering; all other binary atoms populate the car's property dictionary.

### Step 2 — Build Coverage Structures

For efficient candidate filtering, coverage sets are precomputed entirely in Python (no Prolog calls):

**Single-property coverage**
For each `(pred, val)` pair extracted from the data:
- `pos_e[(pred, val)]` — east train indices with ≥1 car having `pred=val`
- `pos_w[(pred, val)]` — west train indices similarly
- `neg_e[(pred, val)]` — east train indices with ≥1 car having `pred≠val` (for `\+` literals)
- `neg_w[(pred, val)]` — west train indices similarly

**Same-car k-tuple coverage** (k = 2, 3, 4, 5)
For each canonical sorted k-tuple of property-value pairs that co-occur on the _same_ car:
- `east_cov[tuple]`, `west_cov[tuple]` — train index sets

### Step 3 — Generate Candidate Rules

Rules are generated in order of increasing complexity, using coverage structures to avoid redundant generation:

| Complexity | Template |
|-----------|----------|
| 1 | `has_car(T,C), pred(C,val)` — single positive literal |
| 1 | `has_car(T,C), \+ pred(C,val)` — single negation literal |
| 2 | `has_car(T,C), p1(C,v1), p2(C,v2)` — same-car pair (positive) |
| 2 | `has_car(T,C1), p1(C1,v1), has_car(T,C2), p2(C2,v2)` — two-car existential conjunction |
| 2 | `has_car(T,C), p1(C,v1), \+ p2(C,v2)` — one positive + one negation on same car |
| 3 | `has_car(T,C), p1(C,v1), p2(C,v2), p3(C,v3)` — same-car triple |
| 3 | `has_car(T,C1), p1(C1,v1), p2(C1,v2), has_car(T,C2), p3(C2,v3)` — pair + single, two cars |
| 4 | `has_car(T,C), p1…p4` — same-car quad |
| 4 | `has_car(T,C1), p1(C1,v1), p2(C1,v2), has_car(T,C2), p3(C2,v3), p4(C2,v4)` — pair + pair |
| 5 | `has_car(T,C), p1…p5` — same-car quint |

Each template's east/west coverage is computed via set operations on the precomputed structures (no Prolog invocation).

### Step 4 — Score and Filter

A **Python consistency score** is computed for each candidate:

```
score = (|east_cov ∩ all_east| + |all_west \ west_cov|) / n_total
```

- `score = 1.0` → perfectly separates east from west (Python-consistent)
- `score < 1.0` → partial credit

Candidates are deduplicated and sorted by `(-score, complexity)` — preferring correct and simple rules.

### Step 5 — Verifier Calls (SWI-Prolog)

The solver calls the SWI-Prolog verifier (`judge.compute` from `rewards/slr_bench.py`) only on candidates that Python-simulation deems consistent (`score == 1.0`), up to a maximum of 100 calls.

The validation program is reconstructed from the training examples (direction labels + background facts).

If the verifier confirms score `= 1.0`, the search halts immediately. Otherwise the best partial-score rule is retained.

If the verifier is unavailable, the best Python-consistent rule is returned.

### Step 6 — Return

Returns a dict with:
- `success` — `True` iff a rule with verifier score `= 1.0` was found
- `program` — the best rule string
- `score` — verifier score (or Python score if verifier unavailable)
- `top_k` — list of up to `k` candidate rules (best first)

---

## Complexity and Efficiency

| Phase | Cost |
|-------|------|
| Parse N examples | O(N · F) where F = facts per train |
| Single coverage | O(N · C · P) where C = cars/train, P = props/car |
| Same-car k-coverage | O(N · C · C(P,k)) |
| Candidate generation | O(|vocab|^k) before pruning |
| Python filtering | O(1) per candidate (set operations) |
| Verifier calls | ≤ 100 × ~300 ms = 30 s max |

For a hard task with 30 trains, 6 cars, 11 properties: same-car coverage up to k=5 processes ~70K combinations — all in pure Python within seconds, before any Prolog subprocess is invoked.

---

## Design Choices

- **Same-car vs two-car rules**: Same-car rules require properties to co-exist on one car (tighter constraint); two-car rules only require both properties to exist somewhere in the train. Both template families are searched.
- **Negation**: `\+ pred(C, val)` is supported at level 1 and level 2, covering cases where the discriminating feature is the _absence_ of a property.
- **No numerical inequalities**: Rules use only specific values (not `N > 2`), since all successful demo examples use equality.
- **Verifier as oracle**: Python simulation handles semantics of the templated rules; the Prolog verifier is used only for final confirmation and to catch edge cases the Python model may miss.
- **Top-K fallback**: When no perfect rule is found (e.g., the true rule uses a template not yet generated), the top-K partial-score rules are returned to support downstream refinement.
