# Search Space Analysis and Naive Enumeration

This document quantifies the search spaces for PBEBench and SLR-Bench to motivate
why brute-force enumeration is infeasible and why the symbolic solvers produced by
the SolverBuilder coding agent represent non-trivial algorithmic contributions.

---

## PBEBench

### DSL structure

Each program is a single `replace(A, B)` call where:
- **A** (predicate/pattern): non-empty string of length 1–3 over alphabet V → `V + V² + V³` choices
- **B** (transform/replacement): string of length 0–3 (empty allowed) → `1 + V + V² + V³` choices
- **Per-program space**: `(V + V² + V³) × (1 + V + V² + V³)`

A task solution is an ordered *cascade* of k programs (k = cascade length). The search
space for a cascade of length k is `(per-program space)^k`, and since the cascade length
is unknown in advance, a brute-force solver must enumerate all cascades of length 2
through max_k.

### Alphabet sizes

PBEBench-Lite uses lowercase letters and a small set of punctuation; the global alphabet
across all tasks has **V = 17** characters (per-task average V = 13). PBEBench-Hard
additionally uses uppercase letters and digits; the global alphabet has **V = 52**
characters (per-task average V ≈ 51).

### Search space table

| Setting | V | Per-program choices | Worst cascade length | Worst-case search space |
|---|---|---|---|---|
| Lite (per-task avg, V=13) | 13 | 5,662,020 | 5 | **≈ 5.8 × 10³³** |
| Lite (global vocab, V=17) | 17 | 27,243,180 | 5 | **≈ 1.5 × 10³⁷** |
| Hard (global vocab, V=52) | 52 | 20,553,379,860 | 20 | **≈ 1.8 × 10²⁰⁶** |

Even at 10⁹ program evaluations per second:
- Exhausting the **Lite** search space would take ~10²⁴ years.
- The **Hard** search space is cosmologically larger (~10²⁰⁶ candidates).

### What makes the symbolic solvers non-trivial

The Claude Code and Qwen3.6 solvers reduce this space to a tractable search by:
1. Inferring the target predicate A directly from input/output diffs rather than
   enumerating all possible patterns.
2. Using the examples to prune the replacement B to a small set of candidates.
3. Applying a greedy or beam strategy over the cascade rather than exhaustive enumeration.

These strategies reduce the effective search space by many orders of magnitude —
the solvers typically evaluate only hundreds to thousands of candidates per task.

---

## SLR-Bench

### DSL structure

SLR-Bench tasks require inducing a Prolog rule of the form:

```prolog
eastbound(T) :- has_car(T, C), body_literal_1, body_literal_2, ...
```

The body is a conjunction of ground literals drawn from four predicates:
`car_color/2`, `car_len/2`, `car_num/2`, `has_wall/2` (each over a small value domain,
~5–10 values per predicate). The `has_car(T, C)` structural glue is always present and
not counted in rule complexity.

Given L candidate ground literals for a task, the number of candidate rules with up to
4 body literals (the effective maximum for this dataset) is:

```
C(L,1) + C(L,2) + C(L,3) + C(L,4)
```

### Search space by curriculum level

L grows with curriculum level as tasks require more predicates and value combinations
(measured empirically from the dataset):

| Curriculum level | Ground literals L | Candidates (up to 4-body) | Wall-clock @ 50 ms/eval | Wall-clock @ 200 ms/eval |
|---|---|---|---|---|
| 1  |  5 |         30 | < 1 s     | < 1 s     |
| 5  | 11 |        561 | < 1 s     | < 1 s     |
| 10 | 19 |      5,035 | ~4 min    | ~17 min   |
| 15 | 33 |     46,937 | ~39 min   | ~2.6 hr   |
| 20 | 50 |    251,175 | ~3.5 hr   | ~14 hr    |
| Theoretical max (L=120) | 120 | ~8.5 × 10⁶ | ~5 days | ~20 days |

Each evaluation requires spawning a SWI-Prolog subprocess to verify the candidate rule
against all training examples. The measured per-call cost ranges from ~50 ms (simple
tasks, fast unification) to ~200 ms+ (complex tasks with many facts), making pure brute
force infeasible at curriculum levels 15–20.

### What makes the symbolic solver non-trivial

Unlike PBEBench, the SLR-Bench search space is in principle finite and tractable for
simple rules. The key bottleneck shifts from *space size* to *evaluation cost*: each
candidate requires SWI-Prolog execution. The Qwen3.6 solver addresses this by:
1. Extracting only task-relevant ground literals from the feature space (pruning L from
   the theoretical maximum of ~120 to the observed 5–50).
2. Searching layer by layer (1-body, then 2-body, etc.) and stopping at the first layer
   that contains a correct rule — avoiding expensive high-arity combinations entirely
   when a simpler rule exists.
3. Sorting and pruning within each layer by syntactic features before evaluation.

This brings most easy/medium tasks down to under a minute, though hard tasks (levels
15–20) can still require tens of thousands of SWI-Prolog calls.
