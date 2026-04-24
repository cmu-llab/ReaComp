# Symbolic Solver Evaluation Findings

**Solver:** `built_libraries/claude_code/Thu_Apr_23_807_PM/SOLVER.py`  
**Algorithm:** Two-phase beam search with dynamic candidate generation (see `SOLVER_ALGORITHM.md`)  
**Date:** 2026-04-23

---

## Summary

| Dataset | N | Pass% | Mean Score | Avg Time/task |
|---|---|---|---|---|
| PBEBench-Lite | 1008 | **80.4%** | 0.944 | 0.004s |
| PBEBench-Hard | 1216 | **69.7%** | 0.987 | 0.62s |

Result files: `evals/solver_results/lite.jsonl`, `evals/solver_results/hard.jsonl`

---

## PBEBench-Lite (max 5 programs)

### By cascade length

| Cascade | N | Pass% | Mean Score |
|---|---|---|---|
| 2 | 246 | 98.4% | 0.990 |
| 3 | 253 | 90.9% | 0.963 |
| 4 | 253 | 83.0% | 0.958 |
| 5 | 256 | 50.0% | 0.866 |

**Key observation:** Performance degrades sharply at cascade=5 (50.0%), which is exactly at the program-count limit. The solver must compress the full 5-step transformation into ≤5 programs, leaving no slack for ordering effects — any feed/bleed interaction that requires a specific intermediate step becomes very hard to satisfy.

### By BFCC category

The lite dataset is overwhelmingly a single category (`bleeding, feeding, counterfeeding, counterbleeding`, 99.4% of tasks) so no meaningful split is possible there.

---

## PBEBench-Hard (max 20 programs)

### By cascade length

| Cascade | N | Pass% | Mean Score |
|---|---|---|---|
| 2  | 64 | 93.8% | 0.999 |
| 3  | 64 | 96.9% | 0.999 |
| 4  | 64 | 92.2% | 0.998 |
| 5  | 64 | 93.8% | 0.999 |
| 6  | 64 | 84.4% | 0.997 |
| 7  | 64 | 84.4% | 0.996 |
| 8  | 64 | 92.2% | 0.998 |
| 9  | 64 | 79.7% | 0.994 |
| 10 | 64 | 81.2% | 0.991 |
| 11 | 64 | 82.8% | 0.996 |
| 12 | 64 | 71.9% | 0.993 |
| 13 | 64 | 73.4% | 0.991 |
| 14 | 64 | 59.4% | 0.983 |
| 15 | 64 | 65.6% | 0.988 |
| 16 | 64 | 67.2% | 0.986 |
| 17 | 64 | 54.7% | 0.984 |
| 18 | 64 | 35.9% | 0.972 |
| 19 | 64 | 12.5% | 0.958 |
| 20 | 64 |  1.6% | 0.935 |

**Key observation:** Pass rate degrades gradually from cascade 5–17, then collapses at 18–20. At cascade=20 (the hard limit), only 1 out of 64 tasks is solved perfectly — the beam search cannot find a 20-step ordering that perfectly satisfies all examples within the 55s time budget. Mean score stays high (0.935+) even at cascade=20, meaning the solver gets most pairs right and fails on only 1–2.

### By BFCC category

| BFCC relationships | N | Pass% |
|---|---|---|
| No BFCC relationships | 55 | **92.7%** |
| Bleeding | 47 | **91.5%** |
| Bleeding, Counterfeeding | 56 | **91.1%** |
| Counterfeeding | 64 | **90.6%** |
| Counterbleeding | 45 | 84.4% |
| Counterfeeding, Counterbleeding | 60 | 81.7% |
| Bleeding, Counterbleeding | 65 | 81.5% |
| Feeding | 64 | 75.0% |
| Feeding, Bleeding, Counterfeeding | 72 | 73.6% |
| Feeding, Counterfeeding, Counterbleeding | 74 | 73.0% |
| Bleeding, Counterfeeding, Counterbleeding | 77 | 71.4% |
| Feeding, Counterfeeding | 76 | 65.8% |
| Feeding, Bleeding, Counterbleeding | 78 | 66.7% |
| Feeding, Counterbleeding | 58 | 67.2% |
| Feeding, Bleeding | 59 | 78.0% |
| Feeding, Bleeding, Counterfeeding, Counterbleeding | 266 | **40.2%** |

**Key observations:**
- Tasks with **no BFCC interactions** are easiest (92.7%) — no ordering dependencies between rules, so any permutation works.
- **Counterfeeding-only** and **Bleeding-only** tasks are nearly as easy — these involve one-directional rule interactions that the dynamic candidate re-extraction handles well.
- **All-four-BFCC** (`Feeding, Bleeding, Counterfeeding, Counterbleeding`) is by far the hardest category at 40.2%, making up 22% of the hard dataset. These tasks have dense mutual interactions between rules where every ordering decision affects every other rule.
- **Feeding** interactions are harder than **Bleeding** ones across the board: feeding requires a rule to create input for a later rule (a positive dependency), which the beam search must discover in the correct order; bleeding (a later rule destroys what an earlier rule would create) is easier to handle by prioritising specific patterns early.

---

## Failure analysis

The overwhelming failure mode across both datasets is **near-misses**: the solver finds a program that correctly transforms all but 1–3 input/output pairs. This is reflected in the high mean scores (0.944 lite, 0.987 hard) despite imperfect pass rates.

The solver's dynamic candidate re-extraction handles shallow ordering dependencies (feed/bleed chains up to ~4 steps) well, but fails when:
1. A correct program must be discovered through a long intermediate chain (cascade ≥ 18).
2. All-four-BFCC tasks require holding multiple interacting constraints simultaneously within the 5-step/20-step beam horizon.
3. A single-character pattern that fixes a changed pair also appears in many unchanged pairs, making any candidate unsafe — the safe-phase beam search discards it and the unrestricted phase doesn't recover in time.
