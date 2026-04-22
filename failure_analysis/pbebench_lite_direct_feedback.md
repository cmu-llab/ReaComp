# Failure Analysis: PBEBench-Lite — DirectFeedback

**File:** `outputs/lite_tasks_full_og_direct_feedback.jsonl`  
**Framework:** `direct_feedback` (correctness only)  
**Tasks:** 77 tasks that exhausted the k=32 attempt budget without hitting reward=1.0  
**Model:** gpt-oss-120b  

The model sees full verifier feedback (reward score + step-by-step cascade trace + mismatch details) on every retry.

---

## Overall reward/attempt breakdown (across 2,464 total iterations)

| Category | Count | % |
|---|---|---|
| Score partial (0 < s < 1) | 1,923 | 78% |
| Constraint violations | 427 | 17% |
| Score = 0.0 (completely wrong) | 86 | 4% |
| Parse failures | 28 | 1% |

All 77 tasks are solvable — GT programs verified constraint-compliant (0 violations, 0 execution errors across all 1008 tasks). There is no hardness ceiling issue; the tasks are synthetically generated to be solvable within the formalism.

---

## Failure type breakdown (at best-reward attempt)

### br=0.8 (57 tasks) — one test case away from correct

| Failure type | Count |
|---|---|
| Under-expansion (output too short) | 41 |
| Over-expansion (output too long) | 7 |
| No rules fired (wrong alphabet) | 7 |
| Wrong substitution (same length, wrong chars) | 2 |

**Under-expansion gap distribution** (how many chars short of expected):

| Gap | Tasks |
|---|---|
| -1 char | 12 |
| -2 chars | 13 |
| -3 chars | 12 |
| -4 chars | 6 |
| -5 to -9 chars | 4 |

### br=0.6 (14 tasks) — two test cases failing
### br=0.4 (4 tasks): tasks 25, 522, 737, 915
### br=0.2 (1 task): task 481
### br=0.0 (1 task): task 739 (stuck_at_zero)

---

## Primary failure mode: strategy lock-in

**76/77 tasks** have the model returning the exact same program sequence 3+ times. This is the dominant failure mode.

| Threshold | Tasks |
|---|---|
| Same sequence ≥ 3 times | 76 / 77 |
| Same sequence ≥ 5 times | 68 / 77 |
| Same sequence ≥ 10 times | 34 / 77 |
| Same sequence ≥ 15 times | 15 / 77 |

**Dominance of modal sequence** (fraction of all parseable iterations that are the same sequence):

| Dominance | Tasks |
|---|---|
| >75% (extreme lock-in) | 9 |
| 50–75% | 10 |
| 25–50% (cycling between 2–3 strategies) | 34 |
| <25% (genuinely exploring) | 24 |

Most extreme lock-in cases:
- task 282: 26/30 iters identical (87%)
- task 151: 24/28 iters identical (86%)
- task 975: 18/21 iters identical (86%)
- task 648: 21/25 iters identical (84%)
- task 710: 22/28 iters identical (79%)

~55 tasks are primarily lost to lock-in. The remaining ~22 tasks are genuinely exploring different strategies and still can't find a working solution — those are harder search-space failures.

---

## Why feedback doesn't break the lock-in

The verifier feedback shows exactly *what* went wrong (step-by-step trace, mismatched output). But fixing the failing input requires **globally restructuring the predicate set** — changing atoms that are currently correct for 4 passing inputs. The model can't see that; it tries to patch the 5th input by tweaking the same wrong predicate set, which breaks the 4 that were passing. The feedback tells the model *where* the cascade fails but not *which atoms to target instead*.

Mean predicate overlap between model's best solution and GT: **61.5%**

| Overlap | Tasks |
|---|---|
| <20% (completely wrong alphabet) | 3 |
| 20–40% | 7 |
| 40–60% | 17 |
| 60–80% | 31 |
| 80–99% | 9 |
| 100% exact pred match | 0 |

The model is typically targeting 2–3 of the 5 GT predicates correctly and inventing 2–3 wrong ones. Crucially, **no task has the right predicate set with wrong ordering/transforms** — so it's not an ordering problem, it's a predicate discovery problem.

Most commonly missed GT predicates: `c` (11 tasks), `a` (10), `k` (8), `x` (7), `u` (7).  
Most commonly invented non-GT predicates: `x` (6 tasks), `z` (5), `q` (4), `u`/`h`/`j`/`w`/`y`/`g` (4 each).

---

## Constraint violations are a symptom, not the cause

427 constraint violations (17% of iters), split:
- `B too long` (len > 3): 256 — model wants e.g. `replace('k', 'fyaj')` (4 chars)
- `Too many programs` (> 5): 170
- `A too long`: 1

The model tries to express transforms that exceed 3 chars because its **wrong predicate hypothesis** forces it to squeeze too much work into a single step. GT never needs >3-char transforms because the correct decomposition distributes the work across intermediate atoms via multi-hop chains.

---

## Illustrative examples of structural gap

**Task 25 (br=0.4):**
- GT: `x→f`, `k→bjb`, `a→xki`, `b→fa`, `a→ya` — routes through `b` and `a` as intermediates
- Model stuck on: `x→f`, `k→F`, `Fj→FF`, `F→fy`, `a→xki` — uses uppercase pivot `F` but misses the `aj` suffix chain
- Failing input `aakje`: model gets `xkixkifyfye`, GT gets `xkixkifyajfyaje`

**Task 151 (br=0.8):**
- GT: `z→ycb`, `w→hz`, `y→jua`, `i→w`, `a→dk` — routes `z` through `y` through `a`
- Model stuck on: `w→hz`, `a→dk`, `y→ju`, `u→udk`, `i→w` — never assigns a rule to `z`
- Failing input `bxbzii`: `z` has no rule, model gets `bxbzww`, GT gets `bxbjudkcbww`

**Task 282 (br=0.8, 87% lock-in):**
- GT: `yk→c`, `b→j`, `c→hji`, `ja→c`, `y→iwc` — fuses bigram `yk` first
- Model stuck on: `c→hj`, `d→id`, `y→iwc`, `ja→c`, `ub→iuj` — treats `y` and `k` independently
- Failing input `uyyka`: GT collapses `yk→c` giving `uyca`, model never fires on `yyk`

**Task 710 (br=0.8, 79% lock-in):**
- GT: `id→vw`, `z→kjb`, `v→gv`, `dh→cvj`, `u→fae` — handles `id` as bigram, handles `u`
- Model stuck on: `v→gv`, `z→kjb`, `dh→cv`, `fu→jff`, `w→aew` — misses `id` bigram and `u`
- Failing input `wuyidx`: model fires `w→aew` first giving `aewuyidx`, GT gets `wfaeygvwx`

---

## What would help

1. **Patience/early-stop for lock-in** (already implemented): `--simplify-patience` terminates Phase 2 when the same sequence repeats. The equivalent for Phase 1 would be even more impactful — if the same sequence has appeared 3 times in Phase 1, force a restart with an explicit instruction to try a different predicate set.

2. **Feedback enrichment for no-rules-fired**: when the failing input contains characters with no matching predicate in the current cascade, this is detectable. The feedback could explicitly flag "character `z` in the failing input has no rule — consider adding one."

3. **Nudge away from locked strategy**: after N repeats of the same sequence, inject a message like "your current approach targets `{preds}`. Try a completely different set of atoms — your current strategy is stuck."

4. **Bigram predicate hints**: the model disproportionately misses bigram predicates (`yk`, `vf`, `cu`, `ei`) and defaults to single-char rules. The feedback could hint "consider whether a 2-character predicate could collapse this step."
