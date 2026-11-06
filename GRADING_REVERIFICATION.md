# Re-verification of symbolic solver grading

## Issue

`scripts/eval_solver.py` recorded each solver's **self-reported** result (`success`, `score`) as
`solved` / `best_reward`. It passed the dataset limits to the solver as arguments, but never
re-checked the returned program with the official verifier. `quick_eval.py` and
`ensemble_outputs.py` read those stored fields, so unions and hybrids inherited them.

Most solvers call the verifier internally with the right limits, so their numbers are unaffected.
One PBEBench solver is not:
`built_solvers/qwen3.6_35b_a3b/Sun_Apr_26_440_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/SOLVER.py`
scores candidates with its own checker and deliberately searches past the budget (docstring:
"The solver tries up to max_programs + 14"; loop `range(max_programs, min(max_programs + 15, 25))`).
Its own checker also does not enforce |A|, |B| <= 3. Many of its reported successes are programs
with more than 5 rules on Lite (20 on Hard) or replacement strings longer than 3 characters.
Unions picked these short invalid programs over valid ones.

On SLR-Bench the CC solver's internal check disagrees with the official judge in both directions
(22 false successes, 10 false failures). Hybrids built from the stored fields never sent the 22
false successes to the LLM fallback.

## Fix

`scripts/eval_solver.py` (PBE datasets):
- passes `max_programs` / `max_pred_len` / `max_transform_len` only if the solver accepts them
  (previously `max_programs` was always passed, so solvers without that argument raised and
  scored 0);
- re-verifies every returned program with `rewards.pbebench.reward` under the dataset budget and
  stores that as `solved` / `best_reward`; the solver's own claim is kept as
  `self_reported_success` / `self_reported_score`;
- scores an empty program directly (fraction of unchanged examples), because the verifier cannot
  parse `[]` and Real-FR has 126 identity tasks.

SLR is not re-verified inside `eval_solver.py` (the judge is slow to load per worker process); use
the post-hoc scripts below.

## Reproduce

```bash
python scripts/reverify_pbe_unions.py      # PBEBench Lite/Hard: solvers, unions, hybrids  -> metrics/reverified_pbe.json
python scripts/reverify_realfr.py          # Real-FR, per-run budget k                     -> metrics/reverified_realfr.json
python scripts/slr_hybrids_verified.py     # SLR-Bench solvers + hybrids rebuilt on verified results -> metrics/slr_hybrids_verified.json
python scripts/slr_verify_robust.py <answers.jsonl ...>   # crash-tolerant SLR re-verification of any file
```

`slr_verify_robust.py` runs the SLR judge in a restartable worker process: some stored answers make
the judge abort the whole Python process (`std::length_error`), which is why a plain loop crashed.
With one persistent judge, BoK and DF re-verify exactly, so re-verification itself is deterministic.

Corrected unions use the same selection rule as `ensemble_outputs.py` (highest verified reward,
ties by lowest complexity, then source order). Corrected hybrids use the solver's answer only if the
verifier accepts it, otherwise the LLM answer (BoK / DF were run on all tasks).

## Results

PBEBench (`metrics/reverified_pbe.json`):

| Row | Stored (self-reported) | Re-verified |
|---|---|---|
| QO `Sun_Apr_26_440_PM` (run 3), Lite | 79.2 | 34.2 |
| QO run 3, Hard | 51.8 | 37.6 |
| Lite All Symbolic: stored file `lite_ensemble_all_solvers` | 91.3 | 82.0 |
| Lite All Symbolic: union rebuilt on verified rewards | | 85.4 |
| Hard All Symbolic: stored file `hard_union_all_solvers` | 84.7 | 80.8 |
| Hard All Symbolic: union rebuilt on verified rewards | | 83.9 |
| Hard BoK + All Symbolic: stored file `hard_union_bok_all_solvers` | 85.8 | 82.9 |
| Hard BoK + All Symbolic: rebuilt | | 85.0 |

Unchanged on re-verification: CC solver (80.4 / 69.7), the other QO runs (QO `Fri_Apr_24_200_AM`
differs by one task on each split because its answers are stored as `[A, B]` pairs), CC + QO, BoK,
DF, all Lite hybrids (BoK + All Symbolic 93.95, DF + All Symbolic 93.15).

SLR-Bench (`metrics/slr_hybrids_verified.json`):

| Row | Stored | Rebuilt on verified results |
|---|---|---|
| CC solver | 68.4 | 67.2 |
| QO solver | 60.7 | 60.7 |
| BoK + CC | 80.3 | 80.7 |
| DF + CC | 86.6 | 86.8 |
| DF + CC + QO | 86.7 | 86.9 |
| Seeded DF + CC | 93.3 | >= 91.1 (lower bound: the 22 false-success tasks have no seeded run yet; ids in the json) |

Real-FR (`metrics/reverified_realfr.json`): every run matches its stored value within 0.2 points,
except the 48-example QO run, which re-verifies higher (61.4 -> 65.5) because it returned the empty
program on identity tasks but reported them as failed.

Note: a solver's induction-time scores are not involved here; all numbers are on the benchmark
test tasks.
