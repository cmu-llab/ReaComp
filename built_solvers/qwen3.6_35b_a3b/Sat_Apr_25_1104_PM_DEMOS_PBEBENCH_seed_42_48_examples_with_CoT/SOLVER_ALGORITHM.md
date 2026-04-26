# SOLVER_ALGORITHM.md — PBE Solver Algorithm Description

## Overview

The solver infers an ordered sequence of `replace(A, B)` programs from a set of
(input, output) example pairs.  Each program is a Python `str.replace(A, B)` call
where `1 ≤ len(A) ≤ 3`, `0 ≤ len(B) ≤ 3`, and the total number of programs is at
most `max_programs` (20 for hard tasks, 5 for PBEBench-Lite).

Programs are applied sequentially to each input string using `str.replace()`.  The
order matters because earlier replacements can create or destroy material that
later replacements match (feeding, bleeding, counterfeeding, counterbleeding
interactions).

## High-Level Strategy

1. **Iterative greedy construction** — at each step, find the single `replace(A, B)`
   program that maximises the fraction of correct (input, output) pairs.
2. **Two types of candidates** at each step:
   - *Exact candidates*: programs that transform the current intermediate string
     directly to the target output.
   - *Progress candidates*: programs that reduce the edit-distance towards the
     target, even if they don't reach it in one step (handles multi-step
     transformations).
3. **Best-ordering search** — after the sequence is built, reorder it to maximise
   correctness using permutations (≤ 8 programs) or random shuffles + local swap
   optimisation.
4. **Multi-start** — repeat the entire process with different random seeds and
   return the best solution found.

## Detailed Algorithm

### Candidate Generation

For a given (intermediate_string, output) pair:

**Exact candidates** (`find_exact_candidates`):
Enumerate every substring `A` of the intermediate string (length 1 to 3) at every
position, and the corresponding substring `B` of the output.  If
`intermediate.replace(A, B) == output`, add `(A, B)` to candidates.

**Progress candidates** (`find_progress_candidates`):
Same enumeration, but instead of requiring exact match, check whether the
replacement *improves* the situation — i.e. the edit distance between the result
and the output is strictly less than the edit distance between the intermediate
and the output.  This captures programs that make partial progress toward the
target, enabling later programs to complete the transformation.

### Greedy Sequence Construction

```
current_programs = []
for step in 1..max_programs:
    failing = pairs where apply(current_programs, inp) != out

    if failing is empty: break

    all_cands = set()
    for (inp, out) in failing:
        intermediate = apply(current_programs, inp)
        all_cands ← all_cands ∪ exact_candidates(intermediate, out)
        all_cands ← all_cands ∪ progress_candidates(intermediate, out)

    Score each candidate (a,b):
        test = current_programs + [(a,b)]
        score = fraction of all pairs transformed correctly by test

    Pick the candidate with highest score.
    Append to current_programs.
```

### Best-Ordering Search

After a sequence is built:

**For ≤ 8 programs**: try all `n!` permutations and keep the one with the highest
score.

**For > 8 programs**:
1. Try `max_shuffles` random orderings.
2. Then apply local optimisation: repeatedly try swapping each adjacent pair of
   programs; if a swap improves the score, keep it and repeat.

### Multi-Start

Run the full greedy + reordering pipeline `num_starts` times with seeds
`0, 1, 2, …`.  Keep the best unique solution found.

### Return Value

- `success`: `True` if any program scored 1.0 (correct on all pairs).
- `program`: the top-ranked program sequence as `replace('A', 'B')` strings.
- `programs`: all top-K unique sequences, sorted by score.
- `scores`: corresponding scores.

## Why It Works

### Multi-step transformations
Some input→output transformations require *multiple* replace programs applied
sequentially.  For example, `rKYNHt → bpBt` needs:
1. `replace('rK', 'bp')` → `bpYNHt`
2. `replace('YNH', 'B')` → `bpBt`

A purely exact-match approach would miss step 1 because
`rKYNHt.replace('rK', 'bp') ≠ bpBt`.  The progress-candidate generation finds it
because the edit distance to `bpBt` decreases.

### Order matters
The same set of programs can produce different results depending on order.
Permutation enumeration (for small sequences) and shuffling + local search
(larger sequences) find the ordering that maximises correctness.

### Handling feedback/bleeding
When an earlier program creates material that a later program acts on
(feeding) or destroys material a later program needs (bleeding), the correct
order is critical.  The reordering step discovers the correct relative ordering
by scoring different permutations.

## Complexity

- Candidate generation per step: `O(P × N × L³)` where `P` is the number of
  failing pairs, `N` is the number of candidate positions per string, `L` is the
  max string length, and `L³` covers all (A, B) lengths.
- Greedy construction: `max_programs` iterations.
- Permutation reordering: `O(n!)` for `n ≤ 8` pairs, `O(max_shuffles)` otherwise.
- Overall: practical for typical PBEBench inputs (≤ 50 examples, strings ≤ 20
  chars) within seconds.

## Limitations

- For very hard tasks where even the LLM achieves < 1.0 score, the solver may
  also fall short because the underlying task has no simple replace-cascade
  solution.
- The edit-distance heuristic for progress candidates can occasionally select
  programs that reduce distance locally but lead to dead ends globally; the
  multi-start strategy mitigates this.
- For PBEBench-Lite (max 5 programs), some tasks may require more programs and
  thus remain unsolvable within the constraint.
