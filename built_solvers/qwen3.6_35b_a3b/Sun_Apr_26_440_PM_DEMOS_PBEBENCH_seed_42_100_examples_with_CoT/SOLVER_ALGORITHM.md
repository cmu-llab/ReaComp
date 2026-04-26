# PBE Solver Algorithm

## Overview

This solver implements **Symbolic Program Induction** for Programming-by-Example (PBE) tasks. Given a set of (input_string, output_string) example pairs, it infers a sequence of `replace(A, B)` operations that transforms every input into its paired output.

## DSL Specification

- Each program is an ordered sequence of `replace(A, B)` calls
- `1 <= len(A) <= 3` (predicate length)
- `0 <= len(B) <= 3` (transform length)
- Maximum of 5 programs for PBEBench-Lite (the solver searches up to 20)
- The `replace` function is Python's built-in `str.replace()` — it replaces **all** occurrences

## Algorithm: Candidate Generation + Multi-Strategy Search

### Phase 1: Candidate Generation

For each changed (input, output) pair, the solver generates candidates:

#### 1a. Complete Single-Operation Candidates
For pairs where `inp.replace(pred, trans) == out` with a single operation, we enumerate all valid `(pred, trans)` pairs:
- Try every substring of `inp` (length 1-3) as potential predicate
- For each, derive the required transform from the output
- Verify the operation produces exactly the output
- These candidates are "grounded" in the data — they directly explain the transformation

#### 1b. Unique vs. Optional Classification
- **Unique candidates**: Pairs with exactly one complete candidate → the operation **must** be in the solution
- **Optional candidates**: Pairs with multiple candidates → need to try different options

#### 1c. Multi-Operation Candidates for Hard Pairs
For pairs where no single operation suffices (the difference cannot be explained by one replace), we find **2-operation sequences**:
- Enumerate all possible first operations `(p1 -> t1)`
- Compute the intermediate string after the first replacement
- Find a second operation `(p2 -> t2)` that transforms the intermediate to the output
- Candidate transforms are substrings of the output (bounded by max length 3)
- This handles cases like `'UNWjaL' -> 'YeWcYjaL'` requiring `replace('NW', 'WcY')` then `replace('UW', 'YeW')`

### Phase 2: Search Strategies

The solver uses four complementary search strategies, trying increasing program sizes:

#### Strategy 1: Unique Operations Permutations
Try all permutations of the required unique operations, with program lengths 1 through max_programs. Since unique operations are fixed (there's only one choice per pair), this is fast and often finds the correct ordering.

#### Strategy 2: Greedy Addition
Starting from the best program found so far, iteratively add the operation from the candidate pool that most improves the score. Each addition is evaluated by checking all (input, output) pairs.

#### Strategy 3: Optional Combinations
For pairs with multiple candidates, try all combinations of choices (Cartesian product). This is only feasible when the total combinations are manageable (≤ 3000). For each combination, try different orderings.

#### Strategy 4: Hard Pair Integration
For hard pairs, try combining the best program with 2-operation sequences. For each sequence, test:
1. The sorted program (by predicate length, longest first)
2. All permutations (for programs ≤ 7 operations)

### Phase 3: Local Optimization

After the main search, try all permutations of the best program to find a better ordering.

## Complexity Analysis

| Component | Complexity | Notes |
|-----------|------------|-------|
| Single-operation candidates per pair | O(n × m × L) | n = input length, m = output length, L = 3 (max pred/trans length) |
| 2-op sequences per hard pair | O(n² × m⁴) | n = input length, m = output length, bounded by max lengths |
| Greedy search | O(k × |C| × |examples|) | k = max_programs, C = candidate pool |
| Optional combinations | O(∏cᵢ × |C|!) | Only when ∏cᵢ ≤ 3000 |

## Key Design Decisions

1. **Multi-operation handling**: The solver doesn't limit itself to single operations per pair. It explicitly searches for 2-operation sequences when needed, which is critical for pairs where the transformation involves multiple distinct changes.

2. **Flexible constraint handling**: The solver tries program sizes from `max_programs` up to `max_programs + 14`, accommodating tasks that may need more operations than the PBEBench-Lite limit of 5.

3. **Ordering-aware search**: Since `replace()` operations can interact (one operation's output may be modified by another), the solver tries multiple orderings. Operations are sorted by predicate length (longest first) as a heuristic for specificity.

4. **Pruning**: The optional combination search is limited to ≤ 3000 combinations. The 2-op sequence search is limited to top 20 sequences per hard pair (sorted by specificity). Time limits prevent runaway search.

## Example Walkthrough

Consider demo 15 with changed pairs like:
- `'BoYF' -> 'BoYWcY'` (single op: `replace('F', 'WcY')`)
- `'UNWjaL' -> 'YeWcYjaL'` (no single op)

**Step 1**: `find_candidates('BoYF', 'BoYWcY')` returns `[('F', 'WcY')]` — unique
**Step 2**: `find_candidates('UNWjaL', 'YeWcYjaL')` returns `[]` — hard pair
**Step 3**: `find_all_2op_sequences('UNWjaL', 'YeWcYjaL')` finds sequences including:
  - `('NW', 'WcY'), ('UW', 'YeW')` (the LLM solution)
  - `('U', 'Ye'), ('NW', 'WcY')`
  - `('U', 'YeW'), ('WNW', 'WcY')`
**Step 4**: Greedy search starts with `('F', 'WcY')` (score 0.979)
**Step 5**: Combines with hard pair ops — `('F', 'WcY'), ('NW', 'WcY'), ('UW', 'YeW')` with ordering `['NW', 'UW', 'F']` scores 1.0
**Result**: 3-operation program that correctly transforms all examples.
