# SOLVER_ALGORITHM.md - PBE Solver Algorithm Description

## Overview

This solver implements a **beam search** approach to find an ordered sequence of
`replace(A, B)` programs that transform input strings into output strings,
satisfying the DSL constraints:

- Each program: `replace(A, B)` where `1 <= len(A) <= 3` and `0 <= len(B) <= 3`
- Maximum of `max_programs` (default 5) programs in the sequence
- Programs are applied **in order** as Python `str.replace()` calls

## Algorithm

### Step 1: Identify Differing Examples

Parse the input-output pairs and identify which pairs differ. Pairs that are
identical require no transformation.

If there are no differing examples, return an empty program list (success=True).

### Step 2: Generate Candidate Programs

For each differing (input, output) pair, generate candidate programs:

**Predicate candidates (A):** All substrings of length 1-3 from all input strings.

**Transform candidates (B):** All substrings of length 0-3 from all output strings,
plus empty string.

For each (A, B) pair, compute a score:

1. **Full explanation:** How many diffs does `replace(A, B)` fully transform?
   (score = fully * 1000)
2. **Partial progress:** How many diffs get closer to their target?
   (score = partial * 10)

A candidate is discarded if it breaks any unchanged example or explains zero diffs.

Candidates are sorted by score and the top `max_candidates` are kept.

### Step 3: Beam Search

The beam search iteratively builds program sequences:

**State representation:** Each beam state has:
- `program_list`: The sequence of programs added so far
- `current_states`: Maps diff index to the current input state after all programs

**Iteration at each step:**
1. For each beam state and candidate program:
   - Skip if already in the program list (no duplicates)
   - Count how many remaining diffs are fully fixed or partially improved
   - Check if it breaks any unchanged example
   - Apply the program to all current states
2. Score all extended states using the verifier
3. Keep top `beam_size` states (deduplicated by program sequence)
4. Track the overall best-scoring program

**Termination:** Stop when:
- Score reaches 1.0 (all examples correct), or
- `max_programs` programs have been added, or
- No new states can be generated

### Step 4: Return Results

Return the best program sequence as a dict with:
- `"success"`: True if score >= 1.0
- `"program"`: List of strings in format `replace('A', 'B')`
- `"num_programs"`: Number of programs
- `"score"`: Fraction of examples correctly transformed

## Key Design Decisions

### Partial Progress Candidates
Some tasks need programs that don't fully explain any single diff but make partial
progress. For example, to transform `AuJw -> Apemh`:
- `replace('Au', 'Ap')` produces `ApJw` (partial progress)
- `replace('Jw', 'emh')` produces `Apemh` (full)

Both programs must be included even though neither alone explains the diff.

### State-Aware Scoring
The beam search evaluates candidates against the **current state** after all
previous programs, not just the original input. This correctly handles programs
whose targets only appear after earlier transformations.

### Verifier-Guided Beam Selection
Heuristic scoring guides the search, but final beam selection uses the actual
verifier (`rewards.pbebench.reward`) score for evaluation.

### Candidate Pruning
To keep search tractable:
- Top `max_candidates` candidates by heuristic score
- Beam size capped at `beam_size` (typically 50)
- Duplicate program sequences eliminated from beam

## Complexity

- **Candidate generation:** O(D * L * 9 * 64) where D=diffs, L=avg string length
- **Beam search per step:** O(B * C * D) where B=beam size, C=candidate count
- **Total:** O(max_programs * B * C * D)

With B=50, C=100, D=20, max_programs=5: each step is milliseconds.

## Edge Cases

1. **No diffs:** Returns empty program, success=True
2. **Complex transformations:** Returns best-effort if full success unattainable
3. **Overlapping replacements:** Order preserved; later programs see earlier results
4. **Empty transform:** `replace('A', '')` is valid (removes pattern A)
