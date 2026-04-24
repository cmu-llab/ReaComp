# Solver Algorithm: Symbolic Program Synthesis for PBE

## Overview

This solver infers an ordered sequence of `str.replace(A, B)` programs that
transform every input string in a PBE (Programming-by-Example) task to its
corresponding output string.  The DSL constraints are:

- Each program has the form `replace(A, B)`.
- `1 ≤ len(A) ≤ 3` (predicate length).
- `0 ≤ len(B) ≤ 3` (transform length).
- Maximum of 20 programs in the sequence.

The solver uses a **greedy iterative search with multi-pass residual fixing**
to find programs that are consistent with all given examples.

---

## Algorithm Steps

### Step 1 – Candidate Extraction

For each example pair `(input, output)` that differs, we identify the **edit
region** — the substring of `input` that must be changed to obtain `output`.

```
1. Find longest common prefix of input and output.
2. Find longest common suffix (after the prefix).
3. The region between them is the edit region.
```

From the edit region we derive candidate `(A, B)` replace programs using three
strategies:

#### a) Standard substitution candidates
All substrings `A` (length 1–3) from the input region paired with all
substrings `B` (length 0–3) from the output region.

#### b) Split candidates (multi-step edits)
If the edit region is ≥ 2 characters long, we try splitting it into two
segments and treating each as a separate program:

```
e.g., 'uJw' → 'pemh' splits into:
      replace('u', 'p') + replace('Jw', 'emh')
```

This captures cases where one complex edit is actually composed of two
simpler replaces applied sequentially.

#### c) Context extension
We also try extending the edit region by one character to the left (including
the first character of the common prefix). This captures patterns like
`replace('Au', 'Ap')` where the prefix character is part of the replacement
pattern.

#### d) Insertion handling
When the output has extra characters at the edit boundary (output longer than
input), we anchor the insertion to the adjacent existing character.

---

### Step 2 – Greedy Program Selection

Starting with an empty program sequence, we iteratively select programs:

1. For each unused candidate `(A, B)`, score it by:
   - **n_fix**: number of currently unfixed examples that become fixed when
     this program is added.
   - **n_break**: number of currently fixed examples that would be broken by
     adding this program.
   - **score = n_fix - 2 × n_break** (heavy penalty for breaking).

2. Select the candidate with the highest score.
3. Tie-breaker: longer predicate (more specific pattern).
4. Stop when no candidate has a positive score or max_programs is reached.

---

### Step 3 – Single-Program Residual Fixing

After greedy selection, check if any examples are still unfixed. For each
unfixed example:

1. Find the edit region between its current (partially transformed) form
   and the target output.
2. Generate candidates from this residual edit region.
3. Try each candidate; if it fixes the example without breaking more than
   2 other examples, add it to the sequence.

---

### Step 4 – Multi-Program Residual Fixing (Pairs)

For stubborn unfixed examples that need multiple programs, try **pairs** of
candidates applied sequentially:

1. For each unfixed example, find the residual edit region.
2. Generate all candidates from this region.
3. Try every pair `(c1, c2)` — check if `input.replace(c1).replace(c2)` equals
   the expected output.
4. If the pair works and breaks ≤ 2 other examples, add both programs.

---

### Step 5 – Output

- **Success**: All examples correctly transformed → return the program sequence.
- **Partial**: Not all examples fixed → return the best programs found along
  with the top-K alternatives.

---

## Complexity Analysis

| Phase | Time Complexity | Space Complexity |
|-------|----------------|------------------|
| Candidate extraction | O(N × L² × L²) | O(N × L⁴) |
| Greedy selection | O(K × C × N) | O(C + N) |
| Residual fixing | O(U × C² × N) | O(C) |

Where:
- N = number of examples
- L = maximum string length (≈ 10–20 in practice)
- C = number of unique candidates (typically 50–200)
- K = max programs (20)
- U = number of unfixed examples (≤ N)

The algorithm is efficient enough for typical PBEBench tasks with 50 examples
and 3–10 programs.

---

## Why This Approach Works

1. **Edit region analysis** focuses the search on relevant patterns, avoiding
   the exponential explosion of trying all possible (A, B) pairs.

2. **Greedy with break penalty** ensures that programs are only selected if
   they help more examples than they harm.

3. **Split candidates** capture multi-step edits that a single replace cannot
   express (e.g., `uJw → pemh` needs two replaces).

4. **Context extension** handles cases where the edit boundary cuts through a
   meaningful pattern (e.g., `Au → Ap`).

5. **Multi-program residual** fixes cases where the greedy phase missed the
   right combination of programs.

---

## Limitations and Failure Modes

- **Overlapping edits**: When different examples need conflicting replacements
  at the same position, the greedy approach may pick the "wrong" program.

- **Deep chains**: Examples requiring 4+ sequential program applications may
  not be fully resolved (only pairs are tried in residual fixing).

- **Long transforms**: Programs requiring `len(B) > 3` are not expressible
  within the DSL.

For such cases the solver still returns the best partial programs found,
which can guide a human or LLM to the correct solution.
