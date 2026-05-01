# SOLVER Algorithm

## Overview

The solver finds a sequence of `replace(A, B)` operations that transforms
every example input string into the corresponding output string.  The design
is informed by studying 100 LLM reasoning traces on PBEBench tasks (25 × 4:
difficulty × success quadrants), identifying where LLMs succeed and fail.

## DSL

A program is an ordered list of operations:

```
replace(A₁, B₁); replace(A₂, B₂); …; replace(Aₙ, Bₙ)
```

Each operation replaces **all** occurrences of `Aᵢ` with `Bᵢ` in the current
string (Python `str.replace` semantics, applied to the output of the previous
step).  Constraints per rule: `1 ≤ len(A) ≤ 3`, `0 ≤ len(B) ≤ 3`.

Complexity: `cascade_complexity = Σ (len(Aᵢ) + len(Bᵢ))`.

---

## What LLM traces reveal

**Easy failures** share a common root cause: spurious rules.  The LLM adds
rules (e.g. `replace('bs','wivA')`) that are not needed but happen to match
a pattern in an *unchanged* input string, silently breaking it.  The safety
filter (see below) eliminates this entire class of error.

**Hard failures** arise when rules interact through phonological cascades
(Feeding, Counterfeeding, Bleeding, Counterbleeding).  The LLM sometimes
picks the wrong rule ordering.  Beam search over ordered sequences naturally
explores all orderings up to the beam horizon.

**Hard successes** show that tasks with 7–8 interacting rules are solvable
when rules are kept contextually specific (long enough patterns to avoid
unintended matches).

---

## Algorithm

### Step 1 — Direct Rule Extraction

`extract_direct_rules(inputs, outputs)` generates *direct* rules: `(A, B)`
pairs where `inp.replace(A, B) == out` for at least one changed example.

For each changed `(inp, out)` pair and each A-substring of length 1–3:
- Try deletion: B = `""`
- Try substitution: B = every substring of `out` of length 1–3

These rules solve at least one example in **one step** and are the most
valuable candidates.

### Step 2 — Safety Filter

`safety_filter(rules, unchanged_inputs)` removes any rule `(A, B)` where `A`
appears in any unchanged input (i.e. a string where `inp == out`).

**Why**: applying such a rule would incorrectly modify strings that must stay
the same.  From the LLM traces, this is the dominant cause of easy failures.
The filter typically reduces the candidate set from ~2000 rules to ~50–100
targeted rules, dramatically tightening the search.

### Step 3 — Beam Search

Each **beam item** stores `(score, complexity, program_tuple, current_strings)`.

`current_strings` tracks the state of all example inputs after applying the
partial program so far, enabling O(n · |string|) incremental scoring without
replaying the whole program.

At each depth level:

1. Expand every beam item by appending each candidate rule.
2. Skip no-ops (rules that leave `current_strings` unchanged).
3. Score: fraction of examples where `current_strings[i] == targets[i]`.
4. **Early exit**: if score = 1.0, return immediately.
5. Deduplicate by program tuple (each unique sequence appears once).
6. Prune to top `beam_width` items, ranked by `(−score, cascade_complexity)`.

This ordering ensures that at a given score, simpler programs (lower
`cascade_complexity`) are preferred, consistent with the Occam's-razor
principle stated in the task spec.

### Step 4 — Three-Phase Search

```
Phase 1:  safety-filtered direct rules,  max_depth=8, beam_width=500
Phase 2:  safety-filtered direct+supp.,  max_depth=8, beam_width=500
Phase 3:  all direct rules (no filter),  max_depth=5, beam_width=200
```

Phase 1 handles the vast majority of tasks quickly.  Phase 2 adds
supplemental rules (all input-substring × output-substring pairs, capped at
2000) for multi-step programs where intermediate rules are not themselves
direct solutions.  Phase 3 is a fallback for edge cases where intermediate
rules temporarily touch unchanged inputs (e.g. "escape-and-restore" patterns
seen in some Counterfeeding tasks).

The first phase that returns `success=True` wins; otherwise the highest-
scoring partial result across all phases is returned.

---

## Complexity

| Variable | Meaning |
|----------|---------|
| E | number of examples |
| L | average string length |
| R | candidate rules (after safety filter, typically 50–100) |
| W | beam width (500) |
| D | max depth (8) |

Per-depth cost: O(W · R · E · L) — apply one rule to all current strings.
Total: O(D · W · R · E · L).

For typical PBEBench tasks (E≈50, L≈6, R≈80, W=500, D=8) ≈ 960 M character
operations; at ~10 ns per Python `str.replace` on a 6-char string ≈ < 1 s.

---

## Output Format

```python
{
    "success": bool,          # True if score == 1.0
    "program": [              # best program found
        {"op": "replace", "from": A, "to": B},
        ...
    ],
    "score":  float,          # fraction of examples satisfied
    "top_k":  [program, ...]  # top-K programs by (score, −complexity)
}
```

---

## Design Choices

| Choice | Rationale |
|--------|-----------|
| Safety filter as primary pass | Eliminates spurious rules — the #1 LLM failure mode observed in CoT traces |
| Direct rules only in phase 1 | Small, high-quality candidate set; most easy tasks solved here |
| Supplemental rules in phase 2 | Handles multi-step programs whose intermediate steps are not direct solutions |
| Fallback without filter | Handles Counterfeeding edge cases where escape-and-restore temporarily modifies unchanged inputs |
| Beam tracks current strings | Incremental O(E·L) scoring per rule; enables deep search without re-applying the full prefix |
| Complexity as tiebreaker | Prefers simpler programs (fewer/shorter rules); prevents over-fitting to training noise |
| max_depth=8 | Hard PBEBench tasks have ground-truth cascade_length up to 17; shorter equivalent programs often need 6–8 steps |
| beam_width=500 with safety filter | Since safety filter reduces candidates to ~50–100, all depth-1 candidates survive, preventing premature pruning of the correct first step |
