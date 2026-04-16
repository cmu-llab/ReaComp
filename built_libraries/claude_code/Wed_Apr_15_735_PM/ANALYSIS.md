# ANALYSIS.md — PBEBench Replace-Sequence Library

---

## SECTION 1: PATTERN SUMMARY

### Task Overview
Each task presents a set of (input, output) string pairs. The goal is to infer an ordered sequence of `replace(pattern, replacement)` calls — applied globally via Python's `str.replace` — that transforms every input into its corresponding output. Constraints: pattern length ≤ 3, replacement length ≤ 3, at most 20 rules. The `bfcc_string` encodes which phonological rule-interaction types are present (Feeding, Bleeding, Counterfeeding, Counterbleeding), which is a proxy for rule ordering complexity.

---

### Recurring Successful Procedures

**P1 — Changed-Pair Isolation**
Scan all (input, output) pairs and separate them into *changed* (input ≠ output) and *unchanged* (input == output) buckets. All successful traces do this first, explicitly listing indices with "(same)" or "(changed)". Appears in: both easy and hard. Worth capturing: focuses hypothesis generation on the few pairs that carry signal; unchanged pairs are used for negative-constraint checking.

**P2 — Minimal Edit Extraction**
For each changed pair, find the shortest substring in the input that needs to become a different substring in the output by trimming longest-common-prefix and longest-common-suffix. Example: `PCklPG → PCnG` becomes edit `klPG → nG` and trimming further to `klP → n`. Appears in: both easy and hard. Worth capturing: produces the tightest possible rule candidate; prevents proposing rules that are longer than necessary and therefore harder to verify.

**P3 — Candidate Rule Hypothesis from Pair**
From the minimal edit `(old, new)`, propose a candidate `replace(old, new)`. Also try one-character shorter/wider variants if the minimal edit exceeds the constraint. Easy traces typically yield 1–2 candidates per changed pair; hard traces may yield more due to ambiguous edits. Worth capturing: standardises the transition from "what changed" to "what rule explains it".

**P4 — Cross-Corpus Consistency Check (negative constraint)**
After hypothesising a candidate rule, apply it to ALL strings and verify that (a) changed strings now produce the correct output, and (b) unchanged strings remain identical. Successful hard traces explicitly check this ("Are there other strings containing G? ... WGIW → WvIW, good."). This is the most critical differentiator between success and failure. Appears in: both easy and hard success; present but applied less carefully in failures. Worth capturing: the #1 failure mode (41% of easy failures, 26% of hard failures) is over-application — a rule fires on a string it should not touch.

**P5 — Rule Ordering / Interaction Simulation**
For Feeding and Bleeding interactions, the output of one rule can create new matches for a later rule. Successful traces explicitly simulate this: apply the rule sequence step-by-step to a sample string and watch what changes at each step. Hard failures (with cascade_length > 10) frequently produce wrong outputs due to unresolved ordering issues. Worth capturing: prevents the 37% of hard failures attributable to wrong outputs from interaction bugs.

**P6 — Proxy-Swap for Counterfeeding Protection**
When a rule A→B must fire on original occurrences of A without being affected by a later rule B→C (counterfeeding), successful traces introduce a temporary placeholder: `replace(A, PROXY)`, then `replace(B, C)`, then `replace(PROXY, desired-output-for-A)`. This protects the original A instances from being consumed by the B→C rule. Appears in: ~56% of easy success, ~64% of hard success. Worth capturing: without it, a two-rule interaction produces the wrong result on the original A strings.

---

### Recurring Failure Modes

**F1 — Over-Application (rule fires on unchanged strings)**
The most common failure mode in easy cases (41%). The model proposes a rule like `replace('M', 'VZ')` that explains one changed pair but also modifies strings that should remain unchanged. Root cause: the model fails to check whether the pattern appears in any of the unchanged strings.

**F2 — Under-Application / Missing Transformations (hard cases)**
37% of hard failures. The model either produces too few rules (missing changed pairs entirely) or uses rules that are too specific (matching an exact 3-char context) when a shorter pattern would cover more cases. Hard-failure entry 1 returned zero rules and scored 26%.

**F3 — Rule Interaction / Ordering Bugs**
30–37% of failures across both categories. Arises when a rule creates a new pattern that a subsequent rule unintentionally transforms, or when the desired counterfeeding protection is absent. Example: `replace('E','Pmm')` in a hard failure over-applies to `UG` (changing `E`→`Pmm` triggers a chain that produces the wrong final string).

**F4 — Bloated Rule Sets**
Hard failures average 13.7 rules vs 8.5 in hard successes. More rules increases interaction risk and over-application surface area. Failures exhibit a pattern of adding compensating rules to fix the side effects of earlier bad rules, creating cascades of fragile fixes.

---

## SECTION 2: LIBRARY DESIGN

Proposed library of **8 helper functions**:

---

### `find_changed_pairs`
**Signature:** `find_changed_pairs(inputs: list[str], outputs: list[str]) -> list[tuple[int, str, str]]`

**Docstring:** Return (index, input_string, output_string) for every pair where input ≠ output.

**Recurring pattern captured:** P1 — Changed-Pair Isolation. Every successful trace begins by enumerating which examples actually change.

**Transfer to harder cases:** Harder tasks have more changed pairs (mean 14.8 vs 7.5 for easy). Isolating them is even more critical to avoid noise from the majority of unchanged strings.

**Failure mode avoided:** Without this step, models propose rules over all strings, generating false hypotheses from coincidental patterns in unchanged strings.

---

### `find_minimal_edit`
**Signature:** `find_minimal_edit(s_in: str, s_out: str) -> tuple[str, str, int]`

**Docstring:** Return (old_substring, new_substring, start_position) identifying the tightest region of change between s_in and s_out by stripping longest-common-prefix and longest-common-suffix.

**Recurring pattern captured:** P2 — Minimal Edit Extraction. Successful traces always reduce each changed pair to the minimal changed region before proposing a rule.

**Transfer to harder cases:** Hard cases have longer strings and subtler edits. Minimal edit extraction prevents proposing a 3-char pattern when the true changed region is 1–2 chars.

**Failure mode avoided:** Reduces risk of proposing overly broad rules; tighter patterns have smaller collateral damage.

---

### `propose_rule_candidates`
**Signature:** `propose_rule_candidates(s_in: str, s_out: str, max_pat_len: int = 3, max_rep_len: int = 3) -> list[tuple[str, str]]`

**Docstring:** Given a changed pair (s_in → s_out), return a ranked list of (pattern, replacement) candidates, from most specific (longest pattern within constraint) to most general (minimal edit). Respects the max_pat_len and max_rep_len constraints.

**Recurring pattern captured:** P3 — Candidate Rule Hypothesis. Bridges P2 output to a usable rule.

**Transfer to harder cases:** Hard cases benefit from seeing multiple candidates at different specificity levels; the model can pick the one that doesn't collide with unchanged strings.

**Failure mode avoided:** Prevents blindly using the full changed region as the rule when a sub-pattern would suffice.

---

### `apply_rules`
**Signature:** `apply_rules(strings: list[str], rules: list[tuple[str, str]]) -> list[str]`

**Docstring:** Apply a sequence of (pattern, replacement) rules in order to each string in the list, using Python's global str.replace semantics.

**Recurring pattern captured:** Core execution primitive used by P4, P5, P6.

**Transfer to harder cases:** Identical semantics on any length of string or rule list.

**Failure mode avoided:** Centralises rule application logic; prevents subtle mistakes like applying rules out of order or forgetting to apply to the full corpus.

---

### `score_rules`
**Signature:** `score_rules(inputs: list[str], outputs: list[str], rules: list[tuple[str, str]]) -> float`

**Docstring:** Apply rules to inputs and return the fraction of examples where the result matches the expected output.

**Recurring pattern captured:** P4 (positive half) — the correctness metric used in every trace to evaluate a proposed sequence.

**Transfer to harder cases:** Works identically regardless of corpus size or cascade depth.

**Failure mode avoided:** Gives a concrete numeric signal to guide rule selection; prevents premature commitment to partially-correct sequences.

---

### `find_over_applications`
**Signature:** `find_over_applications(inputs: list[str], outputs: list[str], rules: list[tuple[str, str]]) -> list[tuple[int, str, str]]`

**Docstring:** Return (index, original_string, incorrectly_modified_string) for every example where input == output (should be unchanged) but the rule sequence modifies it anyway.

**Recurring pattern captured:** P4 (negative half) — the critical cross-corpus consistency check. This is the most important differentiator between successful and failing traces.

**Transfer to harder cases:** Over-application risk grows with cascade length; essential for hard tasks.

**Failure mode avoided:** Directly addresses F1 (41% of easy failures, 26% of hard failures are over-application errors).

---

### `simulate_rule_chain`
**Signature:** `simulate_rule_chain(s: str, rules: list[tuple[str, str]]) -> list[dict]`

**Docstring:** Apply rules one at a time to string s, returning a list of {"rule": (pat, rep), "before": str, "after": str, "changed": bool} dicts showing each step. Useful for debugging rule interaction and ordering.

**Recurring pattern captured:** P5 — Rule Ordering Simulation. Hard success traces explicitly trace the effect of each rule.

**Transfer to harder cases:** Essential for tasks with cascade_length > 5 or multiple BFCC interaction types.

**Failure mode avoided:** Addresses F3 (rule interaction / ordering bugs), especially for Feeding and Bleeding interactions.

---

### `build_proxy_rules`
**Signature:** `build_proxy_rules(pattern: str, final_replacement: str, proxy: str = "__X__") -> list[tuple[str, str]]`

**Docstring:** Build a 3-rule proxy-swap sequence: (1) replace `pattern` with `proxy`, (2) placeholder for caller to insert other rules, (3) replace `proxy` with `final_replacement`. Returns the bookend rules as [pre_rule, post_rule] = [(pattern, proxy), (proxy, final_replacement)]. Used when `pattern` must be protected from being consumed by other rules in the sequence.

**Recurring pattern captured:** P6 — Proxy-Swap for Counterfeeding Protection. Used in 56–64% of successful traces.

**Transfer to harder cases:** Hard tasks with 3–4 BFCC types frequently require multiple proxy swaps; this makes each one reproducible.

**Failure mode avoided:** Addresses F3 for counterfeeding specifically — without protection, rules that produce a string also transform the newly-produced instances unintentionally.

---

## SECTION 3: PYTHON IMPLEMENTATION

See `LIBRARY.py`.

---

## SECTION 4: LIGHTWEIGHT TESTS

See `LIBRARY.py` (bottom section, under `if __name__ == "__main__"`).

---

## SECTION 5: USAGE GUIDE FOR A WEAKER MODEL

See `PROMPTING_GUIDE.md`.

---

## SECTION 6: RISK CHECK

**Functions that might be too specific:**
- `build_proxy_rules`: The proxy pattern is specific to string-replace rule interaction. It won't transfer to non-string or non-sequential-replace tasks. Also: the caller must correctly identify which pattern needs protection — the function won't help if the model doesn't know when to invoke it.

**Functions that might be too broad:**
- `apply_rules`: Works for any list of (str, str) pairs, which is correct. Risk: the model might use it with rules derived by other means (e.g., regex) that aren't plain `str.replace` — it will silently produce wrong results.
- `propose_rule_candidates`: Returns multiple candidates without filtering. On large inputs this could produce candidates that trivially fit a single pair but massively over-apply. Should be followed by `find_over_applications`.

**Parts most likely to fail on unseen tasks:**
- `find_minimal_edit`: Assumes one contiguous changed region. If a string has two independent changes (e.g., two separate substrings change), it returns the region spanning both, producing an overly broad rule. Hard tasks with long chains sometimes produce this.
- `propose_rule_candidates`: Assumes the edit can be expressed as a single replace rule within the length constraint. For multi-change strings, this returns no valid candidates, and the model must fall back to manual reasoning.
- `build_proxy_rules`: Requires the user to know what the `proxy` string is and to ensure it doesn't appear in any input. If it collides with existing corpus strings, it creates new errors.

**Simplification to half the library:**
If forced to 4 functions: keep `find_changed_pairs`, `apply_rules`, `score_rules`, and `find_over_applications`. These cover the two most critical steps (knowing what to change and checking for over-application) and let a model handle the rest with trial-and-error. Drop `find_minimal_edit` and `propose_rule_candidates` (the model can guess rules from the diff), `simulate_rule_chain` (useful but not essential), and `build_proxy_rules` (only needed for counterfeeding which is a minority of cases).
