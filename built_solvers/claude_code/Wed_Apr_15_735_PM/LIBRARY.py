"""
LIBRARY.py — Reusable helper library for string-replace rule induction.

Task context: Given a corpus of (input, output) string pairs, find an ordered
sequence of replace(pattern, replacement) calls — applied globally — that
transforms every input into its corresponding output.
Constraints: pattern length ≤ 3, replacement length ≤ 3, at most 20 rules.

This library provides symbolic primitives that recur across successful
solution traces. It is designed to be called by a weaker model that needs
procedural scaffolding for the induction process.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# 1. find_changed_pairs
# ---------------------------------------------------------------------------

def find_changed_pairs(
    inputs: list[str],
    outputs: list[str],
) -> list[tuple[int, str, str]]:
    """
    Identify which (input, output) pairs actually differ.

    Parameters
    ----------
    inputs  : list of input strings (corpus)
    outputs : list of expected output strings, same length as inputs

    Returns
    -------
    List of (index, input_str, output_str) for every pair where input != output,
    in original corpus order.

    Notes
    -----
    Unchanged pairs (input == output) provide negative constraints for rule
    checking but carry no direct signal for rule induction.
    """
    changed: list[tuple[int, str, str]] = []
    for i, (s_in, s_out) in enumerate(zip(inputs, outputs)):
        if s_in != s_out:
            changed.append((i, s_in, s_out))
    return changed


# ---------------------------------------------------------------------------
# 2. find_minimal_edit
# ---------------------------------------------------------------------------

def find_minimal_edit(s_in: str, s_out: str) -> tuple[str, str, int]:
    """
    Find the tightest contiguous region of change between s_in and s_out.

    Strips the longest common prefix and longest common suffix, leaving only
    the minimal substrings that differ.

    Parameters
    ----------
    s_in  : original string
    s_out : target string

    Returns
    -------
    (old_sub, new_sub, start_pos) where
      - old_sub  is the minimal substring of s_in that changed
      - new_sub  is the minimal substring of s_out it became
      - start_pos is the 0-based index in s_in where the edit begins

    If s_in == s_out, returns ("", "", 0).

    Notes
    -----
    Assumes a single contiguous changed region. If multiple disjoint regions
    changed, the returned span will cover them all (possibly larger than
    optimal). Use propose_rule_candidates to enumerate refinements.
    """
    if s_in == s_out:
        return ("", "", 0)

    # Find longest common prefix
    prefix_len = 0
    for a, b in zip(s_in, s_out):
        if a == b:
            prefix_len += 1
        else:
            break

    # Find longest common suffix (on the remaining tails)
    tail_in = s_in[prefix_len:]
    tail_out = s_out[prefix_len:]
    suffix_len = 0
    for a, b in zip(reversed(tail_in), reversed(tail_out)):
        if a == b:
            suffix_len += 1
        else:
            break

    old_sub = tail_in[:len(tail_in) - suffix_len] if suffix_len else tail_in
    new_sub = tail_out[:len(tail_out) - suffix_len] if suffix_len else tail_out

    return (old_sub, new_sub, prefix_len)


# ---------------------------------------------------------------------------
# 3. propose_rule_candidates
# ---------------------------------------------------------------------------

def propose_rule_candidates(
    s_in: str,
    s_out: str,
    max_pat_len: int = 3,
    max_rep_len: int = 3,
) -> list[tuple[str, str]]:
    """
    Generate (pattern, replacement) rule candidates that explain s_in → s_out.

    Uses find_minimal_edit to get the core change, then enumerates progressively
    shorter prefixes of the old_sub as the pattern (most to least specific),
    subject to max_pat_len and max_rep_len constraints.

    Parameters
    ----------
    s_in        : original string
    s_out       : target string
    max_pat_len : maximum allowed pattern length (default 3)
    max_rep_len : maximum allowed replacement length (default 3)

    Returns
    -------
    List of (pattern, replacement) tuples, ordered from most specific to most
    general, filtered to respect constraints. May be empty if no valid candidate
    exists within constraints.

    Notes
    -----
    The returned candidates are hypotheses from a single pair. Always validate
    them across the full corpus using score_rules and find_over_applications
    before committing.
    """
    if s_in == s_out:
        return []

    old_sub, new_sub, start = find_minimal_edit(s_in, s_out)

    if not old_sub and not new_sub:
        return []

    candidates: list[tuple[str, str]] = []

    # Try expanding context on the left within the max_pat_len constraint
    # so we can propose increasingly specific patterns.
    # Core candidate: minimal edit directly
    if len(old_sub) <= max_pat_len and len(new_sub) <= max_rep_len:
        candidates.append((old_sub, new_sub))

    # Wider candidates: include 1–(max_pat_len - len(old_sub)) chars of prefix context
    context_budget = max_pat_len - len(old_sub)
    for extra in range(1, context_budget + 1):
        ctx_start = max(0, start - extra)
        pat = s_in[ctx_start: start + len(old_sub)]
        # Corresponding replacement must include the same prefix context
        prefix_context = s_in[ctx_start:start]
        rep = prefix_context + new_sub
        if len(pat) <= max_pat_len and len(rep) <= max_rep_len:
            candidates.insert(0, (pat, rep))  # more specific → front

    # Narrower candidates: just the first N chars of old_sub → new_sub prefix
    for trunc in range(1, len(old_sub)):
        pat = old_sub[:trunc]
        rep = new_sub[:trunc] if trunc <= len(new_sub) else new_sub
        if pat and len(pat) <= max_pat_len and len(rep) <= max_rep_len:
            if (pat, rep) not in candidates:
                candidates.append((pat, rep))

    # Deduplicate while preserving order
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    return deduped


# ---------------------------------------------------------------------------
# 4. apply_rules
# ---------------------------------------------------------------------------

def apply_rules(
    strings: list[str],
    rules: list[tuple[str, str]],
) -> list[str]:
    """
    Apply an ordered sequence of (pattern, replacement) rules to each string.

    Uses Python's built-in str.replace, which substitutes all non-overlapping
    occurrences left-to-right (global replace semantics).

    Parameters
    ----------
    strings : list of strings to transform
    rules   : ordered list of (pattern, replacement) pairs

    Returns
    -------
    List of transformed strings, in the same order as inputs.

    Notes
    -----
    Rule order matters. Rule i may create new matches for rule j (i < j) in a
    Feeding interaction, or rule j may destroy matches created by rule i in a
    Bleeding interaction. Use simulate_rule_chain to debug these interactions.
    """
    results: list[str] = []
    for s in strings:
        for pat, rep in rules:
            if pat:  # guard against empty-pattern replace (infinite loop)
                s = s.replace(pat, rep)
        results.append(s)
    return results


# ---------------------------------------------------------------------------
# 5. score_rules
# ---------------------------------------------------------------------------

def score_rules(
    inputs: list[str],
    outputs: list[str],
    rules: list[tuple[str, str]],
) -> float:
    """
    Evaluate how well an ordered rule sequence transforms inputs to outputs.

    Parameters
    ----------
    inputs  : original strings
    outputs : expected target strings
    rules   : ordered (pattern, replacement) sequence to evaluate

    Returns
    -------
    Fraction in [0.0, 1.0] of examples where apply_rules(input) == output.
    Returns 1.0 on an empty corpus.

    Notes
    -----
    Use this to compare candidate rule sequences. A score of 1.0 means every
    example is correctly handled. Use find_over_applications to diagnose
    specific failures even when score < 1.0.
    """
    if not inputs:
        return 1.0
    got = apply_rules(inputs, rules)
    correct = sum(1 for g, o in zip(got, outputs) if g == o)
    return correct / len(inputs)


# ---------------------------------------------------------------------------
# 6. find_over_applications
# ---------------------------------------------------------------------------

def find_over_applications(
    inputs: list[str],
    outputs: list[str],
    rules: list[tuple[str, str]],
) -> list[tuple[int, str, str]]:
    """
    Find cases where a rule sequence modifies a string that should stay unchanged.

    An over-application occurs when input[i] == output[i] (example should be
    unchanged) but apply_rules(input[i]) != input[i].

    Parameters
    ----------
    inputs  : original strings
    outputs : expected target strings (some may equal the corresponding input)
    rules   : ordered (pattern, replacement) sequence to check

    Returns
    -------
    List of (index, original_string, incorrectly_modified_string) for every
    over-application, in corpus order.

    Notes
    -----
    Over-application is the most common failure mode (41% of easy failures,
    26% of hard failures). Always call this after proposing a new rule and
    before accepting it into the sequence.
    """
    got = apply_rules(inputs, rules)
    violations: list[tuple[int, str, str]] = []
    for i, (s_in, s_out, s_got) in enumerate(zip(inputs, outputs, got)):
        if s_in == s_out and s_got != s_in:
            violations.append((i, s_in, s_got))
    return violations


# ---------------------------------------------------------------------------
# 7. simulate_rule_chain
# ---------------------------------------------------------------------------

def simulate_rule_chain(
    s: str,
    rules: list[tuple[str, str]],
) -> list[dict]:
    """
    Trace the effect of applying each rule one at a time to a single string.

    Parameters
    ----------
    s     : string to trace through the rule sequence
    rules : ordered (pattern, replacement) sequence

    Returns
    -------
    List of dicts, one per rule, with keys:
      - "step"    : 1-based step index
      - "rule"    : (pattern, replacement) tuple
      - "before"  : string value before applying this rule
      - "after"   : string value after applying this rule
      - "changed" : True if the rule fired (before != after)

    Notes
    -----
    Use this to debug Feeding interactions (where rule i creates a new match
    consumed by rule j) and Bleeding interactions (where rule j destroys a
    match that rule i would have used). Especially important for tasks with
    cascade_length > 5 or multiple BFCC interaction types.
    """
    trace: list[dict] = []
    current = s
    for step, (pat, rep) in enumerate(rules, start=1):
        before = current
        if pat:
            current = current.replace(pat, rep)
        trace.append({
            "step": step,
            "rule": (pat, rep),
            "before": before,
            "after": current,
            "changed": before != current,
        })
    return trace


# ---------------------------------------------------------------------------
# 8. build_proxy_rules
# ---------------------------------------------------------------------------

def build_proxy_rules(
    pattern: str,
    final_replacement: str,
    proxy: str = "__X__",
) -> tuple[tuple[str, str], tuple[str, str]]:
    """
    Create bookend rules that protect `pattern` from being consumed by
    intermediate rules in a counterfeeding scenario.

    A counterfeeding situation arises when:
      - Rule A: replace(P, Q) should fire on original occurrences of P.
      - Rule B: replace(Q, R) comes after A and would re-transform the Q
        that A just produced — but it should NOT affect the Q produced by A,
        only pre-existing Q occurrences.

    The proxy swap protects against this by replacing P with a placeholder
    before rule B fires, then replacing the placeholder with the desired
    final result afterward.

    Parameters
    ----------
    pattern           : the source pattern to protect (P above)
    final_replacement : what protected instances should become after all
                        other rules have fired
    proxy             : a temporary placeholder string that is guaranteed not
                        to appear in any corpus string. Default "__X__".
                        Caller is responsible for ensuring no collision.

    Returns
    -------
    (pre_rule, post_rule) where
      - pre_rule  = (pattern, proxy)   — insert BEFORE the interfering rules
      - post_rule = (proxy, final_replacement) — insert AFTER the interfering rules

    Usage
    -----
    pre, post = build_proxy_rules("A", "Z", proxy="@@")
    rules = [pre, ("B", "C"), post]   # protected: A→@@ … @@→Z
    # Now "A" instances become Z, while "B"→"C" only affects original B's.

    Notes
    -----
    The proxy string must satisfy:
      1. It does not appear in any input or output corpus string.
      2. It is not created as a substring by any rule in the sequence.
      3. It is unique per protection group (use different proxies for
         different simultaneously protected patterns).
    """
    pre_rule = (pattern, proxy)
    post_rule = (proxy, final_replacement)
    return pre_rule, post_rule


# ---------------------------------------------------------------------------
# SECTION 4: LIGHTWEIGHT TESTS
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    print("Running library tests...\n")

    # -----------------------------------------------------------------------
    # Test 1: find_changed_pairs
    # -----------------------------------------------------------------------
    inputs  = ["cat", "dog", "bird", "fish"]
    outputs = ["cat", "log", "bird", "dish"]
    changed = find_changed_pairs(inputs, outputs)
    assert changed == [(1, "dog", "log"), (3, "fish", "dish")], f"FAIL: {changed}"
    assert find_changed_pairs(["a", "b"], ["a", "b"]) == [], "FAIL: no changes expected"
    print("PASS  find_changed_pairs")

    # -----------------------------------------------------------------------
    # Test 2: find_minimal_edit
    # -----------------------------------------------------------------------
    old, new, pos = find_minimal_edit("PCklPG", "PCnG")
    assert old == "klP", f"FAIL: old={old!r}"
    assert new == "n",   f"FAIL: new={new!r}"
    assert pos == 2,     f"FAIL: pos={pos}"

    old, new, pos = find_minimal_edit("hello", "hello")
    assert old == "" and new == "" and pos == 0, "FAIL: identical strings"

    old, new, pos = find_minimal_edit("abc", "axc")
    assert old == "b" and new == "x" and pos == 1, f"FAIL: single char edit {old!r}->{new!r}@{pos}"
    print("PASS  find_minimal_edit")

    # -----------------------------------------------------------------------
    # Test 3: propose_rule_candidates
    # -----------------------------------------------------------------------
    candidates = propose_rule_candidates("PCklPG", "PCnG")
    # Must include ("klP", "n") as the minimal-edit candidate
    assert ("klP", "n") in candidates, f"FAIL candidates={candidates}"
    # With constraint max_pat_len=3, all patterns must be ≤3 chars
    for pat, rep in candidates:
        assert len(pat) <= 3 and len(rep) <= 3, f"FAIL constraint: {(pat,rep)}"

    # Identical pair → empty
    assert propose_rule_candidates("abc", "abc") == [], "FAIL: no candidates for identical"
    print("PASS  propose_rule_candidates")

    # -----------------------------------------------------------------------
    # Test 4: apply_rules
    # -----------------------------------------------------------------------
    strings = ["cat", "scatter", "dog"]
    rules   = [("at", "og")]
    result  = apply_rules(strings, rules)
    assert result == ["cog", "scogter", "dog"], f"FAIL: {result}"

    # Chained rules
    rules2 = [("a", "b"), ("b", "c")]
    result2 = apply_rules(["abc"], rules2)
    # "abc" → replace a→b: "bbc" → replace b→c: "ccc"
    assert result2 == ["ccc"], f"FAIL: {result2}"

    # Empty rules → identity
    assert apply_rules(["hello"], []) == ["hello"], "FAIL: empty rules"
    print("PASS  apply_rules")

    # -----------------------------------------------------------------------
    # Test 5: score_rules
    # -----------------------------------------------------------------------
    inputs2  = ["cat", "dog", "bird"]
    outputs2 = ["cot", "dog", "bird"]
    rules3   = [("a", "o")]
    score = score_rules(inputs2, outputs2, rules3)
    assert score == 1.0, f"FAIL: score={score}"

    # Partial score
    rules4 = [("x", "y")]
    score2 = score_rules(inputs2, outputs2, rules4)
    assert abs(score2 - 2/3) < 1e-9, f"FAIL: partial score={score2}"

    print("PASS  score_rules")

    # -----------------------------------------------------------------------
    # Test 6: find_over_applications
    # -----------------------------------------------------------------------
    inputs3  = ["cat", "dog", "bat"]   # cat→cot (changed), dog unchanged, bat→bot (changed)
    outputs3 = ["cot", "dog", "bot"]
    rules5   = [("a", "o")]
    over = find_over_applications(inputs3, outputs3, rules5)
    # dog is unchanged, rule doesn't fire on it → no over-application
    assert over == [], f"FAIL: {over}"

    # Now introduce an over-application: "dog" should stay but rule fires
    inputs4  = ["cat", "dat", "bat"]
    outputs4 = ["cot", "dat", "bot"]   # dat should be unchanged
    rules6   = [("a", "o")]
    over2 = find_over_applications(inputs4, outputs4, rules6)
    assert len(over2) == 1, f"FAIL: expected 1 over-application, got {over2}"
    assert over2[0][0] == 1 and over2[0][1] == "dat", f"FAIL: {over2}"
    print("PASS  find_over_applications")

    # -----------------------------------------------------------------------
    # Test 7: simulate_rule_chain
    # -----------------------------------------------------------------------
    trace = simulate_rule_chain("abc", [("a", "b"), ("b", "c")])
    assert len(trace) == 2, f"FAIL: {len(trace)} steps"
    assert trace[0]["before"] == "abc" and trace[0]["after"] == "bbc"
    assert trace[0]["changed"] is True
    assert trace[1]["before"] == "bbc" and trace[1]["after"] == "ccc"
    assert trace[1]["changed"] is True

    # No-op rule
    trace2 = simulate_rule_chain("xyz", [("a", "b")])
    assert trace2[0]["changed"] is False
    print("PASS  simulate_rule_chain")

    # -----------------------------------------------------------------------
    # Test 8: build_proxy_rules
    # -----------------------------------------------------------------------
    pre, post = build_proxy_rules("A", "Z", proxy="@@")
    assert pre  == ("A", "@@"), f"FAIL pre={pre}"
    assert post == ("@@", "Z"),  f"FAIL post={post}"

    # Simulate a counterfeeding scenario:
    # Desired: replace A→Z (independently of whatever B→C does to B)
    # Naive rule [("A","B"),("B","C")] would: A→B→C (wrong, wanted Z)
    # With proxy: [("A","@@"), ("B","C"), ("@@","Z")]
    # "A" → "@@" (pre), then "B"→"C" doesn't touch "@@", then "@@"→"Z"
    strings_proxy = ["A", "B", "AB"]
    rules_proxy   = [pre, ("B", "C"), post]
    result_proxy  = apply_rules(strings_proxy, rules_proxy)
    assert result_proxy[0] == "Z",  f"FAIL: A should become Z, got {result_proxy[0]}"
    assert result_proxy[1] == "C",  f"FAIL: B should become C, got {result_proxy[1]}"
    assert result_proxy[2] == "ZC", f"FAIL: AB should become ZC, got {result_proxy[2]}"
    print("PASS  build_proxy_rules")

    print("\nAll tests passed.")


if __name__ == "__main__":
    _run_tests()
