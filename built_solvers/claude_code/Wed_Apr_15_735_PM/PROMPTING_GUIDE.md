# PROMPTING_GUIDE.md — How to Use LIBRARY.py

## Overview

You have access to a Python library (`LIBRARY.py`) with 8 helper functions for
solving replace-rule induction tasks. Each task gives you a list of input
strings and a list of expected output strings. Your goal is to find an ordered
sequence of `replace(pattern, replacement)` calls — each with pattern ≤ 3 chars
and replacement ≤ 3 chars — that transforms every input into its output.

Import the library before starting:

    from LIBRARY import (
        find_changed_pairs, find_minimal_edit, propose_rule_candidates,
        apply_rules, score_rules, find_over_applications,
        simulate_rule_chain, build_proxy_rules,
    )

---

## Recommended Workflow

**Step 1 — Isolate what actually changes**

    changed = find_changed_pairs(inputs, outputs)

This returns only the pairs where input differs from output. Focus your rule
hypotheses on these pairs. Unchanged pairs are used only as negative constraints.

---

**Step 2 — Extract the minimal edit from each changed pair**

For each `(idx, s_in, s_out)` in `changed`:

    old_sub, new_sub, pos = find_minimal_edit(s_in, s_out)

`old_sub` is what needs to change, `new_sub` is what it should become.

---

**Step 3 — Generate candidate rules**

    candidates = propose_rule_candidates(s_in, s_out)

Returns a list of `(pattern, replacement)` pairs from most specific to most
general. Try the first candidate first. If it over-applies (Step 4), try a
more specific one.

---

**Step 4 — Check for over-application before accepting a rule**

After tentatively adding a rule, immediately check:

    over = find_over_applications(inputs, outputs, current_rules)

If `over` is non-empty, the rule modifies strings that should stay unchanged.
Either make the pattern more specific, or remove the rule and try another
candidate.

---

**Step 5 — Score the current rule sequence**

    sc = score_rules(inputs, outputs, current_rules)

A score of 1.0 means all examples are handled correctly. A score below 1.0
means either some changed pairs are not yet covered (under-application) or
there are still over-applications.

---

**Step 6 — Debug rule interactions**

If a rule sequence gives wrong outputs (not over-application), use:

    trace = simulate_rule_chain(some_input_string, current_rules)

This shows what each rule does step by step. Use it to find where a rule
chain goes wrong — e.g., where a later rule re-transforms something a
previous rule already handled correctly.

---

**Step 7 — Handle counterfeeding (protect a pattern from later rules)**

If you need pattern `P` to produce result `Z` but a later rule `B → C` would
accidentally transform the `Z` you just created, use the proxy swap:

    pre, post = build_proxy_rules("P", "Z", proxy="@@")
    rules = [pre, ("B", "C"), post]

This ensures `P` → `@@` (safe) → `Z`, while `B` → `C` only affects original
`B` instances. Make sure the proxy string (default `"__X__"`) does not appear
in any input or output string. If it might collide, choose a different proxy.

---

## Function Reference Card

| Function | When to call | Input | Output |
|---|---|---|---|
| `find_changed_pairs(inputs, outputs)` | First step always | Two lists of strings | List of `(idx, s_in, s_out)` tuples |
| `find_minimal_edit(s_in, s_out)` | Per changed pair | Two strings | `(old_sub, new_sub, pos)` |
| `propose_rule_candidates(s_in, s_out)` | To get rule hypotheses | Changed pair | List of `(pat, rep)` sorted specific→general |
| `apply_rules(strings, rules)` | To see what a rule sequence produces | List of strings + rule list | Transformed strings |
| `score_rules(inputs, outputs, rules)` | After each rule addition | Corpus + rule list | Float in [0, 1] |
| `find_over_applications(inputs, outputs, rules)` | After each rule addition | Corpus + rule list | List of `(idx, original, wrong_output)` |
| `simulate_rule_chain(s, rules)` | When rule interactions go wrong | Single string + rule list | Step-by-step trace dicts |
| `build_proxy_rules(pattern, replacement, proxy)` | When counterfeeding protection is needed | Pattern string + desired output + proxy | `(pre_rule, post_rule)` pair |

---

## Short Example (toy task)

    inputs  = ["AB", "CD", "AC"]
    outputs = ["ZB", "CD", "ZC"]
    # "A" → "Z"; "CD" unchanged; "AC" → "ZC"

    changed = find_changed_pairs(inputs, outputs)
    # [(0, "AB", "ZB"), (2, "AC", "ZC")]

    # From first changed pair:
    old, new, pos = find_minimal_edit("AB", "ZB")  # old="A", new="Z", pos=0
    candidates = propose_rule_candidates("AB", "ZB")  # [("A","Z")]

    rules = [("A", "Z")]
    print(score_rules(inputs, outputs, rules))   # 1.0
    print(find_over_applications(inputs, outputs, rules))  # [] (CD unchanged)

---

## Common Mistakes to Avoid

1. **Never skip find_over_applications.** The most common error is a rule that
   fixes changed pairs but also modifies strings that should stay the same.

2. **Do not propose more rules than you need.** More rules = more interactions.
   Hard tasks fail partly because the model adds too many rules trying to
   compensate for bad earlier rules. Fix the root cause instead.

3. **Use simulate_rule_chain when score < 1.0 and no over-applications.**
   If the score is low but `find_over_applications` returns empty, you have an
   under-application or ordering problem. Trace a failing example step by step.

4. **Check the proxy is safe before using build_proxy_rules.** Scan all inputs
   and outputs with `any(proxy in s for s in inputs + outputs)`. If it appears,
   choose a longer or different placeholder.

5. **The cascade_length in the task description tells you how many rules the
   ground truth uses.** Use this as a budget hint — if your solution has far
   more rules, you likely have redundant or compensating rules.
