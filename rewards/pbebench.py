"""
Reward function for PBEBench tasks.

Each task is a Programming-by-Example problem: given a set of input strings and
corresponding output strings, the agent must produce an ordered sequence of Python
str.replace() programs that transforms every input into its paired output.

Constraints (from the task prompt):
  1. Each program has the form replace(A, B) with quoted string arguments.
  2. 1 <= len(A) <= 3 characters; len(B) can be 0 (empty string) to 3 characters.
  3. Maximum 5 programs in a sequence.
  4. Only the built-in str.replace() function is used — no other functions or imports.

Score = (number of input→output pairs correctly transformed) / (total pairs).
Value is always in [0.0, 1.0].

The reward also checks the constraints above and returns actionable feedback messages
so BCR can fix invalid programs on the next iteration.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Matches replace('A', 'B') or replace("A", "B") with optional whitespace.
# Group 1 = predicate (A), Group 2 = transform (B).
_REPLACE_RE = re.compile(
    r"""replace\(\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\)""",
    re.IGNORECASE,
)


# ── parsing ───────────────────────────────────────────────────────────────────

def _parse_programs(result: Any) -> Tuple[Optional[List[Tuple[str, str]]], str]:
    """
    Parse the agent output into a list of (predicate, transform) tuples.

    Handles the following result formats produced by BCR:
      - list of strings  e.g. ["replace('a', 'b')", "replace('cd', 'e')"]
      - JSON string       e.g. '["replace(\\'a\\', \\'b\\')"]'
      - plain text        e.g. "replace('a','b')\\nreplace('cd','e')"
      - markdown-fenced   e.g. ```python\\n[...]\\n```

    Returns (programs, "") on success or (None, error_message) on failure.
    """
    if result is None:
        return None, "result is None"

    # Normalise to a single raw text string for regex extraction.
    if isinstance(result, list):
        # Unwrap accidental nesting: [[prog1, prog2, ...]] → [prog1, prog2, ...]
        if len(result) == 1 and isinstance(result[0], list):
            result = result[0]
        raw = "\n".join(str(x) for x in result)
    elif isinstance(result, str):
        raw = result.strip()
        # Strip markdown fences (```python ... ``` or ``` ... ```)
        raw = re.sub(r"```[a-zA-Z]*\n?", "", raw).strip("` \n")
        # Try JSON parse — agent may return a JSON list string
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                raw = "\n".join(str(x) for x in parsed)
            else:
                raw = str(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    else:
        return None, f"Unexpected result type {type(result).__name__!r}"

    matches = _REPLACE_RE.findall(raw)
    if not matches:
        return None, (
            f"No replace(A, B) calls found in output. Got: {raw[:120]!r}. "
            "Return a list like [\"replace('a', 'b')\", \"replace('cd', 'ef')\"]."
        )

    return matches, ""  # list of (predicate, transform) tuples


# ── validation ────────────────────────────────────────────────────────────────

def _validate_programs(programs: List[Tuple[str, str]]) -> List[str]:
    """
    Check each program against the task constraints.
    Returns a list of violation messages (empty list = all constraints satisfied).
    """
    violations: List[str] = []

    if len(programs) > 5:
        violations.append(
            f"Too many programs: {len(programs)} (maximum is 5)"
        )

    for i, (pred, transform) in enumerate(programs):
        idx = i + 1
        if len(pred) < 1:
            violations.append(
                f"Program {idx} replace('{pred}', '{transform}'): "
                "predicate A must have at least 1 character"
            )
        elif len(pred) > 3:
            violations.append(
                f"Program {idx} replace('{pred}', '{transform}'): "
                f"predicate A has {len(pred)} characters (max 3)"
            )
        if len(transform) > 3:
            violations.append(
                f"Program {idx} replace('{pred}', '{transform}'): "
                f"transform B has {len(transform)} characters (max 3)"
            )

    return violations


# ── complexity ────────────────────────────────────────────────────────────────

def cascade_complexity(programs: List[Tuple[str, str]]) -> int:
    """
    Sum of all predicate and transform string lengths in the cascade.
    E.g. [("ab","ac"),("e","fg"),("h","")] → 2+2+1+2+1+0 = 8.
    Not normalised — longer cascades with trivial programs are penalised naturally.
    """
    return sum(len(pred) + len(transform) for pred, transform in programs)


# ── main reward function ───────────────────────────────────────────────────────

def reward(result: Any, execution_ok: bool, entry: Dict) -> Dict:
    """
    Parameters
    ----------
    result       : raw Python value returned by execute_with_library(), or the
                   direct answer string from BCR's direct action; None on failure
    execution_ok : False if the solution code raised a runtime exception
    entry        : full task record dict (contains 'inputs', 'outputs', ...)
    """
    if not execution_ok:
        return {
            "value": 0.0,
            "message": (
                "Solution raised a runtime error during execution. "
                "Make sure your function returns a list of replace() call strings."
            ),
        }

    programs, parse_error = _parse_programs(result)
    if programs is None:
        return {
            "value": 0.0,
            "message": (
                f"Could not parse program sequence from result: {parse_error}. "
                "Your answer must be a list of replace() calls, e.g. "
                "[\"replace('a', 'b')\", \"replace('cd', 'ef')\"]"
            ),
        }

    violations = _validate_programs(programs)
    if violations:
        return {
            "value": 0.0,
            "message": (
                "Program sequence violates task constraints — "
                + "; ".join(violations)
                + ". Constraints: 1<=len(A)<=3, 0<=len(B)<=3, max 5 programs total."
            ),
        }

    inputs: List[str] = entry.get("inputs", [])
    outputs: List[str] = entry.get("outputs", [])

    if not inputs or not outputs or len(inputs) != len(outputs):
        return {
            "value": 0.0,
            "message": "Entry is missing or has mismatched 'inputs'/'outputs' fields.",
        }

    # Apply programs in sequence to each input and compare with expected output.
    correct = 0
    mismatches: List[str] = []
    first_fail_trace: Optional[str] = None  # step-by-step trace for first failure

    for inp, expected in zip(inputs, outputs):
        actual = inp
        for pred, transform in programs:
            actual = actual.replace(pred, transform)
        if actual == expected:
            correct += 1
        else:
            if len(mismatches) < 3:
                mismatches.append(f"'{inp}' → '{actual}' (expected '{expected}')")
            # Build a step-by-step trace for the first failing input only
            if first_fail_trace is None:
                steps = [f"  start: '{inp}'"]
                cur = inp
                for pred, transform in programs:
                    nxt = cur.replace(pred, transform)
                    fired = " (no change)" if nxt == cur else ""
                    steps.append(f"  → replace('{pred}','{transform}'): '{nxt}'{fired}")
                    cur = nxt
                steps.append(f"  expected: '{expected}'")
                first_fail_trace = "\n".join(steps)

    score = correct / len(inputs)
    prog_strs = [f"replace('{p}', '{t}')" for p, t in programs]

    if score >= 1.0:
        return {"value": 1.0}

    msg = (
        f"Score={score:.3f}: {correct}/{len(inputs)} inputs mapped correctly. "
        f"Programs applied: {prog_strs}. "
        f"Mismatches: {mismatches}"
    )
    if first_fail_trace:
        msg += f"\nStep-by-step trace for first failing input:\n{first_fail_trace}"
    return {"value": score, "message": msg}
