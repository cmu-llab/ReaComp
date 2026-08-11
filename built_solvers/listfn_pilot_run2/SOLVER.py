"""
SOLVER_LISTFN.py
Symbolic Program Synthesizer for List-to-List Transformation Tasks.

This solver searches over compositions of list-transformation DSL primitives
(slicing, filtering, sorting, reversing, deduplication, arithmetic, indexing,
interleaving, concatenation with constants, range generation, etc.) to find
a program that reproduces all given input/output examples.
"""

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Verifier import
# ---------------------------------------------------------------------------
try:
    from rewards.list_functions import score_program
except ImportError:
    # Minimal fallback scorer
    def score_program(fn: Callable, inputs: List[List[int]],
                      outputs: List[List[int]]) -> Tuple[float, List[str]]:
        correct = 0
        mismatches = []
        for inp, expected in zip(inputs, outputs):
            try:
                got = fn(list(inp))
                if not isinstance(got, (list, tuple)):
                    got = list(got) if got is not None else None
            except Exception as e:
                got = None
                if len(mismatches) < 3:
                    mismatches.append(f"{inp} -> ERROR: {e}")
                continue
            if isinstance(got, (list, tuple)) and got == list(expected):
                correct += 1
            else:
                if len(mismatches) < 3:
                    mismatches.append(f"{inp} -> {got} (expected {expected})")
        return correct / max(len(inputs), 1), mismatches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _unique_list(lst):
    """Remove consecutive duplicates preserving order."""
    if not lst:
        return []
    out = [lst[0]]
    for x in lst[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def _all_thresholds(inputs):
    """Collect all element values from inputs as candidate thresholds."""
    vals = set()
    for xs in inputs:
        vals.update(xs)
    return sorted(vals)


def _eval_program(src, inputs, outputs):
    """Compile and score a program source string against examples."""
    ns = {}
    try:
        exec(src, {"__builtins__": __builtins__}, ns)
    except SyntaxError:
        return 0.0, ["syntax error"]
    fn = ns.get("program") or ns.get("solve") or ns.get("f") or ns.get("transform")
    if fn is None:
        return 0.0, ["no function found"]
    return score_program(fn, inputs, outputs)


# ---------------------------------------------------------------------------
# Candidate generators
# ---------------------------------------------------------------------------

def _candidate_slices(inputs):
    """Slicing and indexing candidates."""
    cands = []
    lengths = set(len(xs) for xs in inputs)
    max_len = max(lengths) if lengths else 10
    thresholds = list(range(0, max(max_len + 1, 5)))
    for N in thresholds:
        if N == 0:
            continue
        cands.append((f"def program(xs):\n    return xs[:{N}]\n", f"first {N}"))
        cands.append((f"def program(xs):\n    return xs[-{N}:]\n", f"last {N}"))
        cands.append((f"def program(xs):\n    return xs[{N}:]\n", f"from index {N}"))
        for M in thresholds:
            if M > 0:
                cands.append((f"def program(xs):\n    return xs[{N}:-{M}]\n",
                              f"xs[{N}:-{M}]"))
        for k in range(2, max_len + 1):
            cands.append((f"def program(xs):\n    return xs[::{k}]\n",
                          f"every {k}th"))
            for offset in range(1, k):
                cands.append((f"def program(xs):\n    return xs[{offset}::{k}]\n",
                              f"xs[{offset}::{k}]"))
    # Common shortcuts
    for N in [2, 3]:
        cands.append((f"def program(xs):\n    return xs[:{N}]\n", f"first {N}"))
        cands.append((f"def program(xs):\n    return xs[-{N}:]\n", f"last {N}"))
    return cands


def _candidate_filters(inputs):
    """Filter-by-predicate candidates."""
    cands = []
    thresholds = _all_thresholds(inputs)
    for t in [0, 1, 2, 3, 5, 10, 20, 50, 100]:
        if t not in thresholds:
            thresholds.append(t)
    thresholds = sorted(set(thresholds))
    for t in thresholds:
        if t > 0:
            cands.append((f"def program(xs):\n    return [x for x in xs if x >= {t}]\n",
                          f"filter x >= {t}"))
        cands.append((f"def program(xs):\n    return [x for x in xs if x > {t}]\n",
                      f"filter x > {t}"))
        if t > 0:
            cands.append((f"def program(xs):\n    return [x for x in xs if x <= {t}]\n",
                          f"filter x <= {t}"))
        cands.append((f"def program(xs):\n    return [x for x in xs if x < {t}]\n",
                      f"filter x < {t}"))
        cands.append((f"def program(xs):\n    return [x for x in xs if x == {t}]\n",
                      f"filter x == {t}"))
    for k in range(2, 6):
        cands.append((f"def program(xs):\n    return [x for x in xs if x % {k} != 0]\n",
                      f"filter x % {k} != 0"))
        cands.append((f"def program(xs):\n    return [x for x in xs if x % {k} == 0]\n",
                      f"filter x % {k} == 0"))
    cands.append(("def program(xs):\n    return [x for x in xs if x != 0]\n",
                  "filter x != 0"))
    # Unique order-preserving
    cands.append((
        "def program(xs):\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for x in xs:\n"
        "        if x not in seen:\n"
        "            seen.add(x)\n"
        "            out.append(x)\n"
        "    return out\n",
        "unique order-preserving"))
    # Consecutive duplicates
    cands.append(("def program(xs):\n    return _unique_list(xs)\n",
                  "remove consecutive duplicates"))
    return cands


def _candidate_sorting(inputs):
    """Sorting candidates."""
    cands = []
    lengths = set(len(xs) for xs in inputs)
    max_len = max(lengths) if lengths else 10
    for N in range(1, min(max_len + 1, 10)):
        cands.append((f"def program(xs):\n    return sorted(xs)[:{N}]\n",
                      f"smallest {N} sorted"))
        cands.append((f"def program(xs):\n    return sorted(xs)[-{N}:]\n",
                      f"largest {N} sorted"))
    cands.append(("def program(xs):\n    return sorted(xs)\n", "sorted"))
    cands.append(("def program(xs):\n    return list(reversed(xs))\n", "reversed"))
    cands.append(("def program(xs):\n    return sorted(xs, reverse=True)\n",
                  "sorted desc"))
    cands.append(("def program(xs):\n    return sorted(xs)[::-1]\n",
                  "sorted reversed"))
    return cands


def _candidate_positional(inputs):
    """Positional manipulation: delete, replace, rotate, etc."""
    cands = []
    lengths = set(len(xs) for xs in inputs)
    max_len = max(lengths) if lengths else 10
    thresholds = list(range(0, min(max_len + 1, 8)))
    # Delete at position N
    for N in thresholds:
        if N == 0:
            continue
        cands.append((f"def program(xs):\n    return xs[:{N}] + xs[{N+1}:]\n",
                      f"delete index {N}"))
    # Replace first with last
    cands.append(("def program(xs):\n    if not xs: return []\n    return [xs[-1]] + xs[1:]\n",
                  "replace first with last"))
    # Replace first with constant
    for c in range(10):
        cands.append((f"def program(xs):\n    if not xs: return []\n    return [{c}] + xs[1:]\n",
                      f"replace first with {c}"))
    # Replace index N with last
    for N in thresholds:
        if N == 0:
            continue
        cands.append((
            f"def program(xs):\n"
            f"    if not xs: return []\n"
            f"    return xs[:{N}] + [xs[-1]] + xs[{N+1}:]\n",
            f"replace index {N} with last"))
    # Move first to end
    cands.append((
        "def program(xs):\n"
        "    if len(xs) <= 1: return xs\n"
        "    return xs[1:] + [xs[0]]\n",
        "move first to end"))
    # Swap first and last
    cands.append((
        "def program(xs):\n"
        "    if len(xs) <= 1: return xs\n"
        "    return [xs[-1]] + xs[1:-1] + [xs[0]]\n",
        "swap first and last"))
    # Rotate right by N
    for N in range(1, 4):
        cands.append((
            f"def program(xs):\n"
            f"    if len(xs) <= {N}: return xs\n"
            f"    return xs[-{N}:] + xs[:-{N}]\n",
            f"rotate right by {N}"))
    # Rotate left by N
    for N in range(1, 4):
        cands.append((
            f"def program(xs):\n"
            f"    if len(xs) <= {N}: return xs\n"
            f"    return xs[{N}:] + xs[:{N}]\n",
            f"rotate left by {N}"))
    return cands


def _candidate_append(inputs):
    """Append, prepend, index-generation candidates."""
    cands = []
    for c in range(10):
        cands.append((f"def program(xs):\n    return xs + [{c}]\n",
                      f"append {c}"))
        cands.append((f"def program(xs):\n    return [{c}] + xs\n",
                      f"prepend {c}"))
    cands.append(("def program(xs):\n    if not xs: return xs\n    return xs + [xs[-1]]\n",
                  "append last"))
    cands.append(("def program(xs):\n    if not xs: return xs\n    return [xs[0]] + xs\n",
                  "prepend first"))
    cands.append(("def program(xs):\n    return xs + xs\n", "double"))
    cands.append(("def program(xs):\n    return [i for i in range(len(xs))]\n",
                  "indices"))
    cands.append(("def program(xs):\n"
                  "    return [len(xs)-1-i for i in range(len(xs))]\n",
                  "reversed indices"))
    return cands


def _candidate_arithmetic(inputs):
    """Element-wise arithmetic candidates."""
    cands = []
    cands.append(("def program(xs):\n    return [x * 2 for x in xs]\n",
                  "double each"))
    cands.append(("def program(xs):\n    return [x * 3 for x in xs]\n",
                  "triple each"))
    cands.append(("def program(xs):\n    return [x // 2 for x in xs]\n",
                  "halve each"))
    for c in range(1, 6):
        cands.append((f"def program(xs):\n    return [x + {c} for x in xs]\n",
                      f"add {c} to each"))
    cands.append(("def program(xs):\n    return [abs(x) for x in xs]\n",
                  "abs"))
    cands.append(("def program(xs):\n    return [int(x**0.5) for x in xs]\n",
                  "int sqrt"))
    cands.append(("def program(xs):\n    return [sum(xs)]\n", "sum"))
    cands.append(("def program(xs):\n    return [min(xs)] if xs else []\n", "min"))
    cands.append(("def program(xs):\n    return [max(xs)] if xs else []\n", "max"))
    cands.append(("def program(xs):\n    return [len(xs)]\n", "length"))
    cands.append(("def program(xs):\n"
                  "    acc = 0\n"
                  "    return [acc := acc + x for x in xs]\n", "cumsum"))
    cands.append(("def program(xs):\n"
                  "    return [xs[i+1] - xs[i] for i in range(len(xs)-1)]\n",
                  "adjacent diffs"))
    return cands


def _candidate_special(inputs, outputs):
    """Pattern-specific candidates inferred from I/O analysis."""
    cands = []

    # --- Single-element outputs ---
    if all(len(ys) == 1 for ys in outputs) and outputs:
        cands.append((
            "def program(xs):\n"
            "    if not xs: return []\n"
            "    return [xs.count(xs[0]) - 1]\n",
            "count first element - 1"))
        cands.append((
            "def program(xs):\n"
            "    if not xs: return []\n"
            "    return [xs.count(xs[0])]\n",
            "count first element"))
        cands.append((
            "def program(xs):\n"
            "    if not xs: return []\n"
            "    return [xs.count(xs[-1])]\n",
            "count last element"))

    # --- Remove first element ---
    for inp, out in zip(inputs, outputs):
        if out == inp[1:]:
            cands.append(("def program(xs):\n    return xs[1:]\n", "remove first"))
            break

    # --- Remove last element ---
    for inp, out in zip(inputs, outputs):
        if out == inp[:-1]:
            cands.append(("def program(xs):\n    return xs[:-1]\n", "remove last"))
            break

    # --- Reverse ---
    for inp, out in zip(inputs, outputs):
        if out == inp[::-1]:
            cands.append(("def program(xs):\n    return xs[::-1]\n", "reverse"))
            break

    # --- Interleave indices (1-based) with sorted elements ---
    all_match = True
    for inp, out in zip(inputs, outputs):
        if not inp:
            continue
        expected = []
        for i, v in enumerate(sorted(inp)):
            expected.extend([i + 1, v])
        if out != expected:
            all_match = False
            break
    if all_match and any(inp for inp in inputs):
        cands.append((
            "def program(xs):\n"
            "    if not xs: return []\n"
            "    result = []\n"
            "    for i, v in enumerate(sorted(xs)):\n"
            "        result.extend([i + 1, v])\n"
            "    return result\n",
            "interleave indices with sorted"))

    # --- Range interpolation: for each x in xs[1:],
    #    if x > xs[0], insert range(xs[0], x), then append x ---
    all_match = True
    for inp, out in zip(inputs, outputs):
        if not inp:
            expected = []
        else:
            a = inp[0]
            expected = [a]
            for x in inp[1:]:
                if x > a:
                    expected.extend(range(a, x))
                expected.append(x)
        if out != expected:
            all_match = False
            break
    if all_match and any(len(inp) > 1 for inp in inputs):
        cands.append((
            "def program(xs):\n"
            "    if not xs: return []\n"
            "    a = xs[0]\n"
            "    res = [a]\n"
            "    for x in xs[1:]:\n"
            "        if x > a:\n"
            "            res.extend(range(a, x))\n"
            "        res.append(x)\n"
            "    return res\n",
            "range interpolation"))

    # --- Filter multiples of k ---
    for k in range(2, 6):
        all_match = True
        for inp, out in zip(inputs, outputs):
            expected = [x for x in inp if x % k == 0]
            if out != expected:
                all_match = False
                break
        if all_match:
            cands.append((
                f"def program(xs):\n"
                f"    return [x for x in xs if x % {k} == 0]\n",
                f"filter multiples of {k}"))

    # --- Even indices ---
    all_match = True
    for inp, out in zip(inputs, outputs):
        if out != inp[::2]:
            all_match = False
            break
    if all_match and any(inp for inp in inputs):
        cands.append(("def program(xs):\n    return xs[::2]\n", "even indices"))

    # --- Odd indices ---
    all_match = True
    for inp, out in zip(inputs, outputs):
        if out != inp[1::2]:
            all_match = False
            break
    if all_match and any(inp for inp in inputs):
        cands.append(("def program(xs):\n    return xs[1::2]\n", "odd indices"))

    # --- Slice by length formulas ---
    for name, n_fn in [
        ("len//2", lambda n: n // 2),
        ("len-1", lambda n: max(n - 1, 0)),
        ("len-2", lambda n: max(n - 2, 0)),
        ("len-3", lambda n: max(n - 3, 0)),
        ("(len+1)//2", lambda n: (n + 1) // 2),
    ]:
        all_match = True
        for inp, out in zip(inputs, outputs):
            if out != inp[:n_fn(len(inp))]:
                all_match = False
                break
        if all_match and any(inp for inp in inputs):
            if name == "len//2":
                src = "def program(xs):\n    return xs[:len(xs)//2]\n"
            elif name == "len-1":
                src = "def program(xs):\n    return xs[:len(xs)-1]\n"
            elif name == "len-2":
                src = "def program(xs):\n    return xs[:len(xs)-2]\n"
            elif name == "len-3":
                src = "def program(xs):\n    return xs[:len(xs)-3]\n"
            elif name == "(len+1)//2":
                src = "def program(xs):\n    return xs[:len(xs)//2+1]\n"
            cands.append((src, f"first {name}"))

    return cands


def _candidate_compositions(inputs, outputs):
    """Compositions of simpler operations."""
    cands = []
    thresholds = _all_thresholds(inputs)
    # Filter then sort
    for t in thresholds:
        if t > 0:
            cands.append((
                f"def program(xs):\n"
                f"    return sorted([x for x in xs if x >= {t}])\n",
                f"filter x>={t} then sort"))
    # Slice then sort
    for N in range(1, 5):
        cands.append((f"def program(xs):\n    return sorted(xs[:{N}])\n",
                      f"first {N} sorted"))
        cands.append((f"def program(xs):\n    return sorted(xs[-{N}:])\n",
                      f"last {N} sorted"))
    # Sort then every other
    cands.append(("def program(xs):\n    s = sorted(xs)\n    return s[::2]\n",
                  "sorted every other"))
    cands.append(("def program(xs):\n    s = sorted(xs)\n    return s[1::2]\n",
                  "sorted odd indices"))
    # Unique then sorted
    cands.append((
        "def program(xs):\n"
        "    seen = set()\n"
        "    unique = [x for x in xs if not (x in seen or seen.add(x))]\n"
        "    return sorted(unique)\n",
        "unique then sorted"))
    # Cumulative max
    cands.append((
        "def program(xs):\n"
        "    acc = 0\n"
        "    return [acc := max(acc, x) for x in xs]\n",
        "cumulative max"))
    return cands


# ---------------------------------------------------------------------------
# Solver entry point
# ---------------------------------------------------------------------------

def solve_listfn(examples, K=5):
    """
    Solve a List Functions task by searching over composed list-transform DSL programs.

    Parameters
    ----------
    examples : list of (input_list, output_list)
        Each pair is a list of ints.
    K : int
        Number of top candidates to return if no perfect match found.

    Returns
    -------
    dict with at least:
        - "success": bool
        - "program": source string defining program(xs) or a callable
        - "score": float (fraction of examples matched)
    """
    if not examples:
        return {
            "success": False,
            "program": "def program(xs):\n    return []\n",
            "score": 0.0,
        }

    inputs = [list(e[0]) for e in examples]
    outputs = [list(e[1]) for e in examples]

    # --- Phase 1: generate all candidates ---
    all_candidates = []
    seen_srcs = set()

    # Single-arg generators
    for gen_fn in [_candidate_slices, _candidate_filters, _candidate_sorting,
                   _candidate_positional, _candidate_append, _candidate_arithmetic]:
        try:
            for src, desc in gen_fn(inputs):
                if src not in seen_srcs:
                    seen_srcs.add(src)
                    all_candidates.append((src, desc))
        except Exception:
            pass

    # Two-arg generators
    for gen_fn in [_candidate_special, _candidate_compositions]:
        try:
            for src, desc in gen_fn(inputs, outputs):
                if src not in seen_srcs:
                    seen_srcs.add(src)
                    all_candidates.append((src, desc))
        except Exception:
            pass

    # --- Phase 2: score every candidate ---
    scored = []
    for src, desc in all_candidates:
        try:
            score, mismatches = _eval_program(src, inputs, outputs)
            scored.append((score, src, desc, mismatches))
        except Exception:
            continue

    # Sort: highest score first, then shortest source (simplicity tie-break)
    scored.sort(key=lambda x: (-x[0], len(x[1])))

    # --- Phase 3: return best program ---
    for score, src, desc, mismatches in scored:
        if score >= 1.0:
            return {
                "success": True,
                "program": src,
                "score": 1.0,
                "description": desc,
            }

    # No perfect match — return top-K
    top_k = scored[:K]
    best_src = top_k[0][1] if top_k else "def program(xs):\n    return []\n"
    candidates_info = [
        {
            "description": desc,
            "score": score,
            "program": src,
            "mismatches": mismatches,
        }
        for score, src, desc, mismatches in top_k
    ]

    return {
        "success": False,
        "program": best_src,
        "score": top_k[0][0] if top_k else 0.0,
        "candidates": candidates_info,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from rewards.list_functions import reward

    with open("/workspace/DEMOS.json") as f:
        demos = json.load(f)

    for i, demo in enumerate(demos):
        result = solve_listfn(list(zip(demo["input_examples"], demo["output_examples"])))
        r = reward(result, True, {
            "inputs": demo["input_examples"],
            "outputs": demo["output_examples"],
        })
        status = "PASS" if r["value"] == 1.0 else "FAIL"
        print(f"Demo {i} ({demo['task_id']}): {status} score={result.get('score', 0):.2f} reward={r['value']:.2f}")
        if r["value"] < 1.0:
            print(f"  program: {result['program'][:80]}")
            if result.get("candidates"):
                for c in result["candidates"][:3]:
                    print(f"    {c['description']}: score={c['score']}")
