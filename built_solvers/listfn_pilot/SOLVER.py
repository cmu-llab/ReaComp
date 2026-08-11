"""
Symbolic Solver for List Functions Task
========================================

Infers a list-to-list transformation program from input/output examples.

Algorithm: Search over a curated DSL of list transformation primitives,
scoring candidates against examples using the verifier.

Search Strategy:
  1. Generate level-1 (single primitive) programs - focused on common patterns
  2. Generate targeted level-2 compositions for the most useful combinations
  3. Generate specialized level-3 programs for complex patterns
  4. Score candidates and return the best scoring one
  5. Prefer simple, short programs when multiple are correct

DSL Primitives:
  - Slicing: first/last k, skip, drop, every-kth
  - Reversing: reverse
  - Sorting: sort asc/desc, top-k, bottom-k
  - Filtering: by value, by parity, by min/max/first/last
  - Deduplication: stable dedup
  - Structural: replace first/last, drop element at position k, rotate
  - Aggregation: count, length, sum, min, max as singleton
  - Arithmetic: add/subtract constant
  - Complex: range expansion, interleaving, cumulative ops
"""

import sys
sys.path.insert(0, '/workspace')
from rewards.list_functions import _compile_program, score_program


def solve_listfn(examples, K=10):
    """
    Solve a List Functions task by searching over a DSL of list transformations.
    
    Args:
        examples: list of (input_list, output_list) pairs
        K: number of top programs to return if none is fully correct
    
    Returns:
        dict with "success" (bool) and "program" (source string)
    """
    if not examples:
        return {"success": False, "program": "def program(xs): return xs"}

    inputs = [i for i, _ in examples]
    outputs = [o for _, o in examples]

    all_vals = set()
    for x in inputs + outputs:
        all_vals.update(x)
    max_len = max(len(x) for x in inputs) if inputs else 0
    unique_vals = sorted(all_vals)

    candidates = []
    # Generate candidates in priority order (simplest first)
    candidates.extend(_gen_slices(max_len))
    candidates.extend(_gen_reverses())
    candidates.extend(_gen_sorts())
    candidates.extend(_gen_filters(unique_vals))
    candidates.extend(_gen_dedup())
    candidates.extend(_gen_structural(max_len, unique_vals[:8]))
    candidates.extend(_gen_aggregation())
    candidates.extend(_gen_elementwise(unique_vals[:6]))
    candidates.extend(_gen_special(inputs, outputs))
    candidates.extend(_gen_level2(inputs, outputs, unique_vals, max_len))
    candidates.extend(_gen_level3(inputs, outputs))

    # Score candidates
    scored = []
    for code, desc in candidates:
        fn = _compile_program(code)
        if fn is None:
            continue
        try:
            score, _ = score_program(fn, inputs, outputs)
        except Exception:
            score = 0.0
        scored.append((score, code, desc))

    # Deduplicate by code
    seen = set()
    unique = []
    for score, code, desc in scored:
        if code not in seen:
            seen.add(code)
            unique.append((score, code, desc))

    # Sort by score desc, then length asc
    unique.sort(key=lambda x: (-x[0], len(x[1])))

    # Check for perfect solutions
    perfect = [(s, c, d) for s, c, d in unique if s >= 1.0]
    if perfect:
        best = min(perfect, key=lambda x: len(x[1]))
        return {"success": True, "program": best[1], "score": best[0]}

    # Return top-K
    if unique:
        s, c, d = unique[0]
        return {"success": False, "program": c, "score": s, "best_desc": d}

    return {"success": False, "program": "def program(xs): return xs"}


# ===== LEVEL 1: Single primitives =====

def _gen_slices(max_len):
    k_max = min(max_len + 1, 6)
    # Fixed slices
    yield "def program(xs):\n    return xs[:1]", "first 1"
    yield "def program(xs):\n    return xs[:2]", "first 2"
    yield "def program(xs):\n    return xs[:3]", "first 3"
    yield "def program(xs):\n    return xs[-1:]", "last 1"
    yield "def program(xs):\n    return xs[-2:]", "last 2"
    yield "def program(xs):\n    return xs[-3:]", "last 3"
    yield "def program(xs):\n    return xs[1:]", "skip 1"
    yield "def program(xs):\n    return xs[:-1]", "drop last 1"
    yield "def program(xs):\n    return xs[1:-1]", "skip 1, drop last 1"
    yield "def program(xs):\n    return xs[::2]", "every 2nd"
    yield "def program(xs):\n    return xs[1::2]", "every 2nd from 1"
    # Parameterized slices
    for k in range(1, k_max):
        yield (f"def program(xs):\n    return xs[:{k}]", f"first {k}")
        yield (f"def program(xs):\n    return xs[-{k}:]", f"last {k}")
        yield (f"def program(xs):\n    return xs[{k}:]", f"skip {k}")
        yield (f"def program(xs):\n    return xs[:-{k}]", f"drop last {k}")
    # Drop element at position k (use string concat to avoid f-string brace issues)
    for k in range(1, k_max):
        k_end = k + 1
        code = "def program(xs):\n    return xs[:" + str(k) + "] + xs[" + str(k_end) + ":]"
        yield (code, f"drop idx {k}")


def _gen_reverses():
    yield "def program(xs):\n    return list(reversed(xs))", "reverse"
    yield "def program(xs):\n    return xs[::-1]", "reverse slice"


def _gen_sorts():
    k_max = 6
    yield "def program(xs):\n    return sorted(xs)", "sort asc"
    yield "def program(xs):\n    return sorted(xs, reverse=True)", "sort desc"
    for k in range(1, k_max):
        yield (f"def program(xs):\n    return sorted(xs)[:{k}]", f"bottom {k}")
        yield (f"def program(xs):\n    return sorted(xs)[-{k}:]", f"top {k}")
        yield (f"def program(xs):\n    return sorted(xs, reverse=True)[:{k}]", f"top {k} desc")


def _gen_filters(unique_vals):
    # By value from data
    for c in unique_vals[:10]:
        yield (f"def program(xs):\n    return [x for x in xs if x > {c}]", f"x>{c}")
        yield (f"def program(xs):\n    return [x for x in xs if x >= {c}]", f"x>={c}")
        yield (f"def program(xs):\n    return [x for x in xs if x < {c}]", f"x<{c}")
        yield (f"def program(xs):\n    return [x for x in xs if x <= {c}]", f"x<={c}")
        yield (f"def program(xs):\n    return [x for x in xs if x != {c}]", f"x!={c}")
    # Common thresholds
    for c in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
        if c not in unique_vals:
            yield (f"def program(xs):\n    return [x for x in xs if x >= {c}]", f"x>={c}")
            yield (f"def program(xs):\n    return [x for x in xs if x > {c}]", f"x>{c}")
            yield (f"def program(xs):\n    return [x for x in xs if x <= {c}]", f"x<={c}")
    # Parity
    yield "def program(xs):\n    return [x for x in xs if x % 2 == 0]", "even"
    yield "def program(xs):\n    return [x for x in xs if x % 2 == 1]", "odd"
    # Compare to aggregates
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x > min(xs)]", "x>min")
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x < max(xs)]", "x<max")
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x >= min(xs)]", "x>=min")
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x <= max(xs)]", "x<=max")
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x != xs[0]]", "x!=first")
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x != xs[-1]]", "x!=last")
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x > xs[0]]", "x>first")
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x >= xs[0]]", "x>=first")
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x > xs[-1]]", "x>last")
    yield ("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x >= xs[-1]]", "x>=last")


def _gen_dedup():
    yield ("def program(xs):\n    seen = set()\n    result = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result", "dedup")
    yield ("def program(xs):\n    return list(dict.fromkeys(xs))", "dedup")


def _gen_elementwise(const_vals):
    for c in const_vals:
        yield (f"def program(xs):\n    return [x + {c} for x in xs]", f"add {c}")
        if c > 0:
            yield (f"def program(xs):\n    return [x - {c} for x in xs if x > {c}]", f"sub {c}")


def _gen_structural(max_len, const_vals):
    # Replace first/last
    yield ("def program(xs):\n    if not xs: return []\n    return [xs[-1]] + xs[1:]", "replace first w/last")
    yield ("def program(xs):\n    if not xs: return []\n    return xs[:-1] + [xs[0]]", "replace last w/first")
    # Rotate
    yield ("def program(xs):\n    if not xs: return []\n    return [xs[-1]] + xs[:-1]", "rotate right 1")
    yield ("def program(xs):\n    if not xs: return []\n    return xs[1:] + [xs[0]]", "rotate left 1")
    # Drop operations
    yield "def program(xs):\n    return xs[1:]", "drop first"
    yield "def program(xs):\n    return xs[:-1]", "drop last"
    yield "def program(xs):\n    return xs[1:-1]", "drop first+last"
    for k in range(1, min(max_len + 1, 5)):
        k_end = k + 1
        code = "def program(xs):\n    return xs[:" + str(k) + "] + xs[" + str(k_end) + ":]"
        yield (code, f"drop idx {k}")
    # Append/prepend
    for c in const_vals:
        yield (f"def program(xs):\n    return xs + [{c}]", f"append {c}")
        yield (f"def program(xs):\n    return [{c}] + xs", f"prepend {c}")
        yield (f"def program(xs):\n    return [{c}] + xs[1:]", f"replace first={c}")
        yield (f"def program(xs):\n    return xs[:-1] + [{c}]", f"replace last={c}")
    # Replicate
    yield ("def program(xs):\n    if not xs: return []\n    return [xs[0]] * len(xs)", "replicate first")
    yield ("def program(xs):\n    if not xs: return []\n    return [xs[-1]] * len(xs)", "replicate last")


def _gen_aggregation():
    yield ("def program(xs):\n    if not xs: return []\n    return [len(xs)]", "length")
    yield ("def program(xs):\n    if not xs: return []\n    return [sum(xs)]", "sum")
    yield ("def program(xs):\n    if not xs: return []\n    return [min(xs)]", "min")
    yield ("def program(xs):\n    if not xs: return []\n    return [max(xs)]", "max")
    yield ("def program(xs):\n    if not xs: return []\n    return [xs.count(xs[0])]", "count first")
    yield ("def program(xs):\n    if not xs: return []\n    return [xs.count(xs[0]) - 1]", "count first -1")
    yield ("def program(xs):\n    if not xs: return []\n    return [xs.count(xs[-1])]", "count last")
    yield ("def program(xs):\n    if not xs: return []\n    return [xs.count(xs[-1]) - 1]", "count last -1")


def _gen_special(inputs, outputs):
    # Detect range expansion from first element
    expand_ok = True
    for inp, out in zip(inputs, outputs):
        if not inp:
            if len(out) <= 1:
                continue
            expand_ok = False
            break
        a = inp[0]
        exp = [a]
        for x in inp[1:]:
            if x > a:
                exp.extend(range(a, x))
            exp.append(x)
        if list(out) != exp:
            expand_ok = False
            break
    if expand_ok and any(len(i) > 1 for i, _ in zip(inputs, outputs)):
        yield ("def program(xs):\n    if not xs: return []\n    a = xs[0]\n    res = [a]\n    for x in xs[1:]:\n        if x > a:\n            res.extend(range(a, x))\n        res.append(x)\n    return res", "expand from first")

    # Detect interleaving rank with sorted
    interleave_ok = True
    for inp, out in zip(inputs, outputs):
        if not inp:
            interleave_ok = False
            break
        s = sorted(inp)
        exp = []
        for i, v in enumerate(s, 1):
            exp.extend([i, v])
        if list(out) != exp:
            interleave_ok = False
            break
    if interleave_ok:
        yield ("def program(xs):\n    s = sorted(xs)\n    result = []\n    for i, v in enumerate(s, 1):\n        result.extend([i, v])\n    return result", "interleave rank+sorted")

    # Detect even position keeping
    even_ok = all(list(out) == [inp[i] for i in range(0, len(inp), 2)] for inp, out in zip(inputs, outputs))
    if even_ok:
        yield ("def program(xs):\n    return [xs[i] for i in range(0, len(xs), 2)]", "even positions")

    odd_ok = all(list(out) == [inp[i] for i in range(1, len(inp), 2)] for inp, out in zip(inputs, outputs))
    if odd_ok and any(len(inp) > 1 for inp, _ in zip(inputs, outputs)):
        yield ("def program(xs):\n    return [xs[i] for i in range(1, len(xs), 2)]", "odd positions")

    # Detect sort+dedup
    sd_ok = True
    for inp, out in zip(inputs, outputs):
        deduped = []
        seen = set()
        for x in sorted(inp):
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        if list(out) != deduped:
            sd_ok = False
            break
    if sd_ok:
        yield "def program(xs):\n    return sorted(set(xs))", "sort+dedup"


# ===== LEVEL 2: Compositions =====

def _gen_level2(inputs, outputs, unique_vals, max_len):
    candidates = []
    bases = []
    for code, desc in _gen_slices(max_len):
        bases.append((code, desc, "slice"))
    for code, desc in _gen_sorts():
        bases.append((code, desc, "sort"))
    for code, desc in _gen_filters(unique_vals):
        bases.append((code, desc, "filter"))
    for code, desc in _gen_dedup():
        bases.append((code, desc, "dedup"))
    for code, desc in _gen_structural(max_len, unique_vals[:3]):
        bases.append((code, desc, "struct"))

    for code1, desc1, t1 in bases:
        if "return " not in code1:
            continue
        expr1 = code1.split("return ", 1)[1].strip()
        for code2, desc2, t2 in bases:
            if t2 == "filter" and t1 in ("slice", "sort", "struct"):
                expr2 = "[x for x in tmp if x > 0]"
                candidates.append((f"def program(xs):\n    tmp = {expr1}\n    return {expr2}",
                                   f"{desc1}->{desc2}"))
            elif t2 == "sort" and t1 in ("slice", "filter"):
                candidates.append((f"def program(xs):\n    tmp = {expr1}\n    return sorted(tmp)",
                                   f"{desc1}->sort"))
            elif t2 == "dedup" and t1 in ("slice", "sort", "filter"):
                candidates.append((f"def program(xs):\n    tmp = {expr1}\n    return sorted(set(tmp))",
                                   f"{desc1}->dedup"))
    return candidates


# ===== LEVEL 3: Complex patterns =====

def _gen_level3(inputs, outputs):
    candidates = []
    candidates.append(("def program(xs):\n    if not xs: return []\n    res = [xs[0]]\n    for x in xs[1:]:\n        res.append(max(res[-1], x))\n    return res", "cum max"))
    candidates.append(("def program(xs):\n    if not xs: return []\n    res = [xs[0]]\n    for x in xs[1:]:\n        res.append(min(res[-1], x))\n    return res", "cum min"))
    candidates.append(("def program(xs):\n    seen = set()\n    result = []\n    for x in sorted(xs):\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result", "sort dedup"))
    candidates.append(("def program(xs):\n    return [sum(1 for y in xs[i+1:] if y > x) for i, x in enumerate(xs)]", "count larger after"))
    candidates.append(("def program(xs):\n    if not xs: return []\n    res = [xs[0]]\n    for i in range(1, len(xs)):\n        a, b = xs[i-1], xs[i]\n        if b > a:\n            res.extend(range(a, b))\n        res.append(b)\n    return res", "expand consecutive"))
    candidates.append(("def program(xs):\n    if not xs: return []\n    s = sorted(xs)\n    mid = s[len(s)//2]\n    return [mid] + xs[1:]", "replace first median"))
    candidates.append(("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x != xs[0]]", "drop equals first"))
    candidates.append(("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x != xs[-1]]", "drop equals last"))
    return candidates
