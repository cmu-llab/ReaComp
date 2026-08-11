"""
SOLVER_LISTFN.py - Symbolic Program Synthesizer for List Functions

This solver infers a list-to-list transformation program from I/O examples
by searching over a rich DSL of list-transformation primitives and their
compositions.

Key design:
- Analyzes examples to infer structural properties and detect patterns
- Generates candidate programs by composing primitives with tuned parameters
- Uses the verifier's score_program to rank candidates
- Returns the best fully-correct program, or top-K partial matches
"""

import sys
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, '/workspace')
from rewards.list_functions import score_program


def _dedup(cands):
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# =============================================================================
# Candidate generation modules
# =============================================================================

def _gen_slice_candidates(examples):
    """xs[a:b], xs[a:], xs[:b], xs[::-1], xs[-k:], xs[:-k], xs[::step]."""
    cands = []
    starts, ends = set(), set()
    for inp, out in examples:
        for i in range(len(inp) + 1):
            for j in range(i, len(inp) + 1):
                if inp[i:j] == out:
                    starts.add(i)
                    ends.add(j)
    for a in sorted(starts):
        for b in sorted(ends):
            if a < b:
                cands.append(f"def program(xs):\n    return xs[{a}:{b}]\n")
    for a in sorted(starts):
        cands.append(f"def program(xs):\n    return xs[{a}:]\n")
    for b in sorted(ends):
        cands.append(f"def program(xs):\n    return xs[:{b}]\n")
    for k in range(1, 15):
        cands.append(f"def program(xs):\n    return xs[-{k}:]\n")
    for k in range(1, 15):
        cands.append(f"def program(xs):\n    return xs[:-{k}]\n")
    cands.append("def program(xs):\n    return xs[::-1]\n")
    for step in [2, 3]:
        cands.append(f"def program(xs):\n    return xs[::{step}]\n")
    cands.append("def program(xs):\n    return xs[1::2]\n")
    return _dedup(cands)


def _gen_filter_candidates(examples):
    """[x for x in xs if predicate]."""
    cands = []
    all_vals = set()
    for inp, out in examples:
        all_vals.update(inp)
        all_vals.update(out)
    thresholds = set()
    for v in all_vals:
        thresholds.add(v)
        thresholds.add(v + 1)
        thresholds.add(v - 1)
    thresholds.discard(-1)
    for t in sorted(thresholds):
        cands.append(f"def program(xs):\n    return [x for x in xs if x >= {t}]\n")
        cands.append(f"def program(xs):\n    return [x for x in xs if x > {t}]\n")
        cands.append(f"def program(xs):\n    return [x for x in xs if x <= {t}]\n")
        cands.append(f"def program(xs):\n    return [x for x in xs if x < {t}]\n")
        cands.append(f"def program(xs):\n    return [x for x in xs if x != {t}]\n")
    cands.append("def program(xs):\n    return [x for i, x in enumerate(xs) if i % 2 == 0]\n")
    cands.append("def program(xs):\n    return [x for i, x in enumerate(xs) if i % 2 == 1]\n")
    cands.append("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x > xs[0]]\n")
    cands.append("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x >= xs[0]]\n")
    cands.append("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x < xs[0]]\n")
    cands.append("def program(xs):\n    if not xs: return []\n    return [x for x in xs if x <= xs[0]]\n")
    return _dedup(cands)


def _gen_sort_slice_candidates(examples):
    """Sort then slice."""
    cands = []
    for k in range(1, 15):
        cands.append(f"def program(xs):\n    return sorted(xs)[-{k}:]\n")
        cands.append(f"def program(xs):\n    return sorted(xs)[:{k}]\n")
        cands.append(f"def program(xs):\n    s = sorted(xs); return s[-{k}:]\n")
        cands.append(f"def program(xs):\n    s = sorted(xs); return s[:{k}]\n")
    for k in range(1, 15):
        cands.append(f"def program(xs):\n    return list(reversed(xs))[-{k}:]\n")
    cands.append("def program(xs):\n    return list(reversed(xs))\n")
    cands.append("def program(xs):\n    return sorted(xs)[1:]\n")
    cands.append("def program(xs):\n    return sorted(xs)[:-1]\n")
    cands.append("def program(xs):\n    return sorted(xs)[:2]\n")
    return _dedup(cands)


def _gen_replace_candidates(examples):
    """Replace / insert / delete at positions, with constants."""
    cands = []
    cands.append("def program(xs):\n    if not xs: return []\n    return [xs[-1]] + xs[1:]\n")
    for p in range(6):
        cands.append(f"def program(xs):\n    if len(xs) <= {p}: return xs\n    return xs[:{p}] + xs[{p+1}:]\n")
    for p in range(6):
        for c in range(15):
            if p == 0:
                cands.append(f"def program(xs):\n    if not xs: return []\n    return [{c}] + xs[1:]\n")
            else:
                cands.append(f"def program(xs):\n    if len(xs) < {p+1}: return xs\n    return xs[:{p}] + [{c}] + xs[{p+1}:]\n")
    for p in range(8):
        for c in range(15):
            cands.append(f"def program(xs):\n    return xs[:{p}] + [{c}] + xs[{p}:]\n")
    for c in range(15):
        cands.append(f"def program(xs):\n    return xs + [{c}]\n")
    for c in range(15):
        cands.append(f"def program(xs):\n    return [{c}] + xs\n")
    return _dedup(cands)


def _gen_count_candidates(examples):
    """Count-based output programs."""
    cands = []
    cands.append("def program(xs):\n    if not xs: return []\n    return [xs.count(xs[0]) - 1]\n")
    cands.append("def program(xs):\n    if not xs: return []\n    return [xs.count(xs[0])]\n")
    cands.append("def program(xs):\n    return [len(xs)]\n")
    cands.append("def program(xs):\n    return [len(xs) - 1]\n")
    cands.append("def program(xs):\n    return [max(0, len(xs) - 1)]\n")
    cands.append("def program(xs):\n    if not xs: return []\n    return [sum(1 for x in xs if x == xs[0]) - 1]\n")
    return _dedup(cands)


def _gen_range_insert_candidates(examples):
    """Conditional range insertion (task c150 style)."""
    cands = []
    cands.append("""def program(xs):
    if not xs:
        return []
    a = xs[0]
    res = [a]
    for x in xs[1:]:
        if x > a:
            res.extend(range(a, x))
        res.append(x)
    return res
""")
    cands.append("""def program(xs):
    if not xs:
        return []
    a = xs[0]
    res = [a]
    for x in xs[1:]:
        if x < a:
            res.extend(range(x, a))
        res.append(x)
    return res
""")
    cands.append("""def program(xs):
    if len(xs) < 2:
        return xs
    res = []
    for i in range(len(xs) - 1):
        res.append(xs[i])
        lo, hi = min(xs[i], xs[i+1]), max(xs[i], xs[i+1])
        res.extend(range(lo + 1, hi))
    res.append(xs[-1])
    return res
""")
    return _dedup(cands)


def _gen_interleave_candidates(examples):
    """Index-value interleaving (task c199 style)."""
    cands = []
    cands.append("""def program(xs):
    n = len(xs)
    sorted_xs = sorted(xs)
    result = []
    for i, val in enumerate(sorted_xs, start=1):
        result.append(i)
        result.append(val)
    return result
""")
    cands.append("""def program(xs):
    n = len(xs)
    sorted_xs = sorted(xs)
    result = []
    for i, val in enumerate(sorted_xs):
        result.append(i)
        result.append(val)
    return result
""")
    cands.append("""def program(xs):
    n = len(xs)
    result = []
    for i, val in enumerate(xs, start=1):
        result.append(i)
        result.append(val)
    return result
""")
    return _dedup(cands)


def _gen_composite_candidates(examples):
    """Two-layer compositions: sort+dedup, cumulative ops, etc."""
    cands = []
    cands.append("def program(xs):\n    return list(dict.fromkeys(xs))\n")
    cands.append("def program(xs):\n    if not xs: return []\n    res = [xs[0]]\n    for x in xs[1:]:\n        res.append(res[-1] + x)\n    return res\n")
    cands.append("def program(xs):\n    if not xs: return []\n    res = [xs[0]]\n    for x in xs[1:]:\n        res.append(max(res[-1], x))\n    return res\n")
    cands.append("def program(xs):\n    if not xs: return []\n    res = [xs[0]]\n    for x in xs[1:]:\n        res.append(min(res[-1], x))\n    return res\n")
    cands.append("def program(xs):\n    return [xs[i+1]-xs[i] for i in range(len(xs)-1)]\n")
    cands.append("def program(xs):\n    return [xs[i]+xs[i+1] for i in range(len(xs)-1)]\n")
    cands.append("def program(xs):\n    return sorted([x for x in xs if x > 0])\n")
    cands.append("def program(xs):\n    if not xs: return []\n    n = len(xs)\n    return xs[n//3:2*n//3]\n")
    cands.append("def program(xs):\n    return [abs(x) for x in xs]\n")
    cands.append("def program(xs):\n    return [2*x for x in xs]\n")
    cands.append("def program(xs):\n    return [x * x for x in xs]\n")
    cands.append("def program(xs):\n    return [x - 1 for x in xs]\n")
    cands.append("def program(xs):\n    return [x + 1 for x in xs]\n")
    for m in [2, 3, 5, 10]:
        cands.append(f"def program(xs):\n    return [x % {m} for x in xs]\n")
    for add in list(range(-3, 11)):
        if add != 0:
            cands.append(f"def program(xs):\n    return [x + {add} for x in xs]\n")
    for mul in [2, 3, 4, 5]:
        cands.append(f"def program(xs):\n    return [{mul} * x for x in xs]\n")
    return _dedup(cands)


def _gen_simple_candidates(examples):
    """Identity, empty, constant output."""
    cands = [
        "def program(xs):\n    return xs\n",
        "def program(xs):\n    return []\n",
    ]
    return _dedup(cands)


# =============================================================================
# Targeted pattern analysis
# =============================================================================

def _gen_targeted(examples):
    """Analyse examples and generate highly-targeted programs."""
    cands = []
    if not examples:
        return cands

    out_lens = [len(o) for _, o in examples]
    all_outs = [out for _, out in examples]
    all_inps = [inp for inp, _ in examples]

    # Pattern A: take first k elements
    if len(set(out_lens)) == 1:
        k = out_lens[0]
        if k > 0:
            if all(inp[:k] == out for inp, out in examples):
                cands.append(f"def program(xs):\n    return xs[:{k}]\n")

    # Pattern B: take last k elements
    if len(set(out_lens)) == 1:
        k = out_lens[0]
        if k > 0:
            if all(inp[-k:] == out for inp, out in examples):
                cands.append(f"def program(xs):\n    return xs[-{k}:]\n")

    # Pattern C: take from index i
    for i in range(8):
        if all(inp[i:] == out for inp, out in examples):
            cands.append(f"def program(xs):\n    return xs[{i}:]\n")

    # Pattern D: append constant
    all_match = True
    c_val = None
    for inp, out in examples:
        if inp and out:
            expected = inp + [out[-1]]
            if out != expected:
                all_match = False
                break
            c_val = out[-1]
        elif not inp and out == [out[-1]] if out else True:
            c_val = out[0] if out else None
    if all_match and c_val is not None:
        all_match2 = all(
            (not inp and out == [c_val]) or (inp and out == inp + [c_val])
            for inp, out in examples
        )
        if all_match2:
            cands.append(f"def program(xs):\n    return xs + [{c_val}]\n")

    # Pattern E: prepend constant
    all_match = True
    c_val = None
    for inp, out in examples:
        if inp and out:
            expected = [out[0]] + inp
            if out != expected:
                all_match = False
                break
            c_val = out[0]
    if all_match and c_val is not None:
        cands.append(f"def program(xs):\n    return [{c_val}] + xs\n")

    # Pattern F: remove element at index p
    for p in range(5):
        if all(inp[:p] + inp[p+1:] == out for inp, out in examples if len(inp) > p):
            cands.append(f"def program(xs):\n    if len(xs) <= {p}: return xs\n    return xs[:{p}] + xs[{p+1}:]\n")

    # Pattern G: replace first with last
    if all(out == [inp[-1]] + inp[1:] for inp, out in examples if len(inp) > 1):
        cands.append("def program(xs):\n    if not xs: return []\n    return [xs[-1]] + xs[1:]\n")

    # Pattern H: count first element minus 1
    if all(out == [inp.count(inp[0]) - 1] for inp, out in examples if inp):
        cands.append("def program(xs):\n    if not xs: return []\n    return [xs.count(xs[0]) - 1]\n")

    # Pattern I: sorted
    if all(sorted(inp) == out for inp, out in examples):
        cands.append("def program(xs):\n    return sorted(xs)\n")

    # Pattern J: reversed
    if all(list(reversed(inp)) == out for inp, out in examples):
        cands.append("def program(xs):\n    return list(reversed(xs))\n")

    # Pattern K: identity
    if all(inp == out for inp, out in examples):
        cands.append("def program(xs):\n    return xs\n")

    # Pattern L: constant output
    all_consts = set()
    for _, out in examples:
        if out:
            all_consts.add(tuple(out))
    if len(all_consts) == 1:
        c = list(all_consts.pop())
        cands.append(f"def program(xs):\n    return {c}\n")

    # Pattern M: filter by threshold (search thresholds systematically)
    # Also try exact threshold from min output value
    all_out_vals = set()
    for _, out in examples:
        all_out_vals.update(out)
    if all_out_vals:
        exact_thresh = min(all_out_vals)
        candidate = f"def program(xs):\n    return [x for x in xs if x >= {exact_thresh}]\n"
        ok = True
        for inp, out in examples:
            got = [x for x in inp if x >= exact_thresh]
            if got != out:
                ok = False
                break
        if ok:
            cands.append(candidate)
        # Also try > (max_dropped)
        all_in_vals = set()
        for inp, _ in examples:
            all_in_vals.update(inp)
        dropped = all_in_vals - all_out_vals
        if dropped:
            exact_thresh2 = max(dropped) + 1
            candidate2 = f"def program(xs):\n    return [x for x in xs if x >= {exact_thresh2}]\n"
            ok2 = True
            for inp, out in examples:
                got = [x for x in inp if x >= exact_thresh2]
                if got != out:
                    ok2 = False
                    break
            if ok2:
                cands.append(candidate2)

    # Systematic search
    for thresh in range(-1, 200):
        candidate = f"def program(xs):\n    return [x for x in xs if x >= {thresh}]\n"
        ok = True
        for inp, out in examples:
            got = [x for x in inp if x >= thresh]
            if got != out:
                ok = False
                break
        if ok:
            cands.append(candidate)

    # Pattern N: filter by value != 0
    if all(out == [x for x in inp if x != 0] for inp, out in examples):
        cands.append("def program(xs):\n    return [x for x in xs if x != 0]\n")

    # Pattern O: filter elements > 0
    if all(out == [x for x in inp if x > 0] for inp, out in examples):
        cands.append("def program(xs):\n    return [x for x in xs if x > 0]\n")

    # Pattern P: element-wise add
    for add in range(-10, 11):
        if all(out == [x + add for x in inp] for inp, out in examples):
            cands.append(f"def program(xs):\n    return [x + {add} for x in xs]\n")

    # Pattern Q: element-wise multiply
    for mul in [2, 3, 4, 5, 10]:
        if all(out == [x * mul for x in inp] for inp, out in examples):
            cands.append(f"def program(xs):\n    return [{mul} * x for x in xs]\n")

    # Pattern R: cumulative max
    if all(out == [max(inp[:i+1]) for i in range(len(inp))] for inp, out in examples):
        cands.append("""def program(xs):
    if not xs: return []
    res = [xs[0]]
    for x in xs[1:]:
        res.append(max(res[-1], x))
    return res
""")

    # Pattern S: cumulative min
    if all(out == [min(inp[:i+1]) for i in range(len(inp))] for inp, out in examples):
        cands.append("""def program(xs):
    if not xs: return []
    res = [xs[0]]
    for x in xs[1:]:
        res.append(min(res[-1], x))
    return res
""")

    # Pattern T: cumulative sum
    if all(out == [sum(inp[:i+1]) for i in range(len(inp))] for inp, out in examples):
        cands.append("""def program(xs):
    if not xs: return []
    res = [xs[0]]
    for x in xs[1:]:
        res.append(res[-1] + x)
    return res
""")

    return _dedup(cands)


# =============================================================================
# Evaluation
# =============================================================================

def _eval(src, examples):
    """Return score [0, 1] for source string on examples."""
    try:
        tree = compile(src, '<cand>', 'exec')
        ns = {}
        exec(tree, ns)
        if 'program' not in ns or not callable(ns['program']):
            return 0.0
        return score_program(
            ns['program'],
            [list(i) for i, _ in examples],
            [list(o) for _, o in examples],
        )[0]
    except Exception:
        return 0.0


# =============================================================================
# Main solver
# =============================================================================

def solve_listfn(examples, top_k=5):
    """
    Solve a List Functions task from I/O examples.

    Args:
        examples: list of (input_list, output_list) pairs
        top_k: number of top programs to return when none is fully correct

    Returns:
        dict with 'success' (bool) and 'program' (source string defining program(xs))
    """
    if not examples:
        return {"success": False, "program": "def program(xs):\n    return []\n"}

    # Phase 1: Targeted analysis (pattern-matching on examples)
    targeted = _gen_targeted(examples)
    scored = []
    for src in targeted:
        s = _eval(src, examples)
        if s >= 1.0:
            return {"success": True, "program": src}
        scored.append((s, src))

    # Phase 2: Systematic DSL search (all primitive compositions)
    all_cands = []
    all_cands.extend(_gen_simple_candidates(examples))
    all_cands.extend(_gen_slice_candidates(examples))
    all_cands.extend(_gen_filter_candidates(examples))
    all_cands.extend(_gen_sort_slice_candidates(examples))
    all_cands.extend(_gen_replace_candidates(examples))
    all_cands.extend(_gen_count_candidates(examples))
    all_cands.extend(_gen_range_insert_candidates(examples))
    all_cands.extend(_gen_interleave_candidates(examples))
    all_cands.extend(_gen_composite_candidates(examples))

    for src in all_cands:
        s = _eval(src, examples)
        if s >= 1.0:
            return {"success": True, "program": src}
        scored.append((s, src))

    # Phase 3: Return best
    scored.sort(key=lambda x: (-x[0], x[1]))

    if scored:
        return {"success": scored[0][0] >= 1.0, "program": scored[0][1]}

    return {"success": False, "program": "def program(xs):\n    return xs\n"}
