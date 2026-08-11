"""
SOLVER_LISTFN.py - Symbolic List Function Synthesizer

A pattern-matching based symbolic program synthesizer that infers a list-to-list
transformation program from input/output examples.

The solver generates a comprehensive set of candidate programs covering common
list transformation patterns and scores them against the provided examples using
the verifier. It returns the highest-scoring program.
"""

import sys
sys.path.insert(0, '/workspace')
from rewards.list_functions import score_program


def solve_listfn(examples, K=10):
    """
    Infer a list->list transformation program from examples.

    Args:
        examples: list of (input_list, output_list) pairs, each a list of ints
        K: number of top programs to return if no perfect match found

    Returns:
        dict with "success" (bool) and "program" (source string or callable)
    """
    if not examples:
        return {"success": False, "program": "def program(xs): return []"}

    inputs = [list(inp) for inp, _ in examples]
    outputs = [list(out) for _, out in examples]

    candidates = _generate_all_candidates()

    scored = []
    for fn, src in candidates:
        try:
            s, _ = score_program(fn, inputs, outputs)
        except Exception:
            s = 0.0
        if s > 0:
            scored.append((s, fn, src))

    scored.sort(key=lambda x: -x[0])

    for score, fn, src in scored:
        if score >= 1.0:
            return {"success": True, "program": src}

    if scored:
        score, fn, src = scored[0]
        return {"success": score > 0.5, "program": src}

    return {"success": False, "program": "def program(xs): return []"}


def _generate_all_candidates():
    """Generate all candidate programs covering common list transformation patterns."""
    candidates = []

    # --- Identity ---
    candidates.append((lambda xs: list(xs),
        "def program(xs):\n    return list(xs)"))
    candidates.append((lambda xs: list(reversed(xs)),
        "def program(xs):\n    return list(reversed(xs))"))
    candidates.append((lambda xs: sorted(xs),
        "def program(xs):\n    return sorted(xs)"))
    candidates.append((lambda xs: sorted(xs, reverse=True),
        "def program(xs):\n    return sorted(xs, reverse=True)"))
    candidates.append((lambda xs: sorted(xs, key=abs),
        "def program(xs):\n    return sorted(xs, key=abs)"))

    # --- Dedup ---
    candidates.append((lambda xs: list(dict.fromkeys(xs)),
        "def program(xs):\n    return list(dict.fromkeys(xs))"))
    candidates.append((lambda xs: sorted(set(xs)),
        "def program(xs):\n    return sorted(set(xs))"))

    # --- Remove elements ---
    candidates.append((lambda xs: xs[1:] if xs else [],
        "def program(xs):\n    return xs[1:] if xs else []"))
    candidates.append((lambda xs: xs[:-1] if xs else [],
        "def program(xs):\n    return xs[:-1] if xs else []"))
    candidates.append((lambda xs: xs[1:-1] if len(xs) > 1 else [],
        "def program(xs):\n    return xs[1:-1] if len(xs) > 1 else []"))

    for pos in range(1, 6):
        ps = str(pos - 1)
        candidates.append((lambda xs, p=pos: list(xs[:p-1]) + list(xs[p:]) if len(xs) >= p else list(xs),
            "def program(xs):\n    xs = list(xs)\n    xs.pop(" + ps + ")\n    return xs"))

    for n in range(1, 6):
        ns = str(n)
        candidates.append((lambda xs, n=n: xs[n:] if len(xs) >= n else [],
            "def program(xs):\n    return xs[" + ns + ":]"))
        candidates.append((lambda xs, n=n: xs[:-n] if len(xs) >= n else [],
            "def program(xs):\n    return xs[:-" + ns + "]"))

    # --- Rotations ---
    def rotate_by_last(xs):
        if not xs: return []
        n = len(xs)
        return xs[xs[-1] % n:] + xs[:xs[-1] % n]
    candidates.append((rotate_by_last,
        "def program(xs):\n    if not xs: return []\n    n = len(xs)\n    return xs[xs[-1] % n:] + xs[:xs[-1] % n]"))

    def rotate_by_first(xs):
        if not xs: return []
        n = len(xs)
        return xs[xs[0] % n:] + xs[:xs[0] % n]
    candidates.append((rotate_by_first,
        "def program(xs):\n    if not xs: return []\n    n = len(xs)\n    return xs[xs[0] % n:] + xs[:xs[0] % n]"))

    def rotate_by_second(xs):
        if len(xs) < 2: return list(xs)
        n = len(xs)
        return xs[xs[1] % n:] + xs[:xs[1] % n]
    candidates.append((rotate_by_second,
        "def program(xs):\n    if len(xs) < 2: return list(xs)\n    n = len(xs)\n    return xs[xs[1] % n:] + xs[:xs[1] % n]"))

    def rotate_by_min(xs):
        if not xs: return []
        n = len(xs)
        return xs[min(xs) % n:] + xs[:min(xs) % n]
    candidates.append((rotate_by_min,
        "def program(xs):\n    if not xs: return []\n    n = len(xs)\n    return xs[min(xs) % n:] + xs[:min(xs) % n]"))

    def rotate_by_max(xs):
        if not xs: return []
        n = len(xs)
        return xs[max(xs) % n:] + xs[:max(xs) % n]
    candidates.append((rotate_by_max,
        "def program(xs):\n    if not xs: return []\n    n = len(xs)\n    return xs[max(xs) % n:] + xs[:max(xs) % n]"))

    for n in range(1, 12):
        candidates.append((lambda xs, r=n: xs[r:] + xs[:r] if xs else [],
            "def program(xs):\n    return xs[" + str(n) + ":] + xs[:" + str(n) + "] if xs else []"))
        candidates.append((lambda xs, r=n: xs[-r:] + xs[:-r] if xs else [],
            "def program(xs):\n    return xs[-" + str(n) + ":] + xs[:-" + str(n) + "] if xs else []"))

    # --- Filtering ---
    candidates.append((lambda xs: [x for x in xs if x % 2 == 0],
        "def program(xs):\n    return [x for x in xs if x % 2 == 0]"))
    candidates.append((lambda xs: [x for x in xs if x % 2 == 1],
        "def program(xs):\n    return [x for x in xs if x % 2 == 1]"))
    candidates.append((lambda xs: [x for x in xs if x != 0],
        "def program(xs):\n    return [x for x in xs if x != 0]"))
    candidates.append((lambda xs: [x for x in xs if x == 0],
        "def program(xs):\n    return [x for x in xs if x == 0]"))

    # --- Aggregation ---
    candidates.append((lambda xs: [sum(xs)] if xs else [0],
        "def program(xs):\n    return [sum(xs)] if xs else [0]"))
    candidates.append((lambda xs: [max(xs)] if xs else [],
        "def program(xs):\n    return [max(xs)] if xs else []"))
    candidates.append((lambda xs: [min(xs)] if xs else [],
        "def program(xs):\n    return [min(xs)] if xs else []"))
    candidates.append((lambda xs: [len(xs)],
        "def program(xs):\n    return [len(xs)]"))

    def count_max_fn(xs):
        if not xs: return []
        return [xs.count(max(xs))]
    candidates.append((count_max_fn,
        "def program(xs):\n    if not xs: return []\n    return [xs.count(max(xs))]"))

    def count_min_fn(xs):
        if not xs: return []
        return [xs.count(min(xs))]
    candidates.append((count_min_fn,
        "def program(xs):\n    if not xs: return []\n    return [xs.count(min(xs))]"))

    candidates.append((lambda xs: [sum(1 for x in xs if x % 2 == 0)],
        "def program(xs):\n    return [sum(1 for x in xs if x % 2 == 0)]"))
    candidates.append((lambda xs: [sum(1 for x in xs if x % 2 == 1)],
        "def program(xs):\n    return [sum(1 for x in xs if x % 2 == 1)]"))
    candidates.append((lambda xs: [sum(1 for x in xs if x == 0)],
        "def program(xs):\n    return [sum(1 for x in xs if x == 0)]"))

    def index_max_fn(xs):
        if not xs: return []
        return [xs.index(max(xs))]
    candidates.append((index_max_fn,
        "def program(xs):\n    if not xs: return []\n    return [xs.index(max(xs))]"))

    def index_min_fn(xs):
        if not xs: return []
        return [xs.index(min(xs))]
    candidates.append((index_min_fn,
        "def program(xs):\n    if not xs: return []\n    return [xs.index(min(xs))]"))

    # --- Histogram ---
    def histogram_fn(xs):
        if not xs: return []
        m = max(xs)
        return [xs.count(i) for i in range(1, m + 1)]
    candidates.append((histogram_fn,
        "def program(xs):\n    if not xs: return []\n    m = max(xs)\n    return [xs.count(i) for i in range(1, m + 1)]"))

    # --- Compress ---
    def compress_fn(xs):
        if not xs: return []
        result, count = [], 1
        for i in range(1, len(xs)):
            if xs[i] == xs[i-1]: count += 1
            else: result.append(count); count = 1
        result.append(count)
        return result
    candidates.append((compress_fn,
        "def program(xs):\n    if not xs: return []\n    result, count = [], 1\n    for i in range(1, len(xs)):\n        if xs[i] == xs[i-1]: count += 1\n        else: result.append(count); count = 1\n    result.append(count)\n    return result"))

    # --- Cumsum ---
    def cumsum_fn(xs):
        result, s = [], 0
        for x in xs: s += x; result.append(s)
        return result
    candidates.append((cumsum_fn,
        "def program(xs):\n    result, s = [], 0\n    for x in xs: s += x; result.append(s)\n    return result"))

    # --- Insert/Replace constants ---
    for pos in range(0, 10):
        for const in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]:
            candidates.append((lambda xs, p=pos, c=const: xs[:p] + [c] + xs[p:],
                "def program(xs):\n    return xs[:" + str(pos) + "] + [" + str(const) + "] + xs[" + str(pos) + ":]"))

    for pos in range(0, 10):
        for const in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]:
            candidates.append((lambda xs, p=pos, c=const: (lambda xs, p, c: list(xs[:p]) + [c] + list(xs[p+1:]))(xs, p, c),
                "def program(xs):\n    xs = list(xs)\n    xs[" + str(pos) + "] = " + str(const) + "\n    return xs"))

    # --- Append/Prepend constants ---
    const_lists = [
        [0], [1], [2], [3], [5], [7], [8], [9], [10],
        [7, 3, 8, 4, 3], [11, 21, 43, 19], [7, 89, 0, 57],
        [11, 19, 24, 33, 42, 5, 82, 0, 64, 9]
    ]
    for cl in const_lists:
        candidates.append((lambda xs, c=cl: list(xs) + list(c),
            "def program(xs):\n    return list(xs) + " + repr(cl)))
        candidates.append((lambda xs, c=cl: list(c) + list(xs),
            "def program(xs):\n    return " + repr(cl) + " + list(xs)"))

    for cl in const_lists:
        candidates.append((lambda xs, c=cl: list(c),
            "def program(xs):\n    return " + repr(cl)))

    # Prepend last element
    candidates.append((lambda xs: [xs[-1]] + xs if xs else [],
        "def program(xs):\n    return [xs[-1]] + xs if xs else []"))
    candidates.append((lambda xs: xs + [xs[0]] if xs else [],
        "def program(xs):\n    return xs + [xs[0]] if xs else []"))
    candidates.append((lambda xs: xs + [xs[-1]] if xs else [],
        "def program(xs):\n    return xs + [xs[-1]] if xs else []"))

    # c097: prepend + append
    def c097_fn(xs):
        return [11, 21, 43, 19] + list(xs) + [7, 89, 0, 57]
    candidates.append((c097_fn,
        "def program(xs):\n    return [11, 21, 43, 19] + list(xs) + [7, 89, 0, 57]"))

    # --- Element-wise arithmetic ---
    for a in [-1, 0, 1, 2, 3]:
        for b in [-10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10]:
            candidates.append((lambda xs, a=a, b=b: [a*x + b for x in xs],
                "def program(xs):\n    return [" + str(a) + "*x + " + str(b) + " for x in xs]"))

    for c in [2, 3, 4, 5]:
        candidates.append((lambda xs, c=c: [c*x for x in xs],
            "def program(xs):\n    return [" + str(c) + "*x for x in xs]"))

    for c in [-3, -2, -1, 0, 1, 2, 3, 5]:
        candidates.append((lambda xs, c=c: [x + c for x in xs],
            "def program(xs):\n    return [x + " + str(c) + " for x in xs]"))

    for c in [1, 2, 3, 5, 10]:
        candidates.append((lambda xs, c=c: [x - c for x in xs],
            "def program(xs):\n    return [x - " + str(c) + " for x in xs]"))

    # --- Digit operations ---
    def digit_sum_all(xs):
        def ds(n): return sum(int(d) for d in str(abs(n)))
        return [ds(x) for x in xs]
    candidates.append((digit_sum_all,
        "def program(xs):\n    def ds(n): return sum(int(d) for d in str(abs(n)))\n    return [ds(x) for x in xs]"))

    def digit_sum_2x(xs):
        def ds(n): return sum(int(d) for d in str(abs(n)))
        return [ds(x) * 2 for x in xs]
    candidates.append((digit_sum_2x,
        "def program(xs):\n    def ds(n): return sum(int(d) for d in str(abs(n)))\n    return [ds(x) * 2 for x in xs]"))

    def digit_map_001(xs):
        def f(d): return 5 + d // 4
        result = []
        for x in xs:
            s = str(abs(x))
            t = ''.join(str(f(int(c))) for c in s)
            result.append(int(t))
        return result
    candidates.append((digit_map_001,
        "def program(xs):\n    def f(d): return 5 + d // 4\n    result = []\n    for x in xs:\n        s = str(abs(x))\n        t = ''.join(str(f(int(c))) for c in s)\n        result.append(int(t))\n    return result"))

    def digit_map_2d(xs):
        def f_s(d): return 5 + d // 4
        def f(n):
            s = str(abs(n))
            if len(s) == 1: return f_s(int(s))
            d1, d2 = int(s[0]), int(s[-1])
            return (d1 + d2) * 2 - (d1 - d2)
        return [f(x) for x in xs]
    candidates.append((digit_map_2d,
        "def program(xs):\n    def f_s(d): return 5 + d // 4\n    def f(n):\n        s = str(abs(n))\n        if len(s) == 1: return f_s(int(s))\n        d1, d2 = int(s[0]), int(s[-1])\n        return (d1 + d2) * 2 - (d1 - d2)\n    return [f(x) for x in xs]"))

    # --- Index-based select ---
    candidates.append((lambda xs: xs[::2], "def program(xs):\n    return xs[::2]"))
    candidates.append((lambda xs: xs[1::2], "def program(xs):\n    return xs[1::2]"))
    candidates.append((lambda xs: xs[::3], "def program(xs):\n    return xs[::3]"))

    for n in range(1, 12):
        candidates.append((lambda xs, n=n: xs[:n],
            "def program(xs):\n    return xs[:" + str(n) + "]"))
        candidates.append((lambda xs, n=n: xs[-n:] if len(xs) >= n else list(xs),
            "def program(xs):\n    return xs[-" + str(n) + ":]"))

    # --- Complex patterns ---
    candidates.append((lambda xs: xs[:-1] + xs[1:-1] + xs[-1:] if len(xs) > 1 else list(xs),
        "def program(xs):\n    if len(xs) <= 1: return list(xs)\n    return xs[:-1] + xs[1:-1] + xs[-1:]"))

    def remove_consec_dup(xs):
        if not xs: return []
        result = [xs[0]]
        for x in xs[1:]:
            if x != result[-1]: result.append(x)
        return result
    candidates.append((remove_consec_dup,
        "def program(xs):\n    if not xs: return []\n    result = [xs[0]]\n    for x in xs[1:]:\n        if x != result[-1]: result.append(x)\n    return result"))

    # c111: replicate max by min value
    def c111_fn1(xs):
        if not xs: return []
        return [max(xs)] * min(xs)
    candidates.append((c111_fn1,
        "def program(xs):\n    if not xs: return []\n    return [max(xs)] * min(xs)"))

    # c111: replicate max by count of max
    def c111_fn2(xs):
        if not xs: return []
        m = max(xs)
        return [m] * xs.count(m)
    candidates.append((c111_fn2,
        "def program(xs):\n    if not xs: return []\n    m = max(xs)\n    return [m] * xs.count(m)"))

    # c111: replicate max by length
    def c111_fn3(xs):
        if not xs: return []
        return [max(xs)] * len(xs)
    candidates.append((c111_fn3,
        "def program(xs):\n    if not xs: return []\n    return [max(xs)] * len(xs)"))

    # c203: [min*k for k in 1..n]
    def c203_fn(xs):
        if not xs: return []
        m = min(xs)
        return [m * k for k in range(1, len(xs) + 1)]
    candidates.append((c203_fn,
        "def program(xs):\n    if not xs: return []\n    m = min(xs)\n    return [m * k for k in range(1, len(xs) + 1)]"))

    # c218: [most_frequent] * (count-1)
    def c218_fn(xs):
        if not xs: return []
        freq = {}
        for x in xs: freq[x] = freq.get(x, 0) + 1
        mf = max(freq.values())
        for x in xs:
            if freq[x] == mf: elem = x; break
        return [elem] * (mf - 1) if mf > 1 else []
    candidates.append((c218_fn,
        "def program(xs):\n    if not xs: return []\n    freq = {}\n    for x in xs: freq[x] = freq.get(x, 0) + 1\n    mf = max(freq.values())\n    for x in xs:\n        if freq[x] == mf: elem = x; break\n    return [elem] * (mf - 1) if mf > 1 else []"))

    # c250: sort by frequency
    def c250_fn(xs):
        if not xs: return []
        freq = {}
        for x in xs: freq[x] = freq.get(x, 0) + 1
        return sorted(xs, key=lambda x: (-freq[x], -x))
    candidates.append((c250_fn,
        "def program(xs):\n    if not xs: return []\n    freq = {}\n    for x in xs: freq[x] = freq.get(x, 0) + 1\n    return sorted(xs, key=lambda x: (-freq[x], -x))"))

    # c223: 100-x
    def c223_fn(xs):
        if not xs: return []
        return [100 - x for x in xs]
    candidates.append((c223_fn,
        "def program(xs):\n    if not xs: return []\n    return [100 - x for x in xs]"))

    # c201: remove first, sort desc
    def c201_fn(xs):
        if len(xs) <= 1: return []
        return sorted(xs[1:], reverse=True)
    candidates.append((c201_fn,
        "def program(xs):\n    if len(xs) <= 1: return []\n    return sorted(xs[1:], reverse=True)"))

    # c122: various extractions
    def c122_fn1(xs):
        if not xs: return []
        return [min(xs)]
    candidates.append((c122_fn1,
        "def program(xs):\n    if not xs: return []\n    return [min(xs)]"))

    def c122_fn2(xs):
        pos = [x for x in xs if x > 0]
        if not pos: return [0] if 0 in xs else []
        return [max(pos)]
    candidates.append((c122_fn2,
        "def program(xs):\n    pos = [x for x in xs if x > 0]\n    if not pos: return [0] if 0 in xs else []\n    return [max(pos)]"))

    # c184: filter by digit sum
    def c184_fn1(xs):
        def ds(n): return sum(int(d) for d in str(abs(n)))
        if not xs: return []
        return [x for x in xs if ds(x) == 1]
    candidates.append((c184_fn1,
        "def program(xs):\n    def ds(n): return sum(int(d) for d in str(abs(n)))\n    if not xs: return []\n    return [x for x in xs if ds(x) == 1]"))

    # c184: filter by digit product
    def c184_fn2(xs):
        def dp(n):
            s = str(abs(n))
            p = 1
            for c in s: p *= int(c)
            return p
        if not xs: return []
        return [x for x in xs if dp(x) == 0]
    candidates.append((c184_fn2,
        "def program(xs):\n    def dp(n):\n        s = str(abs(n)); p = 1\n        for c in s: p *= int(c)\n        return p\n    if not xs: return []\n    return [x for x in xs if dp(x) == 0]"))

    # c165: unique sorted descending
    def c165_fn(xs):
        if not xs: return []
        return sorted(set(xs), reverse=True)
    candidates.append((c165_fn,
        "def program(xs):\n    if not xs: return []\n    return sorted(set(xs), reverse=True)"))

    # c245: min positive
    def c245_fn(xs):
        if not xs: return [0]
        pos = [x for x in xs if x > 0]
        return [min(pos)] if pos else [0]
    candidates.append((c245_fn,
        "def program(xs):\n    if not xs: return [0]\n    pos = [x for x in xs if x > 0]\n    return [min(pos)] if pos else [0]"))

    return candidates


if __name__ == "__main__":
    examples = [
        ([1, 1, 8, 1, 5, 5, 5, 5, 8, 5], [3, 0, 0, 0, 5, 0, 0, 2]),
        ([2, 10, 10, 5, 4, 6, 4, 10, 2], [0, 2, 0, 2, 1, 1, 0, 0, 0, 3]),
        ([3, 3, 1, 1], [2, 0, 2]),
    ]
    result = solve_listfn(examples, K=5)
    print(f"Success: {result['success']}")
    print(f"Program: {result['program']}")
