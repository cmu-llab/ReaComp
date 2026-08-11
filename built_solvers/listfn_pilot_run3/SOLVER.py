"""
SOLVER_LISTFN.py – Symbolic Program Synthesizer for List Functions Tasks

Performs a structured search over a domain-specific language (DSL) of
list-transformation primitives and their compositions.  Candidates are
scored against all examples via the verifier (rewards/list_functions).

Search proceeds in waves of increasing complexity:
  Wave 0 – identity / no-parameter primitives
  Wave 1 – single primitive with one parameter
  Wave 2 – two-primitive compositions
  Wave 3 – special compound transforms
  Wave 4 – data-adaptive programs inferred from example patterns

The solver returns the first program scoring 100 % (all examples correct);
if none is found it returns the top-K highest-scoring partial matches.
"""

from __future__ import annotations

import random
import sys
from typing import Any, Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Import the verifier
# ---------------------------------------------------------------------------
sys.path.insert(0, "/workspace")
from rewards.list_functions import _compile_program, score_program

# ============================================================================
#  Data-adaptive threshold computation
# ============================================================================
_thresholds: List[int] = []


def _init_thresholds(inputs: List[List[int]]) -> None:
    """Compute interesting threshold values from the data."""
    global _thresholds
    all_vals: List[int] = sorted(set(v for xs in inputs for v in xs))
    base: List[int] = list(range(-3, 20))
    data_vals: List[int] = all_vals[:80]  # cap to avoid explosion
    _thresholds = sorted(set(base + data_vals))


# ============================================================================
#  Source-code builder
# ============================================================================
def _src(name: str, body: str) -> str:
    """Wrap a body string into `def program(xs): body`, handling multiline bodies."""
    # Indent each line of the body with 4 spaces
    indented_body = "\n".join("    " + line if line.strip() else "" for line in body.split("\n"))
    return f"def program(xs):\n{indented_body}"

# ============================================================================
#  Wave 0 – no-parameter primitives
# ============================================================================
def _w0_programs() -> List[Tuple[str, Callable, str]]:
    prims: List[Tuple[str, Callable, str]] = []

    prims.append(("identity", lambda xs: list(xs), _src("identity", "return list(xs)")))
    prims.append(("reverse", lambda xs: list(reversed(xs)), _src("reverse", "return list(reversed(xs))")))
    prims.append(("sort", lambda xs: sorted(xs), _src("sort", "return sorted(xs)")))
    prims.append(("sort_desc", lambda xs: sorted(xs, reverse=True), _src("sort_desc", "return sorted(xs, reverse=True)")))
    prims.append(("dedup_consec", _dedup_consec_fn, _src("dedup_consec", _DEDUP_CONSEC_CODE)))
    prims.append(("unique", _unique_fn, _src("unique", _UNIQUE_CODE)))
    prims.append(("empty", lambda xs: [], _src("empty", "return []")))
    prims.append(("first", lambda xs: [xs[0]] if xs else [], _src("first", "return [xs[0]] if xs else []")))
    prims.append(("last", lambda xs: [xs[-1]] if xs else [], _src("last", "return [xs[-1]] if xs else []")))
    prims.append(("min_val", lambda xs: [min(xs)] if xs else [], _src("min_val", "return [min(xs)] if xs else []")))
    prims.append(("max_val", lambda xs: [max(xs)] if xs else [], _src("max_val", "return [max(xs)] if xs else []")))
    prims.append(("count", lambda xs: [len(xs)], _src("count", "return [len(xs)]")))
    prims.append(("total", lambda xs: [sum(xs)], _src("total", "return [sum(xs)]")))
    prims.append(("count_first_occ_minus1", lambda xs: _count_first_occ_minus1(xs),
                   _src("count_first_occ_minus1", _COUNT_FIRST_CODE)))
    prims.append(("replace_first_last", _replace_first_last_fn,
                   _src("replace_first_last", _REPLACE_FIRST_LAST_CODE)))

    # Slice step variants
    prims.append(("every_2nd_start_0", lambda xs: xs[::2], _src("every_2nd_start_0", "return xs[::2]")))
    prims.append(("every_2nd_start_1", lambda xs: xs[1::2], _src("every_2nd_start_1", "return xs[1::2]")))
    prims.append(("every_3rd", lambda xs: xs[::3], _src("every_3rd", "return xs[::3]")))

    return prims


def _dedup_consec_fn(xs: List[int]) -> List[int]:
    if not xs:
        return []
    out = [xs[0]]
    for x in xs[1:]:
        if x != out[-1]:
            out.append(x)
    return out


_DEDUP_CONSEC_CODE = (
    "out = [xs[0]] if xs else []\n"
    "for x in xs[1:]:\n"
    "    if x != out[-1]:\n"
    "        out.append(x)\n"
    "return out"
)


def _unique_fn(xs: List[int]) -> List[int]:
    seen: set = set()
    out: List[int] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


_UNIQUE_CODE = (
    "seen = set()\n"
    "out = []\n"
    "for x in xs:\n"
    "    if x not in seen:\n"
    "        seen.add(x)\n"
    "        out.append(x)\n"
    "return out"
)


def _count_first_occ_minus1(xs: List[int]) -> List[int]:
    if not xs:
        return []
    count = sum(1 for x in xs if x == xs[0])
    return [count - 1]


_COUNT_FIRST_CODE = (
    "if not xs:\n"
    "    return []\n"
    "count = sum(1 for x in xs if x == xs[0])\n"
    "return [count - 1]"
)


def _replace_first_last_fn(xs: List[int]) -> List[int]:
    if not xs:
        return []
    return [xs[-1]] + xs[1:]


_REPLACE_FIRST_LAST_CODE = (
    "if not xs:\n"
    "    return []\n"
    "return [xs[-1]] + xs[1:]"
)


# ============================================================================
#  Wave 1 – single primitive with one parameter
# ============================================================================
def _w1_programs(thresholds: List[int]) -> List[Tuple[str, Callable, str]]:
    prims: List[Tuple[str, Callable, str]] = []

    # Slicing: take first/last, drop first/last, remove at
    for k in range(0, 11):
        prims.append((f"first_{k}", lambda xs, k=k: xs[:k], _src(f"first_{k}", f"return xs[:{k}]")))
        prims.append((f"last_{k}", lambda xs, k=k: (xs[-k:] if k <= len(xs) else []),
                       _src(f"last_{k}", f"return xs[-{k}:] if {k} <= len(xs) else []")))
        prims.append((f"drop_first_{k}", lambda xs, k=k: xs[k:], _src(f"drop_first_{k}", f"return xs[{k}:]")))
        prims.append((f"drop_last_{k}", lambda xs, k=k: (xs[:-k] if k <= len(xs) else []),
                       _src(f"drop_last_{k}", f"return xs[:-{k}] if {k} <= len(xs) else []")))
        prims.append((f"remove_at_{k}", lambda xs, k=k: (xs[:k] + xs[k + 1:] if k < len(xs) else xs),
                       _src(f"remove_at_{k}", f"return xs[:{k}] + xs[{k}+1:] if {k} < len(xs) else xs")))

    # Sort + slice
    for k in range(0, 11):
        prims.append((f"sort_first_{k}", lambda xs, k=k: sorted(xs)[:k],
                       _src(f"sort_first_{k}", f"return sorted(xs)[:{k}]")))
        prims.append((f"sort_last_{k}", lambda xs, k=k: (sorted(xs)[-k:] if k <= len(xs) else sorted(xs)),
                       _src(f"sort_last_{k}",
                            f"return sorted(xs)[-{k}:] if {k} <= len(sorted(xs)) else sorted(xs)")))
        prims.append((f"sort_desc_first_{k}", lambda xs, k=k: sorted(xs, reverse=True)[:k],
                       _src(f"sort_desc_first_{k}", f"return sorted(xs, reverse=True)[:{k}]")))
        prims.append((f"sort_desc_last_{k}", lambda xs, k=k: (sorted(xs, reverse=True)[-k:] if k <= len(xs) else sorted(xs, reverse=True)),
                       _src(f"sort_desc_last_{k}",
                            f"return sorted(xs, reverse=True)[-{k}:] if {k} <= len(sorted(xs)) else sorted(xs, reverse=True)")))

    # Filter by threshold
    for t in thresholds:
        prims.append((f"filter_ge_{t}", lambda xs, t=t: [x for x in xs if x >= t],
                       _src(f"filter_ge_{t}", f"return [x for x in xs if x >= {t}]")))
        prims.append((f"filter_gt_{t}", lambda xs, t=t: [x for x in xs if x > t],
                       _src(f"filter_gt_{t}", f"return [x for x in xs if x > {t}]")))
        prims.append((f"filter_le_{t}", lambda xs, t=t: [x for x in xs if x <= t],
                       _src(f"filter_le_{t}", f"return [x for x in xs if x <= {t}]")))
        prims.append((f"filter_lt_{t}", lambda xs, t=t: [x for x in xs if x < t],
                       _src(f"filter_lt_{t}", f"return [x for x in xs if x < {t}]")))
        prims.append((f"filter_eq_{t}", lambda xs, t=t: [x for x in xs if x == t],
                       _src(f"filter_eq_{t}", f"return [x for x in xs if x == {t}]")))

    # Map operations
    prims.append(("double", lambda xs: [x * 2 for x in xs], _src("double", "return [x * 2 for x in xs]")))
    prims.append(("negate", lambda xs: [-x for x in xs], _src("negate", "return [-x for x in xs]")))
    prims.append(("increment", lambda xs: [x + 1 for x in xs], _src("increment", "return [x + 1 for x in xs]")))
    prims.append(("decrement", lambda xs: [x - 1 for x in xs], _src("decrement", "return [x - 1 for x in xs]")))

    # Append / prepend constant
    for c in range(-2, 15):
        prims.append((f"append_{c}", lambda xs, c=c: list(xs) + [c],
                       _src(f"append_{c}", f"return list(xs) + [{c}]")))
        prims.append((f"prepend_{c}", lambda xs, c=c: [c] + list(xs),
                       _src(f"prepend_{c}", f"return [{c}] + list(xs)")))

    # Drop first then sort
    for k in range(1, 8):
        prims.append((f"drop_first_then_sorted",
                       lambda xs, k=k: sorted(xs[k:]) if len(xs) > k else [],
                       _src(f"drop_first_then_sorted", f"return sorted(xs[{k}:]) if len(xs) > {k} else []")))

    return prims


# ============================================================================
#  Wave 2 – two-primitive compositions
# ============================================================================
def _w2_programs(thresholds: List[int]) -> List[Tuple[str, Callable, str]]:
    prims: List[Tuple[str, Callable, str]] = []

    # sort + filter / filter + sort
    for t in thresholds:
        prims.append((f"sort_filter_ge_{t}", lambda xs, t=t: sorted([x for x in xs if x >= t]),
                       _src(f"sort_filter_ge_{t}", f"return sorted([x for x in xs if x >= {t}])")))
        prims.append((f"filter_gt_sort_{t}", lambda xs, t=t: sorted([x for x in xs if x > t]),
                       _src(f"filter_gt_sort_{t}", f"return sorted([x for x in xs if x > {t}])")))
        prims.append((f"sort_filter_le_{t}", lambda xs, t=t: sorted([x for x in xs if x <= t]),
                       _src(f"sort_filter_le_{t}", f"return sorted([x for x in xs if x <= {t}])")))
        prims.append((f"filter_lt_sort_{t}", lambda xs, t=t: sorted([x for x in xs if x < t]),
                       _src(f"filter_lt_sort_{t}", f"return sorted([x for x in xs if x < {t}])")))
        prims.append((f"sort_filter_eq_{t}", lambda xs, t=t: sorted([x for x in xs if x == t]),
                       _src(f"sort_filter_eq_{t}", f"return sorted([x for x in xs if x == {t}])")))

    # filter + reverse
    for t in thresholds:
        prims.append((f"filter_ge_{t}_rev", lambda xs, t=t: [x for x in xs if x >= t][::-1],
                       _src(f"filter_ge_{t}_rev", f"return [x for x in xs if x >= {t}][::-1]")))
        prims.append((f"rev_filter_ge_{t}", lambda xs, t=t: [x for x in reversed(xs) if x >= t],
                       _src(f"rev_filter_ge_{t}", f"return [x for x in reversed(xs) if x >= {t}]")))

    # sort + top-k
    for k in range(0, 10):
        prims.append((f"top_{k}_sorted", lambda xs, k=k: (sorted(xs)[-k:] if k <= len(xs) else sorted(xs)),
                       _src(f"top_{k}_sorted",
                            f"return sorted(xs)[-{k}:] if {k} <= len(xs) else sorted(xs)")))

    # reverse + slice
    for k in range(0, 10):
        prims.append((f"rev_first_{k}", lambda xs, k=k: list(reversed(xs))[:k],
                       _src(f"rev_first_{k}", f"return list(reversed(xs))[:{k}]")))
        prims.append((f"first_rev_{k}", lambda xs, k=k: list(reversed(xs[:k])),
                       _src(f"first_rev_{k}", f"return list(reversed(xs[:{k}]))")))

    # filter by comparison with first element
    prims.append(("filter_ge_first",
                   lambda xs: ([x for x in xs[1:] if x >= xs[0]] if xs else []),
                   _src("filter_ge_first", "return [x for x in xs[1:] if x >= xs[0]] if xs else []")))
    prims.append(("filter_gt_first",
                   lambda xs: ([x for x in xs[1:] if x > xs[0]] if xs else []),
                   _src("filter_gt_first", "return [x for x in xs[1:] if x > xs[0]] if xs else []")))
    prims.append(("filter_le_first",
                   lambda xs: ([x for x in xs[1:] if x <= xs[0]] if xs else []),
                   _src("filter_le_first", "return [x for x in xs[1:] if x <= xs[0]] if xs else []")))
    prims.append(("filter_lt_first",
                   lambda xs: ([x for x in xs[1:] if x < xs[0]] if xs else []),
                   _src("filter_lt_first", "return [x for x in xs[1:] if x < xs[0]] if xs else []")))
    prims.append(("filter_eq_first",
                   lambda xs: ([x for x in xs[1:] if x == xs[0]] if xs else []),
                   _src("filter_eq_first", "return [x for x in xs[1:] if x == xs[0]] if xs else []")))

    # dedup + sort
    prims.append(("dedup_sorted", lambda xs: sorted(set(xs)), _src("dedup_sorted", "return sorted(set(xs))")))

    # sort + drop
    for k in range(0, 10):
        prims.append((f"sort_drop_{k}", lambda xs, k=k: sorted(xs)[k:],
                       _src(f"sort_drop_{k}", f"return sorted(xs)[{k}:]")))

    # append / prepend sorted
    for c in range(-2, 15):
        prims.append((f"append_{c}_sorted", lambda xs, c=c: sorted(xs) + [c],
                       _src(f"append_{c}_sorted", f"return sorted(xs) + [{c}]")))
        prims.append((f"prepend_{c}_sorted", lambda xs, c=c: [c] + sorted(xs),
                       _src(f"prepend_{c}_sorted", f"return [{c}] + sorted(xs)")))

    # reverse + sort / sort + reverse
    prims.append(("rev_then_sort", lambda xs: sorted(list(reversed(xs))),
                   _src("rev_then_sort", "return sorted(list(reversed(xs)))")))
    prims.append(("sort_then_rev", lambda xs: list(reversed(sorted(xs))),
                   _src("sort_then_rev", "return list(reversed(sorted(xs)))")))

    return prims


# ============================================================================
#  Wave 3 – special compound transforms
# ============================================================================
def _w3_programs(thresholds: List[int]) -> List[Tuple[str, Callable, str]]:
    prims: List[Tuple[str, Callable, str]] = []

    # Fill gaps: for each element x later in the list where x > first, insert range(first, x) before x
    prims.append(("fill_gaps", _fill_gaps_fn, _src("fill_gaps", _FILL_GAPS_CODE)))

    # Interleave sorted with 1-based indices
    prims.append(("interleave_sorted_indices",
                   lambda xs: _interleave_sorted_idx_fn(xs),
                   _src("interleave_sorted_indices", _INTERLEAVE_SORTED_IDX_CODE)))

    # Remove elements less than first (keep >= first)
    prims.append(("remove_less_than_first",
                   lambda xs: ([xs[0]] + [x for x in xs[1:] if x >= xs[0]]) if xs else [],
                   _src("remove_less_than_first",
                        "return [xs[0]] + [x for x in xs[1:] if x >= xs[0]] if xs else []")))

    # Elements greater than first, sorted
    prims.append(("greater_than_first_sorted",
                   lambda xs: sorted([x for x in xs[1:] if x > xs[0]]) if xs else [],
                   _src("greater_than_first_sorted",
                        "return sorted([x for x in xs[1:] if x > xs[0]]) if xs else []")))

    # Between first and last
    prims.append(("between_first_last",
                   lambda xs: [x for x in xs if xs[0] < x < xs[-1]] if len(xs) >= 2 else [],
                   _src("between_first_last",
                        "return [x for x in xs if xs[0] < x < xs[-1]] if len(xs) >= 2 else []")))

    # Consecutive gap filling
    prims.append(("fill_consec_gaps", _fill_consec_gaps_fn, _src("fill_consec_gaps", _FILL_CONSEC_GAPS_CODE)))

    # Sort without first / drop first then sort
    prims.append(("sort_without_first",
                   lambda xs: sorted(xs[1:]) if len(xs) > 1 else [],
                   _src("sort_without_first", "return sorted(xs[1:]) if len(xs) > 1 else []")))
    prims.append(("drop_first_sorted",
                   lambda xs: sorted(xs[1:]) if len(xs) > 1 else [],
                   _src("drop_first_sorted", "return sorted(xs[1:]) if len(xs) > 1 else []")))

    # Range from first to last
    prims.append(("range_first_last",
                   lambda xs: list(range(xs[0], xs[-1] + 1)) if xs else [],
                   _src("range_first_last", "return list(range(xs[0], xs[-1] + 1)) if xs else []")))

    # Unique sorted
    prims.append(("unique_sorted", lambda xs: sorted(set(xs)), _src("unique_sorted", "return sorted(set(xs))")))

    # Replace first with max/min of rest
    prims.append(("replace_first_with_max_rest",
                   lambda xs: ([max(xs[1:])] + xs[1:]) if len(xs) > 1 else list(xs),
                   _src("replace_first_with_max_rest",
                        "return [max(xs[1:])] + xs[1:] if len(xs) > 1 else list(xs)")))
    prims.append(("replace_first_with_min_rest",
                   lambda xs: ([min(xs[1:])] + xs[1:]) if len(xs) > 1 else list(xs),
                   _src("replace_first_with_min_rest",
                        "return [min(xs[1:])] + xs[1:] if len(xs) > 1 else list(xs)")))

    # Greater than average
    prims.append(("greater_than_avg",
                   lambda xs: [x for x in xs if x > sum(xs) / len(xs)] if xs else [],
                   _src("greater_than_avg",
                        "return [x for x in xs if x > sum(xs) / len(xs)] if xs else []")))

    # First element preserved, rest sorted
    prims.append(("first_preserve_sorted_rest",
                   lambda xs: ([xs[0]] + sorted(xs[1:])) if xs else [],
                   _src("first_preserve_sorted_rest",
                        "return [xs[0]] + sorted(xs[1:]) if xs else []")))

    # Last element preserved, rest sorted
    prims.append(("last_preserve_sorted_rest",
                   lambda xs: ([xs[-1]] + sorted(xs[:-1])) if len(xs) >= 2 else list(xs),
                   _src("last_preserve_sorted_rest",
                        "return [xs[-1]] + sorted(xs[:-1]) if len(xs) >= 2 else list(xs)")))

    return prims


def _fill_gaps_fn(xs: List[int]) -> List[int]:
    """For each element x later in the list where x > first, insert range(first, x) before x."""
    if not xs:
        return []
    a = xs[0]
    res = [a]
    for x in xs[1:]:
        if x > a:
            res.extend(range(a, x))
        res.append(x)
    return res


_FILL_GAPS_CODE = (
    "if not xs:\n"
    "    return []\n"
    "a = xs[0]\n"
    "res = [a]\n"
    "for x in xs[1:]:\n"
    "    if x > a:\n"
    "        res.extend(range(a, x))\n"
    "    res.append(x)\n"
    "return res"
)


def _interleave_sorted_idx_fn(xs: List[int]) -> List[int]:
    """Interleave 1-based indices with sorted elements."""
    sorted_xs = sorted(xs)
    result = []
    for i, val in enumerate(sorted_xs, start=1):
        result.append(i)
        result.append(val)
    return result


_INTERLEAVE_SORTED_IDX_CODE = (
    "sorted_xs = sorted(xs)\n"
    "result = []\n"
    "for i, val in enumerate(sorted_xs, start=1):\n"
    "    result.append(i)\n"
    "    result.append(val)\n"
    "return result"
)


def _fill_consec_gaps_fn(xs: List[int]) -> List[int]:
    """For consecutive pairs where next > prev, insert range(prev+1, next)."""
    if not xs:
        return []
    res = [xs[0]]
    for i in range(1, len(xs)):
        if xs[i] > xs[i - 1]:
            res.extend(range(xs[i - 1] + 1, xs[i]))
        res.append(xs[i])
    return res


_FILL_CONSEC_GAPS_CODE = (
    "if not xs:\n"
    "    return []\n"
    "res = [xs[0]]\n"
    "for i in range(1, len(xs)):\n"
    "    if xs[i] > xs[i-1]:\n"
    "        res.extend(range(xs[i-1] + 1, xs[i]))\n"
    "    res.append(xs[i])\n"
    "return res"
)


# ============================================================================
#  Wave 4 – data-adaptive programs inferred from example patterns
# ============================================================================
def _w4_programs(inputs: List[List[int]], outputs: List[List[int]]) -> List[Tuple[str, Callable, str]]:
    """Infer programs by matching each (input, output) pair directly."""
    prims: List[Tuple[str, Callable, str]] = []

    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        n = len(inp)

        # First k elements
        for k in range(1, min(n, 11)):
            if out == inp[:k]:
                key = f"adapt_first_{k}"
                prims.append((key, lambda xs, k=k: xs[:k], _src(key, f"return xs[:{k}]")))

        # Last k elements
        for k in range(1, min(n, 11)):
            if out == inp[-k:]:
                key = f"adapt_last_{k}"
                prims.append((key, lambda xs, k=k: (xs[-k:] if k <= len(xs) else []),
                               _src(key, f"return xs[-{k}:] if {k} <= len(xs) else []")))

        # Sorted
        if out == sorted(inp):
            prims.append(("adapt_sorted", lambda xs: sorted(xs), _src("adapt_sorted", "return sorted(xs)")))

        # Reversed
        if out == list(reversed(inp)):
            prims.append(("adapt_reversed", lambda xs: list(reversed(xs)),
                           _src("adapt_reversed", "return list(reversed(xs))")))

        # First element only
        if inp and out == [inp[0]]:
            prims.append(("adapt_first_elem", lambda xs: [xs[0]] if xs else [],
                           _src("adapt_first_elem", "return [xs[0]] if xs else []")))

        # Last element only
        if inp and out == [inp[-1]]:
            prims.append(("adapt_last_elem", lambda xs: [xs[-1]] if xs else [],
                           _src("adapt_last_elem", "return [xs[-1]] if xs else []")))

        # Sorted + first k
        for k in range(1, min(n, 11)):
            if out == sorted(inp)[:k]:
                prims.append((f"adapt_sort_first_{k}", lambda xs, k=k: sorted(xs)[:k],
                               _src(f"adapt_sort_first_{k}", f"return sorted(xs)[:{k}]")))

        # Sorted + last k
        for k in range(1, min(n, 11)):
            if out == sorted(inp)[-k:]:
                prims.append((f"adapt_sort_last_{k}", lambda xs, k=k: (sorted(xs)[-k:] if k <= len(xs) else sorted(xs)),
                               _src(f"adapt_sort_last_{k}",
                                    f"return sorted(xs)[-{k}:] if {k} <= len(sorted(xs)) else sorted(xs)")))

        # Sorted descending + first k
        for k in range(1, min(n, 11)):
            if out == sorted(inp, reverse=True)[:k]:
                prims.append((f"adapt_sort_desc_first_{k}", lambda xs, k=k: sorted(xs, reverse=True)[:k],
                               _src(f"adapt_sort_desc_first_{k}", f"return sorted(xs, reverse=True)[:{k}]")))

        # Filter by threshold
        for t in _thresholds:
            if out == [x for x in inp if x >= t]:
                prims.append((f"adapt_filter_ge_{t}", lambda xs, t=t: [x for x in xs if x >= t],
                               _src(f"adapt_filter_ge_{t}", f"return [x for x in xs if x >= {t}]")))
            if out == [x for x in inp if x > t]:
                prims.append((f"adapt_filter_gt_{t}", lambda xs, t=t: [x for x in xs if x > t],
                               _src(f"adapt_filter_gt_{t}", f"return [x for x in xs if x > {t}]")))
            if out == [x for x in inp if x <= t]:
                prims.append((f"adapt_filter_le_{t}", lambda xs, t=t: [x for x in xs if x <= t],
                               _src(f"adapt_filter_le_{t}", f"return [x for x in xs if x <= {t}]")))
            if out == [x for x in inp if x < t]:
                prims.append((f"adapt_filter_lt_{t}", lambda xs, t=t: [x for x in xs if x < t],
                               _src(f"adapt_filter_lt_{t}", f"return [x for x in xs if x < {t}]")))
            if out == [x for x in inp if x == t]:
                prims.append((f"adapt_filter_eq_{t}", lambda xs, t=t: [x for x in xs if x == t],
                               _src(f"adapt_filter_eq_{t}", f"return [x for x in xs if x == {t}]")))

        # Append / prepend constant
        for c in range(-2, 30):
            if out == list(inp) + [c]:
                prims.append((f"adapt_append_{c}", lambda xs, c=c: list(xs) + [c],
                               _src(f"adapt_append_{c}", f"return list(xs) + [{c}]")))
            if out == [c] + list(inp):
                prims.append((f"adapt_prepend_{c}", lambda xs, c=c: [c] + list(xs),
                               _src(f"adapt_prepend_{c}", f"return [{c}] + list(xs)")))

        # Drop first k
        for k in range(1, min(n, 11)):
            if out == inp[k:]:
                prims.append((f"adapt_drop_first_{k}", lambda xs, k=k: xs[k:],
                               _src(f"adapt_drop_first_{k}", f"return xs[{k}:]")))

    # Deduplicate by name while preserving order
    seen: set = set()
    unique: List[Tuple[str, Callable, str]] = []
    for name, fn, src in prims:
        if name not in seen:
            seen.add(name)
            unique.append((name, fn, src))

    return unique


# ============================================================================
#  Main solver
# ============================================================================
def solve_listfn(examples: List[Tuple[List[int], List[int]]],
                 K: int = 5) -> Dict[str, Any]:
    """
    Solve a list functions task by searching over compositions of list
    transformation primitives.

    Args:
        examples: list of (input_list, output_list) pairs
        K: number of top programs to return if no perfect match found

    Returns:
        dict with:
            - 'success' (bool): whether a 100% matching program was found
            - 'program' (str): source code defining `def program(xs): ...`
            - 'score' (float): fraction of examples matched
            - 'name' (str): name of the best program
            - 'top_programs' (list[dict]): top-K partial matches (if not perfect)
    """
    inputs = [list(e[0]) for e in examples]
    outputs = [list(e[1]) for e in examples]

    # Initialize data-adaptive thresholds
    _init_thresholds(inputs)

    # Collect all candidates: (name, function, source_code)
    all_candidates: List[Tuple[str, Callable, str]] = []

    # Generate waves in order (simple to complex)
    for name, fn, src in _w0_programs():
        all_candidates.append((f"W0_{name}", fn, src))

    for name, fn, src in _w1_programs(thresholds=_thresholds):
        all_candidates.append((f"W1_{name}", fn, src))

    for name, fn, src in _w2_programs(thresholds=_thresholds):
        all_candidates.append((f"W2_{name}", fn, src))

    for name, fn, src in _w3_programs(thresholds=_thresholds):
        all_candidates.append((f"W3_{name}", fn, src))

    for name, fn, src in _w4_programs(inputs, outputs):
        all_candidates.append((f"W4_{name}", fn, src))

    # Shuffle for diversity within same-score groups
    random.seed(42)
    waves: Dict[str, List[Tuple[str, Callable, str]]] = {}
    for c in all_candidates:
        wave = c[0].split("_")[0]
        waves.setdefault(wave, []).append(c)
    for wave in waves:
        random.shuffle(waves[wave])
    all_candidates = [c for wc in waves.values() for c in wc]

    # Evaluate candidates and track best
    best_score: float = 0.0
    best_fn: Callable | None = None
    best_src: str | None = None
    best_name: str = ""
    top_k: List[Tuple[str, Callable, str, float]] = []

    for name, fn, src in all_candidates:
        try:
            score, _ = score_program(fn, inputs, outputs)

            if score >= 1.0:
                return {
                    "success": True,
                    "program": src,
                    "score": 1.0,
                    "name": name,
                }

            if score > best_score:
                best_score = score
                best_fn = fn
                best_src = src
                best_name = name

            # Maintain top-K
            entry = (name, fn, src, score)
            if len(top_k) < K:
                top_k.append(entry)
                top_k.sort(key=lambda x: -x[3])
            elif score > top_k[-1][3]:
                top_k.append(entry)
                top_k.sort(key=lambda x: -x[3])
                top_k = top_k[:K]

        except Exception:
            continue

    # Return best program found (partial match or fallback)
    if best_score > 0:
        return {
            "success": False,
            "program": best_src or "def program(xs):\n    return list(xs)",
            "score": best_score,
            "name": best_name,
            "top_programs": [
                {"name": n, "score": s}
                for n, _, s, _ in top_k[:K]
            ],
        }

    return {
        "success": False,
        "program": "def program(xs):\n    return list(xs)",
        "score": 0.0,
        "name": "identity",
        "message": "No matching program found",
    }


# ============================================================================
#  Standalone testing
# ============================================================================
if __name__ == "__main__":
    import json

    with open("/workspace/DEMOS.json", "r") as f:
        demos = json.load(f)

    for i, demo in enumerate(demos):
        task_id = demo.get("task_id", "?")
        success = demo.get("success")

        if not success:
            examples = list(zip(demo["input_examples"], demo["output_examples"]))
            result = solve_listfn(examples)
            v_score, _ = score_program(_compile_program(result["program"]), [e[0] for e in examples], [e[1] for e in examples])
            print(f"Task {i} ({task_id}): reported={result['success']} verified={v_score:.3f}")
            continue

        examples = list(zip(demo["input_examples"], demo["output_examples"]))
        result = solve_listfn(examples)

        # Independent verification
        fn = _compile_program(result["program"])
        v_score, _ = (score_program(fn, [e[0] for e in examples], [e[1] for e in examples]) if fn else (0.0, []))

        status = "PASS" if result["success"] and v_score >= 1.0 else "FAIL"
        print(f"Task {i} ({task_id}): {status} (reported={result['success']}, verified={v_score:.3f}, name={result.get('name')})")
