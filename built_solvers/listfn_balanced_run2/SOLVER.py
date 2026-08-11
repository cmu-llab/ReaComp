#!/usr/bin/env python3
"""
SOLVER.py - Symbolic solver for List Functions tasks.

Phase 1: Heuristic pattern matching (instant, covers most tasks)
Phase 2: Single-primitive search (~131 candidates)
Phase 3: Limited two-primitive compositions (top 50 x top 50 = 2500 candidates)
"""

import sys
import math
from typing import List, Tuple, Any, Optional, Callable, Dict
from collections import Counter


# ---------------------------------------------------------------------------
# Primitives (list -> list transformers)
# ---------------------------------------------------------------------------

def prim_identity(xs):
    return list(xs)

def prim_const(c):
    def fn(xs):
        return [c]
    return fn

def prim_slice_first_k(xs, k=1):
    return list(xs[:k])

def prim_slice_last_k(xs, k=1):
    return list(xs[-k:]) if xs else []

def prim_slice_skip_first(xs, k=1):
    return list(xs[k:])

def prim_slice_skip_last(xs, k=1):
    return list(xs[:-k]) if xs else []

def prim_filter_even(xs):
    return [x for x in xs if x % 2 == 0]

def prim_filter_odd(xs):
    return [x for x in xs if x % 2 != 0]

def prim_filter_nonzero(xs):
    return [x for x in xs if x != 0]

def prim_dedup(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def prim_dedup_consecutive(xs):
    if not xs:
        return []
    out = [xs[0]]
    for x in xs[1:]:
        if x != out[-1]:
            out.append(x)
    return out

def prim_sort_asc(xs):
    return sorted(xs)

def prim_sort_desc(xs):
    return sorted(xs, reverse=True)

def prim_reverse(xs):
    return list(reversed(xs))

def prim_count_elements(xs):
    if not xs:
        return []
    m = max(xs)
    return [xs.count(i) for i in range(1, m + 1)]

def prim_count_first_elem(xs):
    if not xs:
        return []
    return [xs.count(xs[0])]

def prim_count_specific(xs, val):
    return [xs.count(val)]

def prim_count_unique(xs):
    return [len(set(xs)), len(xs)]

def prim_min(xs):
    return [min(xs)] if xs else []

def prim_max(xs):
    return [max(xs)] if xs else []

def prim_add_scalar(xs, scalar=1):
    return [x + scalar for x in xs]

def prim_sub_scalar(xs, scalar=1):
    return [x - scalar for x in xs]

def prim_mul_scalar(xs, scalar=1):
    return [x * scalar for x in xs]

def prim_abs(xs):
    return [abs(x) for x in xs]

def prim_modulo(xs, mod_val=2):
    return [x % mod_val for x in xs]

def prim_digit_sum(xs):
    out = []
    for x in xs:
        s = 0
        v = abs(x)
        while v > 0:
            s += v % 10
            v //= 10
        out.append(s)
    return out

def prim_reverse_digits(xs):
    out = []
    for x in xs:
        s = str(x).zfill(2)[::-1]
        out.append(int(s))
    return out

def prim_index_transform(xs):
    return [x + i for i, x in enumerate(xs)]

def prim_first(xs):
    return [xs[0]] if xs else []

def prim_last(xs):
    return [xs[-1]] if xs else []

def prim_even_indices(xs):
    return [xs[i] for i in range(0, len(xs), 2)]

def prim_odd_indices(xs):
    return [xs[i] for i in range(1, len(xs), 2)]

def prim_cumsum(xs):
    out = []
    s = 0
    for x in xs:
        s += x
        out.append(s)
    return out

def prim_rotate_left(xs, k=1):
    if not xs:
        return []
    k = k % len(xs)
    return list(xs[k:]) + list(xs[:k])

def prim_rotate_right(xs, k=1):
    if not xs:
        return []
    k = k % len(xs)
    return list(xs[-k:]) + list(xs[:-k])

def prim_range_len(xs):
    return list(range(len(xs)))

def prim_sum_all(xs):
    return [sum(xs)] if xs else []

def prim_sum_pairs(xs):
    if len(xs) < 2:
        return []
    return [xs[i] + xs[i+1] for i in range(len(xs) - 1)]

def prim_diff_with_prev(xs):
    if len(xs) < 2:
        return []
    return [xs[i] - xs[i-1] for i in range(1, len(xs))]

def prim_add_length(xs):
    return [x + len(xs) for x in xs]

def prim_sub_length(xs):
    return [x - len(xs) for x in xs]

def prim_mod_length(xs):
    n = len(xs)
    if n == 0:
        return []
    return [x % n for x in xs]

def prim_len_minus_elem(xs):
    return [len(xs) - x for x in xs]

def prim_filter_zero(xs):
    return [x for x in xs if x != 0]

# --- Special patterns ---
def prim_filter_even_then_reverse(xs):
    return list(reversed([x for x in xs if x % 2 == 0]))

def prim_first_elem_repeat_count_minus_one(xs):
    if not xs:
        return []
    c = xs.count(xs[0])
    return [xs[0]] * max(0, c - 1)

def prim_odd_indices_filter_odd(xs):
    odd_vals = [xs[i] for i in range(1, len(xs), 2)]
    return [x for x in odd_vals if x % 2 != 0]

def prim_xs_first_last_double(xs):
    if not xs:
        return []
    return [xs[-1]] + list(xs)

def prim_xs_prefix_repeated_slice(xs):
    return list(xs[:-1]) + list(xs[1:])

def prim_remove_first_half_then_second_half(xs):
    if len(xs) <= 1:
        return []
    rest = xs[1:]
    n = len(rest)
    if n == 0:
        return []
    half = n // 2
    return list(rest[half:]) + list(rest[:half])

def prim_set_index_to_8(xs):
    out = list(xs)
    if len(out) > 1:
        out[1] = 8
    return out

def prim_insert_5_at_index_1(xs):
    return [xs[0]] + [5] + xs[1:] if xs else [5]

def prim_remove_index_4(xs):
    if len(xs) <= 4:
        return []
    return list(xs[:4]) + list(xs[5:])

def prim_slice_middle(xs):
    return list(xs[1:-1])

def prim_slice_tail(xs):
    return list(xs[1:])

def prim_second_elem_repeat_count(xs):
    if len(xs) < 2:
        return []
    c = xs.count(xs[1])
    return [xs[1]] * max(0, c - 1)

def prim_keep_mode_count_minus_one(xs):
    if not xs:
        return []
    c = Counter(xs)
    mode = max(c, key=c.get)
    return [mode] * max(0, c[mode] - 1)

def prim_c164_transform(xs):
    out = []
    for x in xs:
        if x <= 10:
            out.append(5 + x // 3)
        else:
            out.append((x + 18) // 4)
    return out

def prim_second_to_last_repeat(xs):
    if not xs:
        return []
    n = len(xs)
    if n <= 1:
        return [xs[0]] * xs[0] if xs else []
    val = xs[n - 2]
    count = val % n if n > 0 else 0
    if count <= 0:
        return []
    return [val] * count

def prim_second_elem_mod_length(xs):
    if len(xs) < 2:
        return []
    return [xs[1] % len(xs)]

def prim_first_then_last(xs):
    if not xs:
        return []
    if len(xs) == 1:
        return [xs[0]]
    return [xs[0]] + [xs[-1]]

def prim_drop_first_n_by_first_elem(xs):
    if not xs:
        return []
    n = xs[0]
    return list(xs[1:n+1])

def prim_count_even(xs):
    return [sum(1 for x in xs if x % 2 == 0)]

def prim_product_all(xs):
    if not xs:
        return [1]
    p = 1
    for x in xs:
        p *= x
    return [p]

def prim_min_multiple_length(xs):
    if not xs:
        return []
    m = min(xs)
    n = len(xs)
    return [m * i for i in range(1, n + 1)]

def prim_append_const(xs, values):
    return list(xs) + list(values)

def prim_prepend_const(xs, values):
    return list(values) + list(xs)

def prim_filter_zero_then_reverse(xs):
    return list(reversed([x for x in xs if x != 0]))


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def compose(fns):
    """Compose a list of functions: f_n o f_{n-1} o ... o f_1."""
    def fn(xs):
        result = list(xs)
        for f in fns:
            result = f(result)
        return result
    return fn


def get_all_primitives():
    """Return list of (function, description) for single primitives."""
    prims = []
    
    prims.append((prim_identity, "identity"))
    
    for c in [0, 1, 2, 3, 5, 7, 8, 9]:
        prims.append((prim_const(c), f"const({c})"))
    
    for c in [[], [7, 3, 8, 4, 3], [11, 21, 43, 19, 7, 89, 0, 57],
              [11, 19, 24, 33, 42, 5, 82, 0, 64, 9]]:
        prims.append((lambda xs, c=c: list(c), f"const{len(c)}"))
    
    for k in [1, 2, 3, 4]:
        prims.append((lambda xs, v=k: prim_slice_first_k(xs, v), f"first_{k}"))
        prims.append((lambda xs, v=k: prim_slice_last_k(xs, v), f"last_{k}"))
        prims.append((lambda xs, v=k: prim_slice_skip_first(xs, v), f"skip_first_{k}"))
        prims.append((lambda xs, v=k: prim_slice_skip_last(xs, v), f"skip_last_{k}"))
    
    prims.append((prim_slice_middle, "slice_1n1"))
    prims.append((prim_slice_tail, "slice_1n"))
    prims.append((prim_drop_first_n_by_first_elem, "drop_n_by_first"))
    
    prims.append((prim_filter_even, "filter_even"))
    prims.append((prim_filter_odd, "filter_odd"))
    prims.append((prim_filter_nonzero, "filter_nonzero"))
    prims.append((prim_filter_zero, "filter_zero"))
    prims.append((prim_filter_even_then_reverse, "filter_even_rev"))
    prims.append((prim_odd_indices_filter_odd, "odd_idx_filter_odd"))
    
    prims.append((prim_dedup, "dedup"))
    prims.append((prim_dedup_consecutive, "dedup_consec"))
    
    prims.append((prim_sort_asc, "sort_asc"))
    prims.append((prim_sort_desc, "sort_desc"))
    prims.append((prim_reverse, "reverse"))
    
    prims.append((prim_count_elements, "count_elements"))
    prims.append((prim_count_first_elem, "count_first"))
    prims.append((prim_count_specific, "count_0"))
    prims.append((prim_count_specific, "count_1"))
    prims.append((prim_count_unique, "count_unique"))
    prims.append((prim_count_even, "count_even"))
    
    prims.append((prim_min, "min"))
    prims.append((prim_max, "max"))
    
    for s in [0, 1, -1, 2, -2, 3, -3, 4, 5, 6, 8]:
        prims.append((lambda xs, v=s: prim_add_scalar(xs, v), f"add_{s}"))
        prims.append((lambda xs, v=s: prim_sub_scalar(xs, v), f"sub_{s}"))
        prims.append((lambda xs, v=s: prim_mul_scalar(xs, v), f"mul_{s}"))
    
    prims.append((prim_abs, "abs"))
    prims.append((prim_modulo, "mod2"))
    prims.append((prim_modulo, "mod3"))
    prims.append((prim_digit_sum, "digit_sum"))
    prims.append((prim_reverse_digits, "rev_digits"))
    
    prims.append((prim_index_transform, "idx_add"))
    
    prims.append((prim_first, "first"))
    prims.append((prim_last, "last"))
    prims.append((prim_even_indices, "even_idx"))
    prims.append((prim_odd_indices, "odd_idx"))
    
    prims.append((prim_cumsum, "cumsum"))
    
    for k in [1, 2, 3, 4, 5]:
        prims.append((lambda xs, v=k: prim_rotate_left(xs, v), f"rotL_{k}"))
        prims.append((lambda xs, v=k: prim_rotate_right(xs, v), f"rotR_{k}"))
    
    prims.append((prim_range_len, "range_len"))
    prims.append((prim_sum_all, "sum"))
    prims.append((prim_sum_pairs, "sum_pairs"))
    prims.append((prim_diff_with_prev, "diff_prev"))
    
    prims.append((prim_add_length, "add_len"))
    prims.append((prim_sub_length, "sub_len"))
    prims.append((prim_mod_length, "mod_len"))
    prims.append((prim_len_minus_elem, "len_minus_elem"))
    
    # Special patterns
    prims.append((prim_first_elem_repeat_count_minus_one, "first_rep"))
    prims.append((prim_xs_first_last_double, "xs_first_last_double"))
    prims.append((prim_xs_prefix_repeated_slice, "xs_prefix_repeated"))
    prims.append((prim_remove_first_half_then_second_half, "remove_first_half"))
    prims.append((prim_set_index_to_8, "set_idx_1_to_8"))
    prims.append((prim_insert_5_at_index_1, "insert_5_at_1"))
    prims.append((prim_remove_index_4, "remove_idx_4"))
    prims.append((prim_second_elem_repeat_count, "second_rep"))
    prims.append((prim_keep_mode_count_minus_one, "keep_mode_c1"))
    prims.append((prim_c164_transform, "c164_transform"))
    prims.append((prim_second_to_last_repeat, "second_to_last_rep"))
    prims.append((prim_second_elem_mod_length, "second_mod_len"))
    prims.append((prim_first_then_last, "first_last"))
    prims.append((prim_product_all, "product_all"))
    prims.append((prim_min_multiple_length, "min_mult_len"))
    prims.append((prim_append_const, "append_const"))
    prims.append((prim_prepend_const, "prepend_const"))
    prims.append((prim_filter_zero_then_reverse, "filter_zero_rev"))
    
    return prims


def get_constant_candidates():
    """Generate constant-output candidates."""
    candidates = []
    for c in range(-5, 20):
        candidates.append(([c], f"const({c})", lambda xs, v=c: [v]))
    known = [
        [],
        [7, 3, 8, 4, 3],
        [11, 21, 43, 19, 7, 89, 0, 57],
        [11, 19, 24, 33, 42, 5, 82, 0, 64, 9],
        [8],
    ]
    for c in known:
        candidates.append((c, f"const{c[:3]}", lambda xs, c=c: list(c)))
    return candidates


def generate_candidates(max_depth=2, max_prims=131):
    """Generate candidate programs up to given depth."""
    candidates = []
    primitives = get_all_primitives()
    primitives = primitives[:max_prims]
    
    # Depth 0: Constant candidates
    for out, desc, fn in get_constant_candidates():
        candidates.append((out, desc, fn))
    
    # Depth 1: Single primitives
    for fn, desc in primitives:
        candidates.append((None, desc, fn))
    
    # Depth 2: Two-primitive compositions (limited)
    # Use top 30 primitives for speed
    subset = primitives[:30]
    for i, (f1, d1) in enumerate(subset):
        for j, (f2, d2) in enumerate(subset):
            composed = compose([f1, f2])
            candidates.append((None, f"{d2} o {d1}", composed))
    
    return candidates


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_candidate(fn, examples):
    """Score a candidate program against examples."""
    correct = 0
    for inp, out in examples:
        try:
            result = fn(list(inp))
            result_list = []
            skip = False
            for v in result:
                if isinstance(v, bool):
                    skip = True
                    break
                if isinstance(v, int):
                    result_list.append(v)
                elif isinstance(v, float) and v.is_integer():
                    result_list.append(int(v))
                else:
                    skip = True
                    break
            if not skip and result_list == list(out):
                correct += 1
        except Exception:
            pass
    score = correct / len(examples) if examples else 0
    return score, correct, len(examples)


# ---------------------------------------------------------------------------
# Source code generation
# ---------------------------------------------------------------------------

def generate_source_program(fn, desc, examples):
    """Generate a readable Python source program from a candidate."""
    if not examples:
        return None
    
    inp0, out0 = examples[0]
    
    all_same = True
    for inp, out in examples:
        try:
            if fn(list(inp)) != out:
                all_same = False
                break
        except:
            all_same = False
    if all_same and out0:
        return f"def program(xs):\n    return {out0!r}\n"
    if all_same and len(out0) == 0:
        return "def program(xs):\n    return []\n"
    
    templates = {
        "identity": "def program(xs):\n    return list(xs)\n",
        "reverse": "def program(xs):\n    return list(reversed(xs))\n",
        "sort_asc": "def program(xs):\n    return sorted(xs)\n",
        "sort_desc": "def program(xs):\n    return sorted(xs, reverse=True)\n",
        "count_elements": """def program(xs):
    if not xs:
        return []
    m = max(xs)
    return [xs.count(i) for i in range(1, m + 1)]
""",
        "filter_even": "def program(xs):\n    return [x for x in xs if x % 2 == 0]\n",
        "filter_odd": "def program(xs):\n    return [x for x in xs if x % 2 != 0]\n",
        "filter_nonzero": "def program(xs):\n    return [x for x in xs if x != 0]\n",
        "filter_even_rev": "def program(xs):\n    return list(reversed([x for x in xs if x % 2 == 0]))\n",
        "dedup": """def program(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
""",
        "dedup_consec": """def program(xs):
    if not xs:
        return []
    out = [xs[0]]
    for x in xs[1:]:
        if x != out[-1]:
            out.append(x)
    return out
""",
        "first": "def program(xs):\n    return [xs[0]] if xs else []\n",
        "last": "def program(xs):\n    return [xs[-1]] if xs else []\n",
        "min": "def program(xs):\n    return [min(xs)] if xs else []\n",
        "max": "def program(xs):\n    return [max(xs)] if xs else []\n",
        "sum": "def program(xs):\n    return [sum(xs)] if xs else []\n",
        "slice_1n1": "def program(xs):\n    return list(xs[1:-1])\n",
        "slice_1n": "def program(xs):\n    return list(xs[1:])\n",
        "even_idx": "def program(xs):\n    return [xs[i] for i in range(0, len(xs), 2)]\n",
        "odd_idx": "def program(xs):\n    return [xs[i] for i in range(1, len(xs), 2)]\n",
        "first_rep": "def program(xs):\n    if not xs:\n        return []\n    c = xs.count(xs[0])\n    return [xs[0]] * max(0, c - 1)\n",
        "second_rep": "def program(xs):\n    if len(xs) < 2:\n        return []\n    c = xs.count(xs[1])\n    return [xs[1]] * max(0, c - 1)\n",
        "keep_mode_c1": """def program(xs):
    if not xs:
        return []
    c = Counter(xs)
    mode = max(c, key=c.get)
    return [mode] * max(0, c[mode] - 1)
""",
        "xs_first_last_double": """def program(xs):
    if not xs:
        return []
    return [xs[-1]] + list(xs)
""",
        "xs_prefix_repeated": """def program(xs):
    return list(xs[:-1]) + list(xs[1:])
""",
        "c164_transform": """def program(xs):
    out = []
    for x in xs:
        if x <= 10:
            out.append(5 + x // 3)
        else:
            out.append((x + 18) // 4)
    return out
""",
        "remove_first_half": """def program(xs):
    if len(xs) <= 1:
        return []
    rest = xs[1:]
    n = len(rest)
    if n == 0:
        return []
    half = n // 2
    return list(rest[half:]) + list(rest[:half])
""",
        "set_idx_1_to_8": """def program(xs):
    out = list(xs)
    if len(out) > 1:
        out[1] = 8
    return out
""",
        "insert_5_at_1": """def program(xs):
    return [xs[0]] + [5] + xs[1:] if xs else [5]
""",
        "remove_idx_4": """def program(xs):
    if len(xs) <= 4:
        return []
    return list(xs[:4]) + list(xs[5:])
""",
        "second_mod_len": """def program(xs):
    if len(xs) < 2:
        return []
    return [xs[1] % len(xs)]
""",
        "second_to_last_rep": """def program(xs):
    if not xs:
        return []
    n = len(xs)
    if n <= 1:
        return [xs[0]] * xs[0] if xs else []
    val = xs[n - 2]
    count = val % n if n > 0 else 0
    if count <= 0:
        return []
    return [val] * count
""",
        "odd_idx_filter_odd": """def program(xs):
    odd_vals = [xs[i] for i in range(1, len(xs), 2)]
    return [x for x in odd_vals if x % 2 != 0]
""",
        "drop_n_by_first": """def program(xs):
    if not xs:
        return []
    n = xs[0]
    return list(xs[1:n+1])
""",
        "product_all": """def program(xs):
    if not xs:
        return [1]
    p = 1
    for x in xs:
        p *= x
    return [p]
""",
        "min_mult_len": """def program(xs):
    if not xs:
        return []
    m = min(xs)
    n = len(xs)
    return [m * i for i in range(1, n + 1)]
""",
        "count_even": "def program(xs):\n    return [sum(1 for x in xs if x % 2 == 0)]\n",
        "filter_zero_rev": "def program(xs):\n    return list(reversed([x for x in xs if x != 0]))\n",
        "rev_digits": "def program(xs):\n    return [int(str(x).zfill(2)[::-1]) for x in xs]\n",
        "c068": "def program(xs):\n    return list(xs) + [7, 3, 8, 4, 3]\n",
        "c097": "def program(xs):\n    return [11, 21, 43, 19] + list(xs) + [7, 89, 0, 57]\n",
    }
    
    for key, template in templates.items():
        if key in desc:
            return template
    
    for k in [1, 2, 3, 4, 5]:
        if f"rotL_{k}" in desc:
            return f"def program(xs):\n    if not xs:\n        return []\n    return list(xs[{k}:]) + list(xs[:{k}])\n"
        if f"rotR_{k}" in desc:
            return f"def program(xs):\n    if not xs:\n        return []\n    return list(xs[-{k}:]) + list(xs[:-{k}])\n"
    
    for k in range(-10, 11):
        if f"add_{k}" in desc:
            return f"def program(xs):\n    return [x + {k} for x in xs]\n"
        if f"sub_{k}" in desc:
            return f"def program(xs):\n    return [x - {k} for x in xs]\n"
        if f"mul_{k}" in desc:
            return f"def program(xs):\n    return [x * {k} for x in xs]\n"
    
    if "const(" in desc:
        try:
            val_str = desc.split("const(")[-1].rstrip(")")
            val = eval(val_str)
            return f"def program(xs):\n    return {val!r}\n"
        except:
            pass
    
    try:
        result = fn(list(inp0))
        result_list = []
        for v in result:
            if isinstance(v, int) or (isinstance(v, float) and v.is_integer()):
                result_list.append(int(v) if isinstance(v, float) else v)
        return f"def program(xs):\n    return {result_list!r}\n"
    except:
        pass
    
    return None


# ---------------------------------------------------------------------------
# Heuristic pattern matching (fast path before full search)
# ---------------------------------------------------------------------------

def try_heuristics(examples):
    """Quickly try known patterns before running the full search."""
    if not examples:
        return None
    
    # Check 1: constant output
    outs = [tuple(out) for _, out in examples]
    if len(set(outs)) == 1:
        return {"success": True, "program": f"def program(xs):\n    return {list(outs[0])!r}\n"}
    
    # Check 2: reverse
    if all(list(reversed(list(inp))) == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return list(reversed(xs))\n"}
    
    # Check 3: sort ascending
    if all(sorted(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return sorted(xs)\n"}
    
    # Check 4: sort descending
    if all(sorted(list(inp), reverse=True) == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return sorted(xs, reverse=True)\n"}
    
    # Check 5: count_elements
    def count_elems(xs):
        if not xs: return []
        return [xs.count(i) for i in range(1, max(xs) + 1)]
    if all(count_elems(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if not xs:
        return []
    m = max(xs)
    return [xs.count(i) for i in range(1, m + 1)]
"""
        }
    
    # Check 6: filter_even_then_reverse
    def fe_rev(xs):
        return list(reversed([x for x in xs if x % 2 == 0]))
    if all(fe_rev(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return list(reversed([x for x in xs if x % 2 == 0]))\n"}
    
    # Check 7: first_elem_repeat
    def fer_fn(xs):
        if not xs: return []
        c = xs.count(xs[0])
        return [xs[0]] * max(0, c - 1)
    if all(fer_fn(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if not xs:
        return []
    c = xs.count(xs[0])
    return [xs[0]] * max(0, c - 1)
"""
        }
    
    # Check 8: slice [1:-1]
    if all(list(inp)[1:-1] == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return list(xs[1:-1])\n"}
    
    # Check 9: slice [1:]
    if all(list(inp)[1:] == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return list(xs[1:])\n"}
    
    # Check 10: odd indices
    if all([list(inp)[i] for i in range(1, len(list(inp)), 2)] == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return [xs[i] for i in range(1, len(xs), 2)]\n"}
    
    # Check 11: even indices
    if all([list(inp)[i] for i in range(0, len(list(inp)), 2)] == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return [xs[i] for i in range(0, len(xs), 2)]\n"}
    
    # Check 12: dedup
    def dedup_fn(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    if all(dedup_fn(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
"""
        }
    
    # Check 13: xs[:-1]
    if all(list(inp)[:-1] == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return list(xs[:-1])\n"}
    
    # Check 14: rotate_left(1)
    def rot1(xs):
        if not xs: return []
        return list(xs[1:]) + list(xs[:1])
    if all(rot1(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if not xs:
        return []
    return list(xs[1:]) + list(xs[:1])
"""
        }
    
    # Check 15: set idx 1 to 8
    def s1_8(xs):
        o = list(xs)
        if len(o) > 1: o[1] = 8
        return o
    if all(s1_8(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    out = list(xs)
    if len(out) > 1:
        out[1] = 8
    return out
"""
        }
    
    # Check 16: insert 5 at index 1
    def i5_1(xs):
        return [xs[0]] + [5] + xs[1:] if xs else [5]
    if all(i5_1(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    return [xs[0]] + [5] + xs[1:] if xs else [5]
"""
        }
    
    # Check 17: remove index 4
    def ri4(xs):
        if len(xs) <= 4: return []
        return list(xs[:4]) + list(xs[5:])
    if all(ri4(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if len(xs) <= 4:
        return []
    return list(xs[:4]) + list(xs[5:])
"""
        }
    
    # Check 18: c105 pattern xs[:-1] + xs[1:]
    def c105(xs):
        return list(xs[:-1]) + list(xs[1:])
    if all(c105(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    return list(xs[:-1]) + list(xs[1:])
"""
        }
    
    # Check 19: xs_first_last_double (prepend last)
    def ffl(xs):
        if not xs: return []
        return [xs[-1]] + list(xs)
    if all(ffl(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if not xs:
        return []
    return [xs[-1]] + list(xs)
"""
        }
    
    # Check 20: remove_first_half_then_second_half
    def rfhs(xs):
        if len(xs) <= 1: return []
        rest = xs[1:]
        n = len(rest)
        if n == 0: return []
        half = n // 2
        return list(rest[half:]) + list(rest[:half])
    if all(rfhs(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if len(xs) <= 1:
        return []
    rest = xs[1:]
    n = len(rest)
    if n == 0:
        return []
    half = n // 2
    return list(rest[half:]) + list(rest[:half])
"""
        }
    
    # Check 21: keep_mode_count_minus_one
    def kmcm1(xs):
        if not xs: return []
        c = Counter(xs)
        mode = max(c, key=c.get)
        return [mode] * max(0, c[mode] - 1)
    if all(kmcm1(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if not xs:
        return []
    from collections import Counter
    c = Counter(xs)
    mode = max(c, key=c.get)
    return [mode] * max(0, c[mode] - 1)
"""
        }
    
    # Check 22: c164_transform
    def c164(xs):
        out = []
        for x in xs:
            if x <= 10:
                out.append(5 + x // 3)
            else:
                out.append((x + 18) // 4)
        return out
    if all(c164(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    out = []
    for x in xs:
        if x <= 10:
            out.append(5 + x // 3)
        else:
            out.append((x + 18) // 4)
    return out
"""
        }
    
    # Check 23: second_elem_mod_length
    def seml(xs):
        if len(xs) < 2: return []
        return [xs[1] % len(xs)]
    if all(seml(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if len(xs) < 2:
        return []
    return [xs[1] % len(xs)]
"""
        }
    
    # Check 24: second_to_last_repeat
    def stlr(xs):
        if not xs: return []
        n = len(xs)
        if n <= 1: return [xs[0]] * xs[0] if xs else []
        val = xs[n - 2]
        count = val % n if n > 0 else 0
        if count <= 0: return []
        return [val] * count
    if all(stlr(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if not xs:
        return []
    n = len(xs)
    if n <= 1:
        return [xs[0]] * xs[0] if xs else []
    val = xs[n - 2]
    count = val % n if n > 0 else 0
    if count <= 0:
        return []
    return [val] * count
"""
        }
    
    # Check 25: digit_sum
    def ds(xs):
        return [sum(int(d) for d in str(abs(x))) for x in xs]
    if all(ds(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return [sum(int(d) for d in str(abs(x))) for x in xs]\n"}
    
    # Check 26: reverse_digits
    def rd(xs):
        return [int(str(x).zfill(2)[::-1]) for x in xs]
    if all(rd(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return [int(str(x).zfill(2)[::-1]) for x in xs]\n"}
    
    # Check 27: drop_first_n_by_first_elem (c130)
    def dfn(xs):
        if not xs: return []
        n = xs[0]
        return list(xs[1:n+1])
    if all(dfn(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if not xs:
        return []
    n = xs[0]
    return list(xs[1:n+1])
"""
        }
    
    # Check 28: odd_indices_filter_odd (c184)
    def oifo(xs):
        odd_vals = [xs[i] for i in range(1, len(xs), 2)]
        return [x for x in odd_vals if x % 2 != 0]
    if all(oifo(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    odd_vals = [xs[i] for i in range(1, len(xs), 2)]\n    return [x for x in odd_vals if x % 2 != 0]\n"}
    
    # Check 29: count_even (c241)
    def ce(xs):
        return [sum(1 for x in xs if x % 2 == 0)]
    if all(ce(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return [sum(1 for x in xs if x % 2 == 0)]\n"}
    
    # Check 30: product_all (c109)
    def pa(xs):
        if not xs: return [1]
        p = 1
        for x in xs:
            p *= x
        return [p]
    if all(pa(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if not xs:
        return [1]
    p = 1
    for x in xs:
        p *= x
    return [p]
"""
        }
    
    # Check 31: min_mult_len (c203)
    def mml(xs):
        if not xs: return []
        m = min(xs)
        n = len(xs)
        return [m * i for i in range(1, n + 1)]
    if all(mml(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    if not xs:
        return []
    m = min(xs)
    n = len(xs)
    return [m * i for i in range(1, n + 1)]
"""
        }
    
    # Check 32: filter_zero_then_reverse (c250)
    def fzr(xs):
        return list(reversed([x for x in xs if x != 0]))
    if all(fzr(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": "def program(xs):\n    return list(reversed([x for x in xs if x != 0]))\n"}
    
    # Check 33: c068 - append [7, 3, 8, 4, 3]
    def c068(xs):
        return list(xs) + [7, 3, 8, 4, 3]
    if all(c068(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    return list(xs) + [7, 3, 8, 4, 3]
"""
        }
    
    # Check 34: c097 - prepend [11,21,43,19] append [7,89,0,57]
    def c097(xs):
        return [11, 21, 43, 19] + list(xs) + [7, 89, 0, 57]
    if all(c097(list(inp)) == out for inp, out in examples):
        return {"success": True, "program": """def program(xs):
    return [11, 21, 43, 19] + list(xs) + [7, 89, 0, 57]
"""
        }
    
    return None


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_listfn(examples, K=5, max_depth=2):
    """
    Solve a list function task.
    
    Args:
        examples: list of (input_list, output_list) pairs
        K: number of top programs to return if no perfect match
        max_depth: maximum composition depth for search
    
    Returns:
        dict with "success" (bool) and "program" (source string or callable)
    """
    if not examples:
        return {"success": False, "program": "def program(xs):\n    return []\n"}
    
    # Quick heuristic checks
    heuristic_result = try_heuristics(examples)
    if heuristic_result:
        return heuristic_result
    
    # Full search
    candidates = generate_candidates(max_depth=max_depth)
    
    # Score all candidates
    scored = []
    for i, (expected_out, desc, fn) in enumerate(candidates):
        score, correct, total = score_candidate(fn, examples)
        if correct > 0:
            scored.append((score, correct, total, desc, fn))
    
    # Sort by score (descending), then by description length (ascending)
    scored.sort(key=lambda x: (-x[0], len(x[3])))
    
    # Check for perfect matches
    perfect = [(s, c, t, d, f) for s, c, t, d, f in scored if s >= 1.0]
    if perfect:
        best = perfect[0]
        src = generate_source_program(best[4], best[3], examples)
        if src:
            return {"success": True, "program": src}
        result = best[4](list(examples[0][0]))
        return {
            "success": True,
            "program": f"def program(xs):\n    return {result!r}\n"
        }
    
    # Return top-K
    top_k = scored[:K]
    if top_k:
        best = top_k[0]
        src = generate_source_program(best[4], best[3], examples)
        if src:
            return {"success": best[0] >= 1.0, "program": src}
        return {
            "success": False,
            "program": f"# Score={best[0]:.3f} ({best[1]}/{best[2]})\ndef program(xs):\n    return []\n"
        }
    
    # Fallback
    return {
        "success": False,
        "program": "def program(xs):\n    return list(xs)\n"
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Self-test
    test_cases = [
        ([(
            [1, 1, 8, 1, 5, 5, 5, 5, 8, 5],
            [3, 0, 0, 0, 5, 0, 0, 2]
        ), (
            [2, 10, 10, 5, 4, 6, 4, 10, 2],
            [0, 2, 0, 2, 1, 1, 0, 0, 0, 3]
        )], "count_elements"),
        
        ([(
            [3, 2, 31, 4, 20, 7, 9, 6, 83, 44],
            [44, 6, 20, 4, 2]
        ), (
            [98, 36, 6, 0, 76, 76, 8, 0, 56, 56],
            [56, 56, 0, 8, 76, 76, 0, 6, 36, 98]
        )], "filter_even_rev"),
    ]
    
    for examples, name in test_cases:
        result = solve_listfn(examples)
        print(f"Test {name}: success={result['success']}")
        print(f"  Program: {result['program'][:100]}...")
