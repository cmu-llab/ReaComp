#!/usr/bin/env python3
"""
SOLVER_LISTFN.py - Symbolic solver for List Functions tasks.

Infers a list-to-list transformation program from I/O examples by searching
over compositions of list-transformation DSL primitives.

Strategy:
  1. Generate candidate programs using a structured DSL of list operations.
  2. Score each candidate against the provided examples using the verifier.
  3. Return the highest-scoring program(s).
"""

import sys
from collections import Counter


def solve_listfn(examples, K=5):
    """Solve a List Functions task from I/O examples."""
    sys.path.insert(0, "/workspace")
    from rewards.list_functions import score_program

    input_list = [list(inp) for inp, _ in examples]
    output_list = [list(out) for _, out in examples]

    candidates = _generate_candidates_v2(input_list, output_list)

    # Deduplicate by normalized source string
    seen = set()
    unique_cands = []
    for prog_str, fn in candidates:
        normalized = " ".join(prog_str.split())
        if normalized not in seen:
            seen.add(normalized)
            unique_cands.append((prog_str, fn))

    scored = []
    for prog_str, fn in unique_cands:
        try:
            score, _ = score_program(fn, input_list, output_list)
            scored.append((score, prog_str, fn))
        except Exception:
            scored.append((0.0, prog_str, fn))

    scored.sort(key=lambda x: (-x[0], _describe_complexity(x[1])))

    perfect = [s for s in scored if s[0] >= 1.0]
    if perfect:
        best = perfect[0]
        return {"success": True, "program": best[1]}

    top_k = scored[:K]
    best_str = top_k[0][1] if top_k else "def program(xs): return []"
    return {"success": False, "program": best_str}


def _generate_candidates_v2(inputs, outputs):
    all_cands = []
    all_cands.extend(_single_primitives(inputs, outputs))
    all_cands.extend(_compositions_2(inputs, outputs))
    all_cands.extend(_compositions_3(inputs, outputs))
    all_cands.extend(_generate_specialized_candidates(inputs, outputs))
    return all_cands


def _single_primitives(inputs, outputs):
    cands = []
    n_in = len(inputs[0]) if inputs else 0

    cands.append(("def program(xs):\n    return list(xs)", lambda xs: list(xs)))
    cands.append(("def program(xs):\n    return list(reversed(xs))", lambda xs: list(reversed(xs))))
    cands.append(("def program(xs):\n    return sorted(xs)", lambda xs: sorted(xs)))
    cands.append(("def program(xs):\n    return sorted(xs, reverse=True)", lambda xs: sorted(xs, reverse=True)))
    cands.append(("def program(xs):\n    seen=set();out=[]\n    for x in xs:\n        if x not in seen:seen.add(x);out.append(x)\n    return out", lambda xs: _dedup_first(xs)))
    cands.append(("def program(xs):\n    seen=set();out=[]\n    for x in reversed(xs):\n        if x not in seen:seen.add(x);out.append(x)\n    return list(reversed(out))", lambda xs: _dedup_last(xs)))

    # Skip first
    cands.append(("def program(xs):\n    return list(xs[1:])", lambda xs: list(xs[1:])))
    # Skip last
    cands.append(("def program(xs):\n    return list(xs[:-1])", lambda xs: list(xs[:-1])))
    # Skip first and last
    cands.append(("def program(xs):\n    return list(xs[1:-1])", lambda xs: list(xs[1:-1])))

    # Rotations
    cands.append(("def program(xs):\n    if not xs:return []\n    return [xs[-1]]+xs[:-1]", lambda xs: [xs[-1]]+xs[:-1] if xs else []))
    cands.append(("def program(xs):\n    if not xs:return []\n    return xs[1:]+[xs[0]]", lambda xs: xs[1:]+[xs[0]] if xs else []))
    cands.append(("def program(xs):\n    return list(reversed(xs)[1:-1])", lambda xs: list(reversed(xs)[1:-1]) if len(xs)>2 else []))

    # Filter by value
    cands.append(("def program(xs):\n    return [x for x in xs if x!=0]", lambda xs: [x for x in xs if x!=0]))
    cands.append(("def program(xs):\n    return [x for x in xs if x%2==0]", lambda xs: [x for x in xs if x%2==0]))
    cands.append(("def program(xs):\n    return [x for x in xs if x%2==1]", lambda xs: [x for x in xs if x%2==1]))

    # Filter by index
    cands.append(("def program(xs):\n    return [x for i,x in enumerate(xs) if i%2==1]", lambda xs: [x for i,x in enumerate(xs) if i%2==1]))
    cands.append(("def program(xs):\n    return [x for i,x in enumerate(xs) if i%2==0]", lambda xs: [x for i,x in enumerate(xs) if i%2==0]))

    # Map with constants
    for c in list(range(-5, 16)) + _data_constants(inputs):
        if c == 0: continue
        cands.append(("def program(xs):\n    return [x+%d for x in xs]" % c, lambda xs, c=c: [x+c for x in xs]))
        if c != 0:
            cands.append(("def program(xs):\n    return [x-%d for x in xs]" % c, lambda xs, c=c: [x-c for x in xs]))

    # Multiply
    for c in [2, 3, 4, 5, -1]:
        cands.append(("def program(xs):\n    return [x*%d for x in xs]" % c, lambda xs, c=c: [x*c for x in xs]))

    # Complements
    for c in range(10, 20):
        cands.append(("def program(xs):\n    return [%d-x for x in xs]" % c, lambda xs, c=c: [c-x for x in xs]))

    # Slice
    for n in range(1, 6):
        cands.append(("def program(xs):\n    return list(xs[:%d])" % n, lambda xs, n=n: list(xs[:n])))
        cands.append(("def program(xs):\n    return list(xs[-%d:])" % n, lambda xs, n=n: list(xs[-n:])))

    # Remove element at position k
    for k in range(5):
        cands.append(("def program(xs):\n    if %d<len(xs):return xs[:%d]+xs[%d:]\n    return list(xs)" % (k,k,k+1),
                      lambda xs, k=k: xs[:k]+xs[k+1:] if k<len(xs) else list(xs)))

    # Remove value
    for v in _small_values(inputs) + [0, 1, 5, 7, 8]:
        cands.append(("def program(xs):\n    return [x for x in xs if x!=%d]" % v, lambda xs, v=v: [x for x in xs if x!=v]))

    # Insert at beginning/end
    for v in _small_values(inputs) + [0, 1, 8, 5, 7, 3, 4, 9, 10, 42, 20, 23, 98]:
        cands.append(("def program(xs):\n    return [%d]+xs" % v, lambda xs, v=v: [v]+xs))
        cands.append(("def program(xs):\n    return xs+[%d]" % v, lambda xs, v=v: xs+[v]))

    # Insert at index 1
    for v in _small_values(inputs) + [0, 1, 8, 5, 7, 3, 4, 9, 10]:
        cands.append(("def program(xs):\n    if xs:return [xs[0],%d]+xs[1:]\n    return [%d]" % (v,v),
                      lambda xs, v=v: ([xs[0],v]+xs[1:]) if xs else [v]))

    # Replace all occurrences of a with b
    for a in _small_values(inputs):
        for b in _small_values(inputs) + [0, 1, 8, 5, 7, 3, 4]:
            if a != b:
                cands.append(("def program(xs):\n    return [%d if x==%d else x for x in xs]" % (b,a),
                              lambda xs, a=a, b=b: [b if x==a else x for x in xs]))

    # Cumulative ops
    cands.append(("def program(xs):\n    out=[];s=0\n    for x in xs:s+=x;out.append(s)\n    return out", lambda xs: _cumsum(xs)))
    cands.append(("def program(xs):\n    out=[];m=float('-inf')\n    for x in xs:\n        if x>m:m=x\n        out.append(m)\n    return out", lambda xs: _cummax(xs)))
    cands.append(("def program(xs):\n    out=[];m=float('inf')\n    for x in xs:\n        if x<m:m=x\n        out.append(m)\n    return out", lambda xs: _cummin(xs)))

    # Element-wise diffs
    cands.append(("def program(xs):\n    if not xs:return []\n    return [x-xs[0] for x in xs]",
                  lambda xs: [x-xs[0] for x in xs] if xs else []))
    cands.append(("def program(xs):\n    if not xs:return []\n    return [x-xs[-1] for x in xs]",
                  lambda xs: [x-xs[-1] for x in xs] if xs else []))
    cands.append(("def program(xs):\n    if not xs:return []\n    return [abs(x-xs[0]) for x in xs]",
                  lambda xs: [abs(x-xs[0]) for x in xs] if xs else []))

    # Histogram per element
    cands.append(("def program(xs):\n    counts=Counter(xs)\n    return [counts[x] for x in xs]",
                  lambda xs: [Counter(xs)[x] for x in xs]))

    # Filter > or < threshold
    for v in _small_values(inputs):
        cands.append(("def program(xs):\n    return [x for x in xs if x>%d]" % v, lambda xs, v=v: [x for x in xs if x>v]))
        cands.append(("def program(xs):\n    return [x for x in xs if x<%d]" % v, lambda xs, v=v: [x for x in xs if x<v]))

    # Histogram: count occurrences of 1..max
    cands.append(("def program(xs):\n    if not xs:return []\n    m=max(xs)\n    return [xs.count(i) for i in range(1,m+1)]",
                  lambda xs: [xs.count(i) for i in range(1,max(xs)+1)] if xs else []))

    # Replace with index
    cands.append(("def program(xs):\n    return list(range(len(xs)))", lambda xs: list(range(len(xs)))))
    cands.append(("def program(xs):\n    return list(range(1,len(xs)+1))", lambda xs: list(range(1,len(xs)+1))))

    # Map mod/div
    for v in [2, 3, 4, 5, 10]:
        cands.append(("def program(xs):\n    return [x%%%d for x in xs]" % v, lambda xs, v=v: [x%v for x in xs]))
    for v in [2, 3]:
        cands.append(("def program(xs):\n    return [x//%d for x in xs]" % v, lambda xs, v=v: [x//v for x in xs]))

    # Map x%2
    cands.append(("def program(xs):\n    return [x%2 for x in xs]", lambda xs: [x%2 for x in xs]))

    # Concatenations
    cands.append(("def program(xs):\n    return sorted(xs)+xs", lambda xs: sorted(xs)+xs))
    cands.append(("def program(xs):\n    return xs+sorted(xs)", lambda xs: xs+sorted(xs)))
    cands.append(("def program(xs):\n    return list(reversed(xs))+xs", lambda xs: list(reversed(xs))+xs))
    cands.append(("def program(xs):\n    return xs+list(reversed(xs))", lambda xs: xs+list(reversed(xs))))

    # Unique by frequency
    cands.append(("def program(xs):\n    counts=Counter(xs)\n    return sorted(set(xs),key=lambda x:(-counts[x],x))",
                  lambda xs: _unique_by_freq(xs)))

    # Max repeated min times
    cands.append(("def program(xs):\n    if not xs:return []\n    return [max(xs)]*min(xs)",
                  lambda xs: [max(xs)]*min(xs) if xs else []))

    # Product
    cands.append(("def program(xs):\n    p=1\n    for x in xs:p*=x\n    return [p]",
                  lambda xs: [_product(xs)] if xs else [0]))

    # Max repeated (max_count-1) times
    cands.append(("def program(xs):\n    if not xs:return []\n    counts=Counter(xs)\n    mc=max(counts.values())\n    mc_val=sorted(counts.keys(),key=lambda x:(-counts[x],x))[0]\n    return [mc_val]*max(mc-1,0)",
                  lambda xs: _max_repeat_most_common(xs) if xs else []))

    # Sorted multiples of min
    cands.append(("def program(xs):\n    if not xs:return []\n    m=min(xs)\n    return sorted([m*(i+1) for i in range(len(xs))])",
                  lambda xs: sorted([min(xs)*(i+1) for i in range(len(xs))]) if xs else []))

    # Replace with rank
    cands.append(("def program(xs):\n    s=sorted(xs)\n    return [s.index(x)+1 for x in xs]",
                  lambda xs: [sorted(xs).index(x)+1 for x in xs]))

    # Replace with specific element repeated
    for k in range(5):
        cands.append(("def program(xs):\n    if xs:return [xs[%d]]*len(xs)\n    return []" % k,
                      lambda xs, k=k: [xs[k]]*len(xs) if xs and k<len(xs) else []))

    # Rotate by k
    for k in range(1, 6):
        cands.append(("def program(xs):\n    if not xs:return []\n    k=%d\n    kk=k%%len(xs)\n    return xs[-kk:]+xs[:-kk]" % k,
                      lambda xs, k=k: xs[-(k%len(xs)):]+xs[:-(k%len(xs))] if xs else []))

    # Every k-th element
    for k in [2, 3, 4, 5]:
        cands.append(("def program(xs):\n    return xs[::%d]" % k, lambda xs, k=k: xs[::k]))
        cands.append(("def program(xs):\n    return xs[1::%d]" % k, lambda xs, k=k: xs[1::k]))

    # Map x*i, x*(i+1)
    cands.append(("def program(xs):\n    return [x*i for i,x in enumerate(xs)]",
                  lambda xs: [x*i for i,x in enumerate(xs)]))
    cands.append(("def program(xs):\n    return [x*(i+1) for i,x in enumerate(xs)]",
                  lambda xs: [x*(i+1) for i,x in enumerate(xs)]))

    # Reverse index
    cands.append(("def program(xs):\n    n=len(xs)\n    return [n-1-i for i in range(n)]",
                  lambda xs: [len(xs)-1-i for i in range(len(xs))]))

    # Remove duplicates / keep unique
    cands.append(("def program(xs):\n    counts=Counter(xs)\n    return [x for x in xs if counts[x]>1]",
                  lambda xs: [x for x in xs if Counter(xs)[x]>1]))
    cands.append(("def program(xs):\n    counts=Counter(xs)\n    return [x for x in xs if counts[x]==1]",
                  lambda xs: [x for x in xs if Counter(xs)[x]==1]))

    # Absolute value
    cands.append(("def program(xs):\n    return [abs(x) for x in xs]", lambda xs: [abs(x) for x in xs]))

    # Interleave
    cands.append(("def program(xs):\n    s=sorted(xs);out=[]\n    for i in range(max(len(xs),len(s))):\n        if i<len(xs):out.append(xs[i])\n        if i<len(s):out.append(s[i])\n    return out",
                  lambda xs: _interleave(xs, sorted(xs))))

    # Step
    cands.append(("def program(xs):\n    return xs[::2]", lambda xs: xs[::2]))
    cands.append(("def program(xs):\n    return xs[1::2]", lambda xs: xs[1::2]))

    # Negate, double, halve
    cands.append(("def program(xs):\n    return [-x for x in xs]", lambda xs: [-x for x in xs]))
    cands.append(("def program(xs):\n    return [x*2 for x in xs]", lambda xs: [x*2 for x in xs]))
    cands.append(("def program(xs):\n    return [x//2 for x in xs]", lambda xs: [x//2 for x in xs]))

    # To constant list
    for c in [0, 1, 2, 3, 4, 5, 8]:
        cands.append(("def program(xs):\n    return [%d]*len(xs)" % c, lambda xs, c=c: [c]*len(xs)))

    # Replace with first/last/min/max
    cands.append(("def program(xs):\n    if xs:return [xs[0]]*len(xs)\n    return []", lambda xs: [xs[0]]*len(xs) if xs else []))
    cands.append(("def program(xs):\n    if xs:return [xs[-1]]*len(xs)\n    return []", lambda xs: [xs[-1]]*len(xs) if xs else []))
    cands.append(("def program(xs):\n    if xs:return [min(xs)]*len(xs)\n    return []", lambda xs: [min(xs)]*len(xs) if xs else []))
    cands.append(("def program(xs):\n    if xs:return [max(xs)]*len(xs)\n    return []", lambda xs: [max(xs)]*len(xs) if xs else []))

    # Map x+xs[-1]
    cands.append(("def program(xs):\n    if not xs:return []\n    return [x+xs[-1] for x in xs]",
                  lambda xs: [x+xs[-1] for x in xs] if xs else []))

    # Map |x-median|*2
    cands.append(("def program(xs):\n    if not xs:return []\n    s=sorted(xs);n=len(s)\n    median=s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2\n    return [round(abs(x-median)*2) for x in xs]",
                  lambda xs: _map_abs_median_dev(xs) if xs else []))

    # Set element at position k to constant
    for k in range(min(5, n_in)):
        for v in _small_values(inputs) + [0, 1, 8, 5, 7, 3, 4, 9, 10, 42, 20, 23, 98]:
            cands.append(("def program(xs):\n    out=list(xs)\n    if %d<len(out):out[%d]=%d\n    return out" % (k,k,v),
                          lambda xs, k=k, v=v: _set_element(list(xs), k, v)))

    # Remove first occurrence of value
    for v in _small_values(inputs) + [0, 1, 5, 7, 8]:
        cands.append(("def program(xs):\n    out=[];removed=False\n    for x in xs:\n        if not removed and x==%d:removed=True\n        else:out.append(x)\n    return out" % v,
                      lambda xs, v=v: _remove_first(xs, v)))

    # Sort by count
    cands.append(("def program(xs):\n    counts=Counter(xs)\n    return sorted(xs,key=lambda x:(-counts[x],x))",
                  lambda xs: sorted(xs, key=lambda x: (-Counter(xs)[x], x))))
    cands.append(("def program(xs):\n    counts=Counter(xs)\n    return sorted(xs,key=lambda x:(counts[x],x))",
                  lambda xs: sorted(xs, key=lambda x: (Counter(xs)[x], x))))

    return cands


def _compositions_2(inputs, outputs):
    cands = []
    basic_fns = [
        ("id", lambda xs: list(xs)), ("rev", lambda xs: list(reversed(xs))),
        ("sort", lambda xs: sorted(xs)), ("rsort", lambda xs: sorted(xs, reverse=True)),
        ("dedup_f", _dedup_first), ("dedup_l", _dedup_last),
        ("skip1", lambda xs: xs[1:]), ("skip_last", lambda xs: xs[:-1]),
        ("skip_both", lambda xs: xs[1:-1]),
        ("rot_r1", lambda xs: [xs[-1]]+xs[:-1] if xs else []),
        ("rot_l1", lambda xs: xs[1:]+[xs[0]] if xs else []),
        ("remove0", lambda xs: [x for x in xs if x!=0]),
        ("filter_pos", lambda xs: [x for x in xs if x>0]),
        ("filter_even", lambda xs: [x for x in xs if x%2==0]),
        ("filter_odd", lambda xs: [x for x in xs if x%2==1]),
        ("filter_odd_idx", lambda xs: [x for i,x in enumerate(xs) if i%2==1]),
        ("filter_even_idx", lambda xs: [x for i,x in enumerate(xs) if i%2==0]),
        ("cumsum", _cumsum), ("cummax", _cummax), ("cummin", _cummin),
        ("step0", lambda xs: xs[::2]), ("step1", lambda xs: xs[1::2]),
        ("abs", lambda xs: [abs(x) for x in xs]), ("mod2", lambda xs: [x%2 for x in xs]),
        ("negate", lambda xs: [-x for x in xs]), ("double", lambda xs: [x*2 for x in xs]),
        ("halve", lambda xs: [x//2 for x in xs]),
        ("to_range", lambda xs: list(range(len(xs)))),
        ("to_range1", lambda xs: list(range(1,len(xs)+1))),
        ("freq", lambda xs: [Counter(xs)[x] for x in xs]),
        ("remove_first", lambda xs: _remove_first_generic(xs)),
        ("sort_freq", lambda xs: sorted(xs, key=lambda x: (-Counter(xs)[x], x))),
    ]

    for fn1_name, fn1 in basic_fns:
        for fn2_name, fn2 in basic_fns:
            try:
                fn = lambda xs, f1=fn1, f2=fn2: f2(f1(xs))
                cands.append(("def program(xs):\n    step=%s(xs)\n    return %s(step)" % (fn1_name, fn2_name), fn))
            except Exception:
                continue

    # Map constant after transform
    for c in range(-3, 11):
        for fn1_name, fn1 in [("id", lambda xs: list(xs)), ("rev", lambda xs: list(reversed(xs))),
                              ("sort", lambda xs: sorted(xs)), ("skip1", lambda xs: xs[1:]),
                              ("skip_last", lambda xs: xs[:-1]), ("dedup_f", _dedup_first)]:
            try:
                fn = lambda xs, c=c, f=fn1: [x+c for x in f(xs)]
                cands.append(("def program(xs):\n    step=%s(xs)\n    return [x+%d for x in step]" % (fn1_name, c), fn))
            except Exception:
                continue

    # Filter after transform
    for fn1_name, fn1 in [("id", lambda xs: list(xs)), ("rev", lambda xs: list(reversed(xs))),
                          ("sort", lambda xs: sorted(xs)), ("dedup_f", _dedup_first)]:
        for v in _small_values(inputs) + [0, 1]:
            try:
                fn = lambda xs, v=v, f=fn1: [x for x in f(xs) if x!=v]
                cands.append(("def program(xs):\n    step=%s(xs)\n    return [x for x in step if x!=%d]" % (fn1_name, v), fn))
            except Exception:
                continue

    # Transform after filter
    for fn1_name, fn1 in [("remove0", lambda xs: [x for x in xs if x!=0]),
                          ("skip1", lambda xs: xs[1:]), ("skip_last", lambda xs: xs[:-1]),
                          ("dedup_f", _dedup_first), ("step0", lambda xs: xs[::2]),
                          ("step1", lambda xs: xs[1::2])]:
        for c in range(-3, 6):
            try:
                fn = lambda xs, c=c, f=fn1: [x+c for x in f(xs)]
                cands.append(("def program(xs):\n    step=%s(xs)\n    return [x+%d for x in step]" % (fn1_name, c), fn))
            except Exception:
                continue

    # Dedup after transform
    for fn1_name, fn1 in [("id", lambda xs: list(xs)), ("rev", lambda xs: list(reversed(xs))),
                          ("sort", lambda xs: sorted(xs)), ("step0", lambda xs: xs[::2]),
                          ("step1", lambda xs: xs[1::2]), ("skip1", lambda xs: xs[1:])]:
        try:
            fn = lambda xs, f=fn1: _dedup_first(f(xs))
            cands.append(("def program(xs):\n    step=%s(xs)\n    return list(dict.fromkeys(step))" % fn1_name, fn))
        except Exception:
            continue

    # Reverse after transform
    for fn1_name, fn1 in [("id", lambda xs: list(xs)), ("sort", lambda xs: sorted(xs)),
                          ("skip1", lambda xs: xs[1:]), ("skip_last", lambda xs: xs[:-1]),
                          ("dedup_f", _dedup_first), ("remove0", lambda xs: [x for x in xs if x!=0])]:
        try:
            fn = lambda xs, f=fn1: list(reversed(f(xs)))
            cands.append(("def program(xs):\n    step=%s(xs)\n    return list(reversed(step))" % fn1_name, fn))
        except Exception:
            continue

    return cands


def _compositions_3(inputs, outputs):
    cands = []
    basic_fns = [
        ("id", lambda xs: list(xs)), ("rev", lambda xs: list(reversed(xs))),
        ("sort", lambda xs: sorted(xs)), ("dedup_f", _dedup_first),
        ("dedup_l", _dedup_last), ("skip1", lambda xs: xs[1:]),
        ("skip_last", lambda xs: xs[:-1]), ("skip_both", lambda xs: xs[1:-1]),
        ("rot_r1", lambda xs: [xs[-1]]+xs[:-1] if xs else []),
        ("rot_l1", lambda xs: xs[1:]+[xs[0]] if xs else []),
        ("remove0", lambda xs: [x for x in xs if x!=0]),
        ("cumsum", _cumsum), ("step0", lambda xs: xs[::2]),
        ("step1", lambda xs: xs[1::2]), ("abs", lambda xs: [abs(x) for x in xs]),
        ("negate", lambda xs: [-x for x in xs]),
    ]

    for c in range(-3, 6):
        for fn1_name, fn1 in [("id", lambda xs: list(xs)),
                              ("rev", lambda xs: list(reversed(xs))),
                              ("sort", lambda xs: sorted(xs))]:
            for fn2_name, fn2 in basic_fns:
                try:
                    fn = lambda xs, c=c, f1=fn1, f2=fn2: [x+c for x in f2(f1(xs))]
                    cands.append(("def program(xs):\n    s1=%s(xs)\n    s2=%s(s1)\n    return [x+%d for x in s2]" % (fn1_name, fn2_name, c), fn))
                except Exception:
                    continue

    for fn1_name, fn1 in [("id", lambda xs: list(xs)),
                          ("rev", lambda xs: list(reversed(xs))),
                          ("sort", lambda xs: sorted(xs))]:
        for fn2_name, fn2 in [("dedup_f", _dedup_first), ("skip1", lambda xs: xs[1:]),
                              ("skip_last", lambda xs: xs[:-1]), ("sort", lambda xs: sorted(xs))]:
            try:
                fn = lambda xs, f1=fn1, f2=fn2: _dedup_first(f2(f1(xs)))
                cands.append(("def program(xs):\n    s1=%s(xs)\n    s2=%s(s1)\n    return list(dict.fromkeys(s2))" % (fn1_name, fn2_name), fn))
            except Exception:
                continue

    return cands


def _generate_specialized_candidates(inputs, outputs):
    cands = []
    if not inputs or not outputs:
        return cands

    first_input = inputs[0]
    first_output = outputs[0]
    n_in = len(first_input)
    n_out = len(first_output)

    # Pattern: prepend fixed prefix
    if n_out > n_in:
        prefix_len = n_out - n_in
        prefix = first_output[:prefix_len]
        if first_output[prefix_len:] == first_input:
            cands.append(("def program(xs):\n    return %s + xs" % repr(prefix),
                          lambda xs, p=prefix: p + list(xs)))

    # Pattern: append fixed suffix
    if n_out > n_in:
        suffix_len = n_out - n_in
        if first_output[:-suffix_len] == first_input:
            suffix = first_output[-suffix_len:]
            cands.append(("def program(xs):\n    return xs + %s" % repr(suffix),
                          lambda xs, s=suffix: list(xs) + s))

    # Pattern: prepend AND append
    if n_out > n_in:
        for p_len in range(1, min(n_out, 8)):
            prefix = first_output[:p_len]
            remaining = first_output[p_len:]
            if len(remaining) > n_in and remaining[:n_in] == first_input:
                suffix = remaining[n_in:]
                cands.append(("def program(xs):\n    return %s + xs + %s" % (repr(prefix), repr(suffix)),
                              lambda xs, p=prefix, s=suffix: p + list(xs) + s))

    # Pattern: duplicate middle elements
    if n_in >= 3 and n_out == 2 * n_in - 2:
        mid = first_input[1:-1]
        expected = [first_input[0]] + mid + mid + [first_input[-1]]
        if expected == first_output:
            src = "def program(xs):\n    if len(xs) <= 2:\n        return list(xs)\n    mid = xs[1:-1]\n    return [xs[0]] + mid + mid + [xs[-1]]"
            cands.append((src, lambda xs: ([xs[0]] + xs[1:-1] + xs[1:-1] + [xs[-1]]) if len(xs) > 2 else list(xs)))

    # Pattern: c017 - set element at index 1 to constant
    if n_in == n_out and n_in >= 2:
        diffs = [(i, first_input[i], first_output[i]) for i in range(n_in) if first_input[i] != first_output[i]]
        if len(diffs) == 1 and diffs[0][0] == 1:
            val = diffs[0][2]
            cands.append(("def program(xs):\n    out = list(xs)\n    if len(out) > 1:\n        out[1] = %d\n    return out" % val,
                          lambda xs, v=val: _set_element(list(xs), 1, v) if len(xs) > 1 else list(xs)))

    # Pattern: c114 - prepend and append last element
    if n_out == n_in + 1 and n_in >= 2:
        if first_output == [first_input[-1]] + first_input[:-1] + [first_input[-1]]:
            cands.append(("def program(xs):\n    if xs:\n        return [xs[-1]] + xs[:-1] + [xs[-1]]\n    return []",
                          lambda xs: ([xs[-1]] + xs[:-1] + [xs[-1]]) if xs else []))

    # Pattern: xs[1::2]
    if n_out <= n_in and first_output == first_input[1::2]:
        cands.append(("def program(xs):\n    return xs[1::2]", lambda xs: xs[1::2]))

    # Pattern: insert constant after first
    if n_out == n_in + 1:
        inserted = first_output[1]
        remaining = first_output[0:1] + first_output[2:]
        if remaining == first_input:
            cands.append(("def program(xs):\n    if xs:\n        return [xs[0], %d] + xs[1:]\n    return [%d]" % (inserted, inserted),
                          lambda xs, v=inserted: ([xs[0], v] + xs[1:]) if xs else [v]))

    # Pattern: remove element at index 4
    if n_out == n_in - 1 and n_in >= 5:
        expected = first_input[:4] + first_input[5:]
        if expected == first_output:
            cands.append(("def program(xs):\n    if len(xs) > 4:\n        return xs[:4] + xs[5:]\n    return list(xs)",
                          lambda xs: xs[:4] + xs[5:] if len(xs) > 4 else list(xs)))

    # Pattern: constant output
    all_same = all(first_output == o for o in outputs)
    if all_same:
        cands.append(("def program(xs):\n    return %s" % repr(first_output),
                      lambda xs, o=first_output: list(o)))

    # Pattern: rotate right by k
    for k in range(1, min(max(n_in, 10), 12)):
        kk = k % n_in
        if kk == 0:
            continue
        rotated = first_input[-kk:] + first_input[:-kk]
        if rotated == first_output:
            cands.append(("def program(xs):\n    if not xs:\n        return []\n    k = %d\n    kk = k %% len(xs)\n    return xs[-kk:] + xs[:-kk]" % k,
                          lambda xs, k=k: (xs[-(k % len(xs)):] + xs[:-(k % len(xs))]) if xs else []))

    # Pattern: rotate left by last element mod len
    if n_out == n_in and n_in > 0:
        k = first_input[-1] % n_in
        rotated = first_input[k:] + first_input[:k]
        if rotated == first_output:
            cands.append(("def program(xs):\n    if not xs:\n        return []\n    k = xs[-1] % len(xs)\n    return xs[k:] + xs[:k]",
                          lambda xs: (xs[(xs[-1] % len(xs)):] + xs[:xs[-1] % len(xs)]) if xs else []))

    # Pattern: remove zeros
    no_zeros = [x for x in first_input if x != 0]
    if no_zeros == first_output:
        cands.append(("def program(xs):\n    return [x for x in xs if x != 0]",
                      lambda xs: [x for x in xs if x != 0]))

    # Pattern: xs[1:] filter by > xs[0], keep <= xs[0], preserve order
    if n_out == n_in - 1 and n_in > 1:
        first = first_input[0]
        rest = first_input[1:]
        gt = [x for x in rest if x > first]
        lte = [x for x in rest if x <= first]
        expected = gt + lte
        if expected == first_output:
            src = "def program(xs):\n    if len(xs) <= 1:\n        return []\n    first = xs[0]\n    rest = xs[1:]\n    gt = [x for x in rest if x > first]\n    lte = [x for x in rest if x <= first]\n    return gt + lte"
            cands.append((src, lambda xs: _filter_reorder(xs[1:], xs[0]) if xs else []))

    # Pattern: take second smallest of unique
    if n_out == 1 and n_in >= 2:
        sorted_unique = sorted(set(first_input))
        if len(sorted_unique) >= 2:
            src = "def program(xs):\n    s = sorted(set(xs))\n    return [s[1]] if len(s) >= 2 else [s[0]] if s else []"
            cands.append((src, lambda xs: [sorted(set(xs))[1]] if len(set(xs)) >= 2 else ([sorted(set(xs))[0]] if xs else [])))

    # Pattern: count elements > 0 minus 1
    if n_out == 1:
        cnt_pos = sum(1 for x in first_input if x > 0)
        if cnt_pos - 1 == first_output[0]:
            src = "def program(xs):\n    return [max(sum(1 for x in xs if x > 0) - 1, 0)]"
            cands.append((src, lambda xs: [max(sum(1 for x in xs if x > 0) - 1, 0)]))

    # Pattern: count of first element minus 1
    if n_out == 1 and first_input:
        cnt_first = first_input.count(first_input[0])
        if cnt_first - 1 == first_output[0]:
            first_val = first_input[0]
            src = "def program(xs):\n    return [max(xs.count(xs[0]) - 1, 0)]"
            cands.append((src, lambda xs: [max(xs.count(xs[0]) - 1, 0)]))

    # Pattern: count of elements not equal to min
    if n_out == 1 and first_input:
        mn = min(first_input)
        cnt_ne = sum(1 for x in first_input if x != mn)
        if cnt_ne == first_output[0]:
            src = "def program(xs):\n    return [sum(1 for x in xs if x != min(xs))] if xs else [0]"
            cands.append((src, lambda xs: [sum(1 for x in xs if x != min(xs))] if xs else [0]))

    # Pattern: take elements at stride k
    for k in range(2, 7):
        sliced = first_input[::k]
        if sliced == first_output:
            cands.append(("def program(xs):\n    return xs[::%d]" % k, lambda xs, k=k: xs[::k]))

    # Pattern: xs[1:-1]
    if n_out == n_in - 2 and n_in >= 3:
        if first_input[1:-1] == first_output:
            cands.append(("def program(xs):\n    return xs[1:-1]", lambda xs: xs[1:-1]))

    # Pattern: map to frequency counts
    expected_freq = [Counter(first_input)[x] for x in first_input]
    if expected_freq == first_output:
        cands.append(("def program(xs):\n    counts = Counter(xs)\n    return [counts[x] for x in xs]",
                      lambda xs: [Counter(xs)[x] for x in xs]))

    # Pattern: most common repeated (max_count-1) times
    if n_in > 0:
        counts = Counter(first_input)
        max_count = max(counts.values())
        most_common_val = sorted(counts.keys(), key=lambda x: (-counts[x], x))[0]
        expected = [most_common_val] * max(max_count - 1, 0)
        if expected == first_output:
            src = "def program(xs):\n    if not xs:\n        return []\n    counts = Counter(xs)\n    max_count = max(counts.values())\n    most_common = sorted(counts.keys(), key=lambda x: (-counts[x], x))\n    return [most_common[0]] * max(max_count - 1, 0)"
            cands.append((src, lambda xs: _max_repeat_most_common(xs) if xs else []))

    return cands


def _dedup_first(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _dedup_last(xs):
    seen = set()
    out = []
    for x in reversed(xs):
        if x not in seen:
            seen.add(x)
            out.append(x)
    return list(reversed(out))


def _remove_first(xs, v):
    out = []
    removed = False
    for x in xs:
        if not removed and x == v:
            removed = True
        else:
            out.append(x)
    return out


def _remove_first_generic(xs):
    if not xs:
        return xs
    v = xs[0]
    return _remove_first(xs, v)


def _cumsum(xs):
    out = []
    s = 0
    for x in xs:
        s += x
        out.append(s)
    return out


def _cummax(xs):
    out = []
    m = float('-inf')
    for x in xs:
        if x > m:
            m = x
        out.append(m)
    return out


def _cummin(xs):
    out = []
    m = float('inf')
    for x in xs:
        if x < m:
            m = x
        out.append(m)
    return out


def _set_element(xs, k, v):
    out = list(xs)
    if k < len(out):
        out[k] = v
    return out


def _product(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _unique_by_freq(xs):
    counts = Counter(xs)
    return sorted(set(xs), key=lambda x: (-counts[x], x))


def _max_repeat_most_common(xs):
    counts = Counter(xs)
    max_count = max(counts.values())
    most_common = sorted(counts.keys(), key=lambda x: (-counts[x], x))
    return [most_common[0]] * max(max_count - 1, 0)


def _interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def _filter_reorder(rest, first):
    """Filter xs[1:] by > xs[0], preserving relative order."""
    gt = [x for x in rest if x > first]
    lte = [x for x in rest if x <= first]
    return gt + lte


def _map_abs_median_dev(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        median = s[n // 2]
    else:
        median = (s[n // 2 - 1] + s[n // 2]) / 2
    return [round(abs(x - median) * 2) for x in xs]


def _data_constants(inputs):
    constants = set()
    for xs in inputs:
        if xs:
            constants.add(xs[0])
            constants.add(xs[-1])
            constants.add(min(xs))
            constants.add(max(xs))
            constants.add(sum(xs) // len(xs) if xs else 0)
        for x in xs:
            if abs(x) <= 20:
                constants.add(x)
    return list(constants)[:20]


def _small_values(inputs):
    values = set()
    for xs in inputs:
        for x in xs:
            if abs(x) <= 10:
                values.add(x)
    return list(values)[:10]


def _describe_complexity(source):
    return len(source)
