"""
SOLVER.py

Symbolic program synthesis for List Functions tasks.

Searches over a DSL of list-to-list transformation primitives and their
compositions, scoring candidates against provided (input, output) examples
using the verifier from rewards.list_functions.

Returns the first program scoring 1.0 on all examples, or the top-K highest
scoring programs if no perfect match is found.
"""

import sys

sys.path.insert(0, "/workspace")
from rewards.list_functions import score_program


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _dedup(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _count(lst, val):
    return sum(1 for x in lst if x == val)


# ──────────────────────────────────────────────────────────────────────────────
# Level-0: Trivial programs
# ──────────────────────────────────────────────────────────────────────────────

def _trivial_programs(examples):
    programs = []
    if not examples:
        return programs
    outs0 = examples[0][1]
    programs.append((lambda xs: list(xs), "def program(xs):\n    return list(xs)"))
    programs.append((lambda xs: list(outs0), "def program(xs):\n    return " + repr(outs0)))
    programs.append((lambda xs: list(xs), "def program(xs):\n    return xs.copy()"))
    if outs0:
        for v in outs0[:5]:
            fn = lambda xs, v=v: [v]
            src = "def program(xs):\n    return [" + repr(v) + "]"
            programs.append((fn, src))
    return programs


# ──────────────────────────────────────────────────────────────────────────────
# Level-1: Primitives
# ──────────────────────────────────────────────────────────────────────────────

def _elementwise_programs(examples):
    programs = []
    for _inp in [e[0] for e in examples]:
        for a in range(-5, 11):
            for b in range(-20, 21):
                programs.append(
                    (lambda xs, aa=a, bb=b: [aa * x + bb for x in xs],
                     "def program(xs):\n    return [" + repr(a) + "*x+" + repr(b) + " for x in xs]"))
                programs.append(
                    (lambda xs, aa=a, bb=b: [aa * x - bb for x in xs],
                     "def program(xs):\n    return [" + repr(a) + "*x-" + repr(b) + " for x in xs]"))
                if a != 0:
                    programs.append(
                        (lambda xs, aa=a, bb=b: [x // aa + bb for x in xs],
                         "def program(xs):\n    return [x//" + repr(a) + "+" + repr(b) + " for x in xs]"))
                    programs.append(
                        (lambda xs, aa=a, bb=b: [x // aa - bb for x in xs],
                         "def program(xs):\n    return [x//" + repr(a) + "-" + repr(b) + " for x in xs]"))
                programs.append(
                    (lambda xs, bb=b: [x + bb for x in xs],
                     "def program(xs):\n    return [x+" + repr(b) + " for x in xs]"))
                programs.append(
                    (lambda xs, bb=b: [x - bb for x in xs],
                     "def program(xs):\n    return [x-" + repr(b) + " for x in xs]"))
                if a != 0:
                    programs.append(
                        (lambda xs, aa=a: [aa * x for x in xs],
                         "def program(xs):\n    return [" + repr(a) + "*x for x in xs]"))
    return programs


def _sort_reverse_dedup_programs(examples):
    return [
        (lambda xs: sorted(xs), "def program(xs):\n    return sorted(xs)"),
        (lambda xs: sorted(xs, reverse=True),
         "def program(xs):\n    return sorted(xs, reverse=True)"),
        (lambda xs: list(reversed(xs)),
         "def program(xs):\n    return list(reversed(xs))"),
        (lambda xs: _dedup(xs), "def program(xs):\n    return _dedup(xs)"),
        (lambda xs: list(dict.fromkeys(xs)),
         "def program(xs):\n    return list(dict.fromkeys(xs))"),
        (lambda xs: sorted(set(xs)),
         "def program(xs):\n    return sorted(set(xs))"),
    ]


def _filter_programs(examples):
    programs = []
    programs.append((lambda xs: [x for x in xs if x % 2 == 0],
                      "def program(xs):\n    return [x for x in xs if x%2==0]"))
    programs.append((lambda xs: [x for x in xs if x % 2 == 1],
                      "def program(xs):\n    return [x for x in xs if x%2==1]"))
    vals = set()
    for ex in examples:
        for xi in ex[0]:
            vals.add(abs(xi))
            vals.add(abs(xi) + 1)
    for t in sorted(vals)[:40]:
        programs.append((lambda xs, tt=t: [x for x in xs if x > tt],
                          "def program(xs):\n    return [x for x in xs if x>" + repr(t) + "]"))
        programs.append((lambda xs, tt=t: [x for x in xs if x >= tt],
                          "def program(xs):\n    return [x for x in xs if x>=" + repr(t) + "]"))
        programs.append((lambda xs, tt=t: [x for x in xs if x < tt],
                          "def program(xs):\n    return [x for x in xs if x<" + repr(t) + "]"))
        programs.append((lambda xs, tt=t: [x for x in xs if x <= tt],
                          "def program(xs):\n    return [x for x in xs if x<=" + repr(t) + "]"))
        programs.append((lambda xs, tt=t: [x for x in xs if x == tt],
                          "def program(xs):\n    return [x for x in xs if x==" + repr(t) + "]"))
        programs.append((lambda xs, tt=t: [x for x in xs if x != tt],
                          "def program(xs):\n    return [x for x in xs if x!=" + repr(t) + "]"))
    return programs


def _slice_programs(examples):
    programs = []
    for inp in [e[0] for e in examples]:
        n = len(inp)
        if n == 0:
            programs.append((lambda xs: [], "def program(xs):\n    return []"))
            continue
        programs.append((lambda xs: xs[1:], "def program(xs):\n    return xs[1:]"))
        programs.append((lambda xs: xs[:-1], "def program(xs):\n    return xs[:-1]"))
        programs.append((lambda xs: xs[1:-1], "def program(xs):\n    return xs[1:-1]"))
        programs.append((lambda xs: xs[-1:], "def program(xs):\n    return xs[-1:]"))
        programs.append((lambda xs: xs[:1], "def program(xs):\n    return xs[:1]"))
        programs.append((lambda xs: xs[1:xs[0]+1] if len(xs) > 1 else [],
                          "def program(xs):\n    return xs[1:xs[0]+1] if len(xs)>1 else []"))
        programs.append((lambda xs: xs[-min(xs[0],len(xs)):] if xs else [],
                          "def program(xs):\n    if not xs: return []\n    return xs[-min(xs[0],len(xs)):]))"))
        programs.append((lambda xs: xs[::2], "def program(xs):\n    return xs[::2]"))
        programs.append((lambda xs: xs[1::2], "def program(xs):\n    return xs[1::2]"))
        programs.append((lambda xs: xs[::-2], "def program(xs):\n    return xs[::-2]"))
        programs.append((lambda xs: xs[:-1][::-1] if xs else [],
                          "def program(xs):\n    return xs[:-1][::-1] if xs else []"))
        for k in range(1, min(n + 3, 10)):
            programs.append((lambda xs, kk=k: xs[:kk],
                              "def program(xs):\n    return xs[:" + repr(k) + "]"))
            programs.append((lambda xs, kk=k: xs[-kk:] if xs else [],
                              "def program(xs):\n    return xs[-" + repr(k) + ":] if xs else []"))
        for k in range(1, min(n + 3, 10)):
            programs.append((lambda xs, kk=k: xs[k:] if xs else [],
                              "def program(xs):\n    return xs[" + repr(k) + ":] if xs else []"))
        for a in range(0, 4):
            for b in range(0, 4):
                programs.append((lambda xs, aa=a, bb=b: xs[aa:bb] if len(xs) >= bb else xs[aa:min(bb, len(xs))],
                                  "def program(xs):\n    return xs[" + repr(a) + ":" + repr(b) + "]"))
    return programs


def _scalar_output_programs(examples):
    programs = []
    programs.append((lambda xs: [sum(xs)] if xs else [],
                      "def program(xs):\n    return [sum(xs)] if xs else []"))
    programs.append((lambda xs: [max(xs)] if xs else [],
                      "def program(xs):\n    return [max(xs)] if xs else []"))
    programs.append((lambda xs: [min(xs)] if xs else [],
                      "def program(xs):\n    return [min(xs)] if xs else []"))
    programs.append((lambda xs: [len(xs)], "def program(xs):\n    return [len(xs)]"))
    programs.append((lambda xs: ([1 if not xs else __import__('functools').reduce(lambda a,b:a*b, xs)]),
                      "def program(xs):\n    r=1\n    for x in xs: r*=x\n    return [r]"))
    programs.append((lambda xs: ([0] if not xs else [max(__import__('collections').Counter(xs).values()) - 1]),
                      "def program(xs):\n    if not xs: return [0]\n    c={}\n    for x in xs: c[x]=c.get(x,0)+1\n    return [max(c.values())-1]"))
    programs.append((lambda xs: [sum(1 for x in xs if x % 2 == 0)],
                      "def program(xs):\n    return [sum(1 for x in xs if x%2==0)]"))
    programs.append((lambda xs: [sum(1 for x in xs if x % 2 == 1)],
                      "def program(xs):\n    return [sum(1 for x in xs if x%2==1)]"))
    programs.append((lambda xs: ([0] if not xs else [_count(xs, xs[0]) - 1]),
                      "def program(xs):\n    if not xs: return [0]\n    return [sum(1 for x in xs if x==xs[0])-1]"))
    programs.append((lambda xs: ([max(xs)] * min(xs)) if xs else [],
                      "def program(xs):\n    if not xs: return []\n    return [max(xs)]*min(xs)"))
    programs.append((lambda xs: ([xs[0]] * max(_count(xs, xs[0]) - 1, 0)) if xs else [],
                      "def program(xs):\n    if not xs: return []\n    c=sum(1 for x in xs if x==xs[0])-1\n    return [xs[0]]*max(c,0)"))
    return programs


def _histogram_programs(examples):
    def _hist1(xs):
        if not xs:
            return []
        return [xs.count(i) for i in range(1, max(xs) + 1)]

    def _hist0(xs):
        if not xs:
            return []
        return [xs.count(i) for i in range(max(xs))]

    def _freq_sorted(xs):
        if not xs:
            return []
        return [xs.count(v) for v in sorted(set(xs))]

    def _freq_dedup(xs):
        if not xs:
            return []
        return [xs.count(v) for v in _dedup(xs)]

    return [
        (_hist1, "def program(xs):\n    if not xs: return []\n    return [xs.count(i) for i in range(1,max(xs)+1)]"),
        (_hist0, "def program(xs):\n    if not xs: return []\n    m=max(xs)\n    return [xs.count(i) for i in range(m)]"),
        (_freq_sorted, "def program(xs):\n    if not xs: return []\n    return [xs.count(v) for v in sorted(set(xs))]"),
        (_freq_dedup, "def program(xs):\n    if not xs: return []\n    return [xs.count(v) for v in _dedup(xs)]"),
    ]


def _index_programs(examples):
    programs = []
    for i in range(0, 5):
        programs.append((lambda xs, idx=i: ([xs[idx]] if len(xs) > idx else []),
                          "def program(xs):\n    return [xs[" + repr(i) + "]] if len(xs)>" + repr(i) + " else []"))
    for n in range(1, 5):
        programs.append((lambda xs, nn=n: ([xs[-nn]] if len(xs) >= nn else []),
                          "def program(xs):\n    return [xs[-" + repr(n) + "]] if len(xs)>=" + repr(n) + " else []"))
    programs.append((lambda xs: ([xs[-2]] if len(xs) >= 2 else []),
                      "def program(xs):\n    return [xs[-2]] if len(xs)>=2 else []"))
    programs.append((lambda xs: [x + i for i, x in enumerate(xs)],
                      "def program(xs):\n    return [x+i for i,x in enumerate(xs)]"))
    programs.append((lambda xs: [x - i for i, x in enumerate(xs)],
                      "def program(xs):\n    return [x-i for i,x in enumerate(xs)]"))
    programs.append((lambda xs: [x * (i + 1) for i, x in enumerate(xs)],
                      "def program(xs):\n    return [x*(i+1) for i,x in enumerate(xs)]"))
    programs.append((lambda xs: [x for i, x in enumerate(xs) if i % 2 == 1 and x % 2 == 1],
                      "def program(xs):\n    return [x for i,x in enumerate(xs) if i%2==1 and x%2==1]"))
    programs.append((lambda xs: [x for i, x in enumerate(xs) if i % 2 == 1 and x % 2 == 0],
                      "def program(xs):\n    return [x for i,x in enumerate(xs) if i%2==1 and x%2==0]"))
    programs.append((lambda xs: [x for i, x in enumerate(xs) if i % 2 == 0 and x % 2 == 1],
                      "def program(xs):\n    return [x for i,x in enumerate(xs) if i%2==0 and x%2==1]"))
    programs.append((lambda xs: [x for i, x in enumerate(xs) if i % 2 == 0 and x % 2 == 0],
                      "def program(xs):\n    return [x for i,x in enumerate(xs) if i%2==0 and x%2==0]"))
    return programs


def _two_digit_rev_programs(examples):
    def _rev2(xs):
        def rev2(x):
            if x == 0:
                return 0
            s = f"{x:02d}"
            return int(s[::-1])
        return [rev2(x) for x in xs]
    return [(_rev2,
              "def program(xs):\n    def r(x):\n        if x==0: return 0\n        return int(f'{x:02d}'[::-1])\n    return [r(x) for x in xs]")]


def _split_reverse_programs(examples):
    def _split_rev_0(xs):
        result = []
        seg = []
        for x in xs:
            if x == 0:
                result.extend(reversed(seg))
                seg = []
            else:
                seg.append(x)
        result.extend(reversed(seg))
        return result
    return [(_split_rev_0,
              "def program(xs):\n    result=[]; seg=[]\n    for x in xs:\n        if x==0:\n            result.extend(reversed(seg)); seg=[]\n        else:\n            seg.append(x)\n    result.extend(reversed(seg))\n    return result")]


def _pivot_programs(examples):
    def _pivot_part(xs):
        if not xs:
            return []
        p = xs[0]
        r = xs[1:]
        return [x for x in r if x > p] + [x for x in r if x <= p]

    def _pivot_part2(xs):
        if not xs:
            return []
        p = xs[0]
        r = xs[1:]
        return ([x for x in r if x > p] +
                [x for x in r if x < p] +
                [x for x in r if x == p])
    return [(_pivot_part,
              "def program(xs):\n    if not xs: return []\n    p=xs[0]; r=xs[1:]\n    return [x for x in r if x>p]+[x for x in r if x<=p]"),
            (_pivot_part2,
              "def program(xs):\n    if not xs: return []\n    p=xs[0]; r=xs[1:]\n    return [x for x in r if x>p]+[x for x in r if x<p]+[x for x in r if x==p]")]


def _rotate_programs(examples):
    def _rot_last(xs):
        if not xs:
            return xs
        n = len(xs)
        k = xs[-1] % n
        return xs[k:] + xs[:k]

    def _rot_first(xs):
        if not xs:
            return xs
        n = len(xs)
        k = xs[0] % n
        return xs[k:] + xs[:k]
    return [(_rot_last,
              "def program(xs):\n    if not xs: return xs\n    n=len(xs); k=xs[-1]%n\n    return xs[k:]+xs[:k]"),
            (_rot_first,
              "def program(xs):\n    if not xs: return xs\n    n=len(xs); k=xs[0]%n\n    return xs[k:]+xs[:k]")]


def _insert_replace_programs(examples):
    programs = []
    programs.append((lambda xs: ([xs[-1]] + xs if xs else []),
                      "def program(xs):\n    if not xs: return []\n    return [xs[-1]]+xs"))
    programs.append((lambda xs: ([xs[-1]] + xs[:-1] + [xs[-1]] if xs else []),
                      "def program(xs):\n    if not xs: return []\n    return [xs[-1]]+xs[:-1]+[xs[-1]]"))
    programs.append((lambda xs: (xs + [xs[0]] if xs else []),
                      "def program(xs):\n    if not xs: return []\n    return xs+[xs[0]]"))
    programs.append((lambda xs: ([xs[0]] + xs[1:-1] + xs[1:-1] + [xs[-1]] if len(xs) >= 2 else list(xs)),
                      "def program(xs):\n    if len(xs)<2: return list(xs)\n    return [xs[0]]+xs[1:-1]+xs[1:-1]+[xs[-1]]"))
    programs.append((lambda xs: ([xs[-1]] + xs[1:-1] + [xs[0]] if len(xs) >= 2 else list(xs)),
                      "def program(xs):\n    if not xs: return []\n    if len(xs)==1: return list(xs)\n    return [xs[-1]]+xs[1:-1]+[xs[0]]"))
    for val in range(0, 20):
        fn = lambda xs, v=val: (xs[:1] + [v] + xs[1:]) if xs else [v]
        src = "def program(xs):\n    if not xs: return [" + repr(val) + "]\n    return xs[:1]+[" + repr(val) + "]+xs[1:]"
        programs.append((fn, src))
    for pos in range(0, 5):
        for val in range(0, 20):
            fn = lambda xs, p=pos, v=val: (list(xs[:p]) + [v] + list(xs[p + 1:])) if len(xs) > p else list(xs)
            src = "def program(xs):\n    o=list(xs)\n    if len(o)>" + repr(pos) + ": o[" + repr(pos) + "]=" + repr(val) + "\n    return o"
            programs.append((fn, src))
    for pos in range(0, 5):
        fn = lambda xs, p=pos: (xs[:p] + xs[p + 1:]) if len(xs) > p else list(xs)
        src = "def program(xs):\n    if len(xs)>" + repr(pos) + ": return xs[:" + repr(pos) + "]+xs[" + repr(pos + 1) + ":]\n    return list(xs)"
        programs.append((fn, src))
    return programs


def _cumulative_programs(examples):
    def _cummax(xs):
        if not xs:
            return []
        out = [xs[0]]
        for x in xs[1:]:
            out.append(max(out[-1], x))
        return out

    def _cummin(xs):
        if not xs:
            return []
        out = [xs[0]]
        for x in xs[1:]:
            out.append(min(out[-1], x))
        return out

    def _cumsum(xs):
        if not xs:
            return []
        out = [xs[0]]
        for x in xs[1:]:
            out.append(out[-1] + x)
        return out

    return [(_cummax,
              "def program(xs):\n    if not xs: return []\n    out=[xs[0]]\n    for x in xs[1:]: out.append(max(out[-1],x))\n    return out"),
            (_cummin,
              "def program(xs):\n    if not xs: return []\n    out=[xs[0]]\n    for x in xs[1:]: out.append(min(out[-1],x))\n    return out"),
            (_cumsum,
              "def program(xs):\n    if not xs: return []\n    out=[xs[0]]\n    for x in xs[1:]: out.append(out[-1]+x))\n    return out"),
            (lambda xs: ([xs[i + 1] - xs[i] for i in range(len(xs) - 1)]),
              "def program(xs):\n    return [xs[i+1]-xs[i] for i in range(len(xs)-1)]")]


def _modulo_programs(examples):
    return [(lambda xs, mm=m: [x % mm for x in xs],
              "def program(xs):\n    return [x%" + repr(m) + " for x in xs]")
            for m in range(2, 15)]


def _abs_programs(examples):
    return [(lambda xs: [abs(x) for x in xs],
              "def program(xs):\n    return [abs(x) for x in xs]")]


def _pair_sums_programs(examples):
    def _pair_sums(xs):
        n = len(xs)
        return [xs[i] + xs[n - 1 - i] for i in range(n)]
    return [(_pair_sums,
              "def program(xs):\n    n=len(xs)\n    return [xs[i]+xs[n-1-i] for i in range(n)]")]


def _range_programs(examples):
    def _min_mul_range(xs):
        if not xs:
            return []
        m = min(xs)
        return [m * (i + 1) for i in range(len(xs))]
    return [(_min_mul_range,
              "def program(xs):\n    if not xs: return []\n    m=min(xs)\n    return [m*(i+1) for i in range(len(xs))]"),
            (lambda xs: (list(range(len(xs))) if xs else []),
              "def program(xs):\n    return list(range(len(xs))) if xs else []")]


def _prefix_suffix_programs(examples):
    programs = []
    ins = [e[0] for e in examples]
    outs = [e[1] for e in examples]
    for i in range(len(ins) - 1):
        for j in range(i + 1, len(ins)):
            inp_i, out_i = ins[i], outs[i]
            inp_j, out_j = ins[j], outs[j]
            if len(inp_i) < len(out_i) and len(inp_j) < len(out_j):
                for pi_len in range(len(out_i) - len(inp_i) + 1):
                    for si_len in range(max(0, len(out_i) - pi_len - len(inp_i)) + 1):
                        pi = out_i[:pi_len]
                        si = out_i[len(out_i) - si_len:] if si_len > 0 else []
                        if si_len > 0:
                            middle_i = out_i[pi_len:len(out_i) - si_len]
                        else:
                            middle_i = out_i[pi_len:]
                        if list(middle_i) != inp_i:
                            continue
                        pj = out_j[:pi_len]
                        sj = out_j[len(out_j) - si_len:] if si_len > 0 else []
                        if si_len > 0:
                            middle_j = out_j[pi_len:len(out_j) - si_len]
                        else:
                            middle_j = out_j[pi_len:]
                        if list(middle_j) != inp_j:
                            continue
                        if pi == pj and si == sj:
                            fn = lambda xs, pf=pi, sf=sj: list(pf) + list(xs) + list(sf)
                            src = "def program(xs):\n    return " + repr(pi) + "+list(xs)+" + repr(sj)
                            programs.append((fn, src))
                            break
                    else:
                        continue
                    break
        else:
            continue
        break
    return programs


def _compositions(examples):
    programs = []
    programs.append((lambda xs: list(reversed([x for x in xs if x % 2 == 0])),
                      "def program(xs):\n    return list(reversed([x for x in xs if x%2==0]))"))
    programs.append((lambda xs: list(reversed([x for x in xs if x % 2 == 1])),
                      "def program(xs):\n    return list(reversed([x for x in xs if x%2==1]))"))
    programs.append((lambda xs: sorted([x for x in xs if x % 2 == 0]),
                      "def program(xs):\n    return sorted([x for x in xs if x%2==0])"))
    programs.append((lambda xs: sorted([x for x in xs if x % 2 == 0], reverse=True),
                      "def program(xs):\n    return sorted([x for x in xs if x%2==0],reverse=True)"))
    programs.append((lambda xs: sorted([x for x in xs if x % 2 == 1]),
                      "def program(xs):\n    return sorted([x for x in xs if x%2==1])"))
    programs.append((lambda xs: sorted([x for x in xs if x % 2 == 1], reverse=True),
                      "def program(xs):\n    return sorted([x for x in xs if x%2==1],reverse=True)"))
    programs.append((lambda xs: [x for x in reversed(xs) if x % 2 == 0],
                      "def program(xs):\n    return [x for x in reversed(xs) if x%2==0]"))
    programs.append((lambda xs: [x for x in reversed(xs) if x % 2 == 1],
                      "def program(xs):\n    return [x for x in reversed(xs) if x%2==1]"))
    programs.append((lambda xs: sorted(reversed(xs)),
                      "def program(xs):\n    return sorted(reversed(xs))"))
    programs.append((lambda xs: sorted(reversed(xs), reverse=True),
                      "def program(xs):\n    return sorted(reversed(xs),reverse=True)"))
    programs.append((lambda xs: list(reversed(sorted(xs))),
                      "def program(xs):\n    return list(reversed(sorted(xs)))"))
    programs.append((lambda xs: list(reversed(sorted(xs, reverse=True))),
                      "def program(xs):\n    return list(reversed(sorted(xs,reverse=True)))"))
    programs.append((lambda xs: _dedup(reversed(xs)),
                      "def program(xs):\n    return _dedup(reversed(xs))"))
    programs.append((lambda xs: list(reversed(_dedup(xs))),
                      "def program(xs):\n    return list(reversed(_dedup(xs)))"))
    programs.append((lambda xs: sorted(_dedup(xs)),
                      "def program(xs):\n    return sorted(_dedup(xs))"))
    programs.append((lambda xs: list(reversed(xs[1:])) if len(xs) > 1 else [],
                      "def program(xs):\n    return list(reversed(xs[1:])) if len(xs)>1 else []"))
    programs.append((lambda xs: list(reversed(xs[:-1])) if xs else [],
                      "def program(xs):\n    return list(reversed(xs[:-1])) if xs else []"))
    programs.append((lambda xs: list(reversed([x for i, x in enumerate(xs) if i % 2 == 0])),
                      "def program(xs):\n    return list(reversed([x for i,x in enumerate(xs) if i%2==0]))"))
    programs.append((lambda xs: [x for x in xs[1:] if x % 2 == 0] if xs else [],
                      "def program(xs):\n    return [x for x in xs[1:] if x%2==0] if xs else []"))
    programs.append((lambda xs: [x for x in xs[::-1] if x % 2 == 0],
                      "def program(xs):\n    return [x for x in xs[::-1] if x%2==0]"))
    programs.append((lambda xs: xs[1:xs[0]+1] if len(xs) > 1 else [],
                      "def program(xs):\n    return xs[1:xs[0]+1] if len(xs)>1 else []"))
    def _rot_ff(xs):
        if not xs:
            return xs
        n = len(xs)
        k = xs[0] % n
        return [x for x in (xs[k:] + xs[:k]) if x % 2 == 0]
    programs.append((_rot_ff,
                      "def program(xs):\n    if not xs: return xs\n    n=len(xs); k=xs[0]%n\n    r=xs[k:]+xs[:k]\n    return [x for x in r if x%2==0]"))
    return programs


# ──────────────────────────────────────────────────────────────────────────────
# Level-3: Branch programs
# ──────────────────────────────────────────────────────────────────────────────

def _branch_programs(examples):
    programs = []
    if not examples:
        return programs
    ins = [e[0] for e in examples]
    outs = [e[1] for e in examples]

    by_length = {}
    for i, (inp, out) in enumerate(zip(ins, outs)):
        ln = len(inp)
        if ln not in by_length:
            by_length[ln] = []
        by_length[ln].append(i)

    transform_tests = {
        "identity": (lambda xs: list(xs), "return list(xs)"),
        "reverse": (lambda xs: list(reversed(xs)), "return list(reversed(xs))"),
        "sort": (lambda xs: sorted(xs), "return sorted(xs)"),
        "sort_desc": (lambda xs: sorted(xs, reverse=True),
                      "return sorted(xs, reverse=True)"),
        "xs[1:]": (lambda xs: xs[1:], "return xs[1:]"),
        "xs[:-1]": (lambda xs: xs[:-1], "return xs[:-1]"),
        "xs[1:-1]": (lambda xs: xs[1:-1] if len(xs) > 1 else [],
                      "return xs[1:-1] if len(xs)>1 else []"),
        "filter_even": (lambda xs: [x for x in xs if x % 2 == 0],
                         "return [x for x in xs if x%2==0]"),
        "filter_odd": (lambda xs: [x for x in xs if x % 2 == 1],
                        "return [x for x in xs if x%2==1]"),
        "xs[::-1]": (lambda xs: xs[::-1], "return xs[::-1]"),
        "xs[::2]": (lambda xs: xs[::2], "return xs[::2]"),
        "xs[1::2]": (lambda xs: xs[1::2], "return xs[1::2]"),
        "dedup": (lambda xs: _dedup(xs), "return _dedup(xs)"),
        "xs[1:xs[0]+1]": (lambda xs: xs[1:xs[0]+1] if len(xs) > 1 else [],
                           "return xs[1:xs[0]+1] if len(xs)>1 else []"),
        "reversed_xs[1:]": (lambda xs: list(reversed(xs[1:])) if len(xs) > 1 else [],
                             "return list(reversed(xs[1:])) if len(xs)>1 else []"),
        "xs[:-1][::-1]": (lambda xs: xs[:-1][::-1] if xs else [],
                           "return xs[:-1][::-1] if xs else []"),
    }

    length_transforms = {}
    for ln, indices in by_length.items():
        matching = []
        for tname, (tfn, _) in transform_tests.items():
            match_count = 0
            for idx in indices:
                try:
                    if tfn(ins[idx]) == outs[idx]:
                        match_count += 1
                except Exception:
                    pass
            if match_count == len(indices):
                matching.append(tname)
        if matching:
            length_transforms[ln] = matching

    if length_transforms:
        tf_rets = {
            "identity": "return list(xs)",
            "reverse": "return list(reversed(xs))",
            "xs[1:]": "return xs[1:]",
            "xs[:-1]": "return xs[:-1]",
            "xs[::-1]": "return xs[::-1]",
            "filter_even": "return [x for x in xs if x%2==0]",
            "filter_odd": "return [x for x in xs if x%2==1]",
            "sort": "return sorted(xs)",
            "xs[1:-1]": "return xs[1:-1] if len(xs)>1 else []",
            "dedup": "return _dedup(xs)",
            "xs[1:xs[0]+1]": "return xs[1:xs[0]+1] if len(xs)>1 else []",
            "reversed_xs[1:]": "return list(reversed(xs[1:])) if len(xs)>1 else []",
            "xs[:-1][::-1]": "return xs[:-1][::-1] if xs else []",
        }
        for chosen in ["identity", "reverse", "xs[1:]", "xs[:-1]", "xs[::-1]",
                        "filter_even", "filter_odd", "sort", "xs[1:-1]",
                        "dedup", "xs[1:xs[0]+1]", "reversed_xs[1:]",
                        "xs[:-1][::-1]"]:
            branches = []
            for ln in sorted(length_transforms.keys()):
                tfs = length_transforms[ln]
                if chosen in tfs:
                    branches.append((ln, chosen))
                else:
                    branches.append((ln, "identity"))
            src_lines = ["def program(xs):", "    n = len(xs)"]
            for ln, tf in branches:
                ret = tf_rets.get(tf, "return list(xs)")
                src_lines.append("    if n == " + repr(ln) + ":")
                src_lines.append("        " + ret)
            src_lines.append("    return list(xs)")
            full_src = "\n".join(src_lines)
            try:
                ns = {}
                exec(full_src, {"_dedup": _dedup}, ns)
                programs.append((ns["program"], full_src))
            except Exception:
                pass
    return programs


# ──────────────────────────────────────────────────────────────────────────────
# Search orchestration
# ──────────────────────────────────────────────────────────────────────────────

def _deduplicate_programs(programs):
    seen = set()
    result = []
    for fn, src in programs:
        clean = " ".join(src.split())
        if clean not in seen:
            seen.add(clean)
            result.append((fn, src))
    return result


def solve_listfn(examples, K=5):
    """
    Solve a List Functions task by searching over a DSL of list-to-list
    transformation primitives and their compositions.

    Returns dict with "success", "program", "score", and "candidates".
    """
    if not examples:
        return {
            "success": False,
            "program": "def program(xs): return []",
            "score": 0.0,
            "candidates": [("0.0", "def program(xs): return []")]
        }

    ins = [list(e[0]) for e in examples]
    outs = [list(e[1]) for e in examples]
    scored = []

    generators = [
        _trivial_programs,
        _elementwise_programs,
        _sort_reverse_dedup_programs,
        _filter_programs,
        _slice_programs,
        _scalar_output_programs,
        _histogram_programs,
        _index_programs,
        _two_digit_rev_programs,
        _split_reverse_programs,
        _pivot_programs,
        _rotate_programs,
        _insert_replace_programs,
        _cumulative_programs,
        _modulo_programs,
        _abs_programs,
        _pair_sums_programs,
        _range_programs,
        _prefix_suffix_programs,
    ]

    for gen in generators:
        try:
            cands = gen(examples)
            cands = _deduplicate_programs(cands)
            for fn, src in cands:
                try:
                    score, _ = score_program(fn, ins, outs)
                    scored.append((score, src, fn))
                except Exception:
                    scored.append((0.0, src, fn))
        except Exception:
            pass

    for s, src, fn in sorted(scored, key=lambda x: -x[0]):
        if s >= 1.0:
            return {"success": True, "program": src, "score": 1.0,
                     "candidates": [(s, src)]}

    try:
        comp_cands = _compositions(examples)
        comp_cands = _deduplicate_programs(comp_cands)
        for fn, src in comp_cands:
            try:
                score, _ = score_program(fn, ins, outs)
                scored.append((score, src, fn))
                if score >= 1.0:
                    return {"success": True, "program": src, "score": 1.0,
                             "candidates": [(score, src)]}
            except Exception:
                pass
    except Exception:
        pass

    try:
        branch_cands = _branch_programs(examples)
        branch_cands = _deduplicate_programs(branch_cands)
        for fn, src in branch_cands:
            try:
                score, _ = score_program(fn, ins, outs)
                scored.append((score, src, fn))
                if score >= 1.0:
                    return {"success": True, "program": src, "score": 1.0,
                             "candidates": [(score, src)]}
            except Exception:
                pass
    except Exception:
        pass

    scored.sort(key=lambda x: -x[0])
    best_score = scored[0][0] if scored else 0.0
    best_src = scored[0][1] if scored else "def program(xs): return []"
    top_k = [(s, src) for s, src, fn in scored[:K]]

    return {
        "success": best_score >= 1.0,
        "program": best_src,
        "score": best_score,
        "candidates": top_k
    }


if __name__ == "__main__":
    tests = [
        ("c164", [
            ([2, 14, 5, 9, 7, 6, 1], [5, 8, 6, 7, 6, 6, 5]),
            ([23, 7, 8, 97, 15, 55, 0, 49, 92], [10, 6, 7, 29, 8, 18, 5, 17, 28]),
        ]),
        ("c159", [
            ([1, 1, 8, 1, 5, 5, 5, 5, 8, 5], [3, 0, 0, 0, 5, 0, 0, 2]),
            ([2, 10, 10, 5, 4, 6, 4, 10, 2], [0, 2, 0, 2, 1, 1, 0, 0, 0, 3]),
        ]),
        ("c165", [
            ([3, 2, 31, 4, 20, 7, 9, 6, 83, 44], [44, 6, 20, 4, 2]),
            ([98, 36, 6, 0, 76, 76, 8, 0, 56, 56], [56, 56, 0, 8, 76, 76, 0, 6, 36, 98]),
        ]),
        ("c201", [
            ([58, 9, 2, 93, 81, 99, 97, 8, 4, 82], [93, 81, 99, 97, 82, 9, 2, 8, 4]),
            ([87], []),
        ]),
        ("c245", [
            ([36, 47, 90, 4, 23, 92, 93, 1], [0]),
            ([6, 6, 6, 6, 6, 6, 6, 6, 6, 6], [9]),
        ]),
        ("c130", [
            ([6, 3, 68, 8, 85, 5, 97, 61], [3, 68, 8, 85, 5, 97]),
            ([6, 90, 36, 0, 66, 31, 57, 9], [90, 36, 0, 66, 31, 57]),
        ]),
    ]
    for name, exs in tests:
        result = solve_listfn(exs)
        status = "PASS" if result["success"] else "FAIL"
        print(f"{status} {name}: score={result['score']:.3f}")
        if not result["success"]:
            print(f"  Top: {result['candidates'][0][0]} - {result['program'][:100]}")
