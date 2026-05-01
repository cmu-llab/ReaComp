"""
SOLVER_SLR.py — Symbolic rule induction for SLR-Bench.

Algorithm (ILP-style, coverage-based):
  1. Parse each training example into structured car-property dicts.
  2. Precompute coverage: for each candidate literal, which east/west trains
     are "covered" (have at least one car satisfying the literal).
  3. Search rule templates in order of increasing complexity (1 → 5 non-has_car
     body literals), using Python coverage to filter before expensive Prolog calls.
  4. Call the SWI-Prolog verifier only on rules that Python-simulation deems
     perfectly consistent (covers all east, zero west).
  5. Return the first rule with verifier score == 1.0; if none, return top-K
     ranked by Python coverage score then rule complexity.
"""

import itertools
import os
import re
import sys
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_facts(facts_str: str) -> Tuple[Optional[str], List[Dict]]:
    """Return (train_id, ordered list of car-property dicts)."""
    atoms = re.findall(r'(\w+)\(([^)]+)\)', facts_str)
    train_id: Optional[str] = None
    car_order: List[str] = []
    car_data: Dict[str, Dict] = {}

    for pred, args_str in atoms:
        args = [a.strip() for a in args_str.split(',')]
        if pred == 'has_car':
            tid, cid = args[0], args[1]
            if train_id is None:
                train_id = tid
            if cid not in car_data:
                car_order.append(cid)
                car_data[cid] = {}
        elif len(args) == 2:
            cid, val = args[0], args[1]
            car_data.setdefault(cid, {})[pred] = val

    return train_id, [car_data[c] for c in car_order if c in car_data]


def _reconstruct_validation_program(examples: List[Tuple[str, str]]) -> str:
    """Build a Prolog validation_program string from (facts_str, label) pairs."""
    parts = []
    for facts_str, label in examples:
        train_id, _ = _parse_facts(facts_str)
        if train_id:
            parts.append(f'{label}({train_id}).')
        for atom in re.split(r'\.\s*', facts_str.strip()):
            atom = atom.strip()
            if atom:
                parts.append(atom + '.')
        parts.append('')
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Coverage computation (pure Python)
# ---------------------------------------------------------------------------

def _build_coverage(
    trains: List[Dict],
) -> Tuple[Dict, Dict, Dict, Dict, FrozenSet, FrozenSet]:
    """
    Compute single-literal coverage sets.

    pos_e[(pred,val)] = frozenset of east indices that have ≥1 car with pred=val
    pos_w[(pred,val)] = frozenset of west indices similarly
    neg_e[(pred,val)] = frozenset of east indices that have ≥1 car with pred≠val
    neg_w[(pred,val)] = frozenset of west indices similarly
    """
    all_east = frozenset(i for i, t in enumerate(trains) if t['label'] == 'eastbound')
    all_west = frozenset(i for i, t in enumerate(trains) if t['label'] == 'westbound')

    pos_e: Dict[Tuple, Set] = {}
    pos_w: Dict[Tuple, Set] = {}

    for i, t in enumerate(trains):
        in_east = i in all_east
        for car in t['cars']:
            for pred, val in car.items():
                key = (pred, val)
                (pos_e if in_east else pos_w).setdefault(key, set()).add(i)

    # Negation: ∃ car where pred ≠ val
    all_keys = set(pos_e) | set(pos_w)
    neg_e: Dict[Tuple, Set] = {}
    neg_w: Dict[Tuple, Set] = {}
    for i, t in enumerate(trains):
        in_east = i in all_east
        for car in t['cars']:
            for key in all_keys:
                pred, val = key
                if car.get(pred) != val:
                    (neg_e if in_east else neg_w).setdefault(key, set()).add(i)

    return (
        {k: frozenset(v) for k, v in pos_e.items()},
        {k: frozenset(v) for k, v in pos_w.items()},
        {k: frozenset(v) for k, v in neg_e.items()},
        {k: frozenset(v) for k, v in neg_w.items()},
        all_east,
        all_west,
    )


def _build_same_car_coverage(
    trains: List[Dict], arity: int
) -> Tuple[Dict, Dict]:
    """
    Compute same-car k-tuple coverage (arity ∈ {2,3,4,5}).

    key = sorted tuple of (pred,val) pairs of length `arity`
    Returns (east_cov, west_cov) mapping key -> frozenset of train indices.
    """
    east_cov: Dict[Tuple, Set] = {}
    west_cov: Dict[Tuple, Set] = {}

    for i, t in enumerate(trains):
        in_east = t['label'] == 'eastbound'
        for car in t['cars']:
            items = sorted(car.items())
            for combo in itertools.combinations(items, arity):
                key = tuple(x for pair in combo for x in pair)
                (east_cov if in_east else west_cov).setdefault(key, set()).add(i)

    return (
        {k: frozenset(v) for k, v in east_cov.items()},
        {k: frozenset(v) for k, v in west_cov.items()},
    )


# ---------------------------------------------------------------------------
# Rule string builders
# ---------------------------------------------------------------------------

def _make_single_rule(conds: List[Tuple[str, str, bool]]) -> str:
    """Build 'eastbound(T) :- has_car(T, C), ...' from condition list."""
    parts = ['has_car(T, C)']
    for pred, val, neg in conds:
        parts.append(f'\\+ {pred}(C, {val})' if neg else f'{pred}(C, {val})')
    return f"eastbound(T) :- {', '.join(parts)}."


def _make_two_car_rule(
    conds1: List[Tuple[str, str, bool]],
    conds2: List[Tuple[str, str, bool]],
    distinct: bool = False,
) -> str:
    """Build a two-car-variable rule."""
    parts = ['has_car(T, C1)']
    for pred, val, neg in conds1:
        parts.append(f'\\+ {pred}(C1, {val})' if neg else f'{pred}(C1, {val})')
    parts.append('has_car(T, C2)')
    for pred, val, neg in conds2:
        parts.append(f'\\+ {pred}(C2, {val})' if neg else f'{pred}(C2, {val})')
    if distinct:
        parts.append('C1 \\= C2')
    return f"eastbound(T) :- {', '.join(parts)}."


# ---------------------------------------------------------------------------
# Rule complexity (mirrors rewards/slr_bench.py)
# ---------------------------------------------------------------------------

def _rule_complexity(rule: str) -> int:
    if ':-' not in rule:
        return 0
    body = rule.split(':-', 1)[1].rstrip(' .')
    depth = commas = 0
    for ch in body:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            commas += 1
    total = commas + 1
    has_car_count = len(re.findall(r'\bhas_car\s*\(', body))
    return total - has_car_count


# ---------------------------------------------------------------------------
# Verifier integration
# ---------------------------------------------------------------------------

def _try_verify(rule: str, validation_program: str) -> float:
    """
    Call the SWI-Prolog verifier. Returns score ∈ [0,1] or -1.0 on error.
    """
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from rewards.slr_bench import _get_judge, _judge_lock  # noqa
        with _judge_lock:
            judge = _get_judge()
            result = judge.compute(
                predictions=[rule],
                references=[{
                    'validation_program': validation_program,
                    'evaluation_config': {
                        'positive_predicate': 'eastbound',
                        'negative_predicate': 'westbound',
                    },
                }],
            )
        return float(result['detailed_results'][0]['partial_score'])
    except Exception:
        return -1.0


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_slr(
    examples: List[Tuple[str, str]],
    top_k: int = 5,
) -> Dict:
    """
    Induce a Prolog rule from (facts_string, direction_label) training examples.

    Parameters
    ----------
    examples : list of (facts_string, direction_label)
    top_k    : how many candidates to return when no perfect rule is found

    Returns
    -------
    dict with keys:
        success (bool), program (str), score (float), top_k (list[str])
    """
    if not examples:
        default = 'eastbound(T) :- has_car(T, C).'
        return {'success': False, 'program': default, 'score': 0.0, 'top_k': [default]}

    # ── 1. Parse ────────────────────────────────────────────────────────────
    trains = []
    for facts_str, label in examples:
        _, cars = _parse_facts(facts_str)
        trains.append({'cars': cars, 'label': label})

    all_east = frozenset(i for i, t in enumerate(trains) if t['label'] == 'eastbound')
    all_west = frozenset(i for i, t in enumerate(trains) if t['label'] == 'westbound')

    if not all_east or not all_west:
        default = 'eastbound(T) :- has_car(T, C), car_len(C, short).'
        return {'success': False, 'program': default, 'score': 0.0, 'top_k': [default]}

    validation_program = _reconstruct_validation_program(examples)

    # ── 2. Coverage ──────────────────────────────────────────────────────────
    pos_e, pos_w, neg_e, neg_w, all_east, all_west = _build_coverage(trains)
    n_total = len(all_east) + len(all_west)

    def _score(e_cov: FrozenSet, w_cov: FrozenSet) -> float:
        tp = len(e_cov & all_east)
        tn = len(all_west - w_cov)
        return (tp + tn) / n_total

    def _consistent(e_cov: FrozenSet, w_cov: FrozenSet) -> bool:
        return e_cov >= all_east and not w_cov

    # ── 3. Candidate generation ──────────────────────────────────────────────
    # Scored candidates: list of (py_score, complexity, rule_str)
    cands: List[Tuple[float, int, str]] = []

    _cplx_cache: Dict[str, int] = {}

    def _add(rule: str, e_cov: FrozenSet, w_cov: FrozenSet) -> None:
        sc = _score(e_cov, w_cov)
        if rule not in _cplx_cache:
            _cplx_cache[rule] = _rule_complexity(rule)
        cands.append((sc, _cplx_cache[rule], rule))

    all_prop_keys = set(pos_e) | set(pos_w)

    # ── Level 1: single literal (positive) ──────────────────────────────────
    for (pred, val), e_cov in pos_e.items():
        w_cov = pos_w.get((pred, val), frozenset())
        _add(_make_single_rule([(pred, val, False)]), e_cov, w_cov)

    # ── Level 1: single literal (negative / \\+) ────────────────────────────
    for (pred, val), e_cov in neg_e.items():
        w_cov = neg_w.get((pred, val), frozenset())
        _add(_make_single_rule([(pred, val, True)]), e_cov, w_cov)

    # ── Level 1: "no car has property" (nested negation) ────────────────────
    # eastbound(T) :- \+ (has_car(T, C), pred(C, val)).
    # A train fires iff it has NO car with pred=val.
    # e_cov = all_east minus east trains that DO have pred=val.
    for (pred, val) in all_prop_keys:
        e_has = pos_e.get((pred, val), frozenset())
        w_has = pos_w.get((pred, val), frozenset())
        e_no = all_east - e_has
        w_no = all_west - w_has
        rule = f"eastbound(T) :- \\+ (has_car(T, C), {pred}(C, {val}))."
        _add(rule, e_no, w_no)

    # ── Level 2: same-car two positive literals ──────────────────────────────
    pair_e, pair_w = _build_same_car_coverage(trains, 2)
    for key, e_cov in pair_e.items():
        p1, v1, p2, v2 = key
        w_cov = pair_w.get(key, frozenset())
        _add(_make_single_rule([(p1, v1, False), (p2, v2, False)]), e_cov, w_cov)

    # ── Level 2: two-car, one positive literal each (existential conjunction)
    # A rule has_car(T,C1),p1(C1,v1),has_car(T,C2),p2(C2,v2) fires when
    # ∃ car with p1=v1 AND ∃ car with p2=v2 (cars may be the same).
    # Coverage = intersection of individual single-prop coverages.
    good_pos = {(p, v): (pos_e.get((p, v), frozenset()), pos_w.get((p, v), frozenset()))
                for (p, v) in pos_e if pos_e[(p, v)] >= all_east}

    seen_two_car: Set[Tuple] = set()
    for (p1, v1), (e1, w1) in good_pos.items():
        for (p2, v2), (e2, w2) in good_pos.items():
            if (p1, v1) >= (p2, v2):
                continue
            key2 = (p1, v1, p2, v2)
            if key2 in seen_two_car:
                continue
            seen_two_car.add(key2)
            e_cov = e1 & e2  # both conditions must hold → still all_east
            w_cov = w1 & w2  # west train must satisfy BOTH → intersection
            _add(_make_two_car_rule([(p1, v1, False)], [(p2, v2, False)]), e_cov, w_cov)

    # ── Level 2: same-car, one pos + one neg ────────────────────────────────
    # For (p_pos, v_pos, p_neg, v_neg): train has a car where p_pos=v_pos AND p_neg≠v_neg.
    # Must iterate over ALL (pred, val) pairs (east AND west) for the negation slot
    # so we don't miss negating values that appear only in west trains.
    car_pos_neg_e: Dict[Tuple, Set] = {}
    car_pos_neg_w: Dict[Tuple, Set] = {}
    for i, t in enumerate(trains):
        in_east = t['label'] == 'eastbound'
        for car in t['cars']:
            for p_pos, v_pos in car.items():
                for (p_neg, v_neg) in all_prop_keys:
                    if p_neg != p_pos and car.get(p_neg) != v_neg:
                        key = (p_pos, v_pos, p_neg, v_neg)
                        (car_pos_neg_e if in_east else car_pos_neg_w).setdefault(key, set()).add(i)

    for key, e_cov in car_pos_neg_e.items():
        p_pos, v_pos, p_neg, v_neg = key
        w_cov = car_pos_neg_w.get(key, frozenset())
        _add(_make_single_rule([(p_pos, v_pos, False), (p_neg, v_neg, True)]),
             frozenset(e_cov), frozenset(w_cov))

    # ── Level 3: two-distinct-cars same property ─────────────────────────────
    # eastbound(T) :- has_car(T,C1), pred(C1,val), has_car(T,C2), pred(C2,val), C1 \= C2.
    # A train satisfies this iff it has ≥2 different cars with pred=val.
    from collections import Counter as _Counter
    two_distinct_e: Dict[Tuple, Set] = {}
    two_distinct_w: Dict[Tuple, Set] = {}
    for i, t in enumerate(trains):
        in_east = t['label'] == 'eastbound'
        counts: Dict[Tuple, int] = {}
        for car in t['cars']:
            for pred, val in car.items():
                counts[(pred, val)] = counts.get((pred, val), 0) + 1
        for (pred, val), cnt in counts.items():
            if cnt >= 2:
                (two_distinct_e if in_east else two_distinct_w).setdefault((pred, val), set()).add(i)

    for (pred, val), e_cov in two_distinct_e.items():
        w_cov = two_distinct_w.get((pred, val), frozenset())
        rule = _make_two_car_rule([(pred, val, False)], [(pred, val, False)], distinct=True)
        _add(rule, frozenset(e_cov), frozenset(w_cov))

    # ── Level 3: same-car three positive literals ────────────────────────────
    tri_e, tri_w = _build_same_car_coverage(trains, 3)
    for key, e_cov in tri_e.items():
        p1, v1, p2, v2, p3, v3 = key
        w_cov = tri_w.get(key, frozenset())
        _add(_make_single_rule([(p1, v1, False), (p2, v2, False), (p3, v3, False)]),
             e_cov, w_cov)

    # ── Level 3: two-car (pair + single, positive) ──────────────────────────
    # has_car(T,C1),p1(C1,v1),p2(C1,v2),has_car(T,C2),p3(C2,v3)
    # Coverage: train has car with (p1=v1 AND p2=v2) AND train has car with p3=v3
    good_pairs = {(p1, v1, p2, v2): (e_cov, pair_w.get((p1, v1, p2, v2), frozenset()))
                  for (p1, v1, p2, v2), e_cov in pair_e.items()
                  if e_cov >= all_east}

    # Also compute "pair good" from ALL pairs for two-car pair+neg-single
    all_pairs_east = {k: (v, pair_w.get(k, frozenset())) for k, v in pair_e.items()}

    for (p1, v1, p2, v2), (ep, wp) in good_pairs.items():
        for (p3, v3), (e3, w3) in good_pos.items():
            if (p3, v3) in {(p1, v1), (p2, v2)}:
                continue
            e_cov = ep & e3
            w_cov = wp & w3
            if e_cov >= all_east:
                _add(_make_two_car_rule([(p1, v1, False), (p2, v2, False)],
                                        [(p3, v3, False)]), e_cov, w_cov)

    # ── Level 3: two-car (single + neg-single) ───────────────────────────────
    # has_car(T,C1),p1(C1,v1),has_car(T,C2),\+p2(C2,v2)
    # C1 car: single positive; C2 car: single negation.
    # Coverage of C2 side: trains that have a car where p2≠v2
    # (same as neg_e/neg_w for single negation)
    # For the conjunction: both C1 and C2 sides must hold.
    good_neg_single = {}  # (pred, val, neg=True) -> (e_cov, w_cov)
    for (pred, val), e_cov in neg_e.items():
        w_cov = neg_w.get((pred, val), frozenset())
        good_neg_single[(pred, val)] = (frozenset(e_cov), frozenset(w_cov))

    for (p1, v1), (e1, w1) in good_pos.items():
        for (p2, v2), (e2, w2) in good_neg_single.items():
            if (p1, v1) == (p2, v2):
                continue
            # Two-car: C1 has p1=v1 AND C2 has p2≠v2
            e_cov = e1 & e2
            w_cov = w1 & w2
            if e_cov >= all_east:
                _add(_make_two_car_rule([(p1, v1, False)], [(p2, v2, True)]),
                     e_cov, w_cov)

    # ── Level 4: same-car four positive literals ─────────────────────────────
    quad_e, quad_w = _build_same_car_coverage(trains, 4)
    for key, e_cov in quad_e.items():
        p1, v1, p2, v2, p3, v3, p4, v4 = key
        w_cov = quad_w.get(key, frozenset())
        _add(_make_single_rule([(p1, v1, False), (p2, v2, False),
                                (p3, v3, False), (p4, v4, False)]), e_cov, w_cov)

    # ── Level 4: two-car pair+pair ────────────────────────────────────────────
    good_pair_list = list(good_pairs.items())
    for idx1, ((p1, v1, p2, v2), (ep1, wp1)) in enumerate(good_pair_list):
        for ((p3, v3, p4, v4), (ep2, wp2)) in good_pair_list[idx1 + 1:]:
            e_cov = ep1 & ep2
            w_cov = wp1 & wp2
            if e_cov >= all_east:
                _add(_make_two_car_rule([(p1, v1, False), (p2, v2, False)],
                                        [(p3, v3, False), (p4, v4, False)]),
                     e_cov, w_cov)

    # ── Level 5: same-car five positive literals ─────────────────────────────
    quin_e, quin_w = _build_same_car_coverage(trains, 5)
    for key, e_cov in quin_e.items():
        p1, v1, p2, v2, p3, v3, p4, v4, p5, v5 = key
        w_cov = quin_w.get(key, frozenset())
        _add(_make_single_rule([(p1, v1, False), (p2, v2, False), (p3, v3, False),
                                (p4, v4, False), (p5, v5, False)]), e_cov, w_cov)

    # ── 4. Sort and verify ───────────────────────────────────────────────────
    # Sort by (py_score DESC, complexity ASC) — prefer correct and simple
    cands.sort(key=lambda x: (-x[0], x[1]))

    # Deduplicate by rule string
    seen_rules: Set[str] = set()
    deduped: List[Tuple[float, int, str]] = []
    for sc, cplx, rule in cands:
        if rule not in seen_rules:
            seen_rules.add(rule)
            deduped.append((sc, cplx, rule))
    cands = deduped

    # Python-consistent rules are almost always Prolog-correct; limit partial checks.
    MAX_VERIFY_PERFECT = 30   # max verifier calls on Python-consistent rules
    MAX_VERIFY_PARTIAL = 10   # max verifier calls on partial-score rules
    verify_count = 0
    best_rule: Optional[str] = None
    best_score = 0.0
    verifier_available = True

    # Try perfectly Python-consistent rules first (py_score == 1.0)
    perfect_py = [(sc, cplx, rule) for sc, cplx, rule in cands if sc >= 1.0]
    rest = [(sc, cplx, rule) for sc, cplx, rule in cands if sc < 1.0]

    for sc, cplx, rule in perfect_py:
        if verify_count >= MAX_VERIFY_PERFECT:
            break
        v_score = _try_verify(rule, validation_program)
        verify_count += 1
        if v_score < 0:
            verifier_available = False
            best_rule = rule
            best_score = sc
            break
        if v_score >= 1.0:
            best_rule = rule
            best_score = 1.0
            break
        elif v_score > best_score:
            best_score = v_score
            best_rule = rule

    # If still no perfect rule, try top partial-score rules (verifier may catch false negatives)
    if verifier_available and best_score < 1.0:
        for sc, cplx, rule in rest[:MAX_VERIFY_PARTIAL]:
            if verify_count >= MAX_VERIFY_PERFECT + MAX_VERIFY_PARTIAL:
                break
            v_score = _try_verify(rule, validation_program)
            verify_count += 1
            if v_score >= 1.0:
                best_rule = rule
                best_score = 1.0
                break
            elif v_score > best_score:
                best_score = v_score
                best_rule = rule

    if best_rule is None:
        # Verifier unavailable or no candidates; pick best Python rule
        if cands:
            best_rule = cands[0][2]
            best_score = cands[0][0]
        else:
            best_rule = 'eastbound(T) :- has_car(T, C), car_len(C, short).'
            best_score = 0.0

    top_k_rules = [r for _, _, r in cands[:top_k]]
    if best_rule not in top_k_rules:
        top_k_rules.insert(0, best_rule)
    top_k_rules = top_k_rules[:top_k]

    return {
        'success': best_score >= 1.0,
        'program': best_rule,
        'score': best_score,
        'top_k': top_k_rules,
    }
