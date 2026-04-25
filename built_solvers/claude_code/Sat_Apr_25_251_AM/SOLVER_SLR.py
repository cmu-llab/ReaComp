"""
SOLVER_SLR.py — Symbolic rule inducer for SLR-Bench.

Search strategy (ascending complexity):
  1. Single-car existential: eastbound(T) :- has_car(T,C), prop(C,val), ...
  2. Position-specific: eastbound(T) :- has_car(T,C), car_num(C,N), prop(C,val), ...
  3. Two-car rules: eastbound(T) :- has_car(T,C1), conds1, has_car(T,C2), conds2.

Uses a local Python evaluator (mirrors Prolog semantics for the SLR DSL) for
candidate scoring. Returns the simplest perfectly-consistent rule, or top-K
highest-scoring rules if none is perfect.
"""

import re
import sys
import os
from itertools import combinations
from collections import defaultdict

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_ATOM_RE = re.compile(r'(\w+)\(([^)]+)\)')


def _parse_atom(s):
    m = _ATOM_RE.match(s.strip())
    if not m:
        return None
    return m.group(1), [a.strip() for a in m.group(2).split(',')]


def parse_train_model(facts_string):
    """
    Parse a facts_string into {norm_car_id: {pred: val}}.
    Cars are normalised to c1, c2, ... ordered by car_num.
    Predicates discovered dynamically (handles has_roof etc.).
    """
    raw_cars = {}
    for part in re.split(r'\.\s*', facts_string):
        p = _parse_atom(part)
        if not p:
            continue
        pred, args = p
        if pred == 'has_car':
            car_id = args[1]
            raw_cars.setdefault(car_id, {})
        elif len(args) == 2 and pred != 'has_car':
            car_id, val = args[0], args[1]
            raw_cars.setdefault(car_id, {})
            raw_cars[car_id][pred] = val

    def _num(info):
        try:
            return int(info.get('car_num', 0))
        except ValueError:
            return 0

    ordered = sorted(raw_cars.keys(), key=lambda c: _num(raw_cars[c]))
    return {f'c{i+1}': raw_cars[cid] for i, cid in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Local rule evaluator (Prolog existential semantics)
# ---------------------------------------------------------------------------

def _car_ok(car_info, conds):
    return all(car_info.get(p) == v for p, v in conds)


def eval_rule_spec(spec, model):
    """
    spec:
      {'type': 'single', 'conds': [...]}
      {'type': 'two_car', 'conds1': [...], 'conds2': [...]}
    Returns True iff the rule fires for this model.
    """
    cars = list(model.values())
    if spec['type'] == 'single':
        return any(_car_ok(c, spec['conds']) for c in cars)
    # two_car: need two *distinct* cars
    for i, c1 in enumerate(cars):
        if not _car_ok(c1, spec['conds1']):
            continue
        for j, c2 in enumerate(cars):
            if i != j and _car_ok(c2, spec['conds2']):
                return True
    return False


def accuracy(spec, models, labels):
    correct = sum(
        1 for m, lbl in zip(models, labels)
        if eval_rule_spec(spec, m) == (lbl == 'eastbound')
    )
    return correct / len(labels) if labels else 0.0


# ---------------------------------------------------------------------------
# Rule-string builders
# ---------------------------------------------------------------------------

def _single_str(conds):
    parts = ['has_car(T, C)'] + [f'{p}(C, {v})' for p, v in conds]
    return 'eastbound(T) :- ' + ', '.join(parts) + '.'


def _two_car_str(conds1, conds2):
    p1 = ['has_car(T, C1)'] + [f'{p}(C1, {v})' for p, v in conds1]
    p2 = ['has_car(T, C2)'] + [f'{p}(C2, {v})' for p, v in conds2]
    return 'eastbound(T) :- ' + ', '.join(p1 + p2) + '.'


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _prop_vals(models):
    """Return {pred: sorted_vals} for all non-car_num predicates found."""
    pv = defaultdict(set)
    for model in models:
        for car_info in model.values():
            for pred, val in car_info.items():
                pv[pred].add(val)
    return {p: sorted(vs) for p, vs in pv.items()}


def generate_candidates(models, max_complexity=4):
    """
    Yield (spec, rule_str, complexity) in ascending complexity order.
    complexity = rule_complexity() from slr_bench.py (non-has_car literals).
    """
    pv_map = _prop_vals(models)

    # Separate car_num from other properties
    car_nums = sorted(pv_map.pop('car_num', set()), key=lambda x: int(x) if x.isdigit() else 0)
    prop_items = [(p, v) for p, vals in pv_map.items() for v in vals]

    seen_str = set()

    def emit(spec, rule_str, complexity):
        if rule_str not in seen_str and complexity <= max_complexity:
            seen_str.add(rule_str)
            return (spec, rule_str, complexity)
        return None

    def add(lst, spec, rule_str, complexity):
        r = emit(spec, rule_str, complexity)
        if r:
            lst.append(r)

    results = []

    # ---- Complexity 1: single property ----
    for pv in prop_items:
        spec = {'type': 'single', 'conds': [pv]}
        add(results, spec, _single_str([pv]), 1)

    # ---- Complexity 2: two properties on same car ----
    if max_complexity >= 2:
        for pv1, pv2 in combinations(prop_items, 2):
            if pv1[0] != pv2[0]:
                conds = [pv1, pv2]
                add(results, {'type': 'single', 'conds': conds}, _single_str(conds), 2)

        # car_num + one property
        for num in car_nums:
            for pv in prop_items:
                conds = [('car_num', num), pv]
                add(results, {'type': 'single', 'conds': conds}, _single_str(conds), 2)

        # two-car, one property each (no car_num)
        for pv1, pv2 in combinations(prop_items, 2):
            conds1, conds2 = [pv1], [pv2]
            add(results, {'type': 'two_car', 'conds1': conds1, 'conds2': conds2},
                _two_car_str(conds1, conds2), 2)
        # same property, two different cars
        for pv in prop_items:
            add(results, {'type': 'two_car', 'conds1': [pv], 'conds2': [pv]},
                _two_car_str([pv], [pv]), 2)

    # ---- Complexity 3 ----
    if max_complexity >= 3:
        # three properties on same car
        for pvs in combinations(prop_items, 3):
            preds = [p for p, _ in pvs]
            if len(set(preds)) == len(preds):
                conds = list(pvs)
                add(results, {'type': 'single', 'conds': conds}, _single_str(conds), 3)

        # car_num + two properties on same car
        for num in car_nums:
            for pv1, pv2 in combinations(prop_items, 2):
                if pv1[0] != pv2[0]:
                    conds = [('car_num', num), pv1, pv2]
                    add(results, {'type': 'single', 'conds': conds}, _single_str(conds), 3)

        # two-car: one side has car_num + property, other has one property
        for num in car_nums:
            for pv1 in prop_items:
                for pv2 in prop_items:
                    c1 = [('car_num', num), pv1]
                    c2 = [pv2]
                    add(results, {'type': 'two_car', 'conds1': c1, 'conds2': c2},
                        _two_car_str(c1, c2), 3)
                    add(results, {'type': 'two_car', 'conds1': c2, 'conds2': c1},
                        _two_car_str(c2, c1), 3)

        # two-car, two properties each (no car_num)
        for pv1, pv2 in combinations(prop_items, 2):
            for pv3, pv4 in combinations(prop_items, 2):
                if pv1[0] != pv2[0] or pv3[0] != pv4[0]:
                    c1, c2 = [pv1, pv2], [pv3, pv4]
                    add(results, {'type': 'two_car', 'conds1': c1, 'conds2': c2},
                        _two_car_str(c1, c2), 4)

    # ---- Complexity 4: two-car with car_num on both sides ----
    if max_complexity >= 4:
        for n1 in car_nums:
            for n2 in car_nums:
                if n1 == n2:
                    continue
                for pv1 in prop_items:
                    for pv2 in prop_items:
                        c1 = [('car_num', n1), pv1]
                        c2 = [('car_num', n2), pv2]
                        add(results, {'type': 'two_car', 'conds1': c1, 'conds2': c2},
                            _two_car_str(c1, c2), 4)

        # four properties on same car
        for pvs in combinations(prop_items, 4):
            preds = [p for p, _ in pvs]
            if len(set(preds)) == len(preds):
                conds = list(pvs)
                add(results, {'type': 'single', 'conds': conds}, _single_str(conds), 4)

        # car_num + three properties on same car
        for num in car_nums:
            for pvs in combinations(prop_items, 3):
                preds = [p for p, _ in pvs]
                if len(set(preds)) == len(preds):
                    conds = [('car_num', num)] + list(pvs)
                    add(results, {'type': 'single', 'conds': conds}, _single_str(conds), 4)

    return results


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_slr(examples, top_k=5):
    """
    examples: list of (facts_string, direction_label)
              direction_label is 'eastbound' or 'westbound'
    Returns dict with at least:
        'success'       : bool
        'program'       : best Prolog rule string
        'top_k_programs': list of up to top_k rule strings
    """
    if not examples:
        return {'success': False, 'program': None, 'top_k_programs': []}

    models = [parse_train_model(fs) for fs, _ in examples]
    labels = [lbl for _, lbl in examples]

    max_cars = max(len(m) for m in models) if models else 1
    max_complexity = 4 if max_cars <= 3 else 5

    candidates = generate_candidates(models, max_complexity=max_complexity)

    # Score all candidates
    scored = []
    for spec, rule_str, complexity in candidates:
        acc = accuracy(spec, models, labels)
        scored.append((acc, complexity, rule_str, spec))

    # Sort: perfect first, then by accuracy desc, then by complexity asc
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Deduplicate by rule string
    seen = set()
    deduped = []
    for acc, complexity, rule_str, spec in scored:
        if rule_str not in seen:
            seen.add(rule_str)
            deduped.append((acc, complexity, rule_str))

    if not deduped:
        return {'success': False, 'program': None, 'top_k_programs': []}

    best_acc, best_complexity, best_rule = deduped[0]
    top_k_programs = [r for _, _, r in deduped[:top_k]]

    return {
        'success': best_acc >= 1.0,
        'program': best_rule,
        'top_k_programs': top_k_programs,
        'score': best_acc,
        'complexity': best_complexity,
    }


# ---------------------------------------------------------------------------
# Optional: score using the HF verifier when a validation program is available
# ---------------------------------------------------------------------------

def solve_slr_with_entry(examples, entry, top_k=5):
    """
    Like solve_slr but re-scores the top-K candidates using the official
    rewards/slr_bench verifier (requires SWI-Prolog + HF evaluate installed).
    """
    result = solve_slr(examples, top_k=top_k)
    if not result['top_k_programs']:
        return result

    # Import verifier lazily so the solver works even without it
    _rewards_dir = os.path.join(os.path.dirname(__file__), 'rewards')
    if _rewards_dir not in sys.path:
        sys.path.insert(0, os.path.dirname(__file__))

    try:
        from rewards.slr_bench import reward as _reward
    except ImportError:
        return result

    best_score = -1.0
    best_rule = result['program']
    for rule_str in result['top_k_programs']:
        r = _reward(rule_str, execution_ok=True, entry=entry)
        if r['value'] > best_score:
            best_score = r['value']
            best_rule = rule_str
        if best_score >= 1.0:
            break

    result['program'] = best_rule
    result['success'] = best_score >= 1.0
    result['verifier_score'] = best_score
    return result


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    _demo_examples = [
        ("has_car(train0, car0_1). car_num(car0_1, 1). car_color(car0_1, white). car_len(car0_1, short). has_wall(car0_1, full). has_car(train0, car0_2). car_num(car0_2, 2). car_color(car0_2, blue). car_len(car0_2, short). has_wall(car0_2, railing).", "eastbound"),
        ("has_car(train1, car1_1). car_num(car1_1, 1). car_color(car1_1, white). car_len(car1_1, long). has_wall(car1_1, railing). has_car(train1, car1_2). car_num(car1_2, 2). car_color(car1_2, blue). car_len(car1_2, long). has_wall(car1_2, full).", "westbound"),
        ("has_car(train2, car2_1). car_num(car2_1, 1). car_color(car2_1, white). car_len(car2_1, long). has_wall(car2_1, railing). has_car(train2, car2_2). car_num(car2_2, 2). car_color(car2_2, blue). car_len(car2_2, long). has_wall(car2_2, full).", "westbound"),
        ("has_car(train3, car3_1). car_num(car3_1, 1). car_color(car3_1, white). car_len(car3_1, short). has_wall(car3_1, full). has_car(train3, car3_2). car_num(car3_2, 2). car_color(car3_2, blue). car_len(car3_2, short). has_wall(car3_2, full).", "eastbound"),
    ]
    result = solve_slr(_demo_examples)
    print('success:', result['success'])
    print('program:', result['program'])
    print('score:  ', result.get('score'))
    print('top-k:')
    for r in result['top_k_programs']:
        print(' ', r)
