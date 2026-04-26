#!/usr/bin/env python3
"""
SOLVER.py — Symbolic Logic Rule solver for SLR-Bench train-direction tasks.

Infer a Prolog rule of the form eastbound(T) :- Body. from labelled
(background_facts, direction_label) pairs using a multi-stage property-based
candidate generation strategy.  Candidates are generated and filtered purely
in Python.  Only a limited number of the simplest candidates are verified
via the SWI-Prolog judge to minimize wall-clock time.
"""

import re
import itertools
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

sys.path.insert(0, "/workspace")
from rewards.slr_bench import reward, rule_complexity

# Limits
MAX_CONJ_COMBINATIONS = 50000
MAX_MIXED_INT = 10000
MAX_VERIFIER_CALLS = 5


# ======================================================================
# 1. Fact parsing
# ======================================================================

_FACT_RE = re.compile(r"(\w+)\(([^)]*)\)")


def parse_facts(facts_str: str) -> List[Tuple[str, Tuple[str, ...]]]:
    return [
        (m.group(1), tuple(a.strip() for a in m.group(2).split(",")))
        for m in _FACT_RE.finditer(facts_str)
    ]


def extract_train_id(facts):
    for pred, args in facts:
        if pred == "has_car":
            return args[0]
    return None


def extract_cars(facts):
    return [args[1] for pred, args in facts if pred == "has_car"]


# ======================================================================
# 2. Predicate classification
# ======================================================================

_VALUE_PREDICATES = {
    "car_color", "car_len", "has_wall", "has_roof",
    "has_payload", "car_type", "has_window",
}
_INT_PREDICATES = {"car_num", "has_wheel", "load_num", "passenger_num"}

def is_value_pred(p): return p in _VALUE_PREDICATES
def is_int_pred(p): return p in _INT_PREDICATES


# ======================================================================
# 3. In-Python matching helpers
# ======================================================================

def _any_car(facts, cars, pred, value):
    for car in cars:
        for p, a in facts:
            if p == pred and a[0] == car and a[1] == value:
                return True
    return False

def _all_match(FL, CL, pred, value):
    return all(_any_car(f, c, pred, value) for f, c in zip(FL, CL))

def _none_match(FL, CL, pred, value):
    return all(not _any_car(f, c, pred, value) for f, c in zip(FL, CL))

def _any_conj(facts, cars, combo):
    for car in cars:
        if all(any(p == pr and a[0] == car and a[1] == vl for p, a in facts)
               for pr, vl in combo):
            return True
    return False

def _all_conj(FL, CL, combo):
    return all(_any_conj(f, c, combo) for f, c in zip(FL, CL))

def _none_conj(FL, CL, combo):
    return all(not _any_conj(f, c, combo) for f, c in zip(FL, CL))

def _int_any(facts, cars, pred, cond):
    for car in cars:
        for p, a in facts:
            if p == pred and a[0] == car and len(a) == 2:
                try:
                    v = int(a[1])
                    if (cond == ">0" and v > 0) or (cond == "==0" and v == 0):
                        return True
                except ValueError:
                    pass
    return False

def _int_all(FL, CL, pred, cond):
    return all(_int_any(f, c, pred, cond) for f, c in zip(FL, CL))

def _int_none(FL, CL, pred, cond):
    return all(not _int_any(f, c, pred, cond) for f, c in zip(FL, CL))

def _neg_any(facts, cars, pred, value):
    for car in cars:
        for p, a in facts:
            if p == pred and a[0] == car and a[1] != value:
                return True
    return False

def _neg_all(FL, CL, pred, value):
    return all(_neg_any(f, c, pred, value) for f, c in zip(FL, CL))

def _neg_none(FL, CL, pred, value):
    return all(not _neg_any(f, c, pred, value) for f, c in zip(FL, CL))

def _univ_any(facts, cars, pred, value):
    for car in cars:
        for p, a in facts:
            if p == pred and a[0] == car and a[1] == value:
                return True
    return False

def _univ_all(FL, CL, pred, value):
    return all(not _univ_any(f, c, pred, value) for f, c in zip(FL, CL))

def _univ_some(FL, CL, pred, value):
    return any(_univ_any(f, c, pred, value) for f, c in zip(FL, CL))


# ======================================================================
# 4. Candidate generation (returns generator-like lazy list)
# ======================================================================

def _gather_candidates(FL):
    cands = set()
    for facts in FL:
        for p, a in facts:
            if p != "has_car" and len(a) == 2:
                cands.add((p, a[1]))
    return cands


def _gen_single_props(FL, CL, NF, NCL):
    cands = _gather_candidates(FL)
    for pred, value in cands:
        if _all_match(FL, CL, pred, value) and _none_match(NF, NCL, pred, value):
            rule = f"eastbound(T) :- has_car(T, C), {pred}(C, {value})."
            yield (rule, rule_complexity(rule))


def _gen_int_props(FL, CL, NF, NCL):
    vals = defaultdict(set)
    for facts in FL:
        for p, a in facts:
            if is_int_pred(p) and len(a) == 2:
                try: vals[p].add(int(a[1]))
                except: pass
    for pred, values in vals.items():
        for v in sorted(values):
            if _all_match(FL, CL, pred, str(v)) and _none_match(NF, NCL, pred, str(v)):
                rule = f"eastbound(T) :- has_car(T, C), {pred}(C, {v})."
                yield (rule, rule_complexity(rule))
        if _int_all(FL, CL, pred, ">0") and _int_none(NF, NCL, pred, ">0"):
            rule = f"eastbound(T) :- has_car(T, C), {pred}(C, N), N > 0."
            yield (rule, rule_complexity(rule))
        if _all_match(FL, CL, pred, "0") and _none_match(NF, NCL, pred, "0"):
            rule = f"eastbound(T) :- has_car(T, C), {pred}(C, 0)."
            yield (rule, rule_complexity(rule))


def _gen_negation_rules(FL, CL, NF, NCL):
    vals = defaultdict(set)
    for facts in FL:
        for p, a in facts:
            if is_value_pred(p) and len(a) == 2:
                vals[p].add(a[1])
    for pred, values in vals.items():
        for excluded in sorted(values):
            if _neg_all(FL, CL, pred, excluded) and _neg_none(NF, NCL, pred, excluded):
                rule = f"eastbound(T) :- has_car(T, C), {pred}(C, _X), _X \\\\= {excluded}."
                yield (rule, rule_complexity(rule))


def _gen_universal_negation(FL, CL, NF, NCL):
    vals = defaultdict(set)
    for facts in FL:
        for p, a in facts:
            if len(a) == 2: vals[p].add(a[1])
    for facts in NF:
        for p, a in facts:
            if len(a) == 2: vals[p].add(a[1])
    for pred, values in vals.items():
        for val in sorted(values):
            if _univ_all(FL, CL, pred, val) and _univ_some(NF, NCL, pred, val):
                rule = f"eastbound(T) :- \\+ ( has_car(T, C), {pred}(C, {val}) )."
                yield (rule, rule_complexity(rule))


def _gen_conjunctions(FL, CL, NF, NCL, min_size, max_size):
    cands = list(_gather_candidates(FL))
    for size in range(min_size, max_size + 1):
        checked = 0
        for combo in itertools.combinations(cands, size):
            checked += 1
            if checked > MAX_CONJ_COMBINATIONS: break
            if _all_conj(FL, CL, list(combo)) and _none_conj(NF, NCL, list(combo)):
                parts = ["has_car(T, C)"] + [f"{p}(C, {v})" for p, v in combo]
                rule = "eastbound(T) :- " + ", ".join(parts) + "."
                yield (rule, rule_complexity(rule))


def _gen_mixed_int_conjunctions(FL, CL, NF, NCL):
    cv = defaultdict(set)
    for facts in FL:
        for p, a in facts:
            if is_value_pred(p) and len(a) == 2: cv[p].add(a[1])
    ip = set()
    for facts in FL:
        for p, a in facts:
            if is_int_pred(p) and len(a) == 2: ip.add(p)
    checked = 0
    for cat_pred in sorted(cv.keys()):
        for cat_val in cv[cat_pred]:
            for int_pred in sorted(ip):
                checked += 1
                if checked > MAX_MIXED_INT: break
                pos_ok = True
                for pf, pc in zip(FL, CL):
                    found = False
                    for car in pc:
                        cat_ok = any(p == cat_pred and a[0] == car and a[1] == cat_val
                                     for p, a in pf)
                        if not cat_ok: continue
                        int_ok = any(p == int_pred and a[0] == car and len(a) == 2
                                     and a[1].isdigit() and int(a[1]) > 0 for p, a in pf)
                        if cat_ok and int_ok: found = True; break
                    if not found: pos_ok = False; break
                if not pos_ok: continue
                neg_ok = True
                for nf, nc in zip(NF, NCL):
                    for car in nc:
                        cat_ok = any(p == cat_pred and a[0] == car and a[1] == cat_val
                                     for p, a in nf)
                        if not cat_ok: continue
                        int_ok = any(p == int_pred and a[0] == car and len(a) == 2
                                     and a[1].isdigit() and int(a[1]) > 0 for p, a in nf)
                        if cat_ok and int_ok: neg_ok = False; break
                    if not neg_ok: break
                if pos_ok and neg_ok:
                    rule = f"eastbound(T) :- has_car(T, C), {cat_pred}(C, {cat_val}), {int_pred}(C, N), N > 0."
                    yield (rule, rule_complexity(rule))
            if checked > MAX_MIXED_INT: break


# ======================================================================
# 5. Verifier wrapper
# ======================================================================

def _evaluate_rule(rule: str, validation_program: str) -> float:
    try:
        entry = {"validation program": validation_program}
        result = reward(rule, True, entry)
        return float(result.get("value", 0.0))
    except Exception:
        return 0.0


def build_validation_program(examples):
    lines = []
    seen = set()
    for facts_str, label in examples:
        facts = parse_facts(facts_str)
        tid = extract_train_id(facts)
        if not tid or tid in seen: continue
        seen.add(tid)
        lines.append(f"{label}({tid}).")
        for pred, args in facts:
            if pred == "has_car":
                lines.append(f"{pred}({args[0]}, {args[1]}).")
            else:
                lines.append(f"{pred}({', '.join(args)}).")
    return "\n".join(lines) + "\n"


# ======================================================================
# 6. Main solver
# ======================================================================

def solve_slr(examples, top_k=5):
    """
    Parameters
    ----------
    examples : list of (facts_string, direction_label)
    top_k    : number of top candidates to return
    Returns
    -------
    dict with "success", "program", "top_k_programs", "score"
    """
    pos_pairs = [(f, l) for f, l in examples if l == "eastbound"]
    neg_pairs = [(f, l) for f, l in examples if l == "westbound"]
    if not pos_pairs or not neg_pairs:
        return {"success": False, "program": "", "top_k_programs": [], "score": 0.0}

    FL = [parse_facts(f) for f, _ in pos_pairs]
    CL = [extract_cars(pf) for pf in FL]
    NF = [parse_facts(f) for f, _ in neg_pairs]
    NCL = [extract_cars(nf) for nf in NF]
    vp = build_validation_program(examples)

    # ---- Iterative candidate generation with verifier fallback ----
    # We try stages in order of complexity. If no perfect score after
    # verifying, we generate more candidates from higher stages.
    
    candidates: List[Tuple[str, int]] = []
    verified = 0
    scored: List[Tuple[str, float, int]] = []
    
    # Stage 1: Single categorical + integer + negation + universal negation
    seen_r = set()
    
    for gen_fn in [_gen_single_props, _gen_int_props, _gen_negation_rules, _gen_universal_negation]:
        for rule, comp in gen_fn(FL, CL, NF, NCL):
            if rule not in seen_r:
                seen_r.add(rule)
                candidates.append((rule, comp))
    
    # Verify stage 1 candidates
    candidates.sort(key=lambda x: x[1])
    for rule, comp in candidates:
        if verified >= MAX_VERIFIER_CALLS: break
        s = _evaluate_rule(rule, vp)
        scored.append((rule, s, comp))
        verified += 1
    
    # If no perfect score and budget remains, try conjunctions
    if not any(s >= 1.0 for _, s, _ in scored) and verified < MAX_VERIFIER_CALLS:
        for sz in range(2, 4):
            for rule, comp in _gen_conjunctions(FL, CL, NF, NCL, sz, sz):
                if rule not in seen_r:
                    seen_r.add(rule)
                    if verified < MAX_VERIFIER_CALLS:
                        s = _evaluate_rule(rule, vp)
                        scored.append((rule, s, comp))
                        verified += 1
                    else:
                        candidates.append((rule, comp))
                        break
            if verified >= MAX_VERIFIER_CALLS: break
    
    # If still no perfect score, try mixed int conjunctions
    if not any(s >= 1.0 for _, s, _ in scored) and verified < MAX_VERIFIER_CALLS:
        for rule, comp in _gen_mixed_int_conjunctions(FL, CL, NF, NCL):
            if rule not in seen_r:
                seen_r.add(rule)
                if verified < MAX_VERIFIER_CALLS:
                    s = _evaluate_rule(rule, vp)
                    scored.append((rule, s, comp))
                    verified += 1
                else:
                    candidates.append((rule, comp))
                    break
    
    # If still no perfect score, verify a few more from remaining candidates
    if not any(s >= 1.0 for _, s, _ in scored) and verified < MAX_VERIFIER_CALLS:
        for rule, comp in candidates:
            if rule not in {r for r, _, _ in scored}:
                s = _evaluate_rule(rule, vp)
                scored.append((rule, s, comp))
                verified += 1
            if verified >= MAX_VERIFIER_CALLS: break

    scored.sort(key=lambda x: (-x[1], x[2]))

    # Fallback to best in-Python candidate
    if not any(s >= 1.0 for _, s, _ in scored) and seen_r:
        # Find simplest rule
        best_rule = None
        best_comp = float('inf')
        for rule, comp in candidates:
            if comp < best_comp:
                best_comp = comp
                best_rule = rule
        if best_rule is None:
            best_rule = sorted(seen_r, key=lambda r: rule_complexity(r))[0]
        scored.insert(0, (best_rule, 0.0, rule_complexity(best_rule)))

    top = scored[:top_k]
    return {
        "success": top[0][1] >= 1.0 if top else False,
        "program": top[0][0] if top else "",
        "top_k_programs": [r[0] for r in top],
        "score": top[0][1] if top else 0.0,
    }


if __name__ == "__main__":
    with open("/workspace/DEMOS.json") as f:
        import json; data = json.load(f)
    for i in range(min(5, len(data))):
        d = data[i]
        ex = list(zip(d["input_examples"], d["output_examples"]))
        r = solve_slr(ex)
        print(f"Demo {i}: success={r['success']} score={r['score']} rule={r['program'][:80]}")
