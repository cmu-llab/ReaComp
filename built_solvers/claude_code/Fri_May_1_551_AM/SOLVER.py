"""
Symbolic PBE Solver

Searches for a sequence of replace(A, B) operations that transforms
all example inputs to their corresponding outputs.

DSL: programs are ordered lists of replace(A, B) operations applied in
cascade.  Constraints: 1 <= len(A) <= 3, 0 <= len(B) <= 3.

Key design decisions derived from studying LLM reasoning traces on
PBEBench tasks (including both successes and failures):

1. Safety filter — rules where A appears in any *unchanged* input would
   break those inputs.  Filtering to "safe" rules dramatically reduces
   the candidate set and eliminates the spurious rules that cause most
   LLM failures.

2. Rule ordering — for Feeding/Counterfeeding/Bleeding interactions the
   correct order of two rules can differ from the "obvious" order.
   Beam search explores orderings naturally by tracking the current
   string state after each prefix.

3. Direct rules first — rules that solve at least one changed example
   in a single step are explored before generic substring-pair rules,
   enabling early exit on easy tasks.

4. Deeper search (max_depth=8) — hard tasks with cascade_length > 5
   exist; shorter equivalent programs may still require > 5 steps.
"""

import sys
import os
from typing import Any, Dict, List, Optional, Tuple

Program = List[Tuple[str, str]]


# ─── Verifier integration ─────────────────────────────────────────────────────

_verifier: Optional[Any] = None
_verifier_checked: bool = False


def _load_verifier() -> Optional[Any]:
    for d in (
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ):
        if d not in sys.path:
            sys.path.insert(0, d)
    try:
        import rewards.pbebench as _m  # type: ignore
        for attr in ("reward", "compute_reward", "score"):
            fn = getattr(_m, attr, None)
            if callable(fn):
                return fn
    except Exception:
        pass
    return None


def _get_verifier() -> Optional[Any]:
    global _verifier, _verifier_checked
    if not _verifier_checked:
        _verifier = _load_verifier()
        _verifier_checked = True
    return _verifier


# ─── Core helpers ─────────────────────────────────────────────────────────────

def apply_program(program: Program, s: str) -> str:
    """Apply each replace operation in sequence (cascade)."""
    for a, b in program:
        s = s.replace(a, b)
    return s


def program_to_repr(program: Program) -> List[Dict[str, str]]:
    """Structured representation consumed by eval harness / verifier."""
    return [{"op": "replace", "from": a, "to": b} for a, b in program]


def cascade_complexity(program: Program) -> int:
    """Sum of len(A) + len(B) across all operations (lower = simpler)."""
    return sum(len(a) + len(b) for a, b in program)


# ─── Rule extraction ─────────────────────────────────────────────────────────

def extract_direct_rules(
    inputs: List[str],
    outputs: List[str],
) -> List[Tuple[str, str]]:
    """
    Direct rules: (A, B) pairs where applying replace(A, B) to some changed
    input yields the corresponding output in a *single step*.

    These are the highest-value candidates and should be tried first.
    """
    rules: set = set()
    for inp, out in zip(inputs, outputs):
        if inp == out:
            continue
        for a_len in range(1, 4):
            for i in range(len(inp) - a_len + 1):
                a = inp[i : i + a_len]
                # deletion
                if inp.replace(a, "") == out:
                    rules.add((a, ""))
                # substitution with substring of output
                for b_len in range(1, 4):
                    for j in range(len(out) - b_len + 1):
                        b = out[j : j + b_len]
                        if a != b and inp.replace(a, b) == out:
                            rules.add((a, b))
    return list(rules)


def extract_supplemental_rules(
    inputs: List[str],
    outputs: List[str],
    max_rules: int = 2000,
) -> List[Tuple[str, str]]:
    """
    Supplemental rules: cross-product of all input substrings (A pool)
    with all output substrings plus "" (B pool), capped at max_rules.

    Used as fallback when direct rules alone cannot reach a solution
    (e.g., multi-step programs where intermediate rules are not direct).
    """
    rules: set = set()
    a_pool: set = set()
    b_pool: set = {""}
    for inp in inputs:
        for l in range(1, 4):
            for i in range(len(inp) - l + 1):
                a_pool.add(inp[i : i + l])
    for out in outputs:
        for l in range(1, 4):
            for i in range(len(out) - l + 1):
                b_pool.add(out[i : i + l])

    for a in sorted(a_pool, key=len):
        if not any(a in inp for inp in inputs):
            continue
        for b in sorted(b_pool, key=len):
            if a != b:
                rules.add((a, b))
        if len(rules) >= max_rules:
            break
    return list(rules)


def safety_filter(
    rules: List[Tuple[str, str]],
    unchanged_inputs: List[str],
) -> List[Tuple[str, str]]:
    """
    Remove rules where A appears in any unchanged input.

    Applying such a rule would incorrectly modify a string that must stay
    the same.  This is the primary source of spurious-rule failures seen
    in LLM CoT traces: a globally-scoped single-char replacement that
    happens to hit an unrelated string.
    """
    if not unchanged_inputs:
        return rules
    return [
        (a, b)
        for a, b in rules
        if not any(a in s for s in unchanged_inputs)
    ]


# ─── Beam search ─────────────────────────────────────────────────────────────

def beam_search(
    examples: List[Tuple[str, str]],
    candidates: List[Tuple[str, str]],
    max_depth: int = 8,
    beam_width: int = 500,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Beam search over programs (ordered sequences of replace rules).

    Each beam item tracks the *current strings* (after applying the
    partial program so far) for O(n * |string|) incremental scoring.

    Beam is ordered by (-score, cascade_complexity): correct and simple
    programs are prioritised.  Early exit when score == 1.0.
    """
    if not candidates:
        return {"success": False, "program": [], "score": 0.0, "top_k": []}

    n = len(examples)
    targets: Tuple[str, ...] = tuple(out for _, out in examples)

    # Prioritise rules that directly solve at least one example
    direct_set = {
        r
        for r in candidates
        if any(inp.replace(r[0], r[1]) == out for inp, out in examples)
    }
    candidates.sort(key=lambda r: (0 if r in direct_set else 1, len(r[0]) + len(r[1])))

    init_strs: Tuple[str, ...] = tuple(inp for inp, _ in examples)
    # beam: list of (score, complexity, prog_tuple, cur_strs_tuple)
    beam: List[Tuple[float, int, tuple, Tuple[str, ...]]] = [
        (0.0, 0, (), init_strs)
    ]
    # all_results: (score, complexity, prog_tuple) — for final top-K
    all_results: List[Tuple[float, int, tuple]] = []

    for _depth in range(max_depth):
        # next_map: prog_tuple -> (score, complexity, prog_tuple, cur_strs)
        next_map: Dict[tuple, Tuple[float, int, tuple, Tuple[str, ...]]] = {}

        for _sc, cplx, prog, cur in beam:
            for a, b in candidates:
                new_cur = tuple(s.replace(a, b) for s in cur)
                if new_cur == cur:
                    continue  # rule is a no-op — skip

                new_prog = prog + ((a, b),)
                new_score = sum(s == t for s, t in zip(new_cur, targets)) / n
                new_cplx = cplx + len(a) + len(b)

                all_results.append((new_score, new_cplx, new_prog))

                if new_score == 1.0:
                    p = list(new_prog)
                    r = program_to_repr(p)
                    return {
                        "success": True,
                        "program": r,
                        "score": 1.0,
                        "top_k": [r],
                    }

                if new_prog not in next_map:
                    next_map[new_prog] = (new_score, new_cplx, new_prog, new_cur)

        if not next_map:
            break

        items = list(next_map.values())
        items.sort(key=lambda x: (-x[0], x[1]))
        beam = items[:beam_width]

    # ── Build top-K ──────────────────────────────────────────────────────────
    if not all_results:
        return {"success": False, "program": [], "score": 0.0, "top_k": []}

    all_results.sort(key=lambda x: (-x[0], x[1]))
    seen: set = set()
    top: List[Tuple[float, tuple]] = []
    for sc, _, pr in all_results:
        if pr not in seen:
            seen.add(pr)
            top.append((sc, pr))
        if len(top) >= top_k:
            break

    best_score, best_prog = top[0]
    return {
        "success": False,
        "program": program_to_repr(list(best_prog)),
        "score": best_score,
        "top_k": [program_to_repr(list(p)) for _, p in top],
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def solve_pbe(examples, top_k: int = 5) -> Dict[str, Any]:
    """
    Infer a sequence of replace(A, B) operations from input/output examples.

    Search strategy (three phases, earliest success wins):

    Phase 1 — Safety-filtered direct rules, deep beam (max_depth=8,
              beam_width=500).  Direct rules are those that solve at least
              one changed example in a single step.  Safety filter removes
              rules whose pattern appears in any unchanged input (the main
              source of spurious-rule failures in LLM traces).

    Phase 2 — Safety-filtered direct + supplemental rules, same beam
              parameters.  Handles multi-step programs where intermediate
              rules are not direct solutions.

    Phase 3 — All direct rules without safety filter, shallower beam
              (max_depth=5, beam_width=200).  Fallback for edge cases where
              intermediate rules temporarily modify unchanged inputs.

    Parameters
    ----------
    examples : list of (input_string, output_string)
    top_k    : number of programs to return when no perfect solution found

    Returns
    -------
    dict with keys:
        success  : bool  — True iff a perfect (score=1.0) program was found
        program  : list of {"op": "replace", "from": A, "to": B} dicts
        score    : float — fraction of examples the best program satisfies
        top_k    : list of top-K programs ranked by (score, -complexity)
    """
    if not examples:
        return {"success": False, "program": [], "score": 0.0, "top_k": []}

    inputs = [str(e[0]) for e in examples]
    outputs = [str(e[1]) for e in examples]

    # Trivial: identity
    if all(i == o for i, o in zip(inputs, outputs)):
        return {"success": True, "program": [], "score": 1.0, "top_k": [[]]}

    unchanged = [inp for inp, out in zip(inputs, outputs) if inp == out]
    examples_list = list(zip(inputs, outputs))

    direct_rules = extract_direct_rules(inputs, outputs)
    if not direct_rules:
        return {"success": False, "program": [], "score": 0.0, "top_k": []}

    # ── Phase 1: safety-filtered direct rules ────────────────────────────────
    safe_direct = safety_filter(direct_rules, unchanged)

    best_result: Optional[Dict[str, Any]] = None

    if safe_direct:
        r1 = beam_search(
            examples_list, safe_direct,
            max_depth=8, beam_width=500, top_k=top_k,
        )
        if r1["success"]:
            return r1
        best_result = r1

    # ── Phase 2: safety-filtered direct + supplemental rules ─────────────────
    supp_rules = extract_supplemental_rules(inputs, outputs)
    safe_all = safety_filter(list(set(direct_rules) | set(supp_rules)), unchanged)

    if len(safe_all) > len(safe_direct):
        r2 = beam_search(
            examples_list, safe_all,
            max_depth=8, beam_width=500, top_k=top_k,
        )
        if r2["success"]:
            return r2
        if best_result is None or r2["score"] > best_result["score"]:
            best_result = r2

    # ── Phase 3: fallback — all direct rules, no safety filter ───────────────
    r3 = beam_search(
        examples_list, direct_rules,
        max_depth=5, beam_width=200, top_k=top_k,
    )
    if r3["success"]:
        return r3
    if best_result is None or r3["score"] > best_result["score"]:
        best_result = r3

    if best_result is None:
        return {"success": False, "program": [], "score": 0.0, "top_k": []}

    # Attempt verifier re-score on the best program (non-blocking)
    verifier = _get_verifier()
    if verifier is not None and best_result.get("program"):
        try:
            v_score = verifier(best_result["program"], examples)
            if isinstance(v_score, (int, float)):
                best_result["score"] = float(v_score)
                if float(v_score) == 1.0:
                    best_result["success"] = True
        except Exception:
            pass

    return best_result
