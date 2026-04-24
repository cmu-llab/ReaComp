"""
Symbolic PBE Solver: beam search over replace(A,B) program sequences.

DSL constraints are parametric — pass overrides to solve_pbe():
  max_programs      : max cascade length (default 5 for PBEBench-Lite, 20 for hard)
  max_pred_len      : max len(A) in replace(A,B)  (default 3)
  max_transform_len : max len(B) in replace(A,B)  (default 3)
"""
import sys
import os
import difflib
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewards.pbebench import reward as _pbebench_reward

# Search hyper-parameters (not DSL limits)
BEAM_WIDTH = 150
TIME_LIMIT = 55.0  # seconds per solve_pbe call

# Default DSL limits (PBEBench-Lite)
DEFAULT_MAX_PROGRAMS = 5
DEFAULT_MAX_PRED_LEN = 3
DEFAULT_MAX_TRANSFORM_LEN = 3


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def _apply(programs, text):
    for p, r in programs:
        text = text.replace(p, r)
    return text


def _score(programs, examples):
    if not examples:
        return 0.0
    return sum(1 for i, o in examples if _apply(programs, i) == o) / len(examples)


def _verify(programs, examples, max_programs, max_pred_len, max_transform_len):
    strs = [f"replace('{p}', '{r}')" for p, r in programs]
    entry = {
        "inputs": [i for i, _ in examples],
        "outputs": [o for _, o in examples],
    }
    return _pbebench_reward(
        strs, True, entry,
        max_programs=max_programs,
        max_pred_len=max_pred_len,
        max_transform_len=max_transform_len,
    )["value"]


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _extract_candidates_from(pairs, max_pred_len, max_transform_len):
    """
    Generate (pattern, replacement) candidates from (current_string, target) pairs.
    Pairs represent intermediate states: src after partial cascade, tgt = desired output.
    """
    cands = set()

    for src, tgt in pairs:
        if src == tgt:
            continue

        sm = difflib.SequenceMatcher(None, src, tgt)

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                continue

            s_chunk = src[i1:i2]
            t_chunk = tgt[j1:j2]

            # Direct replacement if both fit in the DSL
            if 1 <= len(s_chunk) <= max_pred_len and len(t_chunk) <= max_transform_len:
                cands.add((s_chunk, t_chunk))

            # All substrings of the changed region
            for plen in range(1, min(len(s_chunk), max_pred_len) + 1):
                for pi in range(len(s_chunk) - plen + 1):
                    pat = s_chunk[pi:pi + plen]
                    cands.add((pat, ''))
                    for rlen in range(1, min(len(t_chunk), max_transform_len) + 1):
                        for ri in range(len(t_chunk) - rlen + 1):
                            cands.add((pat, t_chunk[ri:ri + rlen]))

            # Context-extended patterns (1-2 surrounding chars for specificity)
            for lext in range(0, min(i1, 2) + 1):
                for rext in range(0, min(len(src) - i2, 2) + 1):
                    if lext == 0 and rext == 0:
                        continue
                    ext_src = src[i1 - lext:i2 + rext]
                    if not (1 <= len(ext_src) <= max_pred_len):
                        continue
                    lctx = src[i1 - lext:i1]
                    rctx = src[i2:i2 + rext]
                    ext_tgt = lctx + t_chunk + rctx
                    if len(ext_tgt) <= max_transform_len:
                        cands.add((ext_src, ext_tgt))

    return [
        (p, r) for p, r in cands
        if p != r
        and 1 <= len(p) <= max_pred_len
        and len(r) <= max_transform_len
    ]


def _rank_candidates(candidates, examples, partial_progs=None):
    """
    Rank by:
      1. Safety: pattern absent from all already-correct current strings.
      2. Direct fixes: number of wrong pairs the candidate fully corrects.
      3. Partial-fire penalty: fires but doesn't fix.
      4. Pattern length (shorter = more general).
    """
    if partial_progs is None:
        partial_progs = []

    states = [(_apply(partial_progs, inp), out) for inp, out in examples]
    correct_now = {s for s, t in states if s == t}
    wrong = [(s, t) for s, t in states if s != t]

    scored = []
    for p, r in candidates:
        safe = not any(p in s for s in correct_now)
        fixes = sum(1 for s, t in wrong if s.replace(p, r) == t)
        partial = sum(1 for s, t in wrong if p in s and s.replace(p, r) != t)
        scored.append((int(safe), fixes, -partial, -len(p), p, r))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    return [(p, r) for _, _, _, _, p, r in scored]


# ---------------------------------------------------------------------------
# Beam search helpers
# ---------------------------------------------------------------------------

def _safe_beam_search(
    examples, beam_width, max_depth, deadline,
    max_pred_len, max_transform_len,
):
    """Phase 1: only extend with candidates that won't corrupt already-correct pairs."""
    best_prog = []
    best_score = _score([], examples)

    if best_score >= 1.0:
        return [], 1.0

    beam = [([], best_score)]

    for _depth in range(max_depth):
        if time.time() > deadline:
            break

        seen_keys = set()
        next_beam = []

        for progs, _ in beam:
            if time.time() > deadline:
                break

            current_pairs = [(_apply(progs, inp), out) for inp, out in examples]
            correct_now = {s for s, t in current_pairs if s == t}

            cands = _extract_candidates_from(current_pairs, max_pred_len, max_transform_len)
            cands = [(p, r) for p, r in cands if not any(p in s for s in correct_now)]
            cands = _rank_candidates(cands, examples, partial_progs=progs)
            cands = cands[:200]

            for p, r in cands:
                new_progs = progs + [(p, r)]
                key = tuple(new_progs)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                sc = _score(new_progs, examples)
                next_beam.append((new_progs, sc))

                if sc > best_score or (
                    sc == best_score and best_prog and len(new_progs) < len(best_prog)
                ):
                    best_score = sc
                    best_prog = list(new_progs)

                if sc >= 1.0:
                    return new_progs, 1.0

        if not next_beam:
            break

        next_beam.sort(key=lambda x: (-x[1], len(x[0])))
        beam = next_beam[:beam_width]

    return best_prog, best_score


def _beam_search(
    examples, beam_width, max_depth, deadline,
    max_pred_len, max_transform_len,
):
    """Phase 2: unrestricted beam search (no safety filter)."""
    best_prog = []
    best_score = _score([], examples)

    if best_score >= 1.0:
        return [], 1.0

    beam = [([], best_score)]

    for _depth in range(max_depth):
        if time.time() > deadline:
            break

        seen_keys = set()
        next_beam = []

        for progs, _ in beam:
            if time.time() > deadline:
                break

            current_pairs = [(_apply(progs, inp), out) for inp, out in examples]
            cands = _extract_candidates_from(current_pairs, max_pred_len, max_transform_len)
            cands = _rank_candidates(cands, examples, partial_progs=progs)
            cands = cands[:200]

            for p, r in cands:
                new_progs = progs + [(p, r)]
                key = tuple(new_progs)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                sc = _score(new_progs, examples)
                next_beam.append((new_progs, sc))

                if sc > best_score or (
                    sc == best_score and best_prog and len(new_progs) < len(best_prog)
                ):
                    best_score = sc
                    best_prog = list(new_progs)

                if sc >= 1.0:
                    return new_progs, 1.0

        if not next_beam:
            break

        next_beam.sort(key=lambda x: (-x[1], len(x[0])))
        beam = next_beam[:beam_width]

    return best_prog, best_score


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def solve_pbe(
    examples,
    top_k=5,
    max_programs=DEFAULT_MAX_PROGRAMS,
    max_pred_len=DEFAULT_MAX_PRED_LEN,
    max_transform_len=DEFAULT_MAX_TRANSFORM_LEN,
):
    """
    Infer a replace-cascade program from input/output string examples.

    Parameters
    ----------
    examples          : list of (input_string, output_string)
    top_k             : kept for API compatibility; currently returns 1 candidate
    max_programs      : max cascade length (default 5; pass 20 for hard PBEBench)
    max_pred_len      : max len(A) in replace(A,B) (default 3)
    max_transform_len : max len(B) in replace(A,B) (default 3)

    Returns
    -------
    dict with:
      "success"  – bool, True iff a fully correct program was found
      "program"  – list of replace(...) strings (best sequence found)
      "programs" – [program] (list of candidate lists, best first)
      "score"    – float verifier score [0, 1]
    """
    if not examples:
        return {"success": False, "program": [], "programs": [], "score": 0.0}

    if all(i == o for i, o in examples):
        return {"success": True, "program": [], "programs": [[]], "score": 1.0}

    deadline = time.time() + TIME_LIMIT
    half = (deadline + time.time()) / 2

    best_prog, best_score = _safe_beam_search(
        examples,
        beam_width=BEAM_WIDTH,
        max_depth=max_programs,
        deadline=half,
        max_pred_len=max_pred_len,
        max_transform_len=max_transform_len,
    )

    if best_score < 1.0 and time.time() < deadline:
        prog2, score2 = _beam_search(
            examples,
            beam_width=BEAM_WIDTH // 2,
            max_depth=max_programs,
            deadline=deadline,
            max_pred_len=max_pred_len,
            max_transform_len=max_transform_len,
        )
        if score2 > best_score or (score2 == best_score and len(prog2) < len(best_prog)):
            best_prog, best_score = prog2, score2

    if not best_prog:
        return {"success": False, "program": [], "programs": [], "score": 0.0}

    actual_score = _verify(
        best_prog, examples, max_programs, max_pred_len, max_transform_len
    )
    prog_strs = [f"replace('{p}', '{r}')" for p, r in best_prog]

    return {
        "success": actual_score >= 1.0,
        "program": prog_strs,
        "programs": [prog_strs],
        "score": actual_score,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    with open(os.path.join(os.path.dirname(__file__), "DEMOS.json")) as fh:
        demos = json.load(fh)

    t0 = time.time()
    results = []
    for i, demo in enumerate(demos[:25]):
        examples = list(zip(demo["input_examples"], demo["output_examples"]))
        r = solve_pbe(examples)
        status = "PASS" if r["success"] else "FAIL"
        results.append(r["success"])
        print(
            f"Demo {i:2d} [{demo['difficulty']:4s}] cascade={demo['cascade_length']:2d} "
            f"→ {status} score={r['score']:.3f}"
        )

    print(f"\n{sum(results)}/{len(results)} solved in {time.time()-t0:.1f}s")
