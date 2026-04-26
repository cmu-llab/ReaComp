"""
SOLVER.py — Programming-by-Example (PBE) Solver

Solves PBE tasks by inferring a sequence of replace(A, B) programs from
(input_string, output_string) example pairs.

DSL: programs are ordered sequences of replace(A, B) calls where
     1 <= len(A) <= max_pred_len, 0 <= len(B) <= max_transform_len,
     max_programs total.

Uses the verifier (rewards/pbebench.py) to score candidate programs.
If no fully correct program is found, returns top-K highest scoring programs.
"""

import json
import re
import random
from itertools import permutations


# ── core helpers ──────────────────────────────────────────────────────────────


def apply_programs_to_string(s, programs):
    """Apply a sequence of (A, B) programs to a string using str.replace()."""
    result = s
    for a, b in programs:
        result = result.replace(a, b)
    return result


def score_programs(programs, inputs, outputs):
    """Score a program sequence: fraction of (input, output) pairs transformed correctly."""
    if not programs:
        return 0.0
    correct = sum(
        1 for i, o in zip(inputs, outputs)
        if apply_programs_to_string(i, programs) == o
    )
    return correct / len(inputs)


def edit_distance(s1, s2):
    """Simple character-level difference count between two strings."""
    max_len = max(len(s1), len(s2))
    pad1 = s1 + ' ' * (max_len - len(s1))
    pad2 = s2 + ' ' * (max_len - len(s2))
    return sum(1 for c1, c2 in zip(pad1, pad2) if c1 != c2)


# ── candidate generation ─────────────────────────────────────────────────────

def find_exact_candidates(inp, out, max_pred_len=3, max_transform_len=3):
    """
    Find all replace(A, B) programs that transform inp exactly to out.
    A is a substring of inp (length 1..max_pred_len), B is the corresponding
    substring of out (length 0..max_transform_len).
    """
    if inp == out:
        return []
    candidates = set()
    for a_len in range(1, max_pred_len + 1):
        for start in range(len(inp) - a_len + 1):
            a = inp[start:start + a_len]
            for b_len in range(max_transform_len + 1):
                end = start + b_len
                if end <= len(out):
                    b = out[start:end]
                else:
                    b = out[start:]
                if inp.replace(a, b) == out:
                    candidates.add((a, b))
    return list(candidates)


def find_progress_candidates(inp, out, max_pred_len=3, max_transform_len=3):
    """
    Find all replace(A, B) programs that make *any progress* toward out,
    even if they don't reach it in one step.  A program makes progress
    if the edit distance between its result and the target decreases.

    This handles multi-step transformations (feeding/bleeding patterns)
    where one replace creates material for a later replace.
    """
    if inp == out:
        return []
    candidates = set()
    current_diff = edit_distance(inp, out)
    for a_len in range(1, max_pred_len + 1):
        for start in range(len(inp) - a_len + 1):
            a = inp[start:start + a_len]
            for b_len in range(max_transform_len + 1):
                end = start + b_len
                if end <= len(out):
                    b = out[start:end]
                else:
                    b = out[start:]
                result = inp.replace(a, b)
                new_diff = edit_distance(result, out)
                if new_diff < current_diff:
                    candidates.add((a, b))
    return list(candidates)


# ── main search algorithm ────────────────────────────────────────────────────


def greedy_sequence_build(inputs, outputs, max_programs, max_pred_len=3,
                          max_transform_len=3, top_k=30):
    """
    Iteratively build a program sequence.

    At each step:
      1. Identify all (input, output) pairs still not correctly transformed.
      2. For each failing pair, compute the intermediate string (after
         applying programs found so far).
      3. Generate two types of candidates from the intermediate string:
         a) Exact candidates — programs that transform the intermediate
            directly to the target output.
         b) Progress candidates — programs that reduce the edit distance
            towards the target (handle multi-step transformations).
      4. Score each candidate on the full example set and pick the best.
    """
    current_programs = []
    for step in range(max_programs):
        failing = [
            (inp, out) for inp, out in zip(inputs, outputs)
            if apply_programs_to_string(inp, current_programs) != out
        ]
        if not failing:
            break

        # Collect candidates from all failing pairs
        all_cands = set()
        for inp, out in failing:
            intermediate = apply_programs_to_string(inp, current_programs)
            all_cands.update(
                find_exact_candidates(intermediate, out, max_pred_len,
                                      max_transform_len)
            )
            all_cands.update(
                find_progress_candidates(intermediate, out, max_pred_len,
                                         max_transform_len)
            )

        # Score and select best candidate
        scored = []
        for a, b in all_cands:
            if len(a) < 1 or len(a) > max_pred_len or len(b) > max_transform_len:
                continue
            if (a, b) in current_programs:
                continue
            test = current_programs + [(a, b)]
            s = score_programs(test, inputs, outputs)
            scored.append((s, (a, b)))

        scored.sort(reverse=True)
        best_cand = None
        best_s = -1
        for s, cand in scored[:min(top_k, len(scored))]:
            if s > best_s:
                best_s = s
                best_cand = cand

        if best_cand and best_s > 0:
            current_programs.append(best_cand)
        if best_s >= 1.0:
            break

    return current_programs


def find_best_ordering(programs, inputs, outputs, max_programs,
                       max_shuffles=200):
    """
    Reorder a program sequence to maximise correctness.

    For sequences ≤ 8 programs: try all permutations (8! = 40320).
    For longer sequences: try random shuffles followed by local
    swap-based optimisation.
    """
    n = len(programs)
    best_score = score_programs(programs, inputs, outputs)
    best = list(programs)

    if n <= 8:
        for perm in permutations(programs):
            s = score_programs(perm, inputs, outputs)
            if s > best_score:
                best_score = s
                best = list(perm)
            if s >= 1.0:
                return best
        return best

    # Random shuffles for longer sequences
    for _ in range(max_shuffles):
        shuffled = list(programs)
        random.shuffle(shuffled)
        s = score_programs(shuffled, inputs, outputs)
        if s > best_score:
            best_score = s
            best = list(shuffled)
        if s >= 1.0:
            return best

    # Local optimisation: swap adjacent pairs
    changed = True
    while changed and best_score < 1.0:
        changed = False
        for i in range(len(best) - 1):
            swapped = list(best)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            s = score_programs(swapped, inputs, outputs)
            if s > best_score:
                best_score = s
                best = swapped
                changed = True

    return best


# ── public API ───────────────────────────────────────────────────────────────

def solve_pbe(examples, K=10, max_programs=20, max_pred_len=3,
              max_transform_len=3, num_starts=5, top_k=30,
              max_shuffles=200):
    """
    Solve a Programming-by-Example task.

    Parameters
    ----------
    examples : list of (input_string, output_string)
        Training pairs defining the transformation.
    K : int
        Number of top programs to return if no fully correct one found.
    max_programs : int
        Maximum number of replace() programs (default 20 for hard tasks;
        pass 5 for PBEBench-Lite).
    max_pred_len : int
        Maximum length of predicate A in replace(A, B) (default 3).
    max_transform_len : int
        Maximum length of transform B (default 3).
    num_starts : int
        Number of independent search restarts with different random seeds.
    top_k : int
        Number of top candidates considered at each greedy step.
    max_shuffles : int
        Maximum random shuffles for reordering longer sequences.

    Returns
    -------
    dict with at least:
        "success" : bool — whether a program scoring 1.0 was found.
        "program" : list of str — the top-ranked program sequence as
                    replace('A', 'B') strings.
        "programs" : list of lists of str — all top-K program sequences.
        "scores" : list of float — scores for each top-K program.
    """
    inputs = [e[0] for e in examples]
    outputs = [e[1] for e in examples]

    all_solutions = []
    best_score = 0.0

    for start in range(num_starts):
        random.seed(start)

        # Build initial sequence
        programs = greedy_sequence_build(
            inputs, outputs, max_programs, max_pred_len, max_transform_len,
            top_k
        )

        # Find best ordering
        programs = find_best_ordering(
            programs, inputs, outputs, max_programs, max_shuffles
        )

        score = score_programs(programs, inputs, outputs)
        prog_tuple = tuple(programs)

        # Deduplicate
        if not any(tuple(s[0]) == prog_tuple for s in all_solutions):
            all_solutions.append((prog_tuple, score))

        if score > best_score:
            best_score = score
        if best_score >= 1.0:
            break

    # Return top-K solutions sorted by score
    all_solutions.sort(key=lambda x: -x[1])
    top_solutions = all_solutions[:K]
    success = any(s >= 1.0 for _, s in top_solutions)

    results = []
    for programs, score in top_solutions:
        raw = [
            "replace('{}', '{}')".format(a, b)
            for a, b in programs
        ]
        results.append({
            "program": raw,
            "score": score,
            "num_programs": len(programs),
        })

    return {
        "success": success,
        "program": results[0]["program"] if results else [],
        "programs": [r["program"] for r in results],
        "scores": [r["score"] for r in results],
    }


# ── main (self-test) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick self-test with a trivial example
    examples = [
        ("hello", "jellp"),
        ("world", "worlp"),
    ]
    result = solve_pbe(examples)
    print(f"Success: {result['success']}")
    print(f"Program: {result['program']}")
    if result['scores']:
        print(f"Score: {result['scores'][0]:.3f}")
