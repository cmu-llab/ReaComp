#!/usr/bin/env python3
"""
Symbolic Program Synthesizer for Programming-by-Example (PBE) Tasks.

Infers an ordered sequence of str.replace() programs that transforms all
input strings to their corresponding output strings, subject to DSL constraints:
  - Each program is replace(A, B) with 1 <= len(A) <= 3, 0 <= len(B) <= 3
  - Maximum 20 programs in sequence (configurable)
"""

import json
import re
import sys
import copy
import time
from typing import List, Tuple, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Helpers: apply programs, validation, scoring
# ---------------------------------------------------------------------------

def apply_programs(inp: str, programs: List[Tuple[str, str]]) -> str:
    """Apply an ordered sequence of replace programs to an input string."""
    cur = inp
    for pred, trans in programs:
        cur = cur.replace(pred, trans)
    return cur


def validate_programs(programs: List[Tuple[str, str]],
                      max_programs: int = 20,
                      max_pred_len: int = 3,
                      max_transform_len: int = 3) -> bool:
    """Check if a program sequence satisfies DSL constraints."""
    if len(programs) > max_programs:
        return False
    for pred, trans in programs:
        if not (1 <= len(pred) <= max_pred_len):
            return False
        if len(trans) > max_transform_len:
            return False
    return True


def score_programs(programs: List[Tuple[str, str]],
                   examples: List[Tuple[str, str]]) -> float:
    """Return the fraction of examples correctly transformed."""
    if not examples:
        return 0.0
    correct = sum(
        1 for inp, out in examples
        if apply_programs(inp, programs) == out
    )
    return correct / len(examples)


# ---------------------------------------------------------------------------
# Candidate extraction from edit regions
# ---------------------------------------------------------------------------

def extract_candidates_v8(inp: str, out: str) -> set:
    """
    Extract candidate (A, B) replace programs that could explain the
    transformation from inp to out.

    Strategy:
    1. Find the longest common prefix and suffix → defines the edit region.
    2. Generate all sub-string replacements from the edit region.
    3. Split the edit region into segments for multi-step edits.
    4. Extend the context by one character to the left for pattern capture.
    5. Handle insertion cases (empty in_region, non-empty out_region).
    """
    if inp == out:
        return set()

    # Find longest common prefix
    prefix = 0
    while prefix < min(len(inp), len(out)) and inp[prefix] == out[prefix]:
        prefix += 1

    # Find longest common suffix (after the prefix)
    suffix_in = len(inp)
    suffix_out = len(out)
    while (suffix_in > prefix and suffix_out > prefix
           and inp[suffix_in - 1] == out[suffix_out - 1]):
        suffix_in -= 1
        suffix_out -= 1

    in_region = inp[prefix:suffix_in]
    out_region = out[prefix:suffix_out]

    candidates = set()

    def add_standard(in_reg: str, out_reg: str):
        """Add all valid (A, B) candidates from a given edit region."""
        if len(in_reg) > 0 and len(out_reg) > 0:
            # Substitution: replace some chars with others
            for a_len in range(1, min(4, len(in_reg) + 1)):
                for a_start in range(len(in_reg) - a_len + 1):
                    A = in_reg[a_start:a_start + a_len]
                    for b_len in range(min(4, len(out_reg) + 1)):
                        for b_start in range(len(out_reg) - b_len + 1):
                            B = out_reg[b_start:b_start + b_len]
                            candidates.add((A, B))
        elif len(in_reg) > 0 and len(out_reg) == 0:
            # Deletion: remove characters
            for a_len in range(1, min(4, len(in_reg) + 1)):
                for a_start in range(len(in_reg) - a_len + 1):
                    A = in_reg[a_start:a_start + a_len]
                    candidates.add((A, ''))
        elif len(in_reg) == 0 and len(out_reg) > 0:
            # Insertion: add characters (anchor to adjacent existing char)
            if prefix < len(inp):
                anchor = inp[prefix]
                for b_len in range(1, min(4, len(out_reg) + 1)):
                    for b_start in range(len(out_reg) - b_len + 1):
                        B = out_reg[b_start:b_start + b_len] + anchor
                        candidates.add((anchor, B))
            if prefix > 0:
                char_before = inp[prefix - 1]
                for b_len in range(1, min(4, len(out_reg) + 1)):
                    for b_start in range(len(out_reg) - b_len + 1):
                        B = char_before + out_reg[b_start:b_start + b_len]
                        candidates.add((char_before, B))

    # 1. Standard candidates from the edit region
    add_standard(in_region, out_region)

    # 2. Split edit region for multi-step edits
    #    (e.g., 'uJw' → 'pemh' becomes replace('u','p') + replace('Jw','emh'))
    if len(in_region) >= 2:
        for split in range(1, len(in_region)):
            A1, B1 = in_region[:split], out_region[:split]
            A2, B2 = in_region[split:], out_region[split:]
            if 1 <= len(A1) <= 3 and len(B1) <= 3:
                candidates.add((A1, B1))
            if 1 <= len(A2) <= 3 and len(B2) <= 3:
                candidates.add((A2, B2))

    # 3. Extend context one character left (captures patterns like 'Au'→'Ap')
    if prefix > 0:
        extended_in = inp[prefix - 1] + in_region
        extended_out = out[prefix - 1] + out_region
        add_standard(extended_in, extended_out)

    return candidates


# ---------------------------------------------------------------------------
# Program search: greedy + residual fixing
# ---------------------------------------------------------------------------

def find_programs_greedy(examples: List[Tuple[str, str]],
                         max_programs: int = 20) -> List[Tuple[str, str]]:
    """
    Greedy iterative search with residual fixing.

    Phase 1 – Greedy selection:
        At each step, pick the candidate that fixes the most unfixed examples
        while breaking the fewest already-fixed ones.

    Phase 2 – Single-program residual:
        For each remaining unfixed example, try a single program that fixes it.

    Phase 3 – Multi-program residual:
        For each remaining unfixed example, try pairs of programs applied
        sequentially.
    """
    if not examples:
        return []

    # Collect all candidates across all examples
    all_candidates: set = set()
    for inp, out in examples:
        if inp != out:
            all_candidates |= extract_candidates_v8(inp, out)
    all_candidates = list(all_candidates)

    programs: List[Tuple[str, str]] = []

    # ---- Phase 1: Greedy ----
    for _ in range(max_programs):
        best_cand = None
        best_score = -1

        for cand in all_candidates:
            if cand in programs:
                continue
            pred, trans = cand
            n_fix = 0
            n_break = 0
            for inp, out in examples:
                cur = apply_programs(inp, programs)
                if cur == out:
                    new = cur.replace(pred, trans)
                    if new != out:
                        n_break += 1
                else:
                    new = cur.replace(pred, trans)
                    if new == out:
                        n_fix += 1
            # Score: net fixes, penalize breaking existing fixes
            score = n_fix - n_break * 2
            if (score > best_score
                    or (score == best_score and best_cand is not None
                        and len(cand[0]) > len(best_cand[0]))):
                best_score = score
                best_cand = cand

        if best_cand is None or best_score <= 0:
            break
        programs.append(best_cand)

    # ---- Phase 2: Single-program residual ----
    for _ in range(max_programs * 2):
        unfixed = [
            i for i, (inp, out) in enumerate(examples)
            if apply_programs(inp, programs) != out
        ]
        if not unfixed:
            break
        fixed_any = False
        for idx in unfixed:
            if len(programs) >= max_programs:
                break
            inp, out = examples[idx]
            current = apply_programs(inp, programs)
            cands = extract_candidates_v8(current, out)
            for c in cands:
                if c in programs:
                    continue
                if apply_programs(inp, programs + [c]) == out:
                    # Check it doesn't break too many examples
                    n_break = sum(
                        1 for i2, (inp2, out2) in enumerate(examples)
                        if i2 != idx
                        and apply_programs(inp2, programs) == out2
                        and apply_programs(inp2, programs + [c]) != out2
                    )
                    if n_break <= 2:
                        programs.append(c)
                        fixed_any = True
                        break
        if not fixed_any:
            break

    # ---- Phase 3: Multi-program residual (pairs) ----
    unfixed = [
        i for i, (inp, out) in enumerate(examples)
        if apply_programs(inp, programs) != out
    ]
    if unfixed and len(programs) < max_programs - 1:
        for idx in unfixed:
            if len(programs) >= max_programs - 1:
                break
            inp, out = examples[idx]
            current = apply_programs(inp, programs)
            cands = extract_candidates_v8(current, out)
            cands_list = [c for c in cands if c not in programs]
            found = False
            for c1 in cands_list:
                if found:
                    break
                for c2 in cands_list:
                    if c1 == c2 or c2 in programs:
                        continue
                    test = apply_programs(inp, programs + [c1, c2])
                    if test == out:
                        n_break = sum(
                            1 for i2, (inp2, out2) in enumerate(examples)
                            if i2 != idx
                            and apply_programs(inp2, programs) == out2
                            and apply_programs(inp2, programs + [c1, c2])
                            != out2
                        )
                        if n_break <= 2:
                            programs.append(c1)
                            programs.append(c2)
                            found = True
                            break

    return programs


# ---------------------------------------------------------------------------
# Top-K program finder (returns best programs when none are perfect)
# ---------------------------------------------------------------------------

def find_top_k_programs(examples: List[Tuple[str, str]],
                        max_programs: int = 20,
                        k: int = 5) -> List[Tuple[List[Tuple[str, str]], float]]:
    """
    Return the top-K program sequences ranked by verifier score.
    Each element is (program_list, score).
    """
    programs = find_programs_greedy(examples, max_programs)
    score = score_programs(programs, examples)
    return [(programs, score)]


# ---------------------------------------------------------------------------
# Format programs for output
# ---------------------------------------------------------------------------

def format_programs(programs: List[Tuple[str, str]]) -> str:
    """Format a list of (A, B) pairs as a markdown-ready program sequence."""
    lines = [f"replace('{a}', '{b}')" for a, b in programs]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main solver interface
# ---------------------------------------------------------------------------

def solve_pbe(examples: List[Tuple[str, str]],
              max_programs: int = 20,
              k: int = 5) -> Dict[str, Any]:
    """
    Solve a Programming-by-Example task.

    Parameters
    ----------
    examples : list of (input_string, output_string) pairs
    max_programs : maximum number of replace() programs allowed
    k : number of top programs to return (for partial solutions)

    Returns
    -------
    dict with at least:
        - "success": bool – True if all examples are transformed correctly
        - "program": list of (predicate, transform) tuples
        - "score": float – fraction of examples correctly transformed
        - "candidates": list of top-K programs with scores (if not fully successful)
    """
    start_time = time.time()

    # Handle trivial case: all examples unchanged
    if not examples or all(inp == out for inp, out in examples):
        return {
            "success": True,
            "program": [],
            "score": 1.0,
            "candidates": [],
        }

    # Run the synthesizer
    programs = find_programs_greedy(examples, max_programs)
    score = score_programs(programs, examples)
    success = score >= 1.0

    result: Dict[str, Any] = {
        "success": success,
        "program": programs,
        "score": score,
        "candidates": [],
    }

    # If not fully successful, collect top-K candidates
    if not success:
        candidates = find_top_k_programs(examples, max_programs, k)
        result["candidates"] = [
            {
                "program": [format_programs(prog)],
                "score": sc,
                "success": sc >= 1.0,
            }
            for prog, sc in candidates[:k]
        ]

    result["elapsed"] = time.time() - start_time
    return result


# ---------------------------------------------------------------------------
# CLI entry point (for testing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Read examples from stdin as JSON or parse command-line arguments
    if len(sys.argv) >= 3:
        # Format: python SOLVER.py '["a","b","c"]' '["x","y","z"]'
        import json
        inputs = json.loads(sys.argv[1])
        outputs = json.loads(sys.argv[2])
        examples = list(zip(inputs, outputs))
    else:
        print("Usage: python SOLVER.py '<inputs>' '<outputs>'")
        print('Example: python SOLVER.py \'["abc","def"]\' \'["xbc","deg"\']')
        sys.exit(1)

    result = solve_pbe(examples)
    print(json.dumps({
        "success": result["success"],
        "score": result["score"],
        "program": result["program"],
        "formatted_program": format_programs(result["program"]),
        "num_candidates": len(result["candidates"]),
    }, indent=2))
