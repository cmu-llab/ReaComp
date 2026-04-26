"""
Program-by-Example (PBE) Solver using Beam Search.

This solver infers a sequence of replace(A, B) operations that transform
all input strings to their corresponding output strings.

DSL: replace(A, B) with 1 <= len(A) <= 3, 0 <= len(B) <= 3
Constraints: max 5 programs (Lite) or 20 (Hard)

Algorithm: Beam search over program sequences, guided by the verifier.
"""

from collections import Counter
import time
import sys
import os
import random

# Add workspace to path for verifier import
sys.path.insert(0, '/workspace')
from rewards.pbebench import reward


def find_single_replacements(input_str, output_str):
    """
    Find all valid (old, new) pairs where input_str.replace(old, new) == output_str.

    Constraints: 1 <= len(old) <= 3, 0 <= len(new) <= 3.
    Uses substring matching from both input and output for efficiency.
    """
    if input_str == output_str:
        return []

    candidates = set()
    n_in = len(input_str)
    n_out = len(output_str)

    # Try all substrings of input as the "old" pattern
    for i in range(n_in):
        for j in range(i + 1, min(i + 4, n_in + 1)):
            old = input_str[i:j]
            # Try all substrings of output as the "new" replacement
            for i2 in range(n_out):
                for j2 in range(i2 + 1, min(i2 + 4, n_out + 1)):
                    new = output_str[i2:j2]
                    if input_str.replace(old, new) == output_str:
                        candidates.add((old, new))
            # Also try empty replacement (deletion)
            if input_str.replace(old, '') == output_str:
                candidates.add((old, ''))

    return list(candidates)


def apply_programs(s, programs):
    """Apply a sequence of (old, new) replacements to a string."""
    for old, new in programs:
        s = s.replace(old, new)
    return s


def evaluate_programs(inputs, outputs, programs, max_programs=5):
    """
    Evaluate a program sequence against all examples.
    Returns (score, correct_count, total_count).
    """
    n = len(inputs)
    if n == 0:
        return 1.0, 0, 0

    # Check program constraints
    if len(programs) > max_programs:
        return 0.0, 0, n

    correct = 0
    for inp, out in zip(inputs, outputs):
        if apply_programs(inp, programs) == out:
            correct += 1

    return correct / n, correct, n


def collect_candidates(inputs, outputs, programs):
    """
    Collect all candidate (old, new) replacements from unfixed examples.
    Only considers examples that don't yet match their expected output.
    """
    candidates = set()
    for inp, out in zip(inputs, outputs):
        current = apply_programs(inp, programs)
        if current != out:
            cands = find_single_replacements(current, out)
            candidates.update(cands)
    return candidates


def beam_search(inputs, outputs, max_programs=5, beam_width=200, max_time=30):
    """
    Beam search over program sequences.

    At each step, we expand each beam state by trying all unique candidates
    from unfixed examples. We keep the top beam_width states by score.

    Key insight: Programs are applied sequentially using Python's str.replace(),
    which replaces ALL occurrences. The order matters because earlier programs
    can create or destroy patterns for later programs (feeding/bleeding).

    Returns the best program sequence found.
    """
    n = len(inputs)
    if n == 0:
        return [], 1.0

    start_time = time.time()
    best_programs = []
    best_score = 0.0

    # Initial state: (programs_tuple, states_tuple, fixed_count)
    initial_states = tuple(inputs)
    initial_fixed = sum(1 for s, exp in zip(initial_states, outputs) if s == exp)

    # Beam states: (programs_tuple, states_tuple, fixed_count)
    beam = [(tuple(), initial_states, initial_fixed)]

    for step in range(max_programs):
        # Check if we found a perfect solution
        for programs, states, fixed in beam:
            if fixed == n:
                return list(programs), 1.0

        if time.time() - start_time > max_time:
            break

        if not beam:
            break

        # Collect candidates from all unfixed states in beam
        all_candidates = set()
        for programs, states, _ in beam:
            for i, (cur_state, expected) in enumerate(zip(states, outputs)):
                if cur_state == expected:
                    continue
                cands = find_single_replacements(cur_state, expected)
                all_candidates.update(cands)

        if not all_candidates:
            break

        # Expand beam: try each candidate on each beam state
        next_beam = []
        for programs, states, fixed in beam:
            for cand in all_candidates:
                # Skip if this candidate was already used at an earlier position
                # (encourages diversity of program sequences)
                if programs and cand in programs:
                    continue

                # Apply candidate to all states
                new_states = tuple(
                    apply_programs(s, [cand]) for s in states
                )
                new_fixed = sum(
                    1 for ns, exp in zip(new_states, outputs) if ns == exp
                )
                new_programs = programs + (cand,)

                # Only keep if we improved or maintained score
                if new_fixed >= fixed:
                    next_beam.append((new_programs, new_states, new_fixed))

        if not next_beam:
            break

        # Sort by score (descending), then by program length (ascending)
        next_beam.sort(key=lambda x: (-x[2], len(x[0])))
        beam = next_beam[:beam_width]

        # Update best programs
        for programs, states, fixed in beam:
            sc = fixed / n
            if sc > best_score or (sc == best_score and len(programs) < len(best_programs)):
                best_score = sc
                best_programs = list(programs)

    # Final evaluation
    final_score = evaluate_programs(inputs, outputs, best_programs, max_programs)[0]
    return best_programs, final_score


def beam_search_with_seed(inputs, outputs, seed_program, max_programs=5,
                           beam_width=100, max_time=10):
    """
    Beam search that starts from a seed program.
    Used for finding alternative solutions.
    """
    n = len(inputs)
    if n == 0:
        return seed_program, 1.0

    start_time = time.time()
    best_programs = list(seed_program)
    best_score = evaluate_programs(inputs, outputs, seed_program, max_programs)[0]

    initial_states = tuple(apply_programs(inp, seed_program) for inp in inputs)
    initial_fixed = sum(1 for s, exp in zip(initial_states, outputs) if s == exp)

    beam = [(tuple(seed_program), initial_states, initial_fixed)]

    remaining_budget = max_programs - len(seed_program)
    if remaining_budget <= 0:
        return best_programs, best_score

    for step in range(remaining_budget):
        if time.time() - start_time > max_time:
            break

        if not beam:
            break

        for programs, states, fixed in beam:
            if fixed == n:
                return list(programs), 1.0

        # Collect candidates
        all_candidates = set()
        for programs, states, _ in beam:
            for i, (cur_state, expected) in enumerate(zip(states, outputs)):
                if cur_state == expected:
                    continue
                cands = find_single_replacements(cur_state, expected)
                all_candidates.update(cands)

        if not all_candidates:
            break

        # Expand
        next_beam = []
        for programs, states, fixed in beam:
            for cand in all_candidates:
                if cand in programs:
                    continue
                new_states = tuple(
                    apply_programs(s, [cand]) for s in states
                )
                new_fixed = sum(
                    1 for ns, exp in zip(new_states, outputs) if ns == exp
                )
                new_programs = programs + (cand,)

                if new_fixed >= fixed:
                    next_beam.append((new_programs, new_states, new_fixed))

        if not next_beam:
            break

        next_beam.sort(key=lambda x: (-x[2], len(x[0])))
        beam = next_beam[:beam_width]

        for programs, states, fixed in beam:
            sc = fixed / n
            if sc > best_score or (sc == best_score and len(programs) < len(best_programs)):
                best_score = sc
                best_programs = list(programs)

    return best_programs, best_score


def _find_top_k_alternatives(inputs, outputs, best_programs, best_score,
                              max_programs=5, beam_width=200, max_time=30, top_k=10):
    """
    Find top-K alternative programs that differ from the best one.
    Uses multiple beam searches with different starting candidates.
    """
    alternatives = []

    # Collect promising candidates that weren't in the best program
    all_candidates = collect_candidates(inputs, outputs, best_programs)

    # Try starting from different candidates
    candidate_list = list(all_candidates)
    random.seed(42)
    random.shuffle(candidate_list)

    for i, seed in enumerate(candidate_list[:20]):
        if i >= top_k:
            break

        modified_programs, modified_score = beam_search_with_seed(
            inputs, outputs, [seed],
            max_programs=max_programs,
            beam_width=min(beam_width, 100),
            max_time=min(max_time, 10)
        )

        if modified_score > best_score and modified_programs != best_programs:
            prog_strs = [f"replace('{p}', '{t}')" for p, t in modified_programs]
            alternatives.append({
                "programs": modified_programs,
                "program": prog_strs,
                "score": modified_score
            })

    return sorted(alternatives, key=lambda x: -x["score"])[:top_k]


def solve_pbe(examples, max_programs=5, beam_width=200, max_time=30, top_k=10):
    """
    Solve a Programming-by-Example task.

    Parameters:
        examples: list of (input_string, output_string) pairs
        max_programs: maximum number of replace operations allowed (default 5 for Lite)
        beam_width: beam search width (default 200)
        max_time: maximum search time in seconds (default 30)
        top_k: return top-K programs if no perfect solution found

    Returns:
        dict with:
            - success: bool (whether all examples are transformed correctly)
            - program: list of "replace('A', 'B')" strings (for verifier)
            - programs: structured list of (predicate, transform) tuples
            - score: float [0, 1]
            - top_k: list of alternative programs if no perfect solution
    """
    if not examples:
        return {
            "success": True,
            "program": [],
            "programs": [],
            "score": 1.0,
            "top_k": []
        }

    inputs = [ex[0] for ex in examples]
    outputs = [ex[1] for ex in examples]

    # Detect difficulty based on number of changed examples
    changed_count = sum(1 for i, o in zip(inputs, outputs) if i != o)
    total_count = len(inputs)

    # Adaptive search parameters based on task complexity
    if changed_count <= total_count * 0.1:
        adaptive_max = max(max_programs, 3)
        adaptive_beam = min(beam_width, 100)
    elif changed_count <= total_count * 0.3:
        adaptive_max = max(max_programs, 10)
        adaptive_beam = min(beam_width, 200)
    else:
        adaptive_max = max(max_programs, 15)
        adaptive_beam = min(beam_width, 150)

    # Try beam search
    programs, score = beam_search(
        inputs, outputs,
        max_programs=adaptive_max,
        beam_width=adaptive_beam,
        max_time=max_time
    )

    # Format programs as strings (for verifier consumption)
    program_strings = [f"replace('{p}', '{t}')" for p, t in programs]

    # If no perfect solution, try to find top-K alternatives
    top_k_programs = []
    if score < 1.0 and top_k > 0:
        top_k_programs = _find_top_k_alternatives(
            inputs, outputs, programs, score,
            max_programs=adaptive_max,
            beam_width=adaptive_beam,
            max_time=max_time,
            top_k=top_k
        )

    return {
        "success": score >= 1.0,
        "program": program_strings,
        "programs": [(p, t) for p, t in programs],
        "score": score,
        "top_k": top_k_programs
    }


# Main entry point for testing
if __name__ == "__main__":
    # Test with simple examples
    test_examples = [
        ("hello", "hallo"),
        ("world", "worwd"),
        ("python", "python"),
        ("test", "tst"),
    ]

    result = solve_pbe(test_examples, max_programs=5)
    print("Simple test:")
    print(f"  success: {result['success']}")
    print(f"  program: {result['program']}")
    print(f"  score: {result['score']}")

    # Verify with verifier
    entry = {"inputs": [e[0] for e in test_examples], "outputs": [e[1] for e in test_examples]}
    verifier_result = reward(result["program"], True, entry, max_programs=5)
    print(f"  verifier score: {verifier_result['value']}")

    # Test on DEMO examples
    import json
    with open('/workspace/DEMOS.json', 'r') as f:
        demos = json.load(f)

    print("\nDEMO tests:")
    for i in [0, 6, 7]:
        demo = demos[i]
        examples = list(zip(demo['input_examples'], demo['output_examples']))
        result = solve_pbe(examples, max_programs=20, beam_width=150, max_time=20)
        print(f"  DEMO {i}: success={result['success']}, score={result['score']}, programs={len(result['program'])}")
