"""
Programming-by-Example (PBE) Solver

Solves PBE tasks by finding an ordered sequence of replace(A, B) programs
that transforms input strings to corresponding output strings.

Algorithm: Greedy program discovery with multi-step candidate generation.
"""

import time
from rewards.pbebench import reward


def solve_pbe(examples, max_programs=20, K=10):
    """
    Solve a Programming-by-Example task.

    Args:
        examples: list of (input_string, output_string) pairs
        max_programs: maximum number of replace programs in the sequence
        K: number of top candidate programs to return if no perfect solution found

    Returns:
        dict with:
            - "success": bool indicating if all examples are correctly transformed
            - "program": list of program strings like ["replace('a', 'b')", ...]
            - "programs": list of (predicate, transform) tuples
            - "score": float between 0.0 and 1.0
    """
    start_time = time.time()
    time_limit = start_time + 15  # 15 second time limit

    # ── Core utility functions ──────────────────────────────────────────

    def apply_seq(prog_seq, inp):
        """Apply a sequence of (A, B) programs to an input string."""
        s = inp
        for A, B in prog_seq:
            s = s.replace(A, B)
        return s

    def evaluate(prog_seq):
        """Count how many examples are correctly transformed."""
        correct = 0
        for inp, out in examples:
            if apply_seq(prog_seq, inp) == out:
                correct += 1
        return correct, correct / len(examples)

    # ── Separate changed and unchanged pairs ────────────────────────────

    changed = [(i, inp, out) for i, (inp, out) in enumerate(examples) if inp != out]
    unchanged = [(i, inp, out) for i, (inp, out) in enumerate(examples) if inp == out]

    if not changed:
        return {"success": True, "program": [], "programs": []}

    # ── Safety check ────────────────────────────────────────────────────

    def is_safe(prog, prog_seq):
        """Check if adding prog to prog_seq would break any unchanged pair."""
        A, B = prog
        test_seq = prog_seq + [prog]
        for _, uinp, uout in unchanged:
            if apply_seq(test_seq, uinp) != uout:
                return False
        return True

    # ── Substring helpers ───────────────────────────────────────────────

    def get_all_substrings(s, max_len=3):
        """Get all substrings of s with length 0 to max_len."""
        subs = set()
        for lb in range(max_len + 1):
            for start in range(len(s) - lb + 1):
                subs.add(s[start:start + lb])
        return subs

    # ── Candidate generation ───────────────────────────────────────────

    def generate_candidates(prog_seq):
        """
        Generate candidate (A, B) programs from the current state.

        Two types of candidates:
        1. Single-step: programs where current.replace(A, B) == out directly
        2. Multi-step: first step of a 2-step solution where an intermediate
           state can reach the output in one more replace
        """
        candidates = set()
        for _, inp, out in changed:
            current = apply_seq(prog_seq, inp)
            if current == out:
                continue

            # ── Single-step candidates ────────────────────────────────
            # For each possible substring A in current, derive B from output
            for la in range(1, 4):  # A length: 1, 2, 3
                for start in range(len(current) - la + 1):
                    A = current[start:start + la]
                    lb = len(out) - len(current) + la
                    if 0 <= lb <= 3:
                        B = out[start:start + lb]
                        candidates.add((A, B))

            # ── Multi-step candidates ─────────────────────────────────
            # Try all output substrings as possible B values for step 1,
            # then check if the intermediate can reach output in step 2
            output_subs = get_all_substrings(out, 3)
            for la in range(1, 4):
                for start in range(len(current) - la + 1):
                    A1 = current[start:start + la]
                    for B1 in output_subs:
                        intermediate = current.replace(A1, B1)
                        if intermediate != current and intermediate != out:
                            # Check if intermediate can reach out in one more replace
                            for la2 in range(1, 4):
                                found = False
                                for start2 in range(len(intermediate) - la2 + 1):
                                    A2 = intermediate[start2:start2 + la2]
                                    lb2 = len(out) - len(intermediate) + la2
                                    if 0 <= lb2 <= 3:
                                        B2 = out[start2:start2 + lb2]
                                        if intermediate.replace(A2, B2) == out:
                                            candidates.add((A1, B1))
                                            found = True
                                            break
                                if found:
                                    break

        return [p for p in candidates if is_safe(p, prog_seq)]

    # ── Program scoring ────────────────────────────────────────────────

    def improve(prog, prog_seq):
        """How many additional pairs this program would solve."""
        new_seq = prog_seq + [prog]
        before = sum(1 for _, i, o in changed if apply_seq(prog_seq, i) == o)
        after = sum(1 for _, i, o in changed if apply_seq(new_seq, i) == o)
        return after - before

    def total_score(prog, prog_seq):
        """Total number of pairs this program gets correct."""
        new_seq = prog_seq + [prog]
        return sum(1 for inp, out in examples if apply_seq(new_seq, inp) == out)

    # ── Greedy search ──────────────────────────────────────────────────

    prog_seq = []

    for _ in range(max_programs):
        if time.time() > time_limit:
            break

        correct, score = evaluate(prog_seq)
        if correct == len(examples):
            return {
                "success": True,
                "program": [f"replace('{A}', '{B}')" for A, B in prog_seq],
                "programs": list(prog_seq),
                "score": 1.0,
            }

        candidates = generate_candidates(prog_seq)
        if not candidates:
            break

        # Sort by improvement (number of newly solved pairs), then by total score
        candidates.sort(key=lambda p: (improve(p, prog_seq), total_score(p, prog_seq)), reverse=True)

        # Pick the best improving program, or the best overall if none improves
        if candidates[0] and improve(candidates[0], prog_seq) > 0:
            prog_seq = prog_seq + [candidates[0]]
        else:
            prog_seq = prog_seq + [candidates[0]]

    # ── Final evaluation ───────────────────────────────────────────────

    correct, score = evaluate(prog_seq)

    return {
        "success": correct == len(examples),
        "program": [f"replace('{A}', '{B}')" for A, B in prog_seq],
        "programs": list(prog_seq),
        "score": score,
    }
