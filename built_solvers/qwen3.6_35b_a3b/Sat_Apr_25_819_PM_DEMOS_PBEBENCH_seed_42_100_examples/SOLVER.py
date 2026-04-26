"""
SOLVER.py - Symbolic Program Synthesizer for Programming-by-Example (PBE)

Beam-search solver for PBE tasks with str.replace() programs.
"""

from rewards.pbebench import reward


def _str_dist(a, b):
    """Simple character-wise distance between two strings."""
    max_len = max(len(a), len(b))
    dist = 0
    for i in range(max_len):
        c1 = a[i] if i < len(a) else ' '
        c2 = b[i] if i < len(b) else ' '
        if c1 != c2:
            dist += 1
    return dist


def solve_pbe(examples, max_programs=5, max_candidates=100,
              beam_size=50, max_time=30.0):
    """
    Solve a Programming-by-Example task by finding an ordered sequence of
    replace() programs that transforms all input strings to their corresponding
    output strings.

    Parameters
    ----------
    examples : list of (input_string, output_string) pairs
    max_programs : maximum number of replace() programs allowed (default 5)
    max_candidates : maximum number of candidate programs to consider
    beam_size : size of beam search frontier
    max_time : maximum wall-clock time in seconds

    Returns
    -------
    dict with keys:
        - "success": bool - whether all examples are correctly transformed
        - "program": list of str - the program sequence as replace() call strings
        - "num_programs": int - number of programs in the sequence
        - "score": float - fraction of examples correctly transformed
    """
    inputs = [e[0] for e in examples]
    outputs = [e[1] for e in examples]
    n = len(inputs)

    # Identify differing pairs
    diffs = [(i, inputs[i], outputs[i]) for i in range(n) if inputs[i] != outputs[i]]

    if not diffs:
        # All inputs already match outputs - return empty program
        return {
            "success": True,
            "program": [],
            "num_programs": 0,
            "score": 1.0,
        }

    # ── Candidate Generation ────────────────────────────────────────────────
    # Collect all input substrings of length 1-3 (as potential predicates)
    input_subs = set()
    for _, inp, _ in diffs:
        for i in range(len(inp)):
            for j in range(i + 1, min(i + 4, len(inp) + 1)):
                input_subs.add(inp[i:j])

    # Collect all output substrings of length 0-3 (as potential transforms)
    output_subs = set()
    for _, _, out in diffs:
        for i in range(len(out)):
            for j in range(i + 1, min(i + 4, len(out) + 1)):
                output_subs.add(out[i:j])
    output_subs.add("")

    # Score and filter all (pred, trans) pairs
    candidate_scores = []
    for pred in input_subs:
        for trans in output_subs:
            if not (1 <= len(pred) <= 3 and 0 <= len(trans) <= 3):
                continue

            # Count fully explained diffs
            fully = sum(
                1 for _, inp, out in diffs if inp.replace(pred, trans) == out
            )

            if fully > 0:
                # Check if it breaks any unchanged example
                breaks = sum(
                    1 for i in range(n)
                    if inputs[i] == outputs[i]
                    and inputs[i].replace(pred, trans) != inputs[i]
                )
                if breaks == 0:
                    candidate_scores.append((pred, trans, fully * 1000, True))
                continue

            # Check partial progress (makes input closer to output)
            partial = 0
            for _, inp, out in diffs:
                new_state = inp.replace(pred, trans)
                if new_state != inp:
                    old_d = _str_dist(inp, out)
                    new_d = _str_dist(new_state, out)
                    if new_d < old_d:
                        partial += 1

            if partial > 0:
                breaks = sum(
                    1 for i in range(n)
                    if inputs[i] == outputs[i]
                    and inputs[i].replace(pred, trans) != inputs[i]
                )
                if breaks == 0:
                    candidate_scores.append((pred, trans, partial * 10, False))

    # Sort by score descending
    candidate_scores.sort(key=lambda x: -x[2])

    # Take top candidates
    top_n = min(max_candidates, len(candidate_scores))
    final_candidates = [(p[0], p[1]) for p in candidate_scores[:top_n]]

    # ── Beam Search ────────────────────────────────────────────────────────
    # Each beam state: (heuristic_score, program_list, current_states)
    # current_states maps diff index -> current input state after all programs
    initial_states = {idx: inp for idx, _, _ in diffs}
    beam = [(0.0, list(), initial_states)]

    best_program = []
    best_score = 0.0

    for step in range(max_programs):
        new_beam = []

        for _, prog_list, current_states in beam:
            for pred, trans in final_candidates:
                # Skip if already in program list
                if (pred, trans) in prog_list:
                    continue

                # Count fully fixed and partially fixed diffs
                fully_fixed = 0
                partially_fixed = 0
                breaks = False

                for idx, orig_inp, orig_out in diffs:
                    state = current_states[idx]
                    new_state = state.replace(pred, trans)

                    if new_state == orig_out:
                        fully_fixed += 1
                    elif new_state != state:
                        old_d = _str_dist(state, orig_out)
                        new_d = _str_dist(new_state, orig_out)
                        if new_d < old_d:
                            partially_fixed += 1

                # Check if this breaks any unchanged example
                for i in range(n):
                    if (inputs[i] == outputs[i]
                            and inputs[i].replace(pred, trans) != inputs[i]):
                        breaks = True
                        break

                # Skip if no progress or breaks something
                if fully_fixed == 0 and partially_fixed == 0:
                    continue
                if breaks:
                    continue

                # Apply program to all diff states
                new_states = dict(current_states)
                for idx, _, _ in diffs:
                    new_states[idx] = new_states[idx].replace(pred, trans)

                prog_new = prog_list + [(pred, trans)]
                heuristic = fully_fixed * 100 + partially_fixed
                new_beam.append((heuristic, prog_new, new_states))

        if not new_beam:
            break

        # Score each beam state using the verifier
        scored_beam = []
        for _, prog_list, states in new_beam:
            result_prog = [
                "replace('" + p + "', '" + t + "')" for p, t in prog_list
            ]
            entry = {"inputs": inputs, "outputs": outputs}
            r = reward(
                result_prog, True, entry,
                max_programs=max_programs,
                max_pred_len=3,
                max_transform_len=3,
            )
            scored_beam.append((r["value"], prog_list, states))

        # Sort by verifier score (descending), then by program count (ascending)
        scored_beam.sort(key=lambda x: (-x[0], -len(x[1])))

        # Track best overall
        if scored_beam[0][0] > best_score:
            best_score = scored_beam[0][0]
            best_program = scored_beam[0][1]

        # Deduplicate and trim beam
        seen = set()
        trimmed = []
        for val, prog_list, states in scored_beam:
            pk = tuple(prog_list)
            if pk not in seen:
                seen.add(pk)
                trimmed.append((val, prog_list, states))
            if len(trimmed) >= beam_size:
                break

        beam = trimmed

        # Early exit on full success
        if best_score >= 1.0:
            break

    # Format output programs as strings
    program_strs = [
        "replace('" + p + "', '" + t + "')" for p, t in best_program
    ]

    return {
        "success": best_score >= 1.0,
        "program": program_strs,
        "num_programs": len(best_program),
        "score": best_score,
    }
