"""
Programming by Example (PBE) Solver

A symbolic solver for PBEBench tasks that infers a sequence of replace(A, B) operations
from (input_string, output_string) example pairs.

The solver uses the verifier (rewards/pbebench.py reward function) to evaluate candidate
programs and returns the best scoring program(s).

Algorithm: Symbolic Program Induction via Candidate Generation and Search
"""

import time
from itertools import product, permutations


def find_candidates(inp, out, max_pred=3, max_trans=3):
    """
    Find all single replace operations that transform inp exactly to out.
    
    For inp.replace(pred, trans) == out, we need:
    - The prefix before pred in inp must match the prefix of out
    - The remaining parts must be consistent
    
    This is efficient because we only try predicates that appear in inp
    and derive trans from the corresponding part of out.
    """
    if inp == out:
        return [("", "")]
    results = []
    seen = set()
    for start in range(len(inp)):
        for end in range(start + 1, min(start + max_pred + 1, len(inp) + 1)):
            pred = inp[start:end]
            before = inp[:start]
            after = inp[end:]
            if not out.startswith(before):
                continue
            out_middle = out[len(before):]
            for b_len in range(0, max_trans + 1):
                if len(out_middle) < b_len:
                    continue
                B = out_middle[:b_len]
                rest = out_middle[b_len:]
                if after.replace(pred, B) == rest:
                    key = (pred, B)
                    if key not in seen:
                        seen.add(key)
                        results.append((pred, B))
    return results


def _get_output_cands(out, max_trans=3):
    """Get candidate strings of length 0 to max_trans from the output."""
    cands = set([""])
    for i in range(len(out)):
        for j in range(i + 1, min(i + max_trans + 2, len(out) + 1)):
            cands.add(out[i:j])
    return list(cands)


def find_all_2op_sequences(inp, out, max_pred=3, max_trans=3):
    """
    Find all valid 2-operation sequences that transform inp to out.
    
    This handles pairs where no single operation suffices. For each possible
    first operation (p1 -> t1), compute the intermediate string and find a
    second operation that transforms it to out.
    
    We use substrings of the output as candidate transforms (t1, t2).
    """
    results = []
    seen = set()
    output_cands = _get_output_cands(out, max_trans)
    
    for start1 in range(len(inp)):
        for end1 in range(start1 + 1, min(start1 + max_pred + 1, len(inp) + 1)):
            p1 = inp[start1:end1]
            for t1 in output_cands:
                intermediate = inp.replace(p1, t1)
                if intermediate == out:
                    continue  # Handled by find_candidates
                
                for start2 in range(len(intermediate)):
                    for end2 in range(start2 + 1, min(start2 + max_pred + 1, len(intermediate) + 1)):
                        p2 = intermediate[start2:end2]
                        for t2 in output_cands:
                            key = (p1, t1, p2, t2)
                            if key not in seen:
                                seen.add(key)
                                if inp.replace(p1, t1).replace(p2, t2) == out:
                                    results.append(((p1, t1), (p2, t2)))
    return results


def apply_program(program, s):
    """Apply a sequence of replace operations to a string."""
    for pred, trans in program:
        s = s.replace(pred, trans)
    return s


def score_program(program, inputs, outputs):
    """Score a program: fraction of examples correctly transformed."""
    if not inputs:
        return 1.0
    correct = sum(1 for inp, out in zip(inputs, outputs) 
                  if apply_program(program, inp) == out)
    return correct / len(inputs)


def solve_pbe(examples, max_programs=5, max_time=30):
    """
    Solve a PBE task from example pairs.
    
    Parameters
    ----------
    examples : list of (input_string, output_string)
        Training examples showing the desired transformation.
    max_programs : int
        Maximum number of replace operations in a program (default 5 for PBEBench-Lite).
        The solver tries up to max_programs + 14 to handle harder tasks.
    max_time : float
        Maximum time in seconds to search for a solution.
    
    Returns
    -------
    dict with keys:
        - "success": bool - whether a program scoring 1.0 was found
        - "program": list of (pred, trans) tuples - the inferred transformation
        - "score": float - score of the program on examples
    """
    if not examples:
        return {"success": True, "program": [], "score": 1.0}
    
    inputs = [e[0] for e in examples]
    outputs = [e[1] for e in examples]
    
    changed_list = [(idx, inp, out) for idx, (inp, out) in enumerate(zip(inputs, outputs))
                    if inp != out]
    if not changed_list:
        return {"success": True, "program": [], "score": 1.0}
    
    start_time = time.time()
    
    # Step 1: Find complete single-operation candidates for each changed pair
    pair_candidates = {}
    hard_pairs = {}
    for idx, inp, out in changed_list:
        cands = [c for c in find_candidates(inp, out) if c[0] != ""]
        if cands:
            pair_candidates[idx] = cands
        else:
            hard_pairs[idx] = (inp, out)
    
    # Step 2: Separate unique (required) and optional candidates
    unique_ops = set()
    optional = {}
    for idx, cands in pair_candidates.items():
        if len(cands) == 1:
            unique_ops.add(cands[0])
        elif len(cands) > 1:
            optional[idx] = cands
    
    # Step 3: Build candidate pool from all sources
    all_ops = set(unique_ops)
    for cands in optional.values():
        all_ops.update(cands)
    
    # Step 4: For hard pairs, find 2-operation sequences and add operations to pool
    hard_seqs = {}
    for idx, (inp, out) in hard_pairs.items():
        seqs = find_all_2op_sequences(inp, out)
        if seqs:
            # Sort by specificity (prefer longer predicates = more specific)
            seqs.sort(key=lambda x: (-(len(x[0][0]) + len(x[1][0])), x))
            hard_seqs[idx] = seqs[:20]  # Limit to top 20
            for (p1, t1), (p2, t2) in seqs[:20]:
                all_ops.add((p1, t1))
                all_ops.add((p2, t2))
    
    all_ops = sorted(all_ops)
    unique_sorted = sorted(unique_ops, key=lambda x: (-len(x[0]), x))
    
    # Step 5: Search for the best program
    best_prog = []
    best_score = 0
    
    def update(prog):
        nonlocal best_prog, best_score
        score = score_program(prog, inputs, outputs)
        if score > best_score:
            best_score = score
            best_prog = list(prog)
        return best_score == 1.0
    
    # Phase 1: Try different program sizes (from max_programs upward)
    for max_p in range(max(1, max_programs), min(max_programs + 15, 25)):
        if time.time() - start_time > max_time - 5:
            break
        
        # Strategy 1: Permutations of unique operations
        for n in range(1, min(len(unique_sorted) + 1, max_p + 1)):
            if time.time() - start_time > max_time - 5:
                break
            for ordering in permutations(unique_sorted[:n]):
                if update(list(ordering)):
                    return {
                        "success": True,
                        "program": list(ordering),
                        "score": 1.0
                    }
        
        # Strategy 2: Greedy addition from candidate pool
        if best_prog and len(best_prog) < max_p:
            remaining = [op for op in all_ops if op not in set(best_prog)]
            for step in range(max_p - len(best_prog)):
                if time.time() - start_time > max_time - 3:
                    break
                best_step = None
                best_step_score = best_score
                for op in remaining:
                    test = best_prog + [op]
                    s = score_program(test, inputs, outputs)
                    if s > best_step_score:
                        best_step_score = s
                        best_step = test
                if best_step:
                    best_prog = best_step
                    best_score = best_step_score
                    if best_score == 1.0:
                        return {
                            "success": True,
                            "program": list(best_prog),
                            "score": 1.0
                        }
                else:
                    break
        
        # Strategy 3: Try all combinations of unique + optional (if manageable)
        if optional:
            opt_keys = sorted(optional.keys())
            opt_values = [optional[k] for k in opt_keys]
            total = 1
            for vals in opt_values:
                total *= len(vals)
            if total <= 3000 and time.time() - start_time < max_time - 5:
                for combo in product(*opt_values):
                    if time.time() - start_time > max_time - 5:
                        break
                    prog_set = unique_ops | set(combo)
                    prog_list = sorted(prog_set)
                    if len(prog_list) <= max_p:
                        for ordering in permutations(prog_list):
                            if update(list(ordering)):
                                return {
                                    "success": True,
                                    "program": list(ordering),
                                    "score": 1.0
                                }
        
        # Strategy 4: Try 2-op sequences from hard pairs combined with best_prog
        if hard_seqs and best_prog:
            for idx, seqs in hard_seqs.items():
                if time.time() - start_time > max_time - 3:
                    break
                for (p1, t1), (p2, t2) in seqs:
                    prog_set = set(best_prog) | {(p1, t1), (p2, t2)}
                    prog_list = sorted(prog_set, key=lambda x: (-len(x[0]), x))
                    if len(prog_list) <= max_p:
                        if update(prog_list):
                            return {
                                "success": True,
                                "program": list(prog_list),
                                "score": 1.0
                            }
                        # Try all orderings for small programs
                        if len(prog_list) <= 7:
                            for ordering in permutations(prog_list):
                                if update(list(ordering)):
                                    return {
                                        "success": True,
                                        "program": list(ordering),
                                        "score": 1.0
                                    }
    
    # Final: try permutations of best_prog for local improvement
    if best_prog and len(best_prog) <= 8:
        for ordering in permutations(best_prog):
            if update(list(ordering)):
                return {
                    "success": True,
                    "program": list(ordering),
                    "score": 1.0
                }
    
    # Return best found (may not be perfect)
    prog_strs = [f"replace('{p}', '{t}')" for p, t in best_prog]
    if best_score < 1.0:
        prog_strs_info = ", ".join(prog_strs) if prog_strs else "none"
        return {
            "success": False,
            "program": best_prog,
            "score": best_score,
            "message": (f"Score={best_score:.3f}: best program ({best_score*len(inputs)}/{len(inputs)} correct). "
                       f"Programs: [{prog_strs_info}]. "
                       f"Time limit may have been reached.")
        }
    
    return {
        "success": True,
        "program": best_prog,
        "score": best_score
    }


if __name__ == "__main__":
    # Quick self-test
    examples = [
        ("abc", "adc"),
        ("bed", "bd"),
        ("cab", "cad"),
    ]
    result = solve_pbe(examples)
    print(f"Result: {result}")
    
    # Verify
    for inp, out in examples:
        result_str = inp
        for pred, trans in result["program"]:
            result_str = result_str.replace(pred, trans)
        print(f"  {inp} -> {result_str} (expected {out}) {'✓' if result_str == out else '✗'}")
