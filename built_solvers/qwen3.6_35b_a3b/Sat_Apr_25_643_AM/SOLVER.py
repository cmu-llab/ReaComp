"""
Solver for Symbolic Logic Rule (SLR-Bench) tasks.

Infers a Prolog rule of the form `eastbound(T) :- Body.` from training examples
consisting of (facts_string, direction_label) pairs.
"""

import re
import sys
from itertools import combinations

# Import the verifier and scoring utilities
sys.path.insert(0, '/workspace')
from rewards.slr_bench import _get_judge, rule_complexity, parse_prompt_examples


def build_validation_program(examples):
    """Convert (facts_string, direction_label) pairs to a Prolog validation program.

    The validation program contains all ground facts from the examples along with
    eastbound/train declarations.
    """
    blocks = []
    for facts_str, label in examples:
        # Split facts by '.' to get individual atoms
        facts = [f.strip() + '.' for f in facts_str.split('.') if f.strip()]
        train_id = None
        for fact in facts:
            m = re.match(r'has_car\((\w+)', fact)
            if m:
                train_id = m.group(1)
                break
        if train_id:
            block = [f"{label}({train_id})."] + facts
        else:
            block = facts
        blocks.append('\n'.join(block))
    return '\n\n'.join(blocks) + '\n\n'


def build_feature_space(examples):
    """Extract all predicates and their value domains from the examples.

    Returns a dict: {predicate_name: {arg_index: set_of_values}, ...}
    Args are 0-indexed.
    """
    all_preds = {}
    for facts_str, label in examples:
        facts = re.findall(r'(\w+)\(([^)]+)\)', facts_str)
        for pred, args_str in facts:
            args = [a.strip() for a in args_str.split(',')]
            if pred not in all_preds:
                all_preds[pred] = {}
            for i, arg in enumerate(args):
                if i not in all_preds[pred]:
                    all_preds[pred][i] = set()
                all_preds[pred][i].add(arg)
    return all_preds


def generate_body_literals(preds):
    """Generate candidate body literals from the feature space.

    For predicates where arg0 is a car reference and arg1 is a property value,
    generate car-level literals: pred(C, value).

    For predicates where arg0 is a train reference and arg1 is a property value,
    generate train-level literals: pred(T, value).
    """
    literals = []
    for pred, arg_vals in preds.items():
        if pred == 'has_car':
            continue  # Structural predicate, always present

        arg0_vals = arg_vals.get(0, set())
        arg1_vals = arg_vals.get(1, set())

        if not arg0_vals or not arg1_vals:
            continue

        # Check if arg0 contains car references
        is_car_predicate = any(
            'car' in str(v) and str(v)[0].islower() and not str(v).startswith('train')
            for v in arg0_vals
        )
        # Check if arg0 contains train references
        is_train_predicate = any(
            'train' in str(v) for v in arg0_vals
        )
        # Check if arg1 contains property values (not entity IDs)
        arg1_has_props = not any('car' in str(v) for v in arg1_vals)

        if is_car_predicate and arg1_has_props:
            for val in arg1_vals:
                literals.append(f"{pred}(C, {val})")

        if is_train_predicate and arg1_has_props:
            for val in arg1_vals:
                literals.append(f"{pred}(T, {val})")

    return sorted(set(literals))


def evaluate_rule(judge, rule, validation_program):
    """Evaluate a single rule against the validation program.

    Returns a dict with score, syntax_valid, and error.
    """
    try:
        eval_results = judge.compute(
            predictions=[rule],
            references=[{
                "validation_program": validation_program,
                "evaluation_config": {
                    "positive_predicate": "eastbound",
                    "negative_predicate": "westbound",
                },
            }],
        )
        detail = eval_results["detailed_results"][0]
        return {
            "score": float(detail["partial_score"]),
            "syntax_valid": detail.get("syntax_valid", True),
            "error": detail.get("error"),
        }
    except Exception as e:
        return {
            "score": 0.0,
            "syntax_valid": False,
            "error": str(e),
        }


def solve_slr(examples, top_k=5):
    """Solve an SLR-Bench task by inferring the best Prolog rule.

    Parameters
    ----------
    examples : list of (facts_string, direction_label)
        Each facts_string is a space-separated Prolog ground facts string.
        direction_label is "eastbound" or "westbound".
    top_k : int
        Number of top rules to return if no perfect rule is found.

    Returns
    -------
    dict with keys:
        - success (bool): True if a perfect rule (score=1.0) was found.
        - program (str): The best Prolog rule string.
        - top_k_programs (list[str]): Top-K candidate rules.
        - score (float): The score of the best rule.
    """
    # Build validation program from examples
    validation_program = build_validation_program(examples)

    # Build feature space from examples
    preds = build_feature_space(examples)

    # Generate candidate body literals
    literals = generate_body_literals(preds)

    # Initialize the judge
    judge = _get_judge()

    # Evaluate candidates layer by layer (1-literal, 2-literal, ...)
    # Stop after the first layer that contains a perfect rule
    scored_rules = []
    all_rules_seen = set()
    found_perfect = False

    for n in range(1, min(len(literals) + 1, 5)):  # up to 4 body literals
        current_layer = []
        for combo in combinations(literals, n):
            body = ', '.join(sorted(combo))
            rule = f"eastbound(T) :- has_car(T, C), {body}."

            if rule in all_rules_seen:
                continue
            all_rules_seen.add(rule)

            result = evaluate_rule(judge, rule, validation_program)
            complexity = rule_complexity(rule)
            current_layer.append({
                "rule": rule,
                "score": result["score"],
                "complexity": complexity,
                "syntax_valid": result["syntax_valid"],
                "error": result["error"],
            })

            if result["score"] >= 1.0:
                found_perfect = True

        scored_rules.extend(current_layer)

        # Early exit: stop after first layer with a perfect rule
        if found_perfect:
            break

    # Sort by score (descending) then complexity (ascending)
    scored_rules.sort(key=lambda x: (-x["score"], x["complexity"]))

    # Select the best rule
    best = scored_rules[0] if scored_rules else None

    if best is None:
        return {
            "success": False,
            "program": "",
            "top_k_programs": [],
            "score": 0.0,
        }

    top_k_programs = [r["rule"] for r in scored_rules[:top_k]]

    return {
        "success": best["score"] >= 1.0,
        "program": best["rule"],
        "top_k_programs": top_k_programs,
        "score": best["score"],
    }


if __name__ == "__main__":
    # Quick test using a demo
    import json

    with open("/workspace/DEMOS.json", "r") as f:
        demos = json.load(f)

    demo = demos[0]
    result = parse_prompt_examples(demo["prompt"])
    examples = list(zip(result["inputs"], result["outputs"]))

    output = solve_slr(examples, top_k=3)
    print(f"Success: {output['success']}")
    print(f"Score: {output['score']}")
    print(f"Rule: {output['program']}")
    print(f"Top-K: {output['top_k_programs']}")
