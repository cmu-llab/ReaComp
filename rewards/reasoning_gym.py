"""
Reward function for all reasoning_gym datasets.

Uses reasoning_gym's built-in per-dataset scoring via get_score_answer_fn(),
which dispatches to the correct score_answer() implementation for each of the
104 registered task types.

score_answer(answer: str, entry: dict) -> float always returns a value in
[0.0, 1.0], so no clipping or rescaling is applied — the raw score is returned.

Important: the base-class score_answer awards partial credit when the oracle
answer appears as a substring of a longer response:
    reward = len(oracle) / len(response)   if oracle in response
This means a solution returning "The answer is 30" instead of "30" will score
~0.12 instead of 1.0.  BCR is instructed to return minimal, clean answer
strings to avoid this penalty.
"""

from typing import Any, Dict


def reward(result: Any, execution_ok: bool, entry: Dict) -> Dict:
    """
    Parameters
    ----------
    result       : raw Python value from execute_with_library(), or None on failure
    execution_ok : False if the solution code raised a runtime exception
    entry        : full task record dict (question, answer, metadata, ...)
    """
    if not execution_ok:
        return {
            "value": 0.0,
            "message": "Solution raised a runtime error during execution.",
        }

    source_dataset = entry.get("metadata", {}).get("source_dataset", "")
    if not source_dataset:
        return {
            "value": 0.0,
            "message": "Cannot determine source_dataset from entry metadata.",
        }

    try:
        from reasoning_gym import get_score_answer_fn
        score_fn = get_score_answer_fn(source_dataset)
    except Exception as exc:
        return {
            "value": 0.0,
            "message": f"Could not load reasoning_gym scorer for '{source_dataset}': {exc}",
        }

    # Convert raw Python execution result to a string for score_answer().
    # List-of-strings (e.g. tower_of_hanoi move sequences) are joined with
    # newlines since scorers for those tasks call splitlines() on the response.
    # All other types use str() — floats, ints, grids rendered as strings, etc.
    if isinstance(result, list) and all(isinstance(x, str) for x in result):
        model_response = "\n".join(result)
    else:
        model_response = str(result).strip()

    try:
        score = float(score_fn(model_response, entry))
    except Exception as exc:
        return {
            "value": 0.0,
            "message": (
                f"Scoring error for '{source_dataset}': {exc}. "
                f"Response was: {model_response[:120]}"
            ),
        }

    if score >= 1.0:
        return {"value": score}

    return {
        "value": score,
        "message": (
            f"Score={score:.3f} for {source_dataset}. "
            f"Response: '{model_response[:120]}'"
        ),
    }
