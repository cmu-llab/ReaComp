"""
Reward function for SLR-Bench tasks.

Each task presents positive (eastbound) and negative (westbound) train examples
with background knowledge as Prolog ground facts. The agent must induce a rule
of the form 'eastbound(T) :- Body.' that perfectly separates the two classes.

Evaluation delegates to the HuggingFace evaluate metric:
  AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning
which runs SWI-Prolog to verify the rule against the validation program.

Score = partial_score from the evaluate library ∈ [0.0, 1.0].
  - 1.0      : syntactically valid rule that correctly classifies all examples
  - (0.0, 1.0): partially correct (some examples classified correctly)
  - 0.0      : syntax error, evaluation error, or no rule found
"""

import re
import threading
from typing import Any, Dict, List, Optional, Tuple

_EXAMPLES_BLOCK_RE = re.compile(
    r"background knowledge.*?composition\.\s*\n(.*?)\nYour task",
    re.IGNORECASE | re.DOTALL,
)
# Matches a labelled train block: "eastbound(trainN).\n<facts>\n"
_TRAIN_BLOCK_RE = re.compile(
    r"(eastbound|westbound)\((train\d+)\)\.\s*\n((?:(?!eastbound|westbound).+\n?)*)",
    re.IGNORECASE,
)


def parse_prompt_examples(prompt: str) -> Dict[str, List[str]]:
    """
    Extract the train examples from an SLR-Bench prompt as a PBE-style dict.

    Returns
    -------
    {"inputs": [...], "outputs": [...]}
      inputs  : background facts for each train (whitespace-normalised, train-id-agnostic)
      outputs : "eastbound" or "westbound" label for each train, in prompt order
    """
    m = _EXAMPLES_BLOCK_RE.search(prompt)
    block = m.group(1) if m else prompt  # fall back to full prompt if delimiters shift

    inputs, outputs = [], []
    for direction, _train_id, facts_raw in _TRAIN_BLOCK_RE.findall(block):
        # Collapse each fact onto one line, strip trailing whitespace
        facts = " ".join(line.strip() for line in facts_raw.strip().splitlines() if line.strip())
        inputs.append(facts)
        outputs.append(direction.lower())

    return {"inputs": inputs, "outputs": outputs}


_PROLOG_RULE_RE = re.compile(
    r'eastbound\s*\([^)]*\)\s*:-\s*.+?\.',
    re.IGNORECASE | re.DOTALL,
)

_symbolic_judge = None
_judge_lock = threading.Lock()

def no_tqdm(iterable=None, *args, **kwargs):
    return iterable if iterable is not None else []

def _get_judge():
    global _symbolic_judge
    if _symbolic_judge is None:
        from evaluate import load
        _symbolic_judge = load("AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning")
        _symbolic_judge._compute.__globals__["tqdm"] = no_tqdm
    return _symbolic_judge


def _extract_rule(result: Any) -> Tuple[Optional[str], str]:
    if result is None:
        return None, "result is None"

    if isinstance(result, str):
        raw = result.strip()
        raw = re.sub(r'```[a-zA-Z]*\n?', '', raw).strip('` \n')
    else:
        raw = str(result)

    m = _PROLOG_RULE_RE.search(raw)
    if m:
        return m.group(0).strip(), ""

    # Fallback: find any line containing :-
    if ':-' in raw:
        for line in raw.splitlines():
            line = line.strip()
            if ':-' in line:
                return line if line.endswith('.') else line + '.', ""

    return None, f"No Prolog eastbound/1 rule found. Got: {raw[:120]!r}"


def rule_complexity(rule: str) -> int:
    """
    Number of property body literals, matching the dataset's 'rule complexity' label.
    Counts all top-level literals except has_car/2, which is treated as structural glue.
    E.g. has_car(T,C), car_len(C, short) → 1 (not 2).
    """
    if ':-' not in rule:
        return 0
    body = rule.split(':-', 1)[1].rstrip(' .')
    depth = 0
    top_commas = 0
    for ch in body:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            top_commas += 1
    total_literals = top_commas + 1
    has_car_count = len(re.findall(r'\bhas_car\s*\(', body))
    return total_literals - has_car_count


def reward(
    result: Any,
    execution_ok: bool,
    entry: Dict,
    **kwargs,
) -> Dict:
    """
    Parameters
    ----------
    result       : raw value returned by BCR (expected: Prolog rule string)
    execution_ok : False if the agent code raised a runtime exception
    entry        : full task record (must contain 'validation program')
    """
    if not execution_ok:
        return {
            "value": 0.0,
            "message": (
                "Solution raised a runtime error. "
                "Return a valid Prolog rule string, e.g. "
                "'eastbound(T) :- has_car(T, C), car_len(C, short).'"
            ),
        }

    rule, parse_error = _extract_rule(result)
    if rule is None:
        return {
            "value": 0.0,
            "message": (
                f"Could not extract a Prolog rule from result: {parse_error}. "
                "Your answer must be a Prolog rule of the form "
                "'eastbound(T) :- Body.' (ending with a period)."
            ),
        }

    validation_program = entry.get("validation program", "")
    if not validation_program:
        return {"value": 0.0, "message": "Entry missing 'validation program' field."}

    try:
        with _judge_lock:
            judge = _get_judge()
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
    except Exception as e:
        return {"value": 0.0, "message": f"Evaluation error: {e}"}

    detail = eval_results["detailed_results"][0]
    score = float(detail["partial_score"])
    syntax_ok = detail.get("syntax_valid", True)
    error = detail.get("error")

    if score >= 1.0:
        return {"value": 1.0}

    if not syntax_ok:
        msg = (
            f"Prolog syntax error in rule: {error}. "
            f"Rule submitted: {rule!r}. "
            "Ensure the rule ends with '.' and uses only defined predicates."
        )
    else:
        msg = (
            f"Score={score:.3f}: rule does not perfectly separate eastbound from westbound. "
            f"Rule submitted: {rule!r}."
        )
        if error:
            msg += f" Evaluation detail: {error}"

    return {"value": score, "message": msg}
