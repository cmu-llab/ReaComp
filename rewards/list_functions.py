"""Exact verifier for the List Functions pilot domain.

New-files-only: mirrors the rewards/ registry contract used by PBEBench/SLR
(``reward(result, execution_ok, entry, **kwargs) -> {"value": float, "message": str}``,
with ``value == 1.0`` iff every I/O pair is reproduced exactly).

List Functions tasks map an integer list to an integer list. Unlike PBEBench there is
no compact fixed DSL: the reference tasks are defined in a lambda-calculus-with-types
language, so we let the induced program be arbitrary Python over lists (a general
``def program(xs) -> list`` transform), scored by exact list equality. This mirrors the
arbitrary-Python-with-a-primitive-library setting used by recent program-induction work
on non-DSL-complete domains.

The candidate ``result`` may be:
  * a callable ``f(xs) -> list``
  * a source string defining ``program`` / ``solve`` / ``f`` (a function of one list arg)
  * a dict ``{"program": <source or callable>}`` (also accepts key "function"/"f")
  * a list of predicted output lists, one per entry input (direct answers)

Execution of a source program happens in a restricted namespace with no builtins that
touch the filesystem/network. The eval harness already runs each task in a child
process with a timeout, so this verifier does not add its own timeout.
"""
from typing import Any, Dict, List, Optional, Callable
import ast

# Builtins the induced list program is allowed to use. Deliberately small: pure,
# side-effect-free primitives sufficient to express list transforms.
_SAFE_BUILTINS = {
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
    "sum": sum, "min": min, "max": max, "abs": abs, "all": all, "any": any,
    "list": list, "tuple": tuple, "set": set, "dict": dict, "int": int,
    "bool": bool, "str": str, "round": round, "divmod": divmod, "pow": pow,
}


def _coerce_list(x: Any) -> Optional[List[int]]:
    """Best-effort coerce a program output into a list of ints; None if impossible."""
    if x is None:
        return None
    if isinstance(x, tuple):
        x = list(x)
    if not isinstance(x, list):
        return None
    out = []
    for v in x:
        if isinstance(v, bool):  # bool is an int subclass; reject to avoid True==1 traps
            return None
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, float) and v.is_integer():
            out.append(int(v))
        else:
            return None
    return out


def _compile_program(src: str) -> Optional[Callable]:
    """Exec a source string in a restricted namespace and return the transform fn."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    ns: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(compile(tree, "<list_fn_program>", "exec"), ns)  # noqa: S102 (sandboxed ns)
    except Exception:
        return None
    for name in ("program", "solve", "f", "transform", "apply"):
        fn = ns.get(name)
        if callable(fn):
            return fn
    return None


def _extract_callable(result: Any) -> Optional[Callable]:
    if callable(result):
        return result
    if isinstance(result, dict):
        for key in ("program", "function", "f", "solve", "transform"):
            v = result.get(key)
            if callable(v):
                return v
            if isinstance(v, str):
                fn = _compile_program(v)
                if fn is not None:
                    return fn
    if isinstance(result, str):
        return _compile_program(result)
    return None


def score_program(fn: Callable, inputs: List[List[int]],
                  outputs: List[List[int]]) -> (float, List[str]):
    """Fraction of inputs the program maps to the expected output (exact equality)."""
    correct = 0
    mismatches: List[str] = []
    for inp, expected in zip(inputs, outputs):
        try:
            got = _coerce_list(fn(list(inp)))
        except Exception as e:  # program crashed on this input
            got = None
            if len(mismatches) < 3:
                mismatches.append(f"{inp} -> ERROR {type(e).__name__}: {e}")
            continue
        if got is not None and got == list(expected):
            correct += 1
        elif len(mismatches) < 3:
            mismatches.append(f"{inp} -> {got} (expected {expected})")
    return correct / len(inputs), mismatches


def reward(result: Any, execution_ok: bool, entry: Dict, **kwargs) -> Dict:
    if not execution_ok:
        return {"value": 0.0,
                "message": "Solution raised a runtime error during execution."}

    inputs = entry.get("inputs", [])
    outputs = entry.get("outputs", [])
    if not inputs or not outputs or len(inputs) != len(outputs):
        return {"value": 0.0,
                "message": "Entry is missing or has mismatched 'inputs'/'outputs'."}

    # Direct-answer form: result is a list of predicted output lists.
    if isinstance(result, list) and result and all(isinstance(r, (list, tuple)) for r in result):
        preds = [_coerce_list(r) for r in result]
        if len(preds) == len(outputs):
            correct = sum(1 for p, o in zip(preds, outputs) if p == list(o))
            score = correct / len(outputs)
            if score >= 1.0:
                return {"value": 1.0}
            return {"value": score,
                    "message": f"Score={score:.3f}: {correct}/{len(outputs)} correct (direct answers)."}

    fn = _extract_callable(result)
    if fn is None:
        return {"value": 0.0,
                "message": ("Could not obtain a list->list program from result. Return a "
                            "callable, a source string defining program(xs), or a list of "
                            "predicted output lists.")}

    score, mismatches = score_program(fn, inputs, outputs)
    if score >= 1.0:
        return {"value": 1.0}
    return {"value": score,
            "message": f"Score={score:.3f}: {int(score*len(inputs))}/{len(inputs)} "
                       f"inputs mapped correctly. Mismatches: {mismatches}"}
