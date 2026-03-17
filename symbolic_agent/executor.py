"""Safe Python code execution for verifying library functions and solutions."""

import ast
import traceback
from typing import Any, Dict, List, Optional, Tuple

from .models import Function

# Modules that must not be imported inside executed code
_FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "importlib", "socket",
    "pathlib", "tempfile", "signal", "ctypes", "multiprocessing",
}


def _check_safety(code: str) -> Tuple[bool, str]:
    """Return (safe, reason).  Rejects obvious dangerous patterns."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name.split(".")[0] for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module.split(".")[0] if node.module else ""]
            )
            for name in names:
                if name in _FORBIDDEN_MODULES:
                    return False, f"Forbidden import: {name}"

    return True, ""


def safe_exec(code: str, namespace: Optional[Dict] = None) -> Tuple[bool, Optional[Dict], str]:
    """
    Execute code in a namespace.
    Returns (success, namespace_after, error_message).
    """
    if namespace is None:
        namespace = {"__builtins__": __builtins__}

    safe, reason = _check_safety(code)
    if not safe:
        return False, None, reason

    try:
        exec(code, namespace)  # noqa: S102
        return True, namespace, ""
    except Exception:
        return False, None, traceback.format_exc()


def execute_with_library(
    solution_code: str,
    function_name: str,
    args: List[Any],
    library_functions: Optional[List[Function]] = None,
) -> Tuple[bool, Any, str]:
    """
    Load library functions into a namespace, then execute solution_code
    and call function_name(*args).

    Returns (success, result, error_message).
    """
    namespace: Dict = {"__builtins__": __builtins__}

    # Load library dependencies first
    for lib_func in (library_functions or []):
        ok, namespace, err = safe_exec(lib_func.code, namespace)
        if not ok:
            return False, None, f"Failed to load library function '{lib_func.name}': {err}"

    ok, namespace, err = safe_exec(solution_code, namespace)
    if not ok:
        return False, None, err

    if function_name not in namespace:
        return False, None, f"Function '{function_name}' not defined in solution code."

    try:
        result = namespace[function_name](*args)
        return True, result, ""
    except Exception:
        return False, None, traceback.format_exc()
