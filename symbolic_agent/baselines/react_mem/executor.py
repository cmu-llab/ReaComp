"""
Safe code execution for the ReAct+Memory baseline.

Runs a code snippet in a restricted namespace with a timeout.
Returns (ok, result, error_message).
"""

import sys
import io
import traceback
from typing import Any, Tuple


def run_code(code: str, timeout: int = 10) -> Tuple[bool, Any, str]:
    """
    Execute `code` in a fresh namespace.

    The snippet may define any functions and variables.  If it calls
    `print()`, stdout is captured and returned as the result.  If the last
    statement is an expression, its value is the result.  On error, returns
    (False, None, traceback_string).
    """
    namespace: dict = {}
    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        exec(compile(code, "<react_mem>", "exec"), namespace)
        printed = buf.getvalue().strip()
        result = printed if printed else namespace.get("_result", None)
        return True, result, ""
    except Exception:
        tb = traceback.format_exc()
        return False, None, tb
    finally:
        sys.stdout = old_stdout
