"""
Safe code execution for the ReAct+Library baseline.

`run_code_with_library` executes a snippet in a namespace that already
contains all library functions — the agent calls them by name directly.
"""

import io
import sys
import traceback
from typing import Any, Dict, Optional, Tuple


def run_code_with_library(
    code: str,
    library_namespace: Optional[Dict] = None,
    timeout: int = 10,
) -> Tuple[bool, Any, str]:
    """
    Execute ``code`` in a namespace seeded with library functions.

    stdout is captured; if the snippet prints anything that is returned as
    the result string.  Returns (ok, result, error_message).
    """
    import builtins
    ns: dict = dict(library_namespace) if library_namespace else {"__builtins__": builtins}
    # Ensure builtins are always present
    if "__builtins__" not in ns:
        ns["__builtins__"] = builtins

    old_stdout = sys.stdout
    sys.stdout = buf = io.StringIO()
    try:
        exec(compile(code, "<react_library>", "exec"), ns)  # noqa: S102
        printed = buf.getvalue().strip()
        result = printed if printed else ns.get("_result", None)
        return True, result, ""
    except Exception:
        tb = traceback.format_exc()
        return False, None, tb
    finally:
        sys.stdout = old_stdout
