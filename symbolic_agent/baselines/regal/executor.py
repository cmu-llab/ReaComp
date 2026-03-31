"""ReGAL executor — subprocess execution and verification.

Verification: pˆ() == p() (paper §3, Stage 2a).
We compare stdout of the original primitive program to stdout of the
refactored program (codebank helpers + refactored main body).  This is
equivalent to the `old_output == new_output` criterion discussed during
design review.

get_func_names() extracts all called function names from a program, used
to assign blame/credit when updating was_success records.
"""

import ast
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds


def run_program(
    code: str,
    extra_code: str = "",
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple:
    """
    Execute extra_code + code in a subprocess.

    Parameters
    ----------
    code : str
        The main program (solution / refactored call code).
    extra_code : str
        Helper function definitions (codebank source) prepended to code.
    timeout : int

    Returns
    -------
    (is_success, stdout, stderr) : (bool, str, str)
    """
    parts = []
    if extra_code and extra_code.strip():
        parts.append(extra_code.strip())
    if code and code.strip():
        parts.append(code.strip())

    if not parts:
        return False, "", "empty program"

    full = "\n\n".join(parts)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full)
            tmp_path = f.name

        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()

    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as exc:
        return False, "", str(exc)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def verify(
    refactored_code: str,
    original_output: str,
    codebank_code: str = "",
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple:
    """
    Verify that refactored_code produces the same stdout as original_output.

    Returns
    -------
    (passes, actual_output) : (bool, str)
    """
    ok, out, err = run_program(refactored_code, extra_code=codebank_code, timeout=timeout)
    if not ok:
        logger.debug("Verification execution failed: %s", err[:200])
        return False, out
    passes = out == original_output
    if not passes:
        logger.debug(
            "Verification mismatch:\n  expected: %r\n  got:      %r",
            original_output[:100], out[:100],
        )
    return passes, out


def get_func_names(code: str) -> list:
    """
    Return list of function names called in code (used for blame attribution).
    Mirrors get_func_names from original domains/python/utils.py.
    """
    if not code or not code.strip():
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
    return list(set(names))


def split_helpers_and_calls(code: str) -> tuple:
    """
    Split a code block into (helper_function_defs, call_code).

    Helper functions are top-level `def` statements; everything else
    (assignments, calls, imports) goes into call_code.

    Returns
    -------
    (helpers_code, calls_code) : (str, str)
    """
    if not code or not code.strip():
        return "", ""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "", code

    helpers, calls = [], []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            helpers.append(ast.unparse(node))
        else:
            calls.append(ast.unparse(node))

    return "\n\n".join(helpers), "\n".join(calls)
