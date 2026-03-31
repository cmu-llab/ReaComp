"""TroVE executor — runs a solution in a subprocess and captures stdout.

Mirrors TroVE's execution model: the toolbox functions, the **Tools** block,
and the **Solution** block are concatenated into a temporary .py file and
executed via subprocess with a configurable timeout.

The return value is (is_success: bool, output: str) where:
  is_success = subprocess returncode == 0
  output     = stdout stripped of whitespace
"""

import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds, matching TroVE's original


def run_solution(
    solution_code: str,
    tools_code: str,
    toolbox_code: str = "",
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple:
    """
    Execute toolbox_code + tools_code + solution_code in a subprocess.

    Parameters
    ----------
    solution_code : str
        The main program extracted from the **Solution** section.
    tools_code : str
        Imports and/or function definitions from the **Tools** section.
    toolbox_code : str
        Full source of all functions currently in the toolbox (execution
        namespace so imported functions are resolvable).
    timeout : int
        Subprocess wall-clock timeout in seconds.

    Returns
    -------
    (is_success, output) : (bool, str)
    """
    parts = []
    if toolbox_code and toolbox_code.strip():
        parts.append(toolbox_code.strip())
    if tools_code and tools_code.strip():
        parts.append(tools_code.strip())
    if solution_code and solution_code.strip():
        parts.append(solution_code.strip())

    if not parts:
        return False, ""

    full_code = "\n\n".join(parts)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_code)
            tmp_path = f.name

        proc = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        is_success = proc.returncode == 0
        output = proc.stdout.strip()
        if not is_success:
            logger.debug("Execution failed (returncode=%d): %s", proc.returncode, proc.stderr[:200])
        return is_success, output

    except subprocess.TimeoutExpired:
        logger.debug("Execution timed out after %ds", timeout)
        return False, ""
    except Exception as exc:
        logger.debug("Execution error: %s", exc)
        return False, ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
