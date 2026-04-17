"""
Sandboxed code execution for json_agents.

Uses the Apptainer sandbox (same as openhands_agents) when a sif_path is
provided, otherwise falls back to in-process safe_exec (for local testing).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_executor(sif_path: str = "", timeout: int = 30):
    """Return an execute(code, lib_dir=None) callable."""
    if sif_path:
        from openhands_agents.sandbox import ApptainerSandbox
        sandbox = ApptainerSandbox(sif_path=sif_path, timeout=timeout)

        def execute(code: str, lib_dir: str = "") -> tuple[bool, str, str]:
            return sandbox.run_code(code, lib_dir=lib_dir or None)
    else:
        from symbolic_agent.executor import safe_exec

        def execute(code: str, lib_dir: str = "") -> tuple[bool, str, str]:
            import io, contextlib
            ok, ns, err = safe_exec(code)
            if not ok:
                return False, "", err
            return True, "", ""

    return execute
