"""
Apptainer sandbox for safe code execution.

Each run_code() call:
  1. Writes code to a fresh tmpdir (/tmp/oh_exec_XXX/code.py)
  2. Launches: apptainer exec --contain --no-home --writable-tmpfs
               --bind tmpdir:/exec:rw [--bind lib_dir:/exec/<pkg>:ro]
               sandbox.sif python /exec/code.py
  3. Returns (ok, stdout, stderr) and cleans up the tmpdir.

Generated code can import from the library/toolbox package via:
    from library import fn_name   # react_library
    from toolbox import fn_name   # trove
because the package dir is bound at /exec/<basename(lib_dir)> and
we prepend sys.path.insert(0, '/exec') to every code run.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional


_SYS_PATH_PREAMBLE = "import sys; sys.path.insert(0, '/exec')\n"


class ApptainerSandbox:
    def __init__(self, sif_path: str, timeout: int = 30):
        self.sif_path = sif_path
        self.timeout = timeout

    def run_code(
        self,
        code: str,
        lib_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> tuple[bool, str, str]:
        """
        Execute code in the Apptainer sandbox.

        Parameters
        ----------
        code : str
            Python source to execute.
        lib_dir : str, optional
            Host path to a Python package directory (must contain __init__.py).
            Bound into the container as /exec/<basename(lib_dir)>:ro so the
            code can do `from <basename> import fn`.
        timeout : int, optional
            Per-call timeout override (seconds). Falls back to self.timeout.

        Returns
        -------
        (ok, stdout, stderr)
        """
        t = timeout or self.timeout
        exec_dir = tempfile.mkdtemp(prefix="oh_exec_")
        try:
            with open(os.path.join(exec_dir, "code.py"), "w") as f:
                f.write(_SYS_PATH_PREAMBLE + code)

            cmd = [
                "apptainer", "exec",
                "--contain", "--no-home", "--writable-tmpfs",
                "--bind", f"{exec_dir}:/exec:rw",
            ]
            if lib_dir:
                pkg_name = os.path.basename(lib_dir.rstrip("/"))
                cmd += ["--bind", f"{lib_dir}:/exec/{pkg_name}:ro"]

            cmd += [self.sif_path, "python", "/exec/code.py"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=t,
            )
            return result.returncode == 0, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", f"Timeout after {t}s"
        except Exception as exc:
            return False, "", str(exc)
        finally:
            shutil.rmtree(exec_dir, ignore_errors=True)
