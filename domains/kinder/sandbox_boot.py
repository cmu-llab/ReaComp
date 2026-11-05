"""Bootstrap for sandboxed Python (standalone: imports nothing from the repository).

    python -I -B sandbox_boot.py <workdir> <script.py>

Runs ``script.py`` with ``workdir`` first on ``sys.path`` and an import guard: an import
started by code in ``workdir`` may only load the standard library, the packages listed
in $KINDER_SANDBOX_ALLOWED (default numpy, scipy) or modules in ``workdir`` (such as
``kinder_client.py``).  Imports made internally by allowed libraries are not affected.
"""
import importlib
import os
import runpy
import sys

WORKDIR = os.path.realpath(sys.argv[1])
SCRIPT = os.path.realpath(sys.argv[2])
ALLOWED = {p for p in os.environ.get("KINDER_SANDBOX_ALLOWED", "numpy,scipy").split(",") if p}
STDLIB = set(sys.stdlib_module_names)
_IMPORTLIB_DIR = os.path.dirname(os.path.realpath(importlib.__file__))
_THIS = os.path.realpath(__file__)


def _initiator():
    """Filename of the innermost frame outside the import machinery and this file."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_code.co_filename
        if not (name.startswith("<frozen") or name.startswith(_IMPORTLIB_DIR)
                or (not name.startswith("<") and os.path.realpath(name) == _THIS)):
            return name
        frame = frame.f_back
    return None


def _in_workdir(name):
    if name is None or name.startswith("<"):
        return name in ("<string>", "<stdin>")
    return os.path.realpath(name).startswith(WORKDIR + os.sep)


class _Guard:
    def find_spec(self, fullname, path=None, target=None):
        head = fullname.partition(".")[0]
        if head in STDLIB or head in ALLOWED:
            return None
        if os.path.isfile(os.path.join(WORKDIR, head + ".py")) or \
                os.path.isfile(os.path.join(WORKDIR, head, "__init__.py")):
            return None
        if _in_workdir(_initiator()):
            raise ModuleNotFoundError(
                f"import of {fullname!r} is not allowed here: only the Python standard library, "
                f"{', '.join(sorted(ALLOWED))} and files in the working directory are importable")
        return None


sys.meta_path.insert(0, _Guard())
sys.path.insert(0, WORKDIR)
sys.argv = [SCRIPT]
try:
    runpy.run_path(SCRIPT, run_name="__main__")
except SystemExit:
    raise
except BaseException:  # noqa: BLE001 - print the user's traceback without bootstrap frames
    import traceback
    etype, exc, tb = sys.exc_info()
    while tb is not None and not _in_workdir(tb.tb_frame.f_code.co_filename):
        tb = tb.tb_next
    traceback.print_exception(etype, exc, tb)
    sys.exit(1)
