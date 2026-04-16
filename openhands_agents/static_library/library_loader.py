"""
StaticLibrary — read-only pre-built library loader.

Reads a library directory produced by a strong coding agent:
  library_path/
    LIBRARY.py        # all helper functions in one file
    PROMPTING_GUIDE.md  # workflow instructions for the weaker agent
    ANALYSIS.md       # optional context (not used at runtime)

The LIBRARY.py is served to the Apptainer sandbox as a Python package so the
agent can do `from library import fn_name`. No functions can be added or removed
at runtime — the library is fixed for the entire run.

The PROMPTING_GUIDE.md is read and embedded in the agent's system prompt so the
weaker model follows the stronger model's prescribed workflow.
"""

import os
import shutil


class StaticLibrary:
    """
    Wraps a pre-built LIBRARY.py directory for use with the Apptainer sandbox.

    Parameters
    ----------
    library_path : str
        Path to the directory containing LIBRARY.py and PROMPTING_GUIDE.md.
    pkg_dir : str
        Directory where the library package will be staged for bind-mounting.
        Created automatically. Will be cleared and re-populated on init.
    """

    def __init__(self, library_path: str, pkg_dir: str):
        self.library_path = os.path.abspath(library_path)
        self.pkg_dir = os.path.abspath(pkg_dir)

        lib_file = os.path.join(self.library_path, "LIBRARY.py")
        if not os.path.exists(lib_file):
            raise FileNotFoundError(f"LIBRARY.py not found in {self.library_path}")

        self._setup_pkg(lib_file)
        self._guide = self._load_guide()
        self._function_names = self._parse_function_names(lib_file)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def prompting_guide(self) -> str:
        """The PROMPTING_GUIDE.md content, or empty string if not present."""
        return self._guide

    @property
    def function_names(self) -> list[str]:
        """Top-level function names defined in LIBRARY.py."""
        return list(self._function_names)

    def as_listing(self) -> str:
        """One-line listing of available functions for the task prompt."""
        return "\n".join(f"  {name}" for name in self._function_names)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _setup_pkg(self, lib_file: str) -> None:
        """Stage LIBRARY.py as a flat Python package at pkg_dir/library/."""
        pkg = os.path.join(self.pkg_dir, "library")
        if os.path.isdir(pkg):
            shutil.rmtree(pkg)
        os.makedirs(pkg)

        # Copy LIBRARY.py → library/__init__.py so `from library import fn` works
        # directly without needing a submodule.
        shutil.copy(lib_file, os.path.join(pkg, "__init__.py"))

    def _load_guide(self) -> str:
        path = os.path.join(self.library_path, "PROMPTING_GUIDE.md")
        if os.path.exists(path):
            return open(path).read().strip()
        return ""

    @staticmethod
    def _parse_function_names(lib_file: str) -> list[str]:
        """Extract top-level def names from LIBRARY.py (no AST dependency needed)."""
        names = []
        with open(lib_file) as f:
            for line in f:
                if line.startswith("def "):
                    name = line[4:].split("(")[0].strip()
                    if not name.startswith("_"):
                        names.append(name)
        return names
