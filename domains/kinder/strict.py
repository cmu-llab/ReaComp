"""Import allowlist for KinDER solvers and sandboxed agent code.

A solver may use the Python standard library, NumPy, SciPy and sibling files next to
it, which is the strict black-box setting of AgenticGenTAMP (robocode
``utils/strict_blackbox.py``).  Simulator, robotics and geometry packages (``kinder``,
``relational_structs``, ``gymnasium``, ``shapely``, ...) are unavailable, as are the
dynamic-import mechanisms that would sidestep the check.

``check_solver`` is a static check run before a solver is evaluated.  Sandboxed code
is additionally guarded at run time by ``sandbox_boot.py``.  Both are methodological
guardrails, not containment of hostile code.
"""
from __future__ import annotations

import ast
import os
import sys
from typing import List, Optional, Set

ALLOWED_PACKAGES = frozenset({"numpy", "scipy"})
#: standard-library modules a solver may not import (dynamic import mechanisms)
FORBIDDEN_STDLIB = frozenset({"importlib", "runpy"})
#: modules that exist only in development sandboxes, never at evaluation
SANDBOX_ONLY = frozenset({"kinder_client"})


class StrictImportError(ImportError):
    """A solver imports something outside the allowlist."""


def allowlist_description() -> str:
    return ("the Python standard library (except importlib/runpy), numpy, scipy, and other .py files "
            "in the same directory as the solver")


def _imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield node, node.module or "", node.level


def _sibling_files(root: str, head: str) -> List[str]:
    """The files of a sibling module ``head.py`` or of every module in a sibling package ``head/``."""
    single = os.path.join(root, head + ".py")
    if os.path.isfile(single):
        return [single]
    pkg = os.path.join(root, head)
    if os.path.isfile(os.path.join(pkg, "__init__.py")):
        return sorted(os.path.join(d, f) for d, _, fs in os.walk(pkg) for f in fs if f.endswith(".py"))
    return []


def check_solver(entry: str) -> List[str]:
    """Return the allowlist violations of ``entry`` and the sibling modules it reaches."""
    entry = os.path.abspath(entry)
    root = os.path.dirname(entry)
    stdlib = set(sys.stdlib_module_names)
    violations: List[str] = []
    seen: Set[str] = set()
    todo = [entry]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        rel = os.path.relpath(path, root)
        try:
            with open(path) as fh:
                tree = ast.parse(fh.read(), filename=path)
        except SyntaxError as exc:
            violations.append(f"{rel}: syntax error: {exc}")
            continue
        sys_names = {"sys"} | {a.asname for n in ast.walk(tree) if isinstance(n, ast.Import)
                               for a in n.names if a.name == "sys" and a.asname}
        for node, name, level in _imports(tree):
            where = f"{rel}:{node.lineno}"
            if level:  # relative import: only within a sibling package, whose files are all scanned
                if os.path.dirname(path) == root:
                    violations.append(f"{where}: relative import {'.' * level}{name} outside a package")
                continue
            head = name.split(".")[0]
            sibs = _sibling_files(root, head)
            if head in SANDBOX_ONLY:
                violations.append(f"{where}: imports {head} (a development-sandbox helper; solvers receive "
                                  "states and spaces directly and must not import it)")
            elif sibs:
                todo.extend(sibs)
            elif head in FORBIDDEN_STDLIB:
                violations.append(f"{where}: imports {name} (dynamic imports are not allowed)")
            elif head in ALLOWED_PACKAGES or head in stdlib:
                continue
            else:
                violations.append(f"{where}: imports {name}")
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name) and node.id == "__import__") or \
                    (isinstance(node, ast.Attribute) and node.attr == "__import__"):
                violations.append(f"{rel}:{node.lineno}: uses __import__")
            elif (isinstance(node, ast.Attribute) and node.attr == "modules"
                  and isinstance(node.value, ast.Name) and node.value.id in sys_names):
                violations.append(f"{rel}:{node.lineno}: accesses sys.modules")
            elif (isinstance(node, ast.ImportFrom) and node.module == "sys"
                  and any(a.name == "modules" for a in node.names)):
                violations.append(f"{rel}:{node.lineno}: imports sys.modules")
    return violations


def assert_solver_allowed(entry: str) -> None:
    violations = check_solver(entry)
    if violations:
        raise StrictImportError(
            f"{os.path.basename(entry)} may only import {allowlist_description()}:\n  " + "\n  ".join(violations))
