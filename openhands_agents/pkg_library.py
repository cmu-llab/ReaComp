"""
Shared base for library/toolbox package directories.

Each function lives in its own .py file inside a package directory:

    pkg_dir/
        __init__.py          # auto-generated: `from .fn_name import fn_name`
        fn_name.py           # contains exactly one top-level def fn_name(...)

The directory is a valid Python package that can be bind-mounted into the
Apptainer sandbox. Generated code does `from library import fn` or
`from toolbox import fn` depending on the package name (= os.path.basename).

Constraint: function files MUST be standalone — they may import stdlib or
installed packages (numpy, sympy, ...) but must NOT import from the same
package (no `from library import other_fn`). This is required so that
trimming a function never silently breaks other functions.

BM25 retrieval is used to surface the top-K relevant functions for a given
task description. For TroVE the toolbox is small enough that the full listing
is shown in the prompt instead.
"""

import json
import math
import os
import re
from typing import Optional


class PkgLibrary:
    """
    A Python package directory where each function is a separate .py file.

    Parameters
    ----------
    pkg_dir : str
        Absolute path to the package directory (its basename is the import name).
    """

    def __init__(self, pkg_dir: str):
        self.pkg_dir = os.path.abspath(pkg_dir)
        os.makedirs(self.pkg_dir, exist_ok=True)
        # name → {description, usage_count, code}
        self._meta: dict[str, dict] = {}
        self._load_meta()

    # ------------------------------------------------------------------
    # Package management
    # ------------------------------------------------------------------

    def add(self, name: str, description: str, code: str) -> None:
        """Write fn_name.py and update __init__.py. Overwrites if exists."""
        path = os.path.join(self.pkg_dir, f"{name}.py")
        with open(path, "w") as f:
            f.write(code.strip() + "\n")
        self._meta[name] = {
            "description": description,
            "usage_count": self._meta.get(name, {}).get("usage_count", 0),
            "code": code,
        }
        self._update_init()
        self._save_meta()

    def remove(self, name: str) -> None:
        """Delete fn_name.py and update __init__.py."""
        path = os.path.join(self.pkg_dir, f"{name}.py")
        if os.path.exists(path):
            os.remove(path)
        self._meta.pop(name, None)
        self._update_init()
        self._save_meta()

    def increment_usage(self, names: list[str]) -> None:
        """Increment usage count for each function name in the list."""
        for name in names:
            if name in self._meta:
                self._meta[name]["usage_count"] += 1
        self._save_meta()

    def trim(self, n_processed: int, c: float = 0.5) -> list[str]:
        """
        Remove functions used fewer than λ = c × log10(n_processed) times.
        Returns list of removed function names.
        """
        if n_processed < 2:
            return []
        threshold = c * math.log10(n_processed)
        to_remove = [
            name for name, m in self._meta.items()
            if m["usage_count"] < threshold
        ]
        for name in to_remove:
            self.remove(name)
        return to_remove

    def __len__(self) -> int:
        return len(self._meta)

    def __contains__(self, name: str) -> bool:
        return name in self._meta

    def names(self) -> list[str]:
        return list(self._meta.keys())

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def as_listing(self) -> str:
        """Full listing: name + description for all functions (for prompt)."""
        if not self._meta:
            return "(empty)"
        lines = []
        for name, m in self._meta.items():
            lines.append(f"  {name}: {m['description']}")
        return "\n".join(lines)

    def retrieve(self, query: str, k: int) -> list[dict]:
        """
        BM25-style retrieval: return top-k functions by name+description
        token overlap with the query.
        """
        if not self._meta:
            return []

        def tokenize(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+", text.lower())

        q_tokens = set(tokenize(query))
        scored = []
        for name, m in self._meta.items():
            doc_tokens = set(tokenize(name + " " + m["description"]))
            if not doc_tokens:
                continue
            overlap = len(q_tokens & doc_tokens)
            jaccard = overlap / len(q_tokens | doc_tokens) if q_tokens | doc_tokens else 0.0
            scored.append((jaccard, name, m))

        scored.sort(key=lambda x: -x[0])
        return [
            {"name": n, "description": m["description"], "code": m["code"]}
            for _, n, m in scored[:k]
        ]

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"meta": dict(self._meta)}

    @classmethod
    def from_dict(cls, pkg_dir: str, data: dict) -> "PkgLibrary":
        lib = cls(pkg_dir)
        for name, m in data.get("meta", {}).items():
            lib.add(name, m["description"], m["code"])
            lib._meta[name]["usage_count"] = m.get("usage_count", 0)
        lib._save_meta()
        return lib

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all functions and reset metadata. Wipes the pkg_dir contents."""
        for fname in os.listdir(self.pkg_dir):
            if fname.endswith(".py") or fname == "_meta.json":
                os.remove(os.path.join(self.pkg_dir, fname))
        self._meta = {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_init(self) -> None:
        names = sorted(
            f[:-3] for f in os.listdir(self.pkg_dir)
            if f.endswith(".py") and f not in ("__init__.py", "_meta.json")
        )
        lines = ["# Auto-generated by openhands_agents — do not edit\n"]
        for name in names:
            lines.append(f"from .{name} import {name}\n")
        with open(os.path.join(self.pkg_dir, "__init__.py"), "w") as f:
            f.writelines(lines)

    def _meta_path(self) -> str:
        return os.path.join(self.pkg_dir, "_meta.json")

    def _save_meta(self) -> None:
        with open(self._meta_path(), "w") as f:
            json.dump(self._meta, f, indent=2)

    def _load_meta(self) -> None:
        path = self._meta_path()
        if os.path.exists(path):
            with open(path) as f:
                self._meta = json.load(f)
        else:
            # Reconstruct from existing .py files (no descriptions available)
            for fname in os.listdir(self.pkg_dir):
                if fname.endswith(".py") and fname not in ("__init__.py",):
                    name = fname[:-3]
                    code = open(os.path.join(self.pkg_dir, fname)).read()
                    self._meta.setdefault(name, {
                        "description": "", "usage_count": 0, "code": code,
                    })
