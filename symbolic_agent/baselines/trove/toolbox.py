"""TroVE Toolbox — faithful implementation of the TroVE function library.

The library is a plain dict keyed by function name.  Each entry mirrors the
structure from the original TroVE codebase (utils/code.py):

    {
        "name":      str,   # function name
        "signature": str,   # def fn(...) -> ...: (one line, no body)
        "docstr":    str,   # human-readable description
        "function":  str,   # full source code
        "type":      str,   # "function" or "import"
        "frequency": int,   # usage count across examples
        "indices":   list,  # indices of examples that used this function
    }

Retrieval is frequency-based (top-k by frequency), exactly as in the original.
Trimming uses the threshold  C * log_{20}(n)  from the paper (§3.3), where
n is the number of examples processed so far and C = 0.5 by default.
"""

import math
from typing import Optional


class TroVEToolbox:
    """
    In-memory function toolbox with frequency-based retrieval and periodic trimming.
    Mirrors the dict-based toolbox used in the original TroVE code (run_trove.py).
    """

    def __init__(self) -> None:
        self._toolbox: dict = {}  # name -> entry dict

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, func_dict: dict, example_idx: int) -> None:
        """
        Add a new function (called after a successful CREATE-mode response).
        If the name already exists the entry is left unchanged — frequency
        updates are handled separately by update_frequency().

        Faithful to update_library(..., match_old=False) in run_trove.py.
        """
        name = func_dict.get("name", "").strip()
        # Strip "toolbox." prefix that models sometimes produce
        if name.startswith("toolbox."):
            name = name[8:]
        if not name:
            return
        if name not in self._toolbox:
            entry = dict(func_dict)
            entry["name"] = name
            entry["frequency"] = 1
            entry["indices"] = [example_idx]
            self._toolbox[name] = entry

    def update_frequency(self, name: str, example_idx: int) -> None:
        """
        Increment the frequency counter for an existing function.
        Called when IMPORT mode wins and the function was already in the toolbox.

        Faithful to update_library(..., match_old=True) in run_trove.py.
        """
        if name.startswith("toolbox."):
            name = name[8:]
        if name in self._toolbox:
            self._toolbox[name]["frequency"] += 1
            if example_idx not in self._toolbox[name]["indices"]:
                self._toolbox[name]["indices"].append(example_idx)

    def remove(self, name: str) -> None:
        self._toolbox.pop(name, None)

    # ------------------------------------------------------------------
    # Retrieval / formatting
    # ------------------------------------------------------------------

    def format_toolbox(self, topk: int = 10) -> str:
        """
        Return the top-k functions (by frequency) formatted as markdown code
        blocks showing only signature + docstring — NOT the full body.

        Faithful to format_toolbox() in utils/code.py:
            tool_str = f"# {docstr}\\n{signature}"
            toolbox_str_list.append(wrap_code(tool_str))
        """
        if not self._toolbox:
            return ""
        name_freq = sorted(
            [(n, d["frequency"]) for n, d in self._toolbox.items()],
            key=lambda x: -x[1],
        )
        blocks = []
        for tool_name, _ in name_freq[:topk]:
            d = self._toolbox[tool_name]
            tool_str = f"# {d['docstr']}\n{d['signature']}"
            blocks.append(f"```python\n{tool_str}\n```")
        return "\n".join(blocks)

    def get_full_code(self) -> str:
        """
        Return all function source code concatenated, for building the
        execution namespace when running solutions.
        """
        return "\n\n".join(
            d["function"]
            for d in self._toolbox.values()
            if d["type"] == "function"
        )

    # ------------------------------------------------------------------
    # Trimming
    # ------------------------------------------------------------------

    def trim(self, n_processed: int, C: float = 1.0) -> set:
        """
        Remove functions whose frequency is below the threshold
            C * log_{20}(n_processed)
        and return the set of example indices that had used those functions.

        Faithful to trim_library() in run_trove.py:
            threshold = math.log(n, 20)   # log base 20
        C defaults to 1.0, matching the original implementation (C·log_{20}(n)).
        Note: the original uses log base-20 not base-10; we keep base-20.
        """
        if n_processed <= 1:
            return set()
        threshold = C * math.log(n_processed, 20)
        trimmed_indices: set = set()
        to_remove = []
        for name, d in self._toolbox.items():
            if d["frequency"] < threshold:
                trimmed_indices.update(d["indices"])
                to_remove.append(name)
        for name in to_remove:
            del self._toolbox[name]
        return trimmed_indices

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def snapshot(self) -> list:
        """Return a serialisable list of all toolbox entries."""
        return list(self._toolbox.values())

    def to_dict(self) -> dict:
        """Return the raw toolbox dict (for checkpoint saving)."""
        return dict(self._toolbox)

    @classmethod
    def from_dict(cls, data: dict) -> "TroVEToolbox":
        """Restore a toolbox from a previously saved dict."""
        tb = cls()
        tb._toolbox = data
        return tb

    def __len__(self) -> int:
        return len(self._toolbox)

    def __repr__(self) -> str:
        return f"TroVEToolbox({list(self._toolbox.keys())})"
