"""
ReAct Library — a growing collection of reusable Python helper functions.

Functions are stored as {name, description, code} dicts and retrieved by
BM25 on their name + description tokens (same signal used in the main
FunctionLibrary).  The library is shared across all tasks in a session.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9]", " ", text.lower()).split()


def _bm25_scores(
    query_tokens: List[str],
    corpus: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    N = len(corpus)
    if N == 0:
        return []
    avgdl = sum(len(d) for d in corpus) / N
    df: Counter = Counter()
    for doc in corpus:
        for term in set(doc):
            df[term] += 1
    idf = {
        t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
        for t in df
    }
    scores = []
    for doc in corpus:
        dl = len(doc)
        tf: Counter = Counter(doc)
        score = 0.0
        for qt in query_tokens:
            if qt not in tf:
                continue
            f = tf[qt]
            score += idf.get(qt, 0.0) * (f * (k1 + 1)) / (
                f + k1 * (1 - b + b * dl / avgdl)
            )
        scores.append(score)
    return scores


class ReactLibrary:
    """
    Shared library of reusable Python helper functions.

    Each entry: {"name": str, "description": str, "code": str}

    Functions are retrieved by BM25 on ``name + description`` tokens.
    The library grows as the agent adds new helpers during task solving.
    """

    def __init__(self) -> None:
        self._functions: List[Dict] = []
        self._names: set = set()

    def __len__(self) -> int:
        return len(self._functions)

    def add(self, name: str, description: str, code: str) -> bool:
        """Add a function.  Returns False if a function with that name already exists."""
        if name in self._names:
            return False
        self._functions.append({"name": name, "description": description, "code": code})
        self._names.add(name)
        return True

    def update(self, name: str, description: str, code: str) -> None:
        """Add or overwrite a function by name."""
        for entry in self._functions:
            if entry["name"] == name:
                entry["description"] = description
                entry["code"] = code
                return
        self.add(name, description, code)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Return up to k most relevant functions for the query."""
        if not self._functions:
            return []
        corpus = [_tokenize(f["name"] + " " + f["description"]) for f in self._functions]
        query_tokens = _tokenize(query)
        scores = _bm25_scores(query_tokens, corpus)
        ranked = sorted(zip(self._functions, scores), key=lambda x: x[1], reverse=True)
        return [fn for fn, _ in ranked[:k]]

    def all_functions(self) -> List[Dict]:
        return list(self._functions)

    def namespace(self) -> dict:
        """
        Execute all library functions into a single shared namespace.
        Returns the namespace dict (for use as exec globals).
        """
        import builtins
        ns: dict = {"__builtins__": builtins}
        for fn in self._functions:
            try:
                exec(fn["code"], ns)  # noqa: S102
            except Exception:
                pass  # broken function — skip silently
        return ns

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_list(self) -> List[Dict]:
        return [
            {"name": f["name"], "description": f["description"], "code": f["code"]}
            for f in self._functions
        ]

    @classmethod
    def from_list(cls, data: List[Dict]) -> "ReactLibrary":
        lib = cls()
        for d in data:
            lib.add(d.get("name", ""), d.get("description", ""), d.get("code", ""))
        return lib
