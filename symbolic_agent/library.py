import math
import re
from typing import List, Optional

from .models import Function


class FunctionLibrary:
    """Shared library of reusable functions, with text-similarity retrieval."""

    def __init__(self):
        self.functions: List[Function] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, func: Function) -> None:
        """Add or update a function by name."""
        existing = self.get(func.name)
        if existing:
            existing.code = func.code
            existing.description = func.description
        else:
            self.functions.append(func)

    def remove(self, name: str) -> bool:
        before = len(self.functions)
        self.functions = [f for f in self.functions if f.name != name]
        return len(self.functions) < before

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Function]:
        for f in self.functions:
            if f.name == name:
                return f
        return None

    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Function]:
        """Retrieve top-k functions by token-overlap (Jaccard) + usage boost."""
        if not self.functions:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored: List[tuple] = []

        for func in self.functions:
            text = f"{func.name} {func.description} {func.code}".lower()
            func_tokens = set(re.findall(r"\w+", text))

            if query_tokens and func_tokens:
                intersection = len(query_tokens & func_tokens)
                union = len(query_tokens | func_tokens)
                score = intersection / union
            else:
                score = 0.0

            # Boost frequently-used functions
            score += 0.01 * math.log1p(func.usage_count)
            scored.append((score, func))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """One-line summary of the library."""
        if not self.functions:
            return "Library is empty."
        lines = [
            f"- {f.name}: {f.description} (used {f.usage_count}x)"
            for f in self.functions
        ]
        return "\n".join(lines)

    def format_for_prompt(self) -> str:
        """Full formatted library for inclusion in an LLM prompt."""
        if not self.functions:
            return "# Library is empty"
        blocks = ["# Available Library Functions"]
        for f in self.functions:
            blocks.append(f"\n## {f.name}")
            if f.description:
                blocks.append(f"# {f.description}")
            blocks.append(f"```python\n{f.code}\n```")
        return "\n".join(blocks)

    def __len__(self) -> int:
        return len(self.functions)

    def __repr__(self) -> str:
        names = [f.name for f in self.functions]
        return f"FunctionLibrary({names})"
