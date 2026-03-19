import math
import re
from typing import List, Optional, TYPE_CHECKING

from .models import Function

if TYPE_CHECKING:
    from .task_parser import TaskSpec

# --------------------------------------------------------------------------
# Domain affinity matrix
#
# Defines how well functions from one domain transfer to another.
# 1.0 = same domain (full boost); 0.0 = unrelated (no boost).
# Drives the domain-aware component of retrieve_relevant().
# --------------------------------------------------------------------------

DOMAIN_AFFINITY: dict = {
    "list_manipulation": {
        "list_manipulation": 1.0,
        "sequence":          0.6,
        "string_manipulation": 0.4,
        "math":              0.1,
        "logic":             0.1,
        "grid":              0.2,
        "symbolic":          0.3,
        "other":             0.1,
        "general":           0.3,
    },
    "string_manipulation": {
        "string_manipulation": 1.0,
        "list_manipulation": 0.4,
        "sequence":          0.3,
        "math":              0.0,
        "logic":             0.1,
        "grid":              0.1,
        "symbolic":          0.2,
        "other":             0.1,
        "general":           0.3,
    },
    "sequence": {
        "sequence":          1.0,
        "list_manipulation": 0.6,
        "math":              0.5,
        "string_manipulation": 0.3,
        "logic":             0.3,
        "grid":              0.2,
        "symbolic":          0.4,
        "other":             0.1,
        "general":           0.3,
    },
    "math": {
        "math":              1.0,
        "sequence":          0.5,
        "logic":             0.6,
        "list_manipulation": 0.1,
        "string_manipulation": 0.0,
        "grid":              0.2,
        "symbolic":          0.3,
        "other":             0.1,
        "general":           0.3,
    },
    "logic": {
        "logic":             1.0,
        "math":              0.6,
        "sequence":          0.3,
        "symbolic":          0.5,
        "list_manipulation": 0.1,
        "string_manipulation": 0.1,
        "grid":              0.3,
        "other":             0.1,
        "general":           0.3,
    },
    "grid": {
        "grid":              1.0,
        "list_manipulation": 0.2,
        "logic":             0.3,
        "math":              0.2,
        "sequence":          0.2,
        "string_manipulation": 0.1,
        "symbolic":          0.2,
        "other":             0.1,
        "general":           0.2,
    },
    "symbolic": {
        "symbolic":          1.0,
        "logic":             0.5,
        "math":              0.3,
        "sequence":          0.4,
        "list_manipulation": 0.3,
        "string_manipulation": 0.2,
        "grid":              0.2,
        "other":             0.2,
        "general":           0.3,
    },
    "other": {
        "other":             1.0,
        "symbolic":          0.2,
        "general":           0.2,
    },
    "general": {
        "general":           0.5,  # neutral: mildly useful everywhere
    },
}


def _domain_affinity(task_domain: str, func_domain: str) -> float:
    """Return the affinity score between a task domain and a function's domain."""
    row = DOMAIN_AFFINITY.get(task_domain, {})
    return row.get(func_domain, 0.1)


def _type_overlap(task_types: List[str], func_types: List[str]) -> float:
    """
    Soft type-match score in [0, 1].
    Compares base type names (e.g. 'list[int]' → 'list') to be forgiving
    about generic parameters while still distinguishing list vs str vs int.
    """
    if not task_types or not func_types:
        return 0.0

    def base(t: str) -> str:
        return t.split("[")[0].strip().lower()

    task_bases = {base(t) for t in task_types}
    func_bases = {base(t) for t in func_types}
    intersection = len(task_bases & func_bases)
    union = len(task_bases | func_bases)
    return intersection / union if union else 0.0


class FunctionLibrary:
    """
    Shared library of reusable functions.

    Retrieval combines four signals:
      - Text similarity (Jaccard over tokens)
      - Domain affinity (using DOMAIN_AFFINITY matrix)
      - Argument-type overlap (base Python types)
      - Usage popularity (log-scaled)
    """

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
            existing.domain = func.domain or existing.domain
            existing.input_types = func.input_types or existing.input_types
            existing.output_type = func.output_type or existing.output_type
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

    def retrieve_relevant(
        self,
        query: str,
        task_spec: "Optional[TaskSpec]" = None,
        top_k: int = 5,
    ) -> List[Function]:
        """
        Retrieve top-k functions scored by a weighted combination of:
          0.5 × text Jaccard
          0.3 × domain affinity   (requires task_spec)
          0.1 × I/O type overlap  (requires task_spec)
          0.1 × log usage count
        """
        if not self.functions:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        task_domain = task_spec.domain if task_spec else None
        task_itypes = task_spec.input_types if task_spec else []

        scored: List[tuple] = []
        for func in self.functions:
            text = f"{func.name} {func.description} {func.code}".lower()
            func_tokens = set(re.findall(r"\w+", text))

            # --- text similarity ---
            if query_tokens and func_tokens:
                inter = len(query_tokens & func_tokens)
                union = len(query_tokens | func_tokens)
                text_score = inter / union
            else:
                text_score = 0.0

            # --- domain affinity ---
            dom_score = (
                _domain_affinity(task_domain, func.domain) if task_domain else 0.0
            )

            # --- type overlap ---
            type_score = (
                _type_overlap(task_itypes, func.input_types) if task_itypes else 0.0
            )

            # --- usage popularity ---
            pop_score = math.log1p(func.usage_count)

            score = (
                0.5 * text_score
                + 0.3 * dom_score
                + 0.1 * type_score
                + 0.1 * pop_score
            )
            scored.append((score, func))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        if not self.functions:
            return "Library is empty."
        lines = [
            f"- {f.name} [{f.domain}]: {f.description} (used {f.usage_count}x)"
            for f in self.functions
        ]
        return "\n".join(lines)

    def format_for_prompt(self, full_code_for: Optional[List[str]] = None) -> str:
        """
        Format library for inclusion in agent prompts.

        full_code_for : names of functions to show with complete code.
                        All other functions are rendered as compact one-line entries
                        (name, domain, description, type signature) to keep prompts short
                        as the library grows.
                        Pass None to show full code for every function (default, backward compat).
        """
        if not self.functions:
            return "# Library is empty"
        full_set = set(full_code_for) if full_code_for is not None else None
        blocks = ["# Available Library Functions"]
        for f in self.functions:
            sig = ""
            if f.input_types:
                sig += f"  inputs: {', '.join(f.input_types)}"
            if f.output_type:
                sig += f"  →  {f.output_type}"
            domain_tag = f"[{f.domain}]" if f.domain and f.domain != "general" else ""
            if full_set is None or f.name in full_set:
                # Full code block — shown for relevant/active functions
                blocks.append(f"\n## {f.name}  {domain_tag}")
                if f.description or sig:
                    blocks.append(f"# {f.description}{sig}")
                blocks.append(f"```python\n{f.code}\n```")
            else:
                # Compact one-liner — name, domain, description, signature
                blocks.append(f"- {f.name} {domain_tag}: {f.description}{sig}")
        return "\n".join(blocks)

    def __len__(self) -> int:
        return len(self.functions)

    def __repr__(self) -> str:
        names = [f.name for f in self.functions]
        return f"FunctionLibrary({names})"
