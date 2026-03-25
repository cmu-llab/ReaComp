import ast
import math
from typing import List, Literal

from .models import Function

# --------------------------------------------------------------------------
# Cost weights  (α, β, γ, δ, λ)
# --------------------------------------------------------------------------
ALPHA = 1.0   # per new function
BETA = 0.05   # per line of code
GAMMA = 2.0   # redundancy penalty
DELTA = 0.5   # reuse reward
LAMBDA = 0.3  # regularization weight

# Similarity threshold above which a function pair is counted as redundant.
# Applied to both redundancy modes.
REDUNDANCY_THRESHOLD = 0.8

RedundancyMode = Literal["ast_jaccard", "edit_distance"]


# --------------------------------------------------------------------------
# AST helpers shared by both redundancy modes
# --------------------------------------------------------------------------

def _parse_safe(func: Function):
    """Return parsed AST tree or None on SyntaxError."""
    try:
        return ast.parse(func.code)
    except SyntaxError:
        return None


def _ast_node_types(tree) -> frozenset:
    """Set of all AST node-type names present in *tree* (mode A)."""
    return frozenset(type(node).__name__ for node in ast.walk(tree))


def _callee_names(tree) -> frozenset:
    """Set of function/method names called anywhere in *tree* (mode B)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return frozenset(names)


def _jaccard(s1: frozenset, s2: frozenset) -> float:
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def _ast_sequence(tree) -> tuple:
    """DFS-ordered sequence of AST node-type names (mode C)."""
    return tuple(type(node).__name__ for node in ast.walk(tree))


def _edit_similarity(seq1: tuple, seq2: tuple) -> float:
    """Similarity in [0, 1] derived from normalised edit distance between two sequences.

    Uses standard unit-cost insert/delete/substitute edit distance, normalised by
    ``max(len(seq1), len(seq2))``.  Returns 1.0 for identical sequences and 0.0
    when the sequences share nothing in common.
    """
    if not seq1 and not seq2:
        return 1.0
    if not seq1 or not seq2:
        return 0.0
    m, n = len(seq1), len(seq2)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return max(0.0, 1.0 - prev[n] / max(m, n))


# --------------------------------------------------------------------------
# Per-function feature cache helpers
# --------------------------------------------------------------------------

def _features_ab(func: Function):
    """Return (node_type_set, callee_set) for mode ast_jaccard, or (∅, ∅) on error."""
    tree = _parse_safe(func)
    if tree is None:
        return frozenset(), frozenset()
    return _ast_node_types(tree), _callee_names(tree)


def _features_c(func: Function) -> tuple:
    """Return DFS node-type sequence for mode edit_distance, or () on error."""
    tree = _parse_safe(func)
    return _ast_sequence(tree) if tree is not None else ()


class CostTracker:
    """
    Tracks the complexity cost of the solution library.

    TotalCost = α·NumNewFunctions + β·TotalFunctionLength
                + γ·RedundancyPenalty − δ·ReuseReward

    Objective = TaskLoss + λ·TotalCost

    RedundancyPenalty is computed via one of two AST-based modes:

    ``ast_jaccard``   (default)
        For each pair of functions compute
        ``max(jaccard(node_type_sets), jaccard(callee_name_sets))``.
        Captures both structural shape (A) and shared dependencies (B).

    ``edit_distance``
        Compute 1 − normalised_edit_distance on the DFS-linearised
        AST node-type sequences.  More precise than set overlap but
        O(m·n) per pair.
    """

    def __init__(self, lam: float = LAMBDA, redundancy_mode: RedundancyMode = "ast_jaccard"):
        self.lam = lam
        self.redundancy_mode: RedundancyMode = redundancy_mode
        self.num_new_functions: int = 0
        self.total_function_length: int = 0
        self.reuse_count: int = 0
        self.task_loss: float = 0.0
        self.log: List[str] = []

    # ------------------------------------------------------------------
    # Recording events
    # ------------------------------------------------------------------

    def record_new_function(self, func: Function) -> None:
        self.num_new_functions += 1
        length = len(func.code.splitlines())
        self.total_function_length += length
        func.creation_cost = ALPHA + BETA * length
        self.log.append(f"[CREATE] {func.name}  lines={length}  cost={func.creation_cost:.3f}")

    def record_reuse(self, func: Function) -> None:
        self.reuse_count += 1
        func.usage_count += 1
        self.log.append(f"[REUSE]  {func.name}  total_uses={func.usage_count}")

    # ------------------------------------------------------------------
    # Cost components
    # ------------------------------------------------------------------

    def _pair_similarity(self, f1: Function, f2: Function) -> float:
        """Similarity score in [0, 1] between two functions under the active mode."""
        if self.redundancy_mode == "edit_distance":
            s1 = _features_c(f1)
            s2 = _features_c(f2)
            return _edit_similarity(s1, s2)
        else:  # ast_jaccard (A+B)
            nodes1, callees1 = _features_ab(f1)
            nodes2, callees2 = _features_ab(f2)
            return max(_jaccard(nodes1, nodes2), _jaccard(callees1, callees2))

    def redundancy_penalty(self, functions: List[Function]) -> float:
        """Sum of pairwise similarity scores that exceed REDUNDANCY_THRESHOLD."""
        penalty = 0.0
        for i in range(len(functions)):
            for j in range(i + 1, len(functions)):
                sim = self._pair_similarity(functions[i], functions[j])
                if sim > REDUNDANCY_THRESHOLD:
                    penalty += sim
        return penalty

    def reuse_reward(self, functions: List[Function]) -> float:
        return sum(math.log1p(f.usage_count) for f in functions)

    def total_cost(self, functions: List[Function]) -> float:
        cost = (
            ALPHA * self.num_new_functions
            + BETA * self.total_function_length
            + GAMMA * self.redundancy_penalty(functions)
            - DELTA * self.reuse_reward(functions)
        )
        return max(0.0, cost)

    def objective(self, functions: List[Function]) -> float:
        return self.task_loss + self.lam * self.total_cost(functions)

    # ------------------------------------------------------------------
    # Budget awareness (injected into agent prompts)
    # ------------------------------------------------------------------

    def reuse_rate(self) -> float:
        total = self.num_new_functions + self.reuse_count
        return self.reuse_count / total if total > 0 else 0.0

    def library_gate(self, min_functions: int = 5, min_rate: float = 0.7) -> bool:
        """True when the library is growing faster than it is being reused."""
        return self.num_new_functions >= min_functions and self.reuse_rate() < min_rate

    def budget_summary(self, min_functions: int = 5, min_rate: float = 0.7) -> str:
        """Compact one-line status string for injection into SSL / BCR prompts."""
        rate = self.reuse_rate()
        msg = (
            f"Library: {self.num_new_functions} functions created, "
            f"{self.reuse_count} reuse events, reuse rate {rate:.0%}."
        )
        if self.library_gate(min_functions, min_rate):
            msg += (
                " BUDGET GATE ACTIVE: reuse rate is critically low —"
                " action=create is PROHIBITED. You MUST choose reuse or compose."
            )
        return msg

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, functions: List[Function]) -> dict:
        return {
            "num_new_functions": self.num_new_functions,
            "total_function_length": self.total_function_length,
            "reuse_count": self.reuse_count,
            "redundancy_penalty": round(self.redundancy_penalty(functions), 4),
            "reuse_reward": round(self.reuse_reward(functions), 4),
            "total_cost": round(self.total_cost(functions), 4),
            "objective": round(self.objective(functions), 4),
            "redundancy_mode": self.redundancy_mode,
        }
