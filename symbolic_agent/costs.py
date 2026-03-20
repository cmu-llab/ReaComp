import math
from typing import List

from .models import Function

# --------------------------------------------------------------------------
# Cost weights  (α, β, γ, δ, λ)
# --------------------------------------------------------------------------
ALPHA = 1.0   # per new function
BETA = 0.05   # per line of code
GAMMA = 2.0   # redundancy penalty
DELTA = 0.5   # reuse reward
LAMBDA = 0.3  # regularization weight


class CostTracker:
    """
    Tracks the complexity cost of the solution library.

    TotalCost = α·NumNewFunctions + β·TotalFunctionLength
                + γ·RedundancyPenalty − δ·ReuseReward

    Objective = TaskLoss + λ·TotalCost
    """

    def __init__(self, lam: float = LAMBDA):
        self.lam = lam
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

    def redundancy_penalty(self, functions: List[Function]) -> float:
        """Penalise suspiciously similar function names (proxy for duplicates)."""
        penalty = 0.0
        names = [f.name for f in functions]
        for i, n1 in enumerate(names):
            for n2 in names[i + 1 :]:
                common = sum(1 for a, b in zip(n1, n2) if a == b)
                ratio = common / max(len(n1), len(n2), 1)
                if ratio > 0.7:
                    penalty += ratio
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
        }
