from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Function:
    """A library function with metadata for tracking reuse and cost."""
    name: str
    code: str
    description: str = ""
    embedding: Optional[List[float]] = None
    usage_count: int = 0
    creation_cost: float = 0.0

    def usefulness(self) -> float:
        """Higher usage relative to creation cost = more useful."""
        return self.usage_count / (self.creation_cost + 1e-6)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "usage_count": self.usage_count,
            "creation_cost": round(self.creation_cost, 4),
            "usefulness": round(self.usefulness(), 4),
        }


def make_state(
    task_input: Any = None,
    task_type: str = "",
    budget: float = 10.0,
) -> Dict:
    """Create a fresh state object for a task."""
    return {
        "task_input": task_input,
        "task_type": task_type,
        "working_memory": None,
        "library": [],       # snapshot of library function names at solve time
        "trace": [],
        "budget": budget,
        "steps": 0,
        "solved": False,
        "solution": None,
    }
