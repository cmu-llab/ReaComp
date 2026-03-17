"""Controller: orchestrates SSL → BCR cycles until the task is solved."""

import logging
import os
from typing import Any, Dict, List, Optional

import anthropic

from .bcr_agent import BCRAgent
from .costs import CostTracker
from .library import FunctionLibrary
from .models import make_state
from .reporting_agent import ReportingAgent
from .ssl_agent import SSLAgent

logger = logging.getLogger(__name__)

MAX_STEPS = 10
DEFAULT_BUDGET = 15.0


class Controller:
    """
    Main controller loop.

        for step in range(MAX_STEPS):
            if solved(state): break
            if should_call_ssl(state):
                state = SSL_agent(state)
            else:
                state = BCR_agent(state)
        state = Reporting_agent(state)

    The library is shared across tasks in a session, enabling function
    reuse and emergent abstraction over time.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        reporting_model: str = "claude-haiku-4-5-20251001",
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key)
        self.library = FunctionLibrary()
        self.cost_tracker = CostTracker()
        self.ssl_agent = SSLAgent(self.client, model)
        self.bcr_agent = BCRAgent(self.client, model)
        self.reporting_agent = ReportingAgent(self.client, reporting_model)

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _should_call_ssl(self, state: Dict) -> bool:
        """Return True if the SSL agent should run next."""
        # Always start with SSL to prime the library
        if state["steps"] == 0:
            return True

        trace = state["trace"]
        if not trace:
            return True

        last = trace[-1]

        # BCR just decomposed → library needs updating
        if last.get("action") == "decompose":
            return True

        # Alternate: if BCR just ran without solving, try SSL first
        recent_agents = [t.get("agent") for t in trace[-3:]]
        if recent_agents.count("BCR") > recent_agents.count("SSL") and not state["solved"]:
            return True

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        task_input: Any,
        task_type: str = "symbolic",
        budget: float = DEFAULT_BUDGET,
    ) -> Dict:
        """Run the main solve loop for a single task."""
        state = make_state(task_input=task_input, task_type=task_type, budget=budget)
        state["library"] = [f.name for f in self.library.functions]

        logger.info("=== New task: %s ===", task_type)
        logger.info("Input: %s", task_input)

        for step in range(MAX_STEPS):
            if state["solved"]:
                break
            if state["budget"] <= 0:
                logger.warning("Budget exhausted at step %d", step)
                break

            state["steps"] = step

            if self._should_call_ssl(state):
                logger.info("Step %d → SSL", step)
                state = self.ssl_agent.run(state, self.library, self.cost_tracker)
                state["budget"] -= 1.0
            else:
                logger.info("Step %d → BCR", step)
                state = self.bcr_agent.run(state, self.library, self.cost_tracker)
                state["budget"] -= 1.5

        # Final BCR attempt if not yet solved
        if not state["solved"]:
            logger.info("Final BCR attempt")
            state["steps"] += 1
            state = self.bcr_agent.run(state, self.library, self.cost_tracker)

        # Reporting
        if state["solved"]:
            state = self.reporting_agent.run(state, self.library)
        else:
            state["final_output"] = {"error": "Could not solve the task within budget/steps."}

        state["cost_summary"] = self.cost_tracker.summary(self.library.functions)
        state["library_snapshot"] = [f.to_dict() for f in self.library.functions]
        return state

    def solve_batch(self, tasks: List[Dict]) -> List[Dict]:
        """
        Solve multiple tasks in sequence, sharing the function library.
        Each task dict should have keys: "input", "type" (optional).
        """
        results = []
        for i, task in enumerate(tasks):
            task_input = task.get("input", task)
            task_type = task.get("type", "symbolic")
            logger.info("--- Batch task %d/%d ---", i + 1, len(tasks))
            result = self.solve(task_input, task_type)
            results.append(result)
            logger.info(
                "Library size after task %d: %d functions",
                i + 1,
                len(self.library),
            )
        return results

    def library_stats(self) -> Dict:
        """Return current library and cost statistics."""
        return {
            "num_functions": len(self.library),
            "functions": [f.to_dict() for f in self.library.functions],
            "cost_summary": self.cost_tracker.summary(self.library.functions),
            "cost_log": self.cost_tracker.log,
        }
