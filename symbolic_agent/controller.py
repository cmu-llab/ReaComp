"""Controller: orchestrates SSL → BCR cycles until the task is solved."""

import logging
import os
from typing import Any, Dict, List, Optional

from .bcr_agent import BCRAgent
from .costs import CostTracker
from .library import FunctionLibrary
from .llm_client import LLMClient
from .models import make_state
from .reporting_agent import ReportingAgent
from .ssl_agent import SSLAgent
from .task_parser import TaskParser, TaskSpec

logger = logging.getLogger(__name__)

MAX_STEPS = 10
DEFAULT_BUDGET = 15.0


def _extract_prompt(task_input: Any) -> str:
    """Pull the natural-language prompt string out of whatever task_input is."""
    if isinstance(task_input, str):
        return task_input
    if isinstance(task_input, dict):
        return (
            task_input.get("prompt")
            or task_input.get("description")
            or str(task_input)
        )
    return str(task_input)


class Controller:
    """
    Main controller loop.

        task_spec = TaskParser.parse(prompt)
        for step in range(MAX_STEPS):
            if solved(state): break
            if should_call_ssl(state):
                state = SSL_agent(state, task_spec)
            else:
                state = BCR_agent(state, task_spec)
        state = Reporting_agent(state)   # receives original prompt

    The library is shared across tasks in a session, enabling function
    reuse and emergent abstraction over time.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        api_key : str, optional
            Anthropic API key (or any non-empty string for local vLLM).
        model : str
            Model used for all agents (TaskParser, SSL, BCR, Reporting).
        base_url : str, optional
            OpenAI-compatible base URL for local vLLM, e.g. "http://localhost:8000/v1".
        """
        if base_url:
            backend = "openai"
            key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        else:
            backend = "anthropic"
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        client = LLMClient(backend=backend, base_url=base_url, api_key=key, debug_dir=debug_dir)

        self.library = FunctionLibrary()
        self.cost_tracker = CostTracker()
        self.task_parser = TaskParser(client, model=model)
        self.ssl_agent = SSLAgent(client, model)
        self.bcr_agent = BCRAgent(client, model)
        self.reporting_agent = ReportingAgent(client, model)

    # ------------------------------------------------------------------
    # Routing logic
    # ------------------------------------------------------------------

    def _should_call_ssl(self, state: Dict) -> bool:
        """Return True if the SSL agent should run next."""
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
        original_prompt = _extract_prompt(task_input)
        state = make_state(
            task_input=task_input,
            task_type=task_type,
            budget=budget,
            original_prompt=original_prompt,
        )
        state["library"] = [f.name for f in self.library.functions]

        logger.info("=== New task: %s ===", task_type)
        logger.info("Prompt: %s", original_prompt[:120])

        # Parse the task into a structured spec before entering the solve loop
        task_spec: Optional[TaskSpec] = self.task_parser.parse(original_prompt)
        if task_spec:
            logger.info("Task spec: %s", task_spec.summary())
            state["task_spec"] = {
                "domain": task_spec.domain,
                "input_types": task_spec.input_types,
                "output_type": task_spec.output_type,
                "operation_hints": task_spec.operation_hints,
                "symbolic_inputs": task_spec.symbolic_inputs,
            }

        for step in range(MAX_STEPS):
            if state["solved"]:
                break
            if state["budget"] <= 0:
                logger.warning("Budget exhausted at step %d", step)
                break

            state["steps"] = step

            if self._should_call_ssl(state):
                logger.info("Step %d → SSL", step)
                state = self.ssl_agent.run(state, self.library, self.cost_tracker, task_spec)
                state["budget"] -= 1.0
            else:
                logger.info("Step %d → BCR", step)
                state = self.bcr_agent.run(state, self.library, self.cost_tracker, task_spec)
                state["budget"] -= 1.5

        # Final BCR attempt if not yet solved
        if not state["solved"]:
            logger.info("Final BCR attempt")
            state["steps"] += 1
            state = self.bcr_agent.run(state, self.library, self.cost_tracker, task_spec)

        # Reporting — receives original_prompt via state
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
