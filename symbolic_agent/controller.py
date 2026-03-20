"""Controller: orchestrates SSL → BCR cycles until the task is solved."""

import logging
import os
from typing import Any, Callable, Dict, Optional

from .bcr_agent import BCRAgent
from .costs import CostTracker
from .executor import execute_with_library, infer_call_args
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
        lam: float = 0.3,
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

        self.client = client
        self.library = FunctionLibrary()
        self.cost_tracker = CostTracker(lam=lam)
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_task_state(self, task_input: Any, task_type: str, budget: float):
        """Shared setup for solve() and solve_with_reward(). Returns (state, task_spec)."""
        self.client.reset_task_log()
        original_prompt = _extract_prompt(task_input)
        state = make_state(
            task_input=task_input,
            task_type=task_type,
            budget=budget,
            original_prompt=original_prompt,
        )
        state["library"] = [f.name for f in self.library.functions]
        logger.info("Prompt: %s", original_prompt[:120])

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
        return state, task_spec

    def _finalize_state(self, state: Dict) -> Dict:
        """Attach cost/library snapshot and agent message log to state."""
        state["cost_summary"] = self.cost_tracker.summary(self.library.functions)
        state["library_snapshot"] = [f.to_dict() for f in self.library.functions]
        state["agent_messages"] = self.client.get_task_log()
        return state

    def _run_solve_loop(self, state: Dict, task_spec: Optional[TaskSpec], budget: float) -> Dict:
        """Inner SSL/BCR loop shared by solve() and solve_with_reward()."""
        state["budget"] = budget
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

        return state

    def _determine_blame(self, state: Dict, execution_ok: bool, reward_value: float) -> str:
        """
        Categorise what went wrong when reward < 1.0.
        Returns one of: "execution" / "library" / "partial" / "logic"
        """
        if not execution_ok:
            return "execution"
        solution = state.get("solution") or {}
        if solution.get("action") == "direct":
            # BCR answered mentally — wrong answer is a reasoning error, not a library bug
            return "partial" if reward_value > 0.0 else "logic"
        if solution.get("functions_used"):
            return "library"
        if reward_value > 0.0:
            return "partial"
        return "logic"

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
        logger.info("=== New task: %s ===", task_type)
        state, task_spec = self._init_task_state(task_input, task_type, budget)
        state = self._run_solve_loop(state, task_spec, budget)

        if state["solved"]:
            state = self.reporting_agent.run(state, self.library)
        else:
            state["final_output"] = {"error": "Could not solve the task within budget/steps."}

        return self._finalize_state(state)

    def solve_with_reward(
        self,
        task_input: Any,
        task_type: str,
        budget: float,
        reward_fn: Callable,
        entry: Dict,
        max_reward_iters: int = 3,
    ) -> Dict:
        """
        Solve with a reward-feedback loop, iterating until reward=1.0 or max_reward_iters.

        Parameters
        ----------
        reward_fn  : callable matching rewards/{name}.reward(result, execution_ok, entry) -> dict
        entry      : full task record dict from the JSONL (passed as-is to reward_fn)
        """
        logger.info("=== New task (reward loop): %s ===", task_type)
        state, task_spec = self._init_task_state(task_input, task_type, budget)

        final_reward: Dict = {"value": 0.0}
        best_raw_result = None
        call_args = infer_call_args(task_input)

        for reward_iter in range(max_reward_iters):
            logger.info("Reward iteration %d/%d", reward_iter + 1, max_reward_iters)

            state["solved"] = False
            state["solution"] = None
            state["trace"] = []
            state["steps"] = 0

            state = self._run_solve_loop(state, task_spec, budget)

            execution_ok = False
            raw_result = None
            solution = state.get("solution")
            if solution and solution.get("action") == "direct":
                execution_ok = True
                raw_result = solution["answer"]
            elif solution and solution.get("code") and solution.get("function"):
                ok, result, err = execute_with_library(
                    solution_code=solution["code"],
                    function_name=solution["function"],
                    args=call_args,
                    library_functions=self.library.functions,
                )
                execution_ok = ok
                raw_result = result if ok else None

            reward_result = reward_fn(raw_result, execution_ok, entry)
            reward_value = float(reward_result.get("value", 0.0))

            if reward_value > state.get("best_reward", 0.0):
                state["best_reward"] = reward_value
                best_raw_result = raw_result

            state["reward_history"].append({
                "iteration": reward_iter,
                "reward": reward_value,
                "message": reward_result.get("message", ""),
                "blame": self._determine_blame(state, execution_ok, reward_value),
                "solution_summary": (solution.get("reasoning", "")[:200] if solution else ""),
            })
            final_reward = reward_result

            logger.info("Reward iteration %d: value=%.3f  %s", reward_iter + 1, reward_value, reward_result.get("message", ""))

            if reward_value >= 1.0:
                logger.info("Perfect reward achieved, stopping.")
                break

        self.cost_tracker.task_loss += 1.0 - state.get("best_reward", 0.0)
        state["solved"] = state.get("best_reward", 0.0) >= 1.0

        # Cache execution result so Reporting agent doesn't re-execute
        if best_raw_result is not None:
            state["_cached_exec"] = best_raw_result

        if state["solved"]:
            state = self.reporting_agent.run(state, self.library)
        else:
            state["final_output"] = {"error": "Could not solve the task within budget/steps."}

        state["final_reward"] = final_reward
        return self._finalize_state(state)

    def library_stats(self) -> Dict:
        """Return current library and cost statistics."""
        return {
            "num_functions": len(self.library),
            "functions": [f.to_dict() for f in self.library.functions],
            "cost_summary": self.cost_tracker.summary(self.library.functions),
            "cost_log": self.cost_tracker.log,
        }
