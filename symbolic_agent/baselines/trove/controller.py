"""TroVE Controller — online function-induction loop.

Faithful reimplementation of TroVE's run_trove.py:

1.  For each task, run 3-way generation (IMPORT / CREATE / SKIP).
2.  Each mode generates K independent samples (default K=5, paper default).
3.  Best per mode is selected via self-consistency:
      a. Discard execution failures.
      b. Majority vote on stdout.
      c. Tie-break by minimum AST node count (simplest solution).
      d. If all fail, return the first sample.
4.  Best across modes is selected by the same self-consistency criterion
    applied to the three per-mode winners.
5.  Library update:
      - IMPORT wins  → update_frequency() for used functions
      - CREATE wins & execution succeeded → add() new functions
      - SKIP wins    → no library changes
6.  Periodic trimming: every `trim_every` tasks (default 500, paper default),
    remove functions below threshold C·log₂₀(n).  Affected indices are
    re-queued for re-generation with IMPORT|SKIP only (no CREATE, since we
    are processing tasks in a stream rather than replaying the whole dataset;
    the re-generation happens lazily on the next call to solve() if the
    task_input is provided, otherwise it is skipped).

NOTE on K=1 vs K=5:
    Using K>1 requires K LLM API calls per mode (3K calls total per task).
    This is faithful to the paper but expensive with API-based models.
    K=1 is also valid for quick experiments; pass --trove-k 1 to main.py.

NOTE on trim_every:
    The paper uses 500.  For small datasets (≤100 tasks) the library never
    grows large enough for trimming to matter; trim_every can be set high
    (e.g. 9999) to disable it.
"""

import logging
from collections import Counter
from typing import Callable, Dict, List, Optional

from .executor import run_solution
from .llm import TroVELLMClient
from .parse import count_ast_nodes, parse_response
from .prompts import build_create_prompt, build_import_prompt, build_skip_prompt, get_question
from .toolbox import TroVEToolbox

logger = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_TRIM_EVERY = 500
DEFAULT_MAX_TOKENS = 512


class TroVEController:
    """
    Online TroVE controller — processes tasks one at a time and grows the
    shared toolbox across tasks.

    Parameters
    ----------
    api_key : str, optional
    model : str
        LLM model identifier.
    base_url : str, optional
        For OpenAI-compatible (vLLM) backends.
    debug_dir : str, optional
    k : int
        Number of samples per mode (paper default: 5).
    trim_every : int
        Trim toolbox every N tasks (paper default: 500).
    trim_C : float
        Trimming threshold multiplier: threshold = C·log₂₀(n). Default: 0.5.
    temperature : float
        Sampling temperature. Default: 0.3 (TroVE paper).
    top_p : float
        Nucleus sampling top-p. Default: 0.95 (TroVE paper).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
        k: int = DEFAULT_K,
        trim_every: int = DEFAULT_TRIM_EVERY,
        trim_C: float = 0.5,
        temperature: float = 0.3,
        top_p: float = 0.95,
    ):
        self.model = model
        self.k = k
        self.trim_every = trim_every
        self.trim_C = trim_C

        backend = "openai" if base_url else "anthropic"
        self.llm = TroVELLMClient(
            backend=backend,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            debug_dir=debug_dir,
        )
        self.toolbox = TroVEToolbox()
        self._n_processed: int = 0  # number of tasks processed so far

    # ------------------------------------------------------------------
    # Public interface (mirrors symbolic_agent.Controller for main.py)
    # ------------------------------------------------------------------

    def solve(
        self,
        task_input: dict,
        task_type: str = "symbolic",
        **kwargs,
    ) -> dict:
        """
        Process one task through TroVE's 3-way generation pipeline.

        Returns a result dict compatible with main.py's _print_result()
        and _append_task_output().
        """
        self.llm.reset_task_log()
        question = get_question(task_input)
        example_idx = self._n_processed

        # 3-way generation
        best_mode, best_resp = self._multi_way_generation(question, example_idx)

        # Update library
        self._update_library(best_mode, best_resp, example_idx)

        # Periodic trimming
        self._n_processed += 1
        if self._n_processed % self.trim_every == 0:
            trimmed = self.toolbox.trim(self._n_processed, C=self.trim_C)
            if trimmed:
                logger.info(
                    "Trimmed toolbox after %d tasks (affected indices: %s)",
                    self._n_processed,
                    sorted(trimmed)[:10],
                )

        is_success = best_resp.get("is_success", False)
        output = best_resp.get("exec_output", "")

        return self._make_result(
            task_input=task_input,
            task_type=task_type,
            best_mode=best_mode,
            best_resp=best_resp,
            is_success=is_success,
            output=output,
        )

    def solve_with_reward(
        self,
        task_input: dict,
        task_type: str = "symbolic",
        reward_fn: Optional[Callable] = None,
        entry: Optional[dict] = None,
        max_reward_iters: int = 3,
        **kwargs,
    ) -> dict:
        """
        TroVE wrapper for the reward-based evaluation loop in main.py.

        TroVE is a one-shot method: self-consistency across K samples already
        serves as internal refinement.  We run one solve() and evaluate the
        reward.  The reward is recorded in reward_history for eval compatibility
        but no retry loop is performed.
        """
        result = self.solve(task_input, task_type, **kwargs)

        if reward_fn is None or entry is None:
            return result

        output = (result.get("final_output") or {}).get("execution_result", "") or ""
        try:
            reward, message = reward_fn(output, entry)
        except Exception as exc:
            logger.warning("Reward function error: %s", exc)
            reward, message = 0.0, str(exc)

        reward_history = [
            {
                "iteration": 0,
                "reward": reward,
                "blame": "trove",
                "message": str(message)[:120],
            }
        ]
        result["reward_history"] = reward_history
        result["best_reward"] = reward
        result["final_reward"] = reward
        result["solved"] = reward >= 1.0

        return result

    def library_stats(self) -> dict:
        """Return toolbox statistics for main.py --stats output."""
        entries = self.toolbox.snapshot()
        return {
            "num_functions": len(entries),
            "functions": [
                {
                    "name": e.get("name", ""),
                    "usage_count": e.get("frequency", 0),
                    "creation_cost": 0.0,
                    "usefulness": float(e.get("frequency", 0)),
                }
                for e in entries
            ],
            "cost_summary": {},
            "cost_log": [],
        }

    # ------------------------------------------------------------------
    # Core generation pipeline
    # ------------------------------------------------------------------

    def _multi_way_generation(self, question: str, example_idx: int):
        """
        Generate K candidates per mode, pick best per mode, then best overall.
        Returns (winning_mode, best_response_dict).
        """
        toolbox_str = self.toolbox.format_toolbox()

        # --- IMPORT mode ---
        import_candidates = []
        if toolbox_str:
            for _ in range(self.k):
                prompt = build_import_prompt(question, toolbox_str)
                raw = self.llm.call(prompt, self.model, max_tokens=DEFAULT_MAX_TOKENS, tag="trove_import")
                parsed = parse_response(raw)
                is_ok, out = run_solution(
                    parsed["solution_code"],
                    parsed["tools_code"],
                    self.toolbox.get_full_code(),
                )
                import_candidates.append({**parsed, "is_success": is_ok, "exec_output": out})
            best_import = import_candidates[self._select_best(import_candidates)]
        else:
            best_import = {"solution_code": "", "tools_code": "", "functions": [],
                           "is_success": False, "exec_output": ""}

        # --- CREATE mode ---
        create_candidates = []
        for _ in range(self.k):
            prompt = build_create_prompt(question)
            raw = self.llm.call(prompt, self.model, max_tokens=DEFAULT_MAX_TOKENS, tag="trove_create")
            parsed = parse_response(raw)
            # CREATE mode: run without any toolbox (model defines its own functions inline)
            is_ok, out = run_solution(
                parsed["solution_code"],
                parsed["tools_code"],
                toolbox_code="",
            )
            create_candidates.append({**parsed, "is_success": is_ok, "exec_output": out})
        best_create = create_candidates[self._select_best(create_candidates)]

        # --- SKIP mode ---
        skip_candidates = []
        for _ in range(self.k):
            prompt = build_skip_prompt(question)
            raw = self.llm.call(prompt, self.model, max_tokens=DEFAULT_MAX_TOKENS, tag="trove_skip")
            parsed = parse_response(raw)
            is_ok, out = run_solution(
                parsed["solution_code"],
                parsed["tools_code"],
                toolbox_code="",
            )
            skip_candidates.append({**parsed, "is_success": is_ok, "exec_output": out})
        best_skip = skip_candidates[self._select_best(skip_candidates)]

        # --- Select across modes ---
        mode_candidates = []
        mode_names = []
        if toolbox_str:
            mode_candidates.append(best_import)
            mode_names.append("import")
        mode_candidates.append(best_create)
        mode_names.append("create")
        mode_candidates.append(best_skip)
        mode_names.append("skip")

        best_idx = self._select_best(mode_candidates)
        winning_mode = mode_names[best_idx]
        best_resp = mode_candidates[best_idx]

        logger.debug(
            "Task %d: winning_mode=%s, is_success=%s, output=%r",
            self._n_processed, winning_mode, best_resp["is_success"], best_resp["exec_output"][:80],
        )
        return winning_mode, best_resp

    def _select_best(self, candidates: List[dict]) -> int:
        """
        Self-consistency selection over a list of response dicts.

        Algorithm (faithful to select_best_solution in TroVE utils/code.py):
          1. Filter successes (is_success=True).
          2. Majority vote on exec_output.
          3. Among those matching the majority output, pick the one with
             minimum AST node count (simplest solution).
          4. Fall back to index 0 if no successes.
        """
        successes = [(i, c) for i, c in enumerate(candidates) if c.get("is_success")]
        if not successes:
            return 0

        # Majority vote
        output_counter: Counter = Counter(c["exec_output"] for _, c in successes)
        majority_output = output_counter.most_common(1)[0][0]

        # Min AST among majority
        majority = [(i, c) for i, c in successes if c["exec_output"] == majority_output]
        best_i, _ = min(
            majority,
            key=lambda x: count_ast_nodes(x[1].get("solution_code", "")),
        )
        return best_i

    # ------------------------------------------------------------------
    # Library updates
    # ------------------------------------------------------------------

    def _update_library(self, mode: str, resp: dict, example_idx: int) -> None:
        """Update toolbox based on winning mode (faithful to run_trove.py)."""
        if mode == "import":
            # IMPORT: credit existing functions that were used
            for func_dict in resp.get("functions", []):
                name = func_dict.get("name", "")
                if name:
                    self.toolbox.update_frequency(name, example_idx)
        elif mode == "create" and resp.get("is_success"):
            # CREATE: add new functions only when execution succeeded
            for func_dict in resp.get("functions", []):
                self.toolbox.add(func_dict, example_idx)

        # SKIP: no library changes

    # ------------------------------------------------------------------
    # Result formatting
    # ------------------------------------------------------------------

    def _make_result(
        self,
        task_input: dict,
        task_type: str,
        best_mode: str,
        best_resp: dict,
        is_success: bool,
        output: str,
    ) -> dict:
        """
        Build a result dict compatible with main.py's _print_result() and
        _append_task_output().
        """
        return {
            "task_type": task_type,
            "original_prompt": str(task_input),
            "solved": is_success,
            "steps": 1,
            "trace": [
                {
                    "step": 0,
                    "agent": "trove",
                    "action": best_mode,
                    "is_success": is_success,
                }
            ],
            "solution": best_resp.get("solution_code", ""),
            "library_snapshot": self.toolbox.snapshot(),
            "cost_summary": {},  # TroVE has no cost model
            "final_output": {
                "answer": output,
                "explanation": f"TroVE mode={best_mode}",
                "confidence": "high" if is_success else "low",
                "execution_result": output,
            },
            "agent_messages": self.llm.get_task_log(),
            "reward_history": [],
            "best_reward": None,
            "final_reward": None,
        }
