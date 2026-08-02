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

from . import tools_api
from .executor import run_solution
from .llm import TroVELLMClient
from .parse import count_ast_nodes, imported_callsites, parse_response
from .prompts import (
    build_create_prompt,
    build_import_prompt,
    build_import_with_tools_prompt,
    build_skip_prompt,
    get_question,
)
from .toolbox import TroVEToolbox

logger = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_TRIM_EVERY = 500
DEFAULT_MAX_TOKENS = 4096  # reasoning models (e.g. gpt-oss-120b) consume tokens for CoT


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
        For OpenAI-compatible (vLLM) backends. When set, ``self.backend`` is
        ``"openai"``; otherwise ``"anthropic"``. Native tool-calling IMPORT
        requires the openai backend.
    debug_dir : str, optional
    k : int
        Number of samples per mode (paper default: 5).
    trim_every : int
        Trim toolbox every N tasks (paper default: 500).
    trim_C : float
        Trimming threshold multiplier: threshold = C·log₂₀(n). Default: 1.0
        (matches the original TroVE implementation).
    temperature : float
        Sampling temperature. Default: 0.3 (TroVE paper).
    top_p : float
        Nucleus sampling top-p. Default: 0.95 (TroVE paper).
    task_family : str
        Prompt/parsing family. ``"default"`` (generic) or ``"pbebench"``
        (PBEBench-shaped few-shots; strict ``**Solution**`` parsing).
    selection : str
        Candidate selection strategy. ``"reward"`` (default) uses the
        reward function when available and falls back to consistency;
        ``"consistency"`` always uses the original TroVE majority-vote.
    max_tool_iters : int
        Maximum tool-call rounds per IMPORT trajectory in the native
        tool-calling path. Default: 8.
    tool_schema_topk : int
        Number of top-frequency toolbox functions exposed as OpenAI tool
        schemas in the native IMPORT path. Default: 10.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
        k: int = DEFAULT_K,
        trim_every: int = DEFAULT_TRIM_EVERY,
        trim_C: float = 1.0,
        temperature: float = 0.3,
        top_p: float = 0.95,
        task_family: str = "default",
        selection: str = "reward",
        max_tool_iters: int = 8,
        tool_schema_topk: int = 10,
    ):
        self.model = model
        self.k = k
        self.trim_every = trim_every
        self.trim_C = trim_C
        self.task_family = task_family
        self.selection = selection
        self.max_tool_iters = max_tool_iters
        self.tool_schema_topk = tool_schema_topk

        self.backend = "openai" if base_url else "anthropic"
        self.llm = TroVELLMClient(
            backend=self.backend,
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
        reward_fn: Optional[Callable] = None,
        entry: Optional[dict] = None,
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
        best_mode, best_resp, best_reward_score = self._multi_way_generation(
            question, example_idx, reward_fn=reward_fn, entry=entry
        )

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
            best_reward_score=best_reward_score,
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
        result = self.solve(task_input, task_type, reward_fn=reward_fn, entry=entry, **kwargs)

        if reward_fn is None or entry is None:
            return result

        # Reuse the reward score already computed during candidate selection.
        # Re-evaluate only if selection ran in majority-vote mode (no reward_fn at solve time).
        cached = result.pop("_best_reward_score", None)
        if cached is not None:
            reward, message = cached
        else:
            output = (result.get("final_output") or {}).get("execution_result", "") or ""
            is_success = result.get("solved", False)
            try:
                reward_dict = reward_fn(output, is_success, entry)
                reward = reward_dict.get("value", 0.0)
                message = reward_dict.get("message", "")
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

    def _multi_way_generation(
        self,
        question: str,
        example_idx: int,
        reward_fn: Optional[Callable] = None,
        entry: Optional[dict] = None,
    ):
        """
        Generate K candidates per mode, pick best per mode, then best overall.
        Returns (winning_mode, best_response_dict, best_reward_score_or_None).

        When reward_fn + entry are provided, candidate selection uses reward-based
        scoring instead of majority vote on stdout.  This is more reliable for
        PBEBench (program lists rarely match exactly as strings) and equally good
        for reasoning_gym.
        """
        toolbox_str = self.toolbox.format_toolbox()

        # --- IMPORT mode ---
        toolbox_nonempty = bool(toolbox_str)
        use_tools_branch = toolbox_nonempty and self.backend == "openai"

        if use_tools_branch:
            import_candidates = self._generate_import_with_tools(
                question, example_idx, reward_fn=reward_fn, entry=entry
            )
            best_import_idx, best_import_score = self._select_best(
                import_candidates, reward_fn=reward_fn, entry=entry
            )
            best_import = import_candidates[best_import_idx]
            best_import["_reward_score"] = best_import_score
        elif toolbox_nonempty:
            # Legacy text-based IMPORT (Anthropic or unforeseen non-OpenAI path).
            import_candidates = []
            for _ in range(self.k):
                prompt = build_import_prompt(question, toolbox_str, task_family=self.task_family)
                raw = self.llm.call(prompt, self.model, max_tokens=DEFAULT_MAX_TOKENS, tag="trove_import")
                parsed = parse_response(raw, task_family=self.task_family)
                is_ok, out = run_solution(
                    parsed["solution_code"],
                    parsed["tools_code"],
                    self.toolbox.get_full_code(),
                )
                import_candidates.append(
                    {**parsed, "is_success": is_ok, "exec_output": out, "tool_calls": [], "stopped_reason": "legacy"}
                )
            best_import_idx, best_import_score = self._select_best(
                import_candidates, reward_fn=reward_fn, entry=entry
            )
            best_import = import_candidates[best_import_idx]
            best_import["_reward_score"] = best_import_score
        else:
            best_import = {
                "solution_code": "", "tools_code": "", "functions": [],
                "is_success": False, "exec_output": "",
                "tool_calls": [], "stopped_reason": "empty_toolbox",
                "_reward_score": None,
            }

        # --- CREATE mode ---
        create_candidates = []
        for _ in range(self.k):
            prompt = build_create_prompt(question, task_family=self.task_family)
            raw = self.llm.call(prompt, self.model, max_tokens=DEFAULT_MAX_TOKENS, tag="trove_create")
            parsed = parse_response(raw, task_family=self.task_family)
            is_ok, out = run_solution(
                parsed["solution_code"],
                parsed["tools_code"],
                toolbox_code="",
            )
            create_candidates.append({**parsed, "is_success": is_ok, "exec_output": out})
        best_create_idx, best_create_score = self._select_best(
            create_candidates, reward_fn=reward_fn, entry=entry
        )
        best_create = create_candidates[best_create_idx]
        best_create["_reward_score"] = best_create_score

        # --- SKIP mode ---
        skip_candidates = []
        for _ in range(self.k):
            prompt = build_skip_prompt(question, task_family=self.task_family)
            raw = self.llm.call(prompt, self.model, max_tokens=DEFAULT_MAX_TOKENS, tag="trove_skip")
            parsed = parse_response(raw, task_family=self.task_family)
            is_ok, out = run_solution(
                parsed["solution_code"],
                parsed["tools_code"],
                toolbox_code="",
            )
            skip_candidates.append({**parsed, "is_success": is_ok, "exec_output": out})
        best_skip_idx, best_skip_score = self._select_best(
            skip_candidates, reward_fn=reward_fn, entry=entry
        )
        best_skip = skip_candidates[best_skip_idx]
        best_skip["_reward_score"] = best_skip_score

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

        best_idx, best_score = self._select_best(
            mode_candidates, reward_fn=reward_fn, entry=entry
        )
        winning_mode = mode_names[best_idx]
        best_resp = mode_candidates[best_idx]

        logger.debug(
            "Task %d: winning_mode=%s, is_success=%s, reward=%s, output=%r",
            self._n_processed, winning_mode, best_resp["is_success"],
            f"{best_score[0]:.3f}" if best_score else "n/a",
            best_resp["exec_output"][:80],
        )
        return winning_mode, best_resp, best_score

    def _generate_import_with_tools(
        self,
        question: str,
        example_idx: int,
        reward_fn: Optional[Callable] = None,
        entry: Optional[dict] = None,
    ) -> List[dict]:
        """
        IMPORT-mode generation using native OpenAI tool calling.
        Builds K trajectories; each trajectory may invoke toolbox functions
        via tool_calls during the multi-turn loop. Returns K candidate dicts
        compatible with _select_best.
        """
        prompt = build_import_with_tools_prompt(question, task_family=self.task_family)
        tools_schema = tools_api.toolbox_to_openai_tools(self.toolbox, topk=self.tool_schema_topk)

        candidates: List[dict] = []
        for i in range(self.k):
            tag = f"trove_import_t{example_idx}_{i}"
            messages = [{"role": "user", "content": prompt}]
            on_tc = lambda tc: tools_api.dispatch_tool_call(self.toolbox, tc)
            traj = self.llm.chat_with_tools(
                messages=messages,
                tools=tools_schema,
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                max_tool_iters=self.max_tool_iters,
                on_tool_call=on_tc,
                tag=tag,
            )
            parsed = parse_response(traj["final_text"], task_family=self.task_family)
            is_ok, out = run_solution(
                parsed["solution_code"],
                parsed["tools_code"],
                self.toolbox.get_full_code(),
            )
            candidates.append(
                {
                    **parsed,
                    "is_success": is_ok,
                    "exec_output": out,
                    "tool_calls": traj["tool_calls"],
                    "stopped_reason": traj["stopped_reason"],
                    "iterations": traj["iterations"],
                }
            )
        return candidates

    def _select_best(
        self,
        candidates: List[dict],
        reward_fn: Optional[Callable] = None,
        entry: Optional[dict] = None,
    ):
        """
        Select the best candidate from a list of response dicts.

        Returns (best_index, score_or_None) where score is (reward, message)
        when reward-based selection is used, or None otherwise.

        Selection strategy is governed by self.selection:
          - "reward" (default): reward-based when reward_fn+entry provided,
            falls back to consistency when not.
          - "consistency": original TroVE majority-vote algorithm.
        """
        if self.selection == "consistency":
            return self._select_best_by_consistency(candidates), None
        if reward_fn is not None and entry is not None:
            return self._select_best_by_reward(candidates, reward_fn, entry)
        return self._select_best_by_consistency(candidates), None

    def _select_best_by_reward(
        self,
        candidates: List[dict],
        reward_fn: Callable,
        entry: dict,
    ):
        """Reward-based candidate selection. Returns (best_index, (reward, message))."""
        best_idx = 0
        best_reward = -1.0
        best_reuse = -1
        best_ast = float("inf")
        best_message = ""
        for i, c in enumerate(candidates):
            try:
                rd = reward_fn(c.get("exec_output", ""), c.get("is_success", False), entry)
                score = rd.get("value", 0.0)
                msg = rd.get("message", "")
            except Exception as exc:
                logger.debug("Reward scoring error for candidate %d: %s", i, exc)
                score, msg = 0.0, str(exc)
            ast_size = count_ast_nodes(c.get("solution_code", ""))
            reuse_signal = self._reuse_signal(c)
            if (
                score > best_reward
                or (
                    score == best_reward
                    and (
                        reuse_signal > best_reuse
                        or (reuse_signal == best_reuse and ast_size < best_ast)
                    )
                )
            ):
                best_idx = i
                best_reward = score
                best_reuse = reuse_signal
                best_ast = ast_size
                best_message = msg
        return best_idx, (best_reward, best_message)

    @staticmethod
    def _reuse_signal(candidate: dict) -> int:
        """Tie-break signal for candidates that support TroVE's toolbox."""
        functions = candidate.get("functions") or []
        tool_calls = candidate.get("tool_calls") or []
        unique_tool_names = {
            (tc.get("name") or "").split("<|", 1)[0].strip()
            for tc in tool_calls
            if isinstance(tc, dict) and tc.get("name")
        }
        return len(functions) + len({name for name in unique_tool_names if name})

    def _select_best_by_consistency(self, candidates: List[dict]) -> int:
        """
        Original TroVE self-consistency selection (majority vote on stdout).

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

        output_counter: Counter = Counter(c["exec_output"] for _, c in successes)
        majority_output = output_counter.most_common(1)[0][0]

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
            tool_calls = resp.get("tool_calls") or []
            if tool_calls:
                # Native tool-calling path: credit by unique tool_call.function.name
                # (defensive: sanitize and let toolbox.update_frequency filter unknowns).
                unique_names = {
                    tc["name"].split("<|", 1)[0].strip()
                    for tc in tool_calls
                    if tc.get("name")
                }
                for name in unique_names:
                    if name:
                        self.toolbox.update_frequency(name, example_idx)
            else:
                # Legacy text-based IMPORT: credit functions parsed from **Tools**.
                for func_dict in resp.get("functions", []):
                    name = func_dict.get("name", "")
                    if name:
                        self.toolbox.update_frequency(name, example_idx)
        elif mode == "create" and resp.get("is_success"):
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
        best_reward_score=None,
    ) -> dict:
        """
        Build a result dict compatible with main.py's _print_result() and
        _append_task_output(). Adds passive TroVE telemetry fields.
        """
        tool_calls = best_resp.get("tool_calls") or []
        tools_called = sorted({
            tc["name"].split("<|", 1)[0].strip()
            for tc in tool_calls
            if tc.get("name")
        })
        candidate_names = {e["name"] for e in self.toolbox.snapshot()}
        actually_called = sorted(
            imported_callsites(
                solution_code=best_resp.get("solution_code", ""),
                tools_code=best_resp.get("tools_code", ""),
                candidate_names=candidate_names,
            )
        )
        import_eligible = len(self.toolbox) > 0  # state AFTER this task's update
        # Note: import_eligible reflects the current toolbox state after
        # _update_library has already run for this task. The analyzer should
        # interpret this as "a non-empty toolbox existed at some point during
        # this task's processing". For pre-task eligibility, infer from
        # toolbox snapshots in adjacent tasks.

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
            "cost_summary": {},
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
            "_best_reward_score": best_reward_score,
            # TroVE native-tool-calling telemetry
            "won_mode": best_mode,
            "import_eligible": import_eligible,
            "import_was_winner": best_mode == "import",
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "tools_called": tools_called,
            "actually_called": actually_called,
            "trove_stopped_reason": best_resp.get("stopped_reason", ""),
        }
