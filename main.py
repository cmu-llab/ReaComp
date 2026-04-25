"""
Entry point for the Symbolic Library Agent.

Usage:
    python main.py --tasks-file tasks.jsonl      # run tasks from a JSON/JSONL file
    python main.py --output-file results.jsonl   # append each task result live to a JSONL file
    python main.py --tasks-file tasks.jsonl --output-file results.jsonl  # auto-resumes if crashed
    python main.py --stats                       # print library stats after a run
"""

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Set

import functools

from dotenv import load_dotenv

from rewards import load_reward
from symbolic_agent import Controller
from symbolic_agent.models import Function
from symbolic_agent.baselines.trove import TroVEController
from symbolic_agent.baselines.regal import ReGALController
from symbolic_agent.baselines.direct_feedback import DirectFeedbackController
from symbolic_agent.baselines.direct_feedback_simplify import DirectFeedbackSimplifyController

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _print_result(result: dict, task_index: int) -> None:
    print(f"\n{'='*60}")
    print(f"Task {task_index}  type={result.get('task_type', '?')}")
    print(f"{'='*60}")
    print(f"Solved : {result['solved']}")

    # Reward loop summary (only present when solve_with_reward was used)
    if result.get("reward_history"):
        best = result.get("best_reward", 0.0)
        iters = len(result["reward_history"])
        print(f"Reward : best={best:.3f}  iters={iters}")
        for h in result["reward_history"]:
            idx = h.get('iteration', h.get('step', '?'))
            print(f"  iter={idx}  r={h['reward']:.3f}  blame={h.get('blame','?')}  {h.get('message','')[:80]}")

    final = result.get("final_output", {})
    if "error" in final:
        print(f"Error  : {final['error']}")
    else:
        print(f"Answer : {final.get('answer', 'N/A')}")
        print(f"Explain: {final.get('explanation', '')}")
        print(f"Conf.  : {final.get('confidence', '?')}")
        if final.get("execution_result") is not None:
            print(f"Exec   : {final['execution_result']}")

    cost = result.get("cost_summary", {})
    print(f"\nCost summary:")
    for k, v in cost.items():
        print(f"  {k:28s}: {v}")

    trace = result.get("trace", [])
    print(f"\nTrace ({len(trace)} steps):")
    for t in trace:
        agent = t.get("agent", "?")
        action = t.get("action") or t.get("actions", [])
        print(f"  step={t.get('step',0)}  agent={agent}  {action}")


def _load_tasks_file(path: str) -> List[Dict]:
    """
    Load tasks from a JSON or JSONL file.

    Supported formats:
    - JSON:  a single array of task objects  ([ {...}, {...} ])
    - JSONL: one JSON object per line        ({ ... }\\n{ ... }\\n...)

    Each record must have a "prompt" key.  Optional keys:
    - "type"  : task category label (default: "symbolic")
    - any other fields are passed through to the agents as context
    """
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = p.read_text(encoding="utf-8").strip()
    records: List[Dict] = []

    # Try JSON array first, then fall back to JSONL
    if text.startswith("["):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"ERROR: could not parse {path} as JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR: line {lineno} in {path} is not valid JSON: {e}", file=sys.stderr)
                sys.exit(1)

    if not records:
        print(f"ERROR: no records found in {path}", file=sys.stderr)
        sys.exit(1)

    # Validate and normalise
    tasks = []
    for i, rec in enumerate(records):
        if "prompt" not in rec:
            print(f"ERROR: record {i} in {path} is missing required key 'prompt'", file=sys.stderr)
            sys.exit(1)
        # Build a task_input dict the agents can work with.
        # Use an inclusion list so only safe NL fields are visible to agents —
        # oracle answer, metadata, and other dataset internals are never leaked.
        # The full record is preserved in "entry" for the reward function and output.
        _AGENT_KEYS = {"question", "prompt", "task"}
        task_input = {k: v for k, v in rec.items() if k in _AGENT_KEYS}
        tasks.append({
            "input": task_input,
            "type": rec.get("type", "symbolic"),
            "reward": rec.get("reward"),   # name of rewards/{name}.py, or None
            "entry": rec,                  # full original record for reward_fn
        })

    return tasks


def _append_task_output(result: dict, task_index: int, output_file: str) -> None:
    """Append one JSON line to {output_file} immediately after a task completes."""
    final = result.get("final_output", {})
    record = {
        "task_index": task_index,
        "solved": result.get("solved", False),
        "answer": final.get("answer"),
        "best_reward": result.get("best_reward"),
        "reward_history": result.get("reward_history", []),
        "cost_summary": result.get("cost_summary", {}),
        "token_usage": result.get("token_usage", {}),
        "agent_messages": result.get("agent_messages", []),
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    logger.info("Appended task %d → %s", task_index, output_file)


def _save_checkpoint(controller: Controller, last_completed_index: int, checkpoint_file: str) -> None:
    """
    Overwrite {checkpoint_file} with the current controller state so a crashed run
    can be resumed.  Called after every successfully completed task.

    Stores:
      - last_completed_index : last task index that was written to the output file
      - library              : all Function objects (serialised via to_dict())
      - cost_tracker         : cumulative counters and log
      - session_tokens       : accumulated input/output/reasoning token counts
    """
    ckpt = {
        "last_completed_index": last_completed_index,
        "library": [f.to_dict() for f in controller.library.functions],
        "cost_tracker": {
            "num_new_functions": controller.cost_tracker.num_new_functions,
            "total_function_length": controller.cost_tracker.total_function_length,
            "reuse_count": controller.cost_tracker.reuse_count,
            "task_loss": controller.cost_tracker.task_loss,
            "log": controller.cost_tracker.log,
        },
        "session_tokens": controller.client.get_session_token_usage(),
    }
    try:
        Path(checkpoint_file).write_text(
            json.dumps(ckpt, indent=2, default=str), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not write checkpoint %s: %s", checkpoint_file, exc)


def _save_trove_checkpoint(controller, last_completed_index: int, checkpoint_file: str) -> None:
    """
    Save TroVE controller state so a crashed run can be resumed.
    Stores the toolbox, _n_processed counter, and session token counts.
    """
    ckpt = {
        "last_completed_index": last_completed_index,
        "n_processed": controller._n_processed,
        "toolbox": controller.toolbox.to_dict(),
        "session_tokens": controller.llm.get_session_token_usage(),
    }
    try:
        Path(checkpoint_file).write_text(
            json.dumps(ckpt, indent=2, default=str), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not write TroVE checkpoint %s: %s", checkpoint_file, exc)


def _load_trove_checkpoint(controller, checkpoint_file: str) -> int:
    """
    Restore TroVE controller state from {checkpoint_file}.
    Returns the next task index to run (last_completed_index + 1).
    Returns 0 on any error.
    """
    from symbolic_agent.baselines.trove.toolbox import TroVEToolbox
    try:
        ckpt = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))
        controller.toolbox = TroVEToolbox.from_dict(ckpt.get("toolbox", {}))
        controller._n_processed = ckpt.get("n_processed", 0)
        if ckpt.get("session_tokens"):
            controller.llm.restore_session_tokens(ckpt["session_tokens"])
        last_index = ckpt.get("last_completed_index", -1)
        next_index = last_index + 1
        logger.info(
            "TroVE checkpoint loaded: last_completed=%d, toolbox=%d fns, n_processed=%d → resuming from task %d",
            last_index, len(controller.toolbox), controller._n_processed, next_index,
        )
        return next_index
    except Exception as exc:
        logger.warning("Could not load TroVE checkpoint %s: %s — starting from scratch.", checkpoint_file, exc)
        return 0


def _save_react_mem_checkpoint(controller, last_completed_index: int, checkpoint_file: str) -> None:
    """Save ReAct+Memory controller state (episodic memory + session tokens) to checkpoint."""
    ckpt = {
        "last_completed_index": last_completed_index,
        "memory": controller.memory.to_dict(),
        "session_tokens": controller.get_session_token_usage(),
    }
    try:
        Path(checkpoint_file).write_text(json.dumps(ckpt, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write react_mem checkpoint %s: %s", checkpoint_file, exc)


def _load_react_mem_checkpoint(controller, checkpoint_file: str) -> int:
    """Restore ReAct+Memory controller state from checkpoint. Returns next task index."""
    from symbolic_agent.baselines.react_mem.memory import ReActMemory
    try:
        ckpt = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))
        controller.memory = ReActMemory.from_dict(ckpt.get("memory", []), k=controller.memory_k)
        if ckpt.get("session_tokens"):
            controller.restore_session_tokens(ckpt["session_tokens"])
        last_index = ckpt.get("last_completed_index", -1)
        next_index = last_index + 1
        logger.info("react_mem checkpoint loaded: last_completed=%d, memory=%d entries → resuming from task %d",
                    last_index, len(controller.memory), next_index)
        return next_index
    except Exception as exc:
        logger.warning("Could not load react_mem checkpoint %s: %s — starting from scratch.", checkpoint_file, exc)
        return 0


def _save_react_library_checkpoint(controller, last_completed_index: int, checkpoint_file: str) -> None:
    """Save ReAct+Library controller state (function library + session tokens) to checkpoint."""
    ckpt = {
        "last_completed_index": last_completed_index,
        "library": controller.library.to_list(),
        "session_tokens": controller.get_session_token_usage(),
    }
    try:
        Path(checkpoint_file).write_text(json.dumps(ckpt, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write react_library checkpoint %s: %s", checkpoint_file, exc)


def _load_react_library_checkpoint(controller, checkpoint_file: str) -> int:
    """Restore ReAct+Library controller state from checkpoint. Returns next task index."""
    from symbolic_agent.baselines.react_library.library import ReactLibrary
    try:
        ckpt = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))
        controller.library = ReactLibrary.from_list(ckpt.get("library", []))
        if ckpt.get("session_tokens"):
            controller.restore_session_tokens(ckpt["session_tokens"])
        last_index = ckpt.get("last_completed_index", -1)
        next_index = last_index + 1
        logger.info(
            "react_library checkpoint loaded: last_completed=%d, library=%d fns → resuming from task %d",
            last_index, len(controller.library), next_index,
        )
        return next_index
    except Exception as exc:
        logger.warning(
            "Could not load react_library checkpoint %s: %s — starting from scratch.", checkpoint_file, exc
        )
        return 0


def _save_best_of_k_checkpoint(controller, last_completed_index: int, checkpoint_file: str) -> None:
    """Save Best-of-K state (no cross-task library, only token counts) to checkpoint."""
    ckpt = {
        "last_completed_index": last_completed_index,
        "session_tokens": controller.get_session_token_usage(),
    }
    try:
        Path(checkpoint_file).write_text(json.dumps(ckpt, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write best_of_k checkpoint %s: %s", checkpoint_file, exc)


def _load_best_of_k_checkpoint(controller, checkpoint_file: str) -> int:
    """Restore Best-of-K session token counts from checkpoint. Returns next task index."""
    try:
        ckpt = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))
        if ckpt.get("session_tokens"):
            controller.restore_session_tokens(ckpt["session_tokens"])
        last_index = ckpt.get("last_completed_index", -1)
        next_index = last_index + 1
        logger.info("best_of_k checkpoint loaded: last_completed=%d → resuming from task %d",
                    last_index, next_index)
        return next_index
    except Exception as exc:
        logger.warning("Could not load best_of_k checkpoint %s: %s — starting from scratch.", checkpoint_file, exc)
        return 0


class _ParallelCheckpoint:
    """
    Thread-safe checkpoint + output writer for stateless parallel frameworks
    (direct_feedback, best_of_k).

    Uses completed_ids (a set of task indices) so workers can complete out of
    order and resume correctly after a crash.  All file I/O is serialised by
    a single lock so concurrent workers never corrupt outputs.
    """

    def __init__(self, output_file: Optional[str], checkpoint_file: Optional[str]):
        self._lock = threading.Lock()
        self.output_file = output_file
        self.checkpoint_file = checkpoint_file
        self.completed_ids: Set[int] = set()
        self._session_tokens: Dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}

    @classmethod
    def load(cls, output_file: Optional[str], checkpoint_file: Optional[str]) -> "Optional[_ParallelCheckpoint]":
        """Load existing checkpoint if present; return None if nothing to resume."""
        obj = cls(output_file, checkpoint_file)
        if not checkpoint_file or not Path(checkpoint_file).exists():
            return obj
        try:
            ckpt = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))
            ids = set(ckpt.get("completed_ids", []))
            # Back-compat: convert old last_completed_index format
            if not ids and "last_completed_index" in ckpt:
                ids = set(range(int(ckpt["last_completed_index"]) + 1))
            obj.completed_ids = ids
            if ckpt.get("session_tokens"):
                for k in ("input", "output", "reasoning"):
                    obj._session_tokens[k] = int(ckpt["session_tokens"].get(k, 0))
            logger.info(
                "Parallel checkpoint loaded: %d tasks already done → skipping them.",
                len(obj.completed_ids),
            )
        except Exception as exc:
            logger.warning("Could not load parallel checkpoint %s: %s — starting fresh.", checkpoint_file, exc)
        return obj

    def record(self, result: Dict, task_index: int) -> None:
        """Record one completed task result: append to output + update checkpoint. Thread-safe."""
        with self._lock:
            # Accumulate session tokens from per-task usage
            tu = result.get("token_usage", {})
            for k in ("input", "output", "reasoning"):
                self._session_tokens[k] += int(tu.get(k, 0))

            self.completed_ids.add(task_index)

            if self.output_file:
                _append_task_output(result, task_index, self.output_file)

            if self.checkpoint_file:
                ckpt = {
                    "completed_ids": sorted(self.completed_ids),
                    "session_tokens": dict(self._session_tokens),
                }
                try:
                    Path(self.checkpoint_file).write_text(
                        json.dumps(ckpt, indent=2, default=str), encoding="utf-8"
                    )
                except Exception as exc:
                    logger.warning("Could not write parallel checkpoint: %s", exc)

    def get_session_token_usage(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._session_tokens)


def _run_parallel_stateless(
    controller,
    tasks: List[Dict],
    args,
    reward_name_default: Optional[str],
    ckpt: "_ParallelCheckpoint",
    workers: int,
) -> None:
    """
    Run a stateless framework (direct_feedback / best_of_k) with N worker threads.
    Each task is independent; results are written to the checkpoint as they complete.
    """
    pending = [
        (i, task) for i, task in enumerate(tasks)
        if i not in ckpt.completed_ids
    ]
    logger.info(
        "Parallel run: %d tasks pending (%d already done), %d workers",
        len(pending), len(ckpt.completed_ids), workers,
    )

    def _solve_one(i: int, task: Dict) -> Dict:
        task_input = task.get("input", task)
        task_type = task.get("type", "symbolic")
        reward_name = task.get("reward") or reward_name_default
        logger.info("--- Task %d (worker) ---", i + 1)
        if reward_name:
            reward_fn = load_reward(reward_name)
            if args.max_programs is not None:
                reward_fn = functools.partial(reward_fn, max_programs=args.max_programs)
            return controller.solve_with_reward(
                task_input=task_input,
                task_type=task_type,
                budget=args.budget,
                reward_fn=reward_fn,
                entry=task.get("entry", {}),
                max_reward_iters=args.max_reward_iters,
            )
        return controller.solve(task_input, task_type, budget=args.budget)

    if workers == 1:
        for i, task in pending:
            result = _solve_one(i, task)
            _print_result(result, i)
            ckpt.record(result, i)
    else:
        futs = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, task in pending:
                futs[pool.submit(_solve_one, i, task)] = i
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.error("Task %d failed with exception: %s", i, exc)
                    result = {
                        "solved": False, "task_type": "unknown", "original_prompt": "",
                        "answer": None, "best_reward": 0.0, "final_reward": {},
                        "reward_history": [], "trace": [],
                        "final_output": {"error": str(exc)},
                        "agent_messages": [], "token_usage": {},
                        "cost_summary": {}, "library_snapshot": [],
                    }
                _print_result(result, i)
                ckpt.record(result, i)


def _save_regal_checkpoint(controller, last_completed_index: int, checkpoint_file: str) -> None:
    """Save ReGAL session token counts to checkpoint (codebank doesn't change during test)."""
    ckpt = {
        "last_completed_index": last_completed_index,
        "session_tokens": controller.llm.get_session_token_usage(),
    }
    try:
        Path(checkpoint_file).write_text(json.dumps(ckpt, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write regal checkpoint %s: %s", checkpoint_file, exc)


def _load_regal_checkpoint(controller, checkpoint_file: str) -> int:
    """Restore ReGAL session token counts from checkpoint. Returns next task index."""
    try:
        ckpt = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))
        if ckpt.get("session_tokens"):
            controller.llm.restore_session_tokens(ckpt["session_tokens"])
        last_index = ckpt.get("last_completed_index", -1)
        next_index = last_index + 1
        logger.info("regal checkpoint loaded: last_completed=%d → resuming from task %d",
                    last_index, next_index)
        return next_index
    except Exception as exc:
        logger.warning("Could not load regal checkpoint %s: %s — starting from scratch.", checkpoint_file, exc)
        return 0


def _load_checkpoint(controller: Controller, checkpoint_file: str) -> int:
    """
    Restore controller state from {checkpoint_file}.
    Returns the next task index to run (last_completed_index + 1).
    Returns 0 and leaves controller untouched on any error.
    """
    try:
        ckpt = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))

        # Restore library
        controller.library.functions = []
        for fd in ckpt.get("library", []):
            controller.library.functions.append(Function(
                name=fd["name"],
                code=fd["code"],
                description=fd.get("description", ""),
                domain=fd.get("domain", "general"),
                input_types=fd.get("input_types", []),
                output_type=fd.get("output_type", ""),
                usage_count=fd.get("usage_count", 0),
                creation_cost=fd.get("creation_cost", 0.0),
            ))

        # Restore cost tracker
        ct = ckpt.get("cost_tracker", {})
        controller.cost_tracker.num_new_functions = ct.get("num_new_functions", 0)
        controller.cost_tracker.total_function_length = ct.get("total_function_length", 0)
        controller.cost_tracker.reuse_count = ct.get("reuse_count", 0)
        controller.cost_tracker.task_loss = ct.get("task_loss", 0.0)
        controller.cost_tracker.log = ct.get("log", [])

        # Recompute embeddings for restored functions if semantic retrieval is active.
        controller.library.recompute_embeddings()

        # Restore session token counts so cumulative totals survive resume
        if ckpt.get("session_tokens"):
            controller.client.restore_session_tokens(ckpt["session_tokens"])

        last_index = ckpt.get("last_completed_index", -1)
        next_index = last_index + 1
        logger.info(
            "Checkpoint loaded: last_completed=%d, library=%d functions → resuming from task %d",
            last_index, len(controller.library), next_index,
        )
        return next_index
    except Exception as exc:
        logger.warning("Could not load checkpoint %s: %s — starting from scratch.", checkpoint_file, exc)
        return 0


def _print_library_stats(controller: Controller) -> None:
    stats = controller.library_stats()
    print(f"\n{'='*60}")
    print(f"Library stats  ({stats['num_functions']} functions)")
    print(f"{'='*60}")
    for f in stats["functions"]:
        print(
            f"  {f['name']:30s}  uses={f['usage_count']:2d}  "
            f"cost={f['creation_cost']:.3f}  useful={f['usefulness']:.2f}"
        )
    print("\nCost summary:")
    for k, v in stats["cost_summary"].items():
        print(f"  {k:28s}: {v}")
    print("\nCost log (last 20 entries):")
    for entry in stats["cost_log"][-20:]:
        print(f"  {entry}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Symbolic Library Agent")
    parser.add_argument("--stats", action="store_true", help="Print library stats after run")
    parser.add_argument(
        "--framework",
        default="ssl_bcr",
        choices=["ssl_bcr", "trove", "regal", "react_mem", "react_library", "best_of_k", "direct_feedback", "direct_feedback_simplify"],
        help="Solution framework to use. 'ssl_bcr': symbolic library agent (default). "
             "'trove': TroVE online function induction baseline. "
             "'regal': ReGAL offline refactoring baseline. "
             "'react_mem': ReAct agent with episodic memory. "
             "'react_library': ReAct agent with a shared growing Python function library. "
             "'best_of_k': K independent sampling attempts, best-of-K by reward. "
             "'direct_feedback': sequential single-turn calls with verifier feedback history. "
             "'direct_feedback_simplify': direct_feedback + simplification phase to minimise cascade complexity. "
             "(default: ssl_bcr)",
    )
    parser.add_argument("--model", default="claude-sonnet-4-5", help=(
        "Model name to use. Common values: "
        "claude-sonnet-4-5 (Anthropic, default), claude-opus-4-6 (Anthropic), "
        "gpt-4o / gpt-4o-mini (OpenAI — requires --backend openai), "
        "openai/gpt-oss-120b (vLLM — requires --base-url). "
        "Short aliases supported: sonnet→claude-sonnet-4-5, opus→claude-opus-4-6, "
        "haiku→claude-haiku-4-5-20251001, gpt4o→gpt-4o, gpt4omini→gpt-4o-mini."
    ))
    parser.add_argument(
        "--backend",
        default=None,
        choices=["anthropic", "openai", "vllm"],
        help=(
            "LLM backend to use. 'anthropic': Anthropic API (default when model starts with 'claude'). "
            "'openai': OpenAI API (uses OPENAI_API_KEY; auto-set when model starts with 'gpt' or 'o1/o3'). "
            "'vllm': local vLLM OpenAI-compatible server (requires --base-url). "
            "If not set, backend is inferred from model name and --base-url."
        ),
    )
    parser.add_argument("--budget", type=float, default=15.0, help="Step budget per task")
    parser.add_argument("--lam", type=float, default=0.3, help="λ: regularisation weight for library cost in Objective = TaskLoss + λ·TotalCost (default: 0.3)")
    parser.add_argument(
        "--redundancy-mode",
        default="ast_jaccard",
        choices=["ast_jaccard", "edit_distance"],
        help=(
            "Algorithm used to compute the redundancy penalty between library functions. "
            "'ast_jaccard': max(Jaccard on AST node-type sets, Jaccard on callee-name sets) — "
            "fast, captures structural shape and shared dependencies. "
            "'edit_distance': 1 − normalised edit distance on DFS-linearised AST node sequences — "
            "more precise, O(m·n) per pair. (default: ast_jaccard)"
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL for local vLLM (e.g. http://localhost:8000/v1). "
             "When set, ANTHROPIC_API_KEY is not required.",
    )
    parser.add_argument(
        "--tasks-file",
        default=None,
        metavar="FILE",
        help="Path to a JSON or JSONL file of tasks.  Each record must have a 'prompt' key. "
             "Optional keys: 'type' (task category, default 'symbolic'). "
             "Runs all records in batch mode, sharing one library across tasks.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        metavar="FILE",
        help="Path to a JSONL file where each completed task is appended immediately as "
             "a single JSON line.  Each record contains the response, full trajectory, "
             "and all component-agent LLM messages (request + response) for training data.",
    )
    parser.add_argument(
        "--max-reward-iters",
        type=int,
        default=3,
        metavar="N",
        help="Maximum reward-feedback iterations per task when the task record has a 'reward' field. "
             "The agent retries until it achieves reward=1.0 or exhausts N iterations. (default: 3)",
    )
    # ---- Token budget flags ----
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Base max_tokens per LLM call for simple tasks. "
             "Overrides per-agent defaults (SSL: 2048, BCR: 4096, Reporting: 1024). "
             "Set equal to --max-tokens-complex to disable complexity scaling.",
    )
    parser.add_argument(
        "--max-tokens-complex",
        type=int,
        default=None,
        metavar="N",
        help="Max_tokens per call for complex tasks (those with bfs/dfs/recursion hints). "
             "Overrides per-agent defaults (SSL: 4096, BCR: 8192, Reporting: 2048).",
    )
    parser.add_argument(
        "--max-tokens-patch",
        type=int,
        default=16384,
        metavar="N",
        help="Max_tokens for the neural patch call (default: 16384).",
    )
    parser.add_argument(
        "--max-tokens-parser",
        type=int,
        default=512,
        metavar="N",
        help="Max_tokens for the TaskParser call (default: 512).",
    )
    parser.add_argument(
        "--show-projected-budget",
        action="store_true",
        default=False,
        help="Print the projected maximum token budget per task before the run, then exit.",
    )
    parser.add_argument(
        "--default-reward",
        default=None,
        metavar="NAME",
        help="Reward module name to use for tasks that do not have a 'reward' field "
             "(e.g. 'reasoning_gym'). Enables the reward loop for the whole run.",
    )
    parser.add_argument(
        "--semantic-retrieval",
        action="store_true",
        default=False,
        help="Use sentence_transformers for library retrieval (cosine similarity on "
             "name+description embeddings) instead of token Jaccard. Requires: "
             "pip install sentence_transformers. (default: off)",
    )
    parser.add_argument(
        "--semantic-model",
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help="Sentence transformer model for --semantic-retrieval. "
             "(default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        metavar="DIR",
        help="Directory to write per-call LLM debug logs (request, raw response including "
             "chain-of-thought, extracted tool calls).  Each run creates a timestamped "
             "subdirectory so logs from multiple runs are never overwritten.",
    )
    # ---- ReAct+Memory-specific flags ----
    parser.add_argument(
        "--react-mem-k",
        type=int,
        default=3,
        metavar="K",
        help="[react_mem] Number of memory examples retrieved for each task. (default: 3)",
    )
    parser.add_argument(
        "--react-max-steps",
        type=int,
        default=5,
        metavar="N",
        help="[react_mem/react_library] Maximum ReAct steps (thought/code/observe cycles) per task. (default: 5)",
    )
    # ---- ReAct+Library-specific flags ----
    parser.add_argument(
        "--react-lib-k",
        type=int,
        default=5,
        metavar="K",
        help="[react_library] Number of library functions retrieved for each task. (default: 5)",
    )
    # ---- Best-of-K-specific flags ----
    parser.add_argument(
        "--bok-k",
        type=int,
        default=5,
        metavar="K",
        help="[best_of_k] Number of independent sampling attempts per task. (default: 5)",
    )
    # ---- Direct-feedback-specific flags ----
    parser.add_argument(
        "--df-k",
        type=int,
        default=3,
        metavar="K",
        help="[direct_feedback] Maximum sequential attempts per task. "
             "Attempt 1 is the raw task; attempt 2+ include verifier feedback history. "
             "(default: 3)",
    )
    parser.add_argument(
        "--dfs-k",
        type=int,
        default=32,
        metavar="K",
        help="[direct_feedback_simplify] Total attempt budget shared across both phases. "
             "All attempts go to correctness first; once correct, the remainder go to "
             "simplification. (default: 32)",
    )
    parser.add_argument(
        "--simplify-patience",
        type=int,
        default=3,
        metavar="N",
        help="[direct_feedback_simplify] Stop Phase 2 early if the model produces the same "
             "replace() program sequence N times in a row. (default: 3)",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        default=None,
        metavar="N",
        help="Override the max_programs constraint passed to the pbebench reward function. "
             "Use 5 for PBEBench-Lite, 20 for full PBEBench. (default: reward function default)",
    )
    # ---- Parallelism (stateless frameworks only) ----
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel worker threads. Only supported for stateless frameworks "
             "(direct_feedback, best_of_k). Sequential frameworks ignore this flag. (default: 1)",
    )
    # TroVE-specific flags
    parser.add_argument(
        "--trove-k",
        type=int,
        default=5,
        metavar="K",
        help="[TroVE] Number of samples per mode (IMPORT/CREATE/SKIP). Paper default: 5. "
             "Set to 1 for fast/cheap runs. (default: 5)",
    )
    parser.add_argument(
        "--trove-trim-every",
        type=int,
        default=500,
        metavar="N",
        help="[TroVE] Trim low-frequency toolbox functions every N tasks. "
             "Paper default: 500. Set to 9999 to disable for small datasets. (default: 500)",
    )
    parser.add_argument(
        "--trove-selection",
        choices=["reward", "consistency"],
        default="reward",
        help="[TroVE] Candidate selection strategy. 'reward' (default) uses "
             "the per-task reward function with AST tie-breaking. "
             "'consistency' uses the original TroVE majority-vote algorithm. "
             "(default: reward)",
    )
    parser.add_argument(
        "--trove-task-family",
        choices=["default", "pbebench"],
        default="default",
        help="[TroVE] Task family for prompt selection and parser strictness. "
             "'pbebench' uses PBEBench-shaped few-shots and strict **Solution** "
             "parsing (no fallback to any python block). (default: default)",
    )
    # ReGAL-specific flags
    parser.add_argument(
        "--regal-train-file",
        default=None,
        metavar="FILE",
        help="[ReGAL] Path to JSONL training file with 'program' key in each record. "
             "Triggers offline training before the test run. (default: None — test-only mode)",
    )
    parser.add_argument(
        "--regal-retrieval",
        default="sentence_transformers",
        choices=["sentence_transformers", "chromadb"],
        help="[ReGAL] Vector retrieval backend for CodeBank/DemoBank. "
             "'sentence_transformers': local cosine similarity (default). "
             "'chromadb': chromadb PersistentClient with sentence_transformer embeddings.",
    )
    parser.add_argument(
        "--regal-embedding-model",
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help="[ReGAL] Sentence transformer model for query embedding (clustering + retrieval). "
             "(default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--regal-codebank-dir",
        default=None,
        metavar="DIR",
        help="[ReGAL] Directory to save/load trained CodeBank and DemoBank. "
             "If set and the directory contains regal_codebank.json, it is loaded before the test run. "
             "After training (--regal-train-file), the banks are saved here.",
    )
    parser.add_argument(
        "--regal-batch-size",
        type=int,
        default=4,
        metavar="N",
        help="[ReGAL] Examples per refactoring batch during training. Paper: 3–5. (default: 4)",
    )
    parser.add_argument(
        "--regal-edit-codebank",
        action="store_true",
        default=False,
        help="[ReGAL] Enable Stage 3a editCodeBank: periodically prompt the LLM to improve "
             "failing helper functions. Off by default. (default: off)",
    )
    parser.add_argument(
        "--regal-edit-every",
        type=int,
        default=5,
        metavar="N",
        help="[ReGAL] Run editCodeBank every N training batches. (default: 5)",
    )
    parser.add_argument(
        "--regal-prune-every",
        type=int,
        default=5,
        metavar="N",
        help="[ReGAL] Run pruneCodeBank every N training batches. (default: 5)",
    )
    parser.add_argument(
        "--regal-icl-budget",
        type=int,
        default=10,
        metavar="N",
        help="[ReGAL] Total number of ICL examples in the test-time agent prompt. (default: 10)",
    )
    parser.add_argument(
        "--regal-icl-split",
        type=float,
        default=0.5,
        metavar="R",
        help="[ReGAL] Fraction of ICL budget drawn from DemoBank (refactored demos) vs "
             "primitive training examples. Paper default: 0.5. (default: 0.5)",
    )
    args = parser.parse_args()

    # ---- Model alias resolution ----
    _MODEL_ALIASES = {
        "sonnet":     "claude-sonnet-4-6",
        "opus":       "claude-opus-4-6",
        "haiku":      "claude-haiku-4-5-20251001",
        "gpt4o":      "gpt-4o",
        "gpt4omini":  "gpt-4o-mini",
        "gpt-4o-mini-alias": "gpt-4o-mini",
    }
    model = _MODEL_ALIASES.get(args.model, args.model)

    # ---- Backend resolution ----
    # Priority: --backend > --base-url (vllm) > model-name inference
    if args.backend == "vllm" or (args.backend is None and args.base_url):
        resolved_backend = "vllm"
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        base_url = args.base_url
        if not base_url:
            print("ERROR: --base-url is required for --backend vllm.", file=sys.stderr)
            sys.exit(1)
        logger.info("Backend: vLLM at %s  model=%s", base_url, model)
    elif args.backend == "openai" or (
        args.backend is None and (model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3"))
    ):
        resolved_backend = "openai"
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = "https://api.openai.com/v1"
        if not api_key:
            print("ERROR: OPENAI_API_KEY is not set.  Add it to .env or export it.", file=sys.stderr)
            sys.exit(1)
        logger.info("Backend: OpenAI API  model=%s", model)
    else:
        resolved_backend = "anthropic"
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        base_url = None
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY is not set.  Add it to .env or export it.", file=sys.stderr)
            sys.exit(1)
        logger.info("Backend: Anthropic  model=%s", model)

    # --show-projected-budget: print budget and exit (only for ssl_bcr)
    if args.show_projected_budget:
        from symbolic_agent import Controller as _C
        _c = _C(
            model=model,
            max_tokens_base=args.max_tokens,
            max_tokens_complex=args.max_tokens_complex,
            max_tokens_patch=args.max_tokens_patch,
            max_tokens_parser=args.max_tokens_parser,
        )
        pb = _c.projected_budget(max_reward_iters=args.max_reward_iters)
        print("Projected token budget per task:")
        for k, v in pb.items():
            print(f"  {k}: {v}")
        return

    if args.framework == "react_mem":
        from symbolic_agent.baselines.react_mem import ReActMemController
        controller = ReActMemController(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=args.debug_dir,
            memory_k=args.react_mem_k,
            max_steps=args.react_max_steps,
            max_tokens=args.max_tokens or 4096,
        )
        logger.info("Framework: ReAct+Memory (k=%d, max_steps=%d)", args.react_mem_k, args.react_max_steps)
    elif args.framework == "react_library":
        from symbolic_agent.baselines.react_library import ReActLibraryController
        controller = ReActLibraryController(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=args.debug_dir,
            library_k=args.react_lib_k,
            max_steps=args.react_max_steps,
            max_tokens=args.max_tokens or 4096,
        )
        logger.info(
            "Framework: ReAct+Library (library_k=%d, max_steps=%d)",
            args.react_lib_k, args.react_max_steps,
        )
    elif args.framework == "best_of_k":
        from symbolic_agent.baselines.best_of_k import BestOfKController
        controller = BestOfKController(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=args.debug_dir,
            k=args.bok_k,
            max_tokens=args.max_tokens or 4096,
        )
        logger.info("Framework: Best-of-K (k=%d)", args.bok_k)
    elif args.framework == "direct_feedback":
        controller = DirectFeedbackController(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=args.debug_dir,
            k=args.df_k,
            max_tokens=args.max_tokens or 4096,
        )
        logger.info("Framework: direct_feedback (k=%d)", args.df_k)
    elif args.framework == "direct_feedback_simplify":
        controller = DirectFeedbackSimplifyController(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=args.debug_dir,
            k=args.dfs_k,
            max_tokens=args.max_tokens or 4096,
            simplify_patience=args.simplify_patience,
        )
        logger.info("Framework: direct_feedback_simplify (k=%d, patience=%d)", args.dfs_k, args.simplify_patience)
    elif args.framework == "trove":
        controller = TroVEController(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=args.debug_dir,
            k=args.trove_k,
            trim_every=args.trove_trim_every,
            task_family=args.trove_task_family,
            selection=args.trove_selection,
        )
        logger.info(
            "Framework: TroVE (k=%d, trim_every=%d, task_family=%s, selection=%s)",
            args.trove_k, args.trove_trim_every, args.trove_task_family, args.trove_selection,
        )
    elif args.framework == "regal":
        from pathlib import Path as _Path
        controller = ReGALController(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=args.debug_dir,
            retrieval=args.regal_retrieval,
            embedding_model=args.regal_embedding_model,
            chroma_path=(
                str(_Path(args.regal_codebank_dir) / "chroma")
                if args.regal_codebank_dir and args.regal_retrieval == "chromadb"
                else None
            ),
            edit_codebank=args.regal_edit_codebank,
            edit_every=args.regal_edit_every,
            prune_every=args.regal_prune_every,
            icl_budget=args.regal_icl_budget,
            icl_split=args.regal_icl_split,
        )
        # Load pre-trained banks if available
        if args.regal_codebank_dir:
            cb_path = str(_Path(args.regal_codebank_dir) / "regal_codebank.json")
            db_path = str(_Path(args.regal_codebank_dir) / "regal_demobank.json")
            if _Path(cb_path).exists() and _Path(db_path).exists():
                controller.load(cb_path, db_path)
                logger.info("ReGAL: loaded pre-trained banks from %s", args.regal_codebank_dir)
        # Offline training
        if args.regal_train_file:
            train_tasks = _load_tasks_file(args.regal_train_file)
            logger.info("ReGAL: training on %d examples (batch_size=%d)", len(train_tasks), args.regal_batch_size)
            controller.train(train_tasks, batch_size=args.regal_batch_size)
            if args.regal_codebank_dir:
                _Path(args.regal_codebank_dir).mkdir(parents=True, exist_ok=True)
                controller.save(
                    str(_Path(args.regal_codebank_dir) / "regal_codebank.json"),
                    str(_Path(args.regal_codebank_dir) / "regal_demobank.json"),
                )
        logger.info(
            "Framework: ReGAL (retrieval=%s, codebank=%d fns, demobank=%d demos)",
            args.regal_retrieval, len(controller.codebank), len(controller.demobank),
        )
    else:
        controller = Controller(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=args.debug_dir,
            lam=args.lam,
            redundancy_mode=args.redundancy_mode,
            semantic_retrieval=args.semantic_retrieval,
            semantic_model=args.semantic_model,
            max_tokens_base=args.max_tokens,
            max_tokens_complex=args.max_tokens_complex,
            max_tokens_patch=args.max_tokens_patch,
            max_tokens_parser=args.max_tokens_parser,
        )
        logger.info("Framework: ssl_bcr")

    if args.tasks_file:
        tasks = _load_tasks_file(args.tasks_file)
        logger.info("Loaded %d tasks from %s", len(tasks), args.tasks_file)

        # Checkpoint file lives alongside the output file: results.jsonl → results.ckpt.json
        ckpt_file = (
            str(Path(args.output_file).with_suffix(".ckpt.json"))
            if args.output_file else None
        )

        # ---- Stateless parallel path (direct_feedback, best_of_k) ----
        # Uses completed_ids checkpointing so workers can finish out of order.
        _STATELESS_FRAMEWORKS = ("direct_feedback", "direct_feedback_simplify", "best_of_k")
        if args.framework in _STATELESS_FRAMEWORKS:
            par_ckpt = _ParallelCheckpoint.load(args.output_file, ckpt_file)
            workers = max(1, args.workers)
            _run_parallel_stateless(
                controller=controller,
                tasks=tasks,
                args=args,
                reward_name_default=args.default_reward,
                ckpt=par_ckpt,
                workers=workers,
            )
            su = par_ckpt.get_session_token_usage()
            logger.info(
                "Session token usage: input=%d  output=%d  reasoning=%d  total=%d",
                su["input"], su["output"], su["reasoning"],
                su["input"] + su["output"] + su["reasoning"],
            )
        else:
            # ---- Sequential path for stateful frameworks ----
            # Auto-resume via last_completed_index.
            start_index = 0
            if args.framework in ("ssl_bcr", "trove", "react_mem", "react_library", "regal"):
                if (
                    ckpt_file
                    and Path(ckpt_file).exists()
                    and args.output_file
                    and Path(args.output_file).exists()
                ):
                    if args.framework == "trove":
                        start_index = _load_trove_checkpoint(controller, ckpt_file)
                    elif args.framework == "react_mem":
                        start_index = _load_react_mem_checkpoint(controller, ckpt_file)
                    elif args.framework == "react_library":
                        start_index = _load_react_library_checkpoint(controller, ckpt_file)
                    elif args.framework == "regal":
                        start_index = _load_regal_checkpoint(controller, ckpt_file)
                    else:
                        start_index = _load_checkpoint(controller, ckpt_file)
                elif ckpt_file and Path(ckpt_file).exists():
                    logger.warning(
                        "Checkpoint found but output file %s is missing — ignoring checkpoint and starting fresh.",
                        args.output_file,
                    )

            for i, task in enumerate(tasks):
                if i < start_index:
                    continue
                task_input = task.get("input", task)
                task_type = task.get("type", "symbolic")
                reward_name = task.get("reward") or args.default_reward
                logger.info("--- Task %d/%d ---", i + 1, len(tasks))

                if reward_name:
                    reward_fn = load_reward(reward_name)
                    if args.max_programs is not None:
                        reward_fn = functools.partial(reward_fn, max_programs=args.max_programs)
                    result = controller.solve_with_reward(
                        task_input=task_input,
                        task_type=task_type,
                        budget=args.budget,
                        reward_fn=reward_fn,
                        entry=task.get("entry", {}),
                        max_reward_iters=args.max_reward_iters,
                    )
                else:
                    result = controller.solve(task_input, task_type, budget=args.budget)
                _print_result(result, i)
                if args.output_file:
                    _append_task_output(result, i, args.output_file)
                if ckpt_file:
                    if args.framework == "ssl_bcr":
                        _save_checkpoint(controller, i, ckpt_file)
                    elif args.framework == "trove":
                        _save_trove_checkpoint(controller, i, ckpt_file)
                    elif args.framework == "react_mem":
                        _save_react_mem_checkpoint(controller, i, ckpt_file)
                    elif args.framework == "react_library":
                        _save_react_library_checkpoint(controller, i, ckpt_file)
                    elif args.framework == "regal":
                        _save_regal_checkpoint(controller, i, ckpt_file)
                # Library/memory size logging
                if args.framework == "trove":
                    lib_size = len(controller.toolbox)
                elif args.framework == "regal":
                    lib_size = len(controller.codebank)
                elif args.framework == "react_mem":
                    lib_size = len(controller.memory)
                else:
                    lib_size = len(controller.library)
                logger.info("Library size after task %d: %d functions", i + 1, lib_size)
            if args.stats:
                _print_library_stats(controller)
            # Session token usage for sequential path
            _su_src = None
            if hasattr(controller, "get_session_token_usage"):
                _su_src = controller
            else:
                _inner = getattr(controller, "client", None) or getattr(controller, "llm", None)
                if _inner and hasattr(_inner, "get_session_token_usage"):
                    _su_src = _inner
            if _su_src:
                su = _su_src.get_session_token_usage()
                logger.info(
                    "Session token usage: input=%d  output=%d  reasoning=%d  total=%d",
                    su["input"], su["output"], su["reasoning"],
                    su["input"] + su["output"] + su["reasoning"],
                )
    else:
        print("ERROR: --tasks-file is required.", file=sys.stderr)
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
