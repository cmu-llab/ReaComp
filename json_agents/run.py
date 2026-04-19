"""
json_agents/run.py — entry point for JSON-mode agent baselines.

Frameworks:
  static_library  — pre-built library (LIBRARY.py) + iterative check_reward loop
  react           — no library, pure ReAct reasoning loop

Compatible with gpt-oss-120b (JSON output, no tool-calling).
Output JSONL is compatible with scripts/eval.py.

Usage:
    python -m json_agents.run \\
        --framework static_library \\
        --library-path built_libraries/claude_code/Wed_Apr_15_735_PM \\
        --dataset-path data/pbebench_lite.jsonl \\
        --output-path outputs/json_sl_pbebench.jsonl \\
        --base-url http://localhost:8000/v1 \\
        --model openai/gpt-oss-120b \\
        --workers 4

    python -m json_agents.run \\
        --framework react \\
        --dataset-path data/pbebench_lite.jsonl \\
        --output-path outputs/json_react_pbebench.jsonl \\
        --base-url http://localhost:8000/v1 \\
        --model openai/gpt-oss-120b
"""

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_dataset(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_reward(name: str):
    import importlib
    mod = importlib.import_module(f"rewards.{name}")
    return mod.reward


def _task_input(rec: dict) -> dict:
    _AGENT_KEYS = {"question", "prompt", "task"}
    return {k: v for k, v in rec.items() if k in _AGENT_KEYS}


def _load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


class _OutputWriter:
    """Thread-safe JSONL writer + checkpoint."""

    def __init__(self, output_path: str, ckpt_path: str):
        self._output_path = output_path
        self._ckpt_path = ckpt_path
        self._lock = threading.Lock()
        self._completed_ids: set = set()

    def load_completed(self, ckpt: dict) -> set:
        ids = set(ckpt.get("completed_ids", []))
        self._completed_ids = ids
        return ids

    def write(self, result: dict) -> None:
        with self._lock:
            with open(self._output_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
            self._completed_ids.add(result["task_id"])
            ckpt_data = {"completed_ids": sorted(self._completed_ids)}
            with open(self._ckpt_path, "w") as f:
                json.dump(ckpt_data, f, indent=2)


def _write_debug(debug_dir: str, task_id: Any, data: dict) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"task_{task_id}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _make_execute_fn(sif_path: str, timeout: int):
    """Return execute(code, lib_dir='') -> (ok, stdout, stderr)."""
    if sif_path:
        from openhands_agents.sandbox import ApptainerSandbox
        sandbox = ApptainerSandbox(sif_path=sif_path, timeout=timeout)

        def execute(code: str, lib_dir: str = "") -> tuple:
            return sandbox.run_code(code, lib_dir=lib_dir or None)
    else:
        logger.warning("No --sif-path provided; code execution disabled.")

        def execute(code: str, lib_dir: str = "") -> tuple:
            return False, "", "Sandbox not configured (--sif-path missing)"

    return execute


# ──────────────────────────────────────────────────────────────────────────────
# Framework runners
# ──────────────────────────────────────────────────────────────────────────────

def run_static_library(args, records, reward_fn, ckpt_path):
    from symbolic_agent.llm_client import LLMClient
    from .static_library_agent import StaticLibraryJsonAgent

    if not args.library_path:
        raise ValueError("--library-path required for --framework static_library")

    execute_fn = _make_execute_fn(args.sif_path, args.sandbox_timeout)

    # Stage library as a package for sandbox bind-mount
    lib_dir = ""
    if args.sif_path and args.pkg_dir:
        import ast, shutil
        pkg_dir = os.path.join(args.pkg_dir, "json_sl_library")
        pkg = os.path.join(pkg_dir, "library")
        if os.path.isdir(pkg):
            shutil.rmtree(pkg)
        os.makedirs(pkg)
        shutil.copy(os.path.join(args.library_path, "LIBRARY.py"), os.path.join(pkg, "__init__.py"))
        lib_dir = pkg

    ckpt = _load_checkpoint(ckpt_path)
    writer = _OutputWriter(args.output_path, ckpt_path)
    completed_ids = writer.load_completed(ckpt)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    pending = [(i, rec) for i, rec in enumerate(records)
               if not (args.skip_existing and rec.get("task_id", i) in completed_ids)]
    total = len(records)

    # Each worker needs its own LLM + agent instance (LLMClient is not thread-safe)
    def _make_agent():
        _llm = LLMClient(
            backend=args.backend,
            base_url=args.base_url or None,
            api_key=args.api_key,
            debug_dir=args.debug_dir or None,
        )
        return StaticLibraryJsonAgent(
            llm=_llm,
            model=args.model,
            library_path=args.library_path,
            execute_fn=execute_fn,
            lib_dir=lib_dir,
            max_iters=args.max_iters,
            max_tokens=args.max_tokens,
        )

    _local = threading.local()

    def _solve_one(i: int, rec: dict) -> None:
        if not hasattr(_local, "agent"):
            _local.agent = _make_agent()
        task_id = rec.get("task_id", i)
        task_input = _task_input(rec)
        result = _local.agent.solve(task_input, reward_fn, rec)
        result["task_id"] = task_id
        result["dataset"] = rec.get("dataset", "")
        agent_messages = result.pop("agent_messages", [])
        writer.write(result)
        if args.debug_dir:
            _write_debug(args.debug_dir, task_id, {
                "task_id": task_id,
                "task_input": task_input,
                "answer": result.get("answer"),
                "best_reward": result.get("best_reward"),
                "reward_history": result.get("reward_history"),
                "token_usage": result.get("token_usage"),
                "agent_messages": agent_messages,
            })
        logger.info("[%d/%d] task_id=%s reward=%.3f iters=%d",
                    i + 1, total, task_id, result["best_reward"],
                    len(result.get("reward_history", [])))

    workers = max(1, args.workers)
    if workers == 1:
        for i, rec in pending:
            _solve_one(i, rec)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_solve_one, i, rec): (i, rec) for i, rec in pending}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    i, rec = futures[fut]
                    logger.error("Task %s failed: %s", rec.get("task_id", i), exc)


def run_react(args, records, reward_fn, ckpt_path):
    from symbolic_agent.llm_client import LLMClient
    from .react_agent import ReActJsonAgent

    execute_fn = _make_execute_fn(args.sif_path, args.sandbox_timeout)

    ckpt = _load_checkpoint(ckpt_path)
    writer = _OutputWriter(args.output_path, ckpt_path)
    completed_ids = writer.load_completed(ckpt)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    pending = [(i, rec) for i, rec in enumerate(records)
               if not (args.skip_existing and rec.get("task_id", i) in completed_ids)]
    total = len(records)

    _local = threading.local()

    def _make_agent():
        _llm = LLMClient(
            backend=args.backend,
            base_url=args.base_url or None,
            api_key=args.api_key,
            debug_dir=args.debug_dir or None,
        )
        return ReActJsonAgent(
            llm=_llm,
            model=args.model,
            execute_fn=execute_fn,
            max_iters=args.max_iters,
            max_tokens=args.max_tokens,
        )

    def _solve_one(i: int, rec: dict) -> None:
        if not hasattr(_local, "agent"):
            _local.agent = _make_agent()
        task_id = rec.get("task_id", i)
        task_input = _task_input(rec)
        result = _local.agent.solve(task_input, reward_fn, rec)
        result["task_id"] = task_id
        result["dataset"] = rec.get("dataset", "")
        agent_messages = result.pop("agent_messages", [])
        writer.write(result)
        if args.debug_dir:
            _write_debug(args.debug_dir, task_id, {
                "task_id": task_id,
                "task_input": task_input,
                "answer": result.get("answer"),
                "best_reward": result.get("best_reward"),
                "reward_history": result.get("reward_history"),
                "token_usage": result.get("token_usage"),
                "agent_messages": agent_messages,
            })
        logger.info("[%d/%d] task_id=%s reward=%.3f iters=%d",
                    i + 1, total, task_id, result["best_reward"],
                    len(result.get("reward_history", [])))

    workers = max(1, args.workers)
    if workers == 1:
        for i, rec in pending:
            _solve_one(i, rec)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_solve_one, i, rec): (i, rec) for i, rec in pending}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    i, rec = futures[fut]
                    logger.error("Task %s failed: %s", rec.get("task_id", i), exc)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="json_agents — JSON-mode baselines (gpt-oss-120b compatible)")
    parser.add_argument("--framework", required=True, choices=["static_library", "react"])
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--default-reward", required=True, help="Reward module (e.g. pbebench)")

    # LLM
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--backend", default="gpt_oss", choices=["openai", "gpt_oss", "anthropic"],
                        help="LLM backend (gpt_oss for Harmony-format models, openai for standard)")
    parser.add_argument("--max-tokens", type=int, default=4096)

    # Sandbox
    parser.add_argument("--sif-path", default=os.environ.get("SANDBOX_SIF", ""),
                        help="Apptainer .sif path for sandboxed code execution (optional)")
    parser.add_argument("--sandbox-timeout", type=int, default=30)
    parser.add_argument("--pkg-dir", default=os.path.expanduser("~/oh_packages"),
                        help="Directory for staging library packages for sandbox")

    # Framework-specific
    parser.add_argument("--library-path", default="",
                        help="static_library: path to directory containing LIBRARY.py + PROMPTING_GUIDE.md")
    parser.add_argument("--max-iters", type=int, default=8,
                        help="Max reasoning iterations per task (default 8)")

    # Run control
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--debug-dir", default="")

    args = parser.parse_args()

    ckpt_path = args.checkpoint_path or args.output_path.replace(".jsonl", ".ckpt.json")

    records = _load_dataset(args.dataset_path)
    logger.info("Loaded %d records from %s", len(records), args.dataset_path)

    reward_fn = _load_reward(args.default_reward)

    dispatch = {
        "static_library": run_static_library,
        "react": run_react,
    }
    dispatch[args.framework](args, records, reward_fn, ckpt_path)
    logger.info("Done. Results -> %s", args.output_path)


if __name__ == "__main__":
    main()
