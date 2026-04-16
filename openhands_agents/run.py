"""
openhands_agents/run.py — entry point for sandboxed baselines.

Usage:
    python -m openhands_agents.run \
        --framework {react_library,trove,best_of_k} \
        --dataset-path data/interleaved/pbebench_rg_string_pilot.jsonl \
        --output-path outputs/oh_trove.jsonl \
        --sif-path /scratch/$USER/sif_images/sandbox.sif \
        --pkg-dir /scratch/$USER/oh_packages \
        --base-url http://gpu-node:8000/v1 \
        --model Qwen/Qwen3-Coder-480B-A22B-Instruct \
        --default-reward reasoning_gym \
        --workers 8

Output: one JSONL line per task, compatible with scripts/eval.py.
Checkpoint: <output-path>.ckpt.json, saved after every task (tracks completed IDs
            so runs can be parallelised and resumed out-of-order).
Debug: --debug-dir DIR writes one JSON file per task with prompt, answer, reward,
       library retrieved, and (for react_library) conversation trajectory.

Parallelism notes:
  react_library / trove: tasks processed in parallel threads (--workers N).
    Parallel tasks share the library — each will see the library state at the
    moment it starts its conversation. Library writes are thread-safe.
  best_of_k: already uses asyncio internally; --workers is ignored.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Thread-safe JSONL writer + checkpoint manager."""

    def __init__(self, output_path: str, ckpt_path: str, controller_fn: Callable):
        self._output_path = output_path
        self._ckpt_path = ckpt_path
        self._controller_fn = controller_fn  # returns dict for checkpoint
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
            ckpt_data = {
                "completed_ids": sorted(self._completed_ids),
                "controller": self._controller_fn(),
            }
            with open(self._ckpt_path, "w") as f:
                json.dump(ckpt_data, f, indent=2)


def _write_debug(debug_dir: str, task_id: Any, data: dict) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"task_{task_id}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ──────────────────────────────────────────────────────────────────────────────
# Framework runners
# ──────────────────────────────────────────────────────────────────────────────

def run_trove(args, records, reward_fn, sandbox, ckpt_path):
    from .pkg_library import PkgLibrary
    from .trove.controller import TroVEController
    from .trove.prompts import build_import_prompt, build_create_prompt

    pkg_dir = os.path.join(args.pkg_dir, "toolbox")
    toolbox = PkgLibrary(pkg_dir)
    controller = TroVEController(
        base_url=args.base_url,
        model=args.model,
        toolbox=toolbox,
        sandbox=sandbox,
        api_key=args.api_key,
        k=args.k,
        max_tokens=args.max_tokens,
        trim_every=args.trim_every,
    )

    ckpt = _load_checkpoint(ckpt_path)
    completed_ids: set = set(ckpt.get("completed_ids", []))
    # Legacy compat: if old checkpoint used last_completed_index
    if not completed_ids and "last_completed_index" in ckpt:
        completed_ids = set(range(ckpt["last_completed_index"] + 1))
    if ckpt.get("controller"):
        controller.from_dict(ckpt["controller"])
        logger.info("Resumed TroVE checkpoint (n_processed=%d)", controller._n_processed)

    writer = _OutputWriter(
        args.output_path, ckpt_path,
        controller_fn=controller.to_dict,
    )
    writer._completed_ids = completed_ids

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    pending = [
        (i, rec) for i, rec in enumerate(records)
        if not (args.skip_existing and rec.get("task_id", i) in completed_ids)
    ]
    total = len(records)

    def _solve_one(i: int, rec: dict) -> None:
        task_id = rec.get("task_id", i)
        task_input = _task_input(rec)
        listing = toolbox.as_listing_with_signatures()
        result = controller.solve(task_input, reward_fn, rec)
        result["task_id"] = task_id
        result["dataset"] = rec.get("dataset", "")
        writer.write(result)
        if args.debug_dir:
            _write_debug(args.debug_dir, task_id, {
                "task_id": task_id,
                "task_input": task_input,
                "toolbox_listing": listing,
                "answer": result.get("answer"),
                "best_reward": result.get("best_reward"),
                "mode": result.get("mode"),
            })
        logger.info("[%d/%d] task_id=%s reward=%.3f toolbox=%d",
                    i + 1, total, task_id, result["best_reward"], result["toolbox_size"])

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


def run_react_library(args, records, reward_fn, sandbox, ckpt_path):
    from .pkg_library import PkgLibrary
    from .react_library.controller import ReActLibraryController

    pkg_dir = os.path.join(args.pkg_dir, "library")
    library = PkgLibrary(pkg_dir)
    controller = ReActLibraryController(
        base_url=args.base_url,
        model=args.model,
        library=library,
        sandbox=sandbox,
        api_key=args.api_key,
        library_k=args.library_k,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        max_reward_iters=args.max_reward_iters,
    )

    ckpt = _load_checkpoint(ckpt_path)
    completed_ids: set = set(ckpt.get("completed_ids", []))
    if not completed_ids and "last_completed_index" in ckpt:
        completed_ids = set(range(ckpt["last_completed_index"] + 1))
    if ckpt.get("controller"):
        controller.from_dict(ckpt["controller"])
        logger.info("Resumed react_library checkpoint (library=%d fns)", len(library))

    writer = _OutputWriter(
        args.output_path, ckpt_path,
        controller_fn=controller.to_dict,
    )
    writer._completed_ids = completed_ids

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    pending = [
        (i, rec) for i, rec in enumerate(records)
        if not (args.skip_existing and rec.get("task_id", i) in completed_ids)
    ]
    total = len(records)

    def _solve_one(i: int, rec: dict) -> None:
        task_id = rec.get("task_id", i)
        task_input = _task_input(rec)
        result = controller.solve(task_input, reward_fn, rec)
        result["task_id"] = task_id
        result["dataset"] = rec.get("dataset", "")
        writer.write(result)
        if args.debug_dir:
            _write_debug(args.debug_dir, task_id, {
                "task_id": task_id,
                "task_input": task_input,
                "answer": result.get("answer"),
                "best_reward": result.get("best_reward"),
                "library_size": result.get("library_size"),
                "library_additions_this_task": result.get("library_additions_this_task"),
                "reward_history": result.get("reward_history"),
            })
        logger.info("[%d/%d] task_id=%s reward=%.3f library=%d",
                    i + 1, total, task_id, result["best_reward"], result["library_size"])

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


def run_static_library(args, records, reward_fn, sandbox, ckpt_path):
    from .static_library.library_loader import StaticLibrary
    from .static_library.controller import StaticLibraryController

    if not args.library_path:
        raise ValueError("--library-path is required for --framework static_library")

    pkg_dir = os.path.join(args.pkg_dir, "static_library")
    os.makedirs(pkg_dir, exist_ok=True)

    static_lib = StaticLibrary(library_path=args.library_path, pkg_dir=pkg_dir)
    controller = StaticLibraryController(
        base_url=args.base_url,
        model=args.model,
        static_library=static_lib,
        sandbox=sandbox,
        api_key=args.api_key,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
    )

    ckpt = _load_checkpoint(ckpt_path)
    completed_ids: set = set(ckpt.get("completed_ids", []))
    if not completed_ids and "last_completed_index" in ckpt:
        completed_ids = set(range(ckpt["last_completed_index"] + 1))

    writer = _OutputWriter(
        args.output_path, ckpt_path,
        controller_fn=controller.to_dict,
    )
    writer._completed_ids = completed_ids

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    pending = [
        (i, rec) for i, rec in enumerate(records)
        if not (args.skip_existing and rec.get("task_id", i) in completed_ids)
    ]
    total = len(records)

    def _solve_one(i: int, rec: dict) -> None:
        task_id = rec.get("task_id", i)
        task_input = _task_input(rec)
        result = controller.solve(task_input, reward_fn, rec)
        result["task_id"] = task_id
        result["dataset"] = rec.get("dataset", "")
        writer.write(result)
        if args.debug_dir:
            _write_debug(args.debug_dir, task_id, {
                "task_id": task_id,
                "task_input": task_input,
                "answer": result.get("answer"),
                "best_reward": result.get("best_reward"),
                "reward_history": result.get("reward_history"),
                "token_usage": result.get("token_usage"),
            })
        logger.info("[%d/%d] task_id=%s reward=%.3f",
                    i + 1, total, task_id, result["best_reward"])

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


def run_best_of_k(args, records, reward_fn, sandbox, ckpt_path):
    from .best_of_k.controller import BestOfKController

    controller = BestOfKController(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        k=args.k,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent,
    )

    ckpt = _load_checkpoint(ckpt_path)
    completed_ids: set = set(ckpt.get("completed_ids", []))
    if not completed_ids and "last_completed_index" in ckpt:
        completed_ids = set(range(ckpt["last_completed_index"] + 1))

    pending = [(i, r) for i, r in enumerate(records)
               if not (args.skip_existing and r.get("task_id", i) in completed_ids)]
    pending_indices = [i for i, _ in pending]
    pending_recs = [r for _, r in pending]

    writer = _OutputWriter(
        args.output_path, ckpt_path,
        controller_fn=lambda: {},
    )
    writer._completed_ids = completed_ids

    if pending_recs:
        logger.info("best_of_k: generating %d×%d samples async...", len(pending_recs), args.k)
        task_inputs = [_task_input(r) for r in pending_recs]
        all_samples = asyncio.run(controller.generate_all(task_inputs))
    else:
        all_samples = []

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    for local_i, (orig_i, rec, samples) in enumerate(zip(pending_indices, pending_recs, all_samples)):
        result = controller.score_and_pick(samples, reward_fn, rec, sandbox)
        result["task_id"] = rec.get("task_id", orig_i)
        result["dataset"] = rec.get("dataset", "")
        writer.write(result)
        logger.info("[%d/%d] reward=%.3f", local_i + 1, len(pending_recs), result["best_reward"])


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="openhands_agents — sandboxed baselines")
    parser.add_argument("--framework", required=True, choices=["react_library", "trove", "best_of_k", "static_library"])
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--default-reward", required=True, help="Reward module name (e.g. reasoning_gym)")

    # Sandbox
    parser.add_argument("--sif-path", default=os.environ.get("SANDBOX_SIF", ""),
                        help="Path to sandbox.sif (or set SANDBOX_SIF env var)")
    parser.add_argument("--sandbox-timeout", type=int, default=30)
    parser.add_argument("--pkg-dir", default=os.path.expanduser("~/oh_packages"),
                        help="Directory for library/toolbox package dirs")

    # LLM
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-480B-A22B-Instruct")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)

    # Framework-specific
    parser.add_argument("--library-path", default="",
                        help="static_library: path to built_libraries/... directory containing LIBRARY.py and PROMPTING_GUIDE.md")
    parser.add_argument("--k", type=int, default=5, help="K per mode (trove) or K samples (best_of_k)")
    parser.add_argument("--max-steps", type=int, default=100, help="react_library: max agent steps per conversation")
    parser.add_argument("--max-reward-iters", type=int, default=3,
                        help="react_library: kept for API compat; agent iterates inline via check_reward")
    parser.add_argument("--library-k", type=int, default=5, help="react_library: BM25 retrieval top-k")
    parser.add_argument("--trim-every", type=int, default=200, help="trove: trim period (tasks)")
    parser.add_argument("--max-concurrent", type=int, default=64, help="best_of_k: async concurrency limit")

    # Parallelism & debug
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker threads (react_library/trove). "
                             "Parallel tasks share the library; additions may not be visible "
                             "to concurrently running tasks.")
    parser.add_argument("--debug-dir", default="",
                        help="Directory for per-task debug JSON files (prompt, answer, reward, trajectory)")

    # Run control
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--checkpoint-path", default="",
                        help="Override checkpoint path (default: <output>.ckpt.json)")
    parser.add_argument("--clear", action="store_true",
                        help="Delete output file, checkpoint, and pkg_dir library before starting")

    args = parser.parse_args()

    if not args.sif_path:
        parser.error("--sif-path required (or set SANDBOX_SIF env var)")

    ckpt_path = args.checkpoint_path or args.output_path.replace(".jsonl", ".ckpt.json")

    if args.clear:
        from .pkg_library import PkgLibrary
        for path in (args.output_path, ckpt_path):
            if os.path.exists(path):
                os.remove(path)
                logger.info("Cleared %s", path)
        pkg_subdir = {"react_library": "library", "trove": "toolbox", "best_of_k": None, "static_library": None}[args.framework]
        if pkg_subdir:
            pkg_dir = os.path.join(args.pkg_dir, pkg_subdir)
            if os.path.isdir(pkg_dir):
                PkgLibrary(pkg_dir).clear()
                logger.info("Cleared library at %s", pkg_dir)

    records = _load_dataset(args.dataset_path)
    logger.info("Loaded %d records from %s", len(records), args.dataset_path)

    reward_fn = _load_reward(args.default_reward)

    from .sandbox import ApptainerSandbox
    sandbox = ApptainerSandbox(sif_path=args.sif_path, timeout=args.sandbox_timeout)

    os.makedirs(args.pkg_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    dispatch = {
        "trove": run_trove,
        "react_library": run_react_library,
        "best_of_k": run_best_of_k,
        "static_library": run_static_library,
    }
    dispatch[args.framework](args, records, reward_fn, sandbox, ckpt_path)
    logger.info("Done. Results → %s", args.output_path)


if __name__ == "__main__":
    main()
