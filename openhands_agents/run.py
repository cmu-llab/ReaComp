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
        --default-reward reasoning_gym

Output: one JSONL line per task, compatible with scripts/eval.py.
Checkpoint: <output-path>.ckpt.json, saved after every task.
"""

import argparse
import asyncio
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Allow running as `python openhands_agents/run.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def _save_checkpoint(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_result(path: str, result: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Framework runners
# ──────────────────────────────────────────────────────────────────────────────

def run_trove(args, records, reward_fn, sandbox, ckpt_path):
    from .pkg_library import PkgLibrary
    from .trove.controller import TroVEController

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
    last_completed = ckpt.get("last_completed_index", -1)
    if ckpt.get("controller"):
        controller.from_dict(ckpt["controller"])
        logger.info("Resumed TroVE checkpoint (n_processed=%d)", controller._n_processed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    for i, rec in enumerate(records):
        if args.skip_existing and i <= last_completed:
            continue

        task_input = _task_input(rec)
        result = controller.solve(task_input, reward_fn, rec)
        result["task_id"] = rec.get("task_id", i)
        result["dataset"] = rec.get("dataset", "")
        _write_result(args.output_path, result)

        _save_checkpoint(ckpt_path, {
            "last_completed_index": i,
            "controller": controller.to_dict(),
        })
        logger.info("[%d/%d] reward=%.3f toolbox=%d", i + 1, len(records), result["best_reward"], result["toolbox_size"])


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
    last_completed = ckpt.get("last_completed_index", -1)
    if ckpt.get("controller"):
        controller.from_dict(ckpt["controller"])
        logger.info("Resumed react_library checkpoint (library=%d fns)", len(library))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    for i, rec in enumerate(records):
        if args.skip_existing and i <= last_completed:
            continue

        task_input = _task_input(rec)
        result = controller.solve(task_input, reward_fn, rec)
        result["task_id"] = rec.get("task_id", i)
        result["dataset"] = rec.get("dataset", "")
        _write_result(args.output_path, result)

        _save_checkpoint(ckpt_path, {
            "last_completed_index": i,
            "controller": controller.to_dict(),
        })
        logger.info("[%d/%d] reward=%.3f library=%d", i + 1, len(records), result["best_reward"], result["library_size"])


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
    last_completed = ckpt.get("last_completed_index", -1)

    # Stage 1: generate all K samples for all tasks concurrently
    pending = [r for i, r in enumerate(records) if not (args.skip_existing and i <= last_completed)]
    pending_indices = [i for i, r in enumerate(records) if not (args.skip_existing and i <= last_completed)]

    if pending:
        logger.info("best_of_k Stage 1: generating %d×%d samples async...", len(pending), args.k)
        task_inputs = [_task_input(r) for r in pending]
        all_samples = asyncio.run(controller.generate_all(task_inputs))
    else:
        all_samples = []

    # Stage 2: score and pick best per task
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    for local_i, (orig_i, rec, samples) in enumerate(zip(pending_indices, pending, all_samples)):
        result = controller.score_and_pick(samples, reward_fn, rec, sandbox)
        result["task_id"] = rec.get("task_id", orig_i)
        result["dataset"] = rec.get("dataset", "")
        _write_result(args.output_path, result)

        _save_checkpoint(ckpt_path, {"last_completed_index": orig_i})
        logger.info("[%d/%d] reward=%.3f", local_i + 1, len(pending), result["best_reward"])


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="openhands_agents — sandboxed baselines")
    parser.add_argument("--framework", required=True, choices=["react_library", "trove", "best_of_k"])
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
    parser.add_argument("--k", type=int, default=5, help="K per mode (trove) or K samples (best_of_k)")
    parser.add_argument("--max-steps", type=int, default=8, help="react_library: max agent steps per conversation")
    parser.add_argument("--max-reward-iters", type=int, default=3, help="react_library: reward iterations")
    parser.add_argument("--library-k", type=int, default=5, help="react_library: BM25 retrieval top-k")
    parser.add_argument("--trim-every", type=int, default=200, help="trove: trim period (tasks)")
    parser.add_argument("--max-concurrent", type=int, default=64, help="best_of_k: async concurrency limit")

    # Run control
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--checkpoint-path", default="", help="Override checkpoint path (default: <output>.ckpt.json)")

    args = parser.parse_args()

    if not args.sif_path:
        parser.error("--sif-path required (or set SANDBOX_SIF env var)")

    ckpt_path = args.checkpoint_path or args.output_path.replace(".jsonl", ".ckpt.json")

    records = _load_dataset(args.dataset_path)
    logger.info("Loaded %d records from %s", len(records), args.dataset_path)

    reward_fn = _load_reward(args.default_reward)

    from .sandbox import ApptainerSandbox
    sandbox = ApptainerSandbox(sif_path=args.sif_path, timeout=args.sandbox_timeout)

    os.makedirs(args.pkg_dir, exist_ok=True)

    dispatch = {
        "trove": run_trove,
        "react_library": run_react_library,
        "best_of_k": run_best_of_k,
    }
    dispatch[args.framework](args, records, reward_fn, sandbox, ckpt_path)
    logger.info("Done. Results → %s", args.output_path)


if __name__ == "__main__":
    main()
