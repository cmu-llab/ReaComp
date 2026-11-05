#!/usr/bin/env python3
"""KinDER DirectSolve — one OpenHands conversation per KinDER instance.

The per-task agent of the KinDER domains (``kinder_<family>``): it writes a policy for
one instance, tests it against a simulator of that instance, and submits it.  Run on
the ``train`` split it produces the reasoning traces that become solver-induction demos;
run on the ``test`` split it is the LLM baseline and the fallback of the solver-first
hybrid.  Output rows, checkpoints (``<output>.ckpt.json``), ``--skip-existing``,
sharding and ``--debug-dir`` behave as in ``openhands_agents.run --framework direct_solve``.

Usage (from the repository root, with an OpenAI-compatible server such as vLLM):
    python -m openhands_agents.run_kinder_direct_solve \\
        --domain kinder_obstruction2d --split train \\
        --output-path outputs/kinder_obstruction2d_train_direct_solve_openhands.jsonl \\
        --base-url http://localhost:8000/v1 --model openai/Qwen/Qwen3.6-35B-A3B \\
        --workers 8 --skip-existing

Each worker runs its own simulator server and sandbox processes on this machine (CPU);
the LLM is the only GPU consumer.
"""
import argparse
import logging
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains import get_domain, list_domains  # noqa: E402
from openhands_agents.run import _OutputWriter, _load_checkpoint, _write_debug  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    kinder_domains = [d for d in list_domains() if d.startswith("kinder_")]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True, choices=kinder_domains)
    parser.add_argument("--split", required=True, help="train (traces) or test (baseline / hybrid fallback)")
    parser.add_argument("--output-path", required=True)

    # LLM
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="openai/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=None, help="default: the server's")

    # Agent
    parser.add_argument("--max-steps", type=int, default=40, help="LLM turns per task")
    parser.add_argument("--code-timeout", type=float, default=120, help="seconds per python call")
    parser.add_argument("--eval-timeout", type=float, default=600,
                        help="seconds to evaluate the submitted policy (rollout + replay)")
    parser.add_argument("--workers", type=int, default=4)

    # Bookkeeping
    parser.add_argument("--debug-dir", default="",
                        help="Directory for per-task debug JSON (task record, row, full event trajectory)")
    parser.add_argument("--start-index", type=int, default=None,
                        help="First record index to process (inclusive, 0-based). Use with --end-index to shard.")
    parser.add_argument("--end-index", type=int, default=None, help="Last record index (exclusive).")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many tasks.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--checkpoint-path", default="",
                        help="Override checkpoint path (default: <output>.ckpt.json)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # On SIGTERM/SIGHUP (e.g. a scheduler's time limit) stop the simulator servers and remove the
    # sandboxes before exiting; finished rows and the checkpoint are already on disk.
    def _terminate(signum, _frame):
        from domains.kinder.sandbox import close_all_sandboxes
        logger.warning("signal %d: closing sandboxes and exiting", signum)
        close_all_sandboxes()
        os._exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _terminate)

    domain = get_domain(args.domain)
    split = domain.split(args.split)
    records = domain.load_tasks(args.split)
    logger.info("Loaded %d records from %s", len(records), split.path)
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index or 0
        end = args.end_index if args.end_index is not None else len(records)
        records = records[start:end]
        logger.info("Sliced to indices [%d, %d) → %d records", start, end, len(records))
    if args.limit is not None:
        records = records[:args.limit]

    from openhands_agents.kinder_direct_solve.controller import KinderDirectSolveController
    controller = KinderDirectSolveController(
        domain, base_url=args.base_url, model=args.model, api_key=args.api_key, max_steps=args.max_steps,
        max_tokens=args.max_tokens, temperature=args.temperature, code_timeout=args.code_timeout,
        eval_timeout=args.eval_timeout)

    ckpt_path = args.checkpoint_path or args.output_path.replace(".jsonl", ".ckpt.json")
    completed_ids = set(_load_checkpoint(ckpt_path).get("completed_ids", []))
    writer = _OutputWriter(args.output_path, ckpt_path, controller_fn=lambda: {})
    writer._completed_ids = completed_ids
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)

    pending = [rec for rec in records if not (args.skip_existing and rec["task_id"] in completed_ids)]
    total = len(pending)

    def solve_one(n: int, rec: dict) -> None:
        result = controller.solve(rec)
        trajectory = result.pop("_trajectory", [])
        result["task_id"] = rec["task_id"]
        result["task_index"] = rec["task_index"]
        result["dataset"] = split.label
        result["domain"] = domain.name
        result.update(domain.row_extras(rec))
        writer.write(result)
        if args.debug_dir:
            _write_debug(args.debug_dir, rec["task_id"], {
                "task_id": rec["task_id"], "task_record": rec, "row": {k: v for k, v in result.items()
                                                                       if k != "trajectory"},
                "trajectory": trajectory})
        logger.info("[%d/%d] %s count=%s reward=%.1f steps=%d submitted=%s tokens=%s", n, total, rec["task_id"],
                    rec["count"], result["best_reward"], result["steps_used"], result["submitted"],
                    result["token_usage"])

    if args.workers <= 1:
        for n, rec in enumerate(pending, 1):
            solve_one(n, rec)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(solve_one, n, rec): rec for n, rec in enumerate(pending, 1)}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    logger.error("Task %s failed: %s", futures[fut]["task_id"], exc)


if __name__ == "__main__":
    main()
