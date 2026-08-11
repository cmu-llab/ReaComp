"""
Standalone runner for the PBEBench / SLR-Bench TroVE baseline.

ADDITIVE entry point (new file) — does not modify openhands_agents/run.py.
Reuses PkgLibrary, ApptainerSandbox, and the paper-faithful TroVE algorithm via
PbeSlrTroVEController. Passes the FULL task record as both task_input (for
DSL-aware prompt building) and entry (for the reward function), fixing the
empty-task bug in the generic run_trove path.

Usage:
    python -m openhands_agents.run_trove_pbe_slr \
        --framework-task-type pbe \
        --dataset-path data/pbebench/lite_tasks_full_og.jsonl \
        --output-path outputs/oh_trove_qwen_lite.jsonl \
        --default-reward pbebench \
        --max-programs 5 \
        --sif-path /scratch/$USER/sif_images/sandbox.sif \
        --pkg-dir /scratch/$USER/oh_packages_trove_lite \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3.6-35B-A3B \
        --k 3 --workers 8 --skip-existing

Output: one JSONL line per task (task_id, solved, best_reward, answer, mode,
        toolbox_size, token_usage). Checkpoint: <output>.ckpt.json.
"""

import argparse
import functools
import importlib
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openhands_agents.pkg_library import PkgLibrary          # noqa: E402
from openhands_agents.trove.mem_limited_sandbox import MemoryLimitedApptainerSandbox  # noqa: E402
from openhands_agents.trove.pbe_slr_controller import PbeSlrTroVEController  # noqa: E402


def _load_dataset(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_reward(name):
    return importlib.import_module(f"rewards.{name}").reward


def _load_ckpt(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser(description="TroVE baseline for PBEBench / SLR-Bench (Qwen).")
    ap.add_argument("--framework-task-type", required=True, choices=["pbe", "slr"])
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--default-reward", required=True, help="pbebench | slr_bench")
    ap.add_argument("--max-programs", type=int, default=5, help="pbe cascade budget (5 Lite / 20 Hard)")

    ap.add_argument("--sif-path", default=os.environ.get("SANDBOX_SIF", ""))
    ap.add_argument("--sandbox-timeout", type=int, default=30)
    ap.add_argument("--sandbox-mem-gb", type=float, default=4.0,
                    help="per-exec RLIMIT_AS cap (GiB); prevents brute-force candidates from OOM-killing the job")
    ap.add_argument("--pkg-dir", required=True, help="toolbox package dir (per-run, to avoid collisions)")

    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--k", type=int, default=3, help="candidates per mode (total 3K/task)")
    ap.add_argument("--trim-every", type=int, default=200)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen thinking mode ON (DirectSolve-matched; use with --max-tokens 16384)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="sampling temperature; omit/None -> vLLM default (DirectSolve-matched)")
    ap.add_argument("--request-timeout", type=float, default=120,
                    help="per-request HTTP timeout (s); raise for large K + thinking mode")

    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--start-index", type=int, default=None)
    ap.add_argument("--end-index", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--checkpoint-path", default="")
    args = ap.parse_args()

    if not args.sif_path:
        ap.error("--sif-path required (or set SANDBOX_SIF)")

    ckpt_path = args.checkpoint_path or args.output_path.replace(".jsonl", ".ckpt.json")

    records = _load_dataset(args.dataset_path)
    for gi, rec in enumerate(records):
        rec.setdefault("task_id", gi)
        rec.setdefault("task_index", gi)
    if args.start_index is not None or args.end_index is not None:
        s = args.start_index or 0
        e = args.end_index if args.end_index is not None else len(records)
        records = records[s:e]
        logger.info("Sliced to [%d, %d) -> %d records", s, e, len(records))
    total = len(records)
    logger.info("Loaded %d task records", total)

    reward_fn = _load_reward(args.default_reward)
    if args.framework_task_type == "pbe":
        reward_fn = functools.partial(reward_fn, max_programs=args.max_programs)

    toolbox = PkgLibrary(os.path.join(args.pkg_dir, "toolbox"))
    sandbox = MemoryLimitedApptainerSandbox(
        sif_path=args.sif_path, timeout=args.sandbox_timeout,
        mem_limit_gb=args.sandbox_mem_gb,
    )

    controller = PbeSlrTroVEController(
        base_url=args.base_url, model=args.model, toolbox=toolbox, sandbox=sandbox,
        api_key=args.api_key, k=args.k, max_tokens=args.max_tokens, trim_every=args.trim_every,
        task_type=args.framework_task_type, max_programs=args.max_programs,
        enable_thinking=args.enable_thinking, temperature=args.temperature,
        request_timeout=args.request_timeout,
    )

    ckpt = _load_ckpt(ckpt_path)
    completed = set(ckpt.get("completed_ids", []))
    if ckpt.get("controller"):
        controller.from_dict(ckpt["controller"])
        logger.info("Resumed checkpoint (n_processed=%d, toolbox=%d)",
                    controller._n_processed, len(toolbox))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    write_lock = threading.Lock()

    def _write(result):
        with write_lock:
            with open(args.output_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
            completed.add(result["task_id"])
            with open(ckpt_path, "w") as f:
                json.dump({"completed_ids": sorted(completed),
                           "controller": controller.to_dict()}, f)

    pending = [(i, r) for i, r in enumerate(records)
               if not (args.skip_existing and r.get("task_id", i) in completed)]
    logger.info("%d pending (of %d)", len(pending), total)

    def _solve_one(i, rec):
        task_id = rec.get("task_id", i)
        # Pass the FULL record as task_input (for DSL prompts) AND entry (for reward).
        result = controller.solve(rec, reward_fn, rec)
        result.pop("candidates", None)  # keep output compact
        result["task_id"] = task_id
        result["task_index"] = rec.get("task_index", i)
        result["dataset"] = rec.get("dataset", "")
        if args.framework_task_type == "pbe":
            result["cascade_length"] = rec.get("cascade_length")
        else:
            result["curriculum_level"] = rec.get("curriculum level")
            result["curriculum_tier"] = rec.get("curriculum tier")
        _write(result)
        logger.info("[%d/%d] task_id=%s reward=%.3f mode=%s toolbox=%d",
                    i + 1, total, task_id, result["best_reward"],
                    result.get("mode"), result.get("toolbox_size", 0))

    workers = max(1, args.workers)
    if workers == 1:
        for i, rec in pending:
            _solve_one(i, rec)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_solve_one, i, rec): (i, rec) for i, rec in pending}
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc:
                    i, rec = futs[fut]
                    logger.error("Task %s failed: %s", rec.get("task_id", i), exc)

    logger.info("Done. Results -> %s (toolbox=%d)", args.output_path, len(toolbox))


if __name__ == "__main__":
    main()
