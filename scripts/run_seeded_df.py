"""
Solver-seeded Direct Feedback ablation (additive, orXJ Concern 3).

Runs seeded DF only on the tasks the symbolic solver FAILS (best_reward < 1.0),
seeding each with the solver's near-miss program. Writes one JSONL row per
solver-failed task. Plain-DF numbers for comparison come from the existing DF
output files; we do not re-run plain DF.

Usage (SLR, CC solver, k=32, vLLM on :PORT serving gpt-oss-120b):
  python scripts/run_seeded_df.py \
    --benchmark slr \
    --solver-results evals/solver_results/slr_claude_code/slr.jsonl \
    --tasks data/slr_bench/v1_All_full.jsonl \
    --reward slr_bench \
    --out outputs/seeded_df_slr_cc.jsonl \
    --base-url http://localhost:8000/v1 --model openai/gpt-oss-120b \
    --k 32 --max-tokens 32768 --workers 8

  # Lite:
  python scripts/run_seeded_df.py \
    --benchmark lite \
    --solver-results evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl \
    --tasks data/pbebench/lite_tasks_full_og.jsonl \
    --reward pbebench --max-programs 5 \
    --out outputs/seeded_df_lite_cc.jsonl ...
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbolic_agent.baselines.direct_feedback.seeded_controller import (  # noqa: E402
    SeededDirectFeedbackController,
)


def _load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _difficulty(benchmark, srow):
    if benchmark == "slr":
        return srow.get("curriculum_tier") or "?"
    return srow.get("cascade_length")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["lite", "slr"])
    ap.add_argument("--solver-results", required=True, help="solver JSONL with per-task best_reward/answer")
    ap.add_argument("--tasks", required=True, help="task JSONL (row i == task_index i)")
    ap.add_argument("--reward", required=True, help="reward module (pbebench | slr_bench)")
    ap.add_argument("--max-programs", type=int, default=None, help="pbe reward budget (5 Lite)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap number of failed tasks (debug)")
    args = ap.parse_args()

    reward_fn = importlib.import_module(f"rewards.{args.reward}").reward
    if args.max_programs is not None:
        reward_fn = functools.partial(reward_fn, max_programs=args.max_programs)

    solver_rows = _load(args.solver_results)
    tasks = _load(args.tasks)

    # Solver-failed tasks only.
    failed = [r for r in solver_rows if float(r.get("best_reward", 0.0)) < 1.0]
    if args.limit:
        failed = failed[: args.limit]
    logger.info("%s: %d solver-failed tasks (of %d)", args.benchmark, len(failed), len(solver_rows))

    ckpt_path = args.out.replace(".jsonl", ".ckpt.json")
    done = set()
    if os.path.exists(ckpt_path):
        done = set(json.load(open(ckpt_path)).get("completed", []))
        logger.info("resuming: %d already done", len(done))

    ctrl = SeededDirectFeedbackController(
        model=args.model, base_url=args.base_url, api_key=args.api_key,
        k=args.k, max_tokens=args.max_tokens,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    lock = threading.Lock()

    def _write(row):
        with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            done.add(row["task_index"])
            json.dump({"completed": sorted(done)}, open(ckpt_path, "w"))

    pending = [r for r in failed if r.get("task_index") not in done]
    logger.info("%d pending", len(pending))

    def _one(srow):
        ti = srow["task_index"]
        task = tasks[ti]
        res = ctrl.solve_seeded(
            task_input=task, reward_fn=reward_fn, entry=task,
            seed_answer=srow.get("answer"),
            task_type=args.benchmark,
        )
        out = {
            "task_index": ti,
            "difficulty": _difficulty(args.benchmark, srow),
            "solver_seed_reward": res.get("seed_reward"),
            "seeded_best_reward": res.get("best_reward"),
            "seeded_solved": res.get("solved"),
            "attempts": res.get("cost_summary", {}).get("actual_attempts"),
            "token_usage": res.get("token_usage"),
            "answer": res.get("answer"),
        }
        _write(out)
        logger.info("task %s (diff=%s): seed=%.3f -> seeded=%.3f solved=%s",
                    ti, out["difficulty"], out["solver_seed_reward"] or 0.0,
                    out["seeded_best_reward"] or 0.0, out["seeded_solved"])

    if args.workers == 1:
        for r in pending:
            _one(r)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_one, r): r for r in pending}
            for fut in as_completed(futs):
                exc = fut.exception()
                if exc:
                    logger.error("task %s failed: %s", futs[fut].get("task_index"), exc)

    logger.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
