"""
Evaluate a SOLVER.py file on PBEBench or SLR-Bench datasets.

Usage:
    python scripts/eval_solver.py --solver built_libraries/claude_code/Thu_Apr_23_807_PM/SOLVER.py
    python scripts/eval_solver.py --solver SOLVER.py --dataset lite
    python scripts/eval_solver.py --solver SOLVER.py --dataset hard
    python scripts/eval_solver.py --solver SOLVER.py --dataset slr
    python scripts/eval_solver.py --solver SOLVER.py --limit 100
    python scripts/eval_solver.py --output-dir evals/solver_results/
    python scripts/eval_solver.py --output-dir evals/solver_results/ --workers 8

    # SLR solver (expose solve_slr as the entry point):
    python scripts/eval_solver.py --solver built_solvers/claude_code/Sat_Apr_25_251_AM/SOLVER_SLR.py --dataset slr
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)  # needed by SOLVER.py to import rewards.*


def _default_solver_path():
    paths = []
    for search_root in ("built_solvers", "built_libraries"):
        d = os.path.join(REPO_ROOT, search_root)
        for dirpath, _, filenames in os.walk(d):
            for name in ("SOLVER.py", "SOLVER_SLR.py"):
                if name in filenames:
                    paths.append(os.path.join(dirpath, name))
    if not paths:
        raise FileNotFoundError("No SOLVER.py found under built_solvers/ or built_libraries/")
    return max(paths, key=os.path.getmtime)


def _load_solver(path):
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SOLVER not found: {path}")
    spec = importlib.util.spec_from_file_location("SOLVER", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Support both PBEBench (solve_pbe) and SLR-Bench (solve_slr) solvers
    if hasattr(mod, "solve_slr"):
        return mod.solve_slr, path, "slr"
    return mod.solve_pbe, path, "pbe"


DATASETS = {
    "lite": {
        "path": os.path.join(REPO_ROOT, "data", "pbebench", "lite_tasks_full_og.jsonl"),
        "label": "PBEBench-Lite",
        "kind": "pbe",
        "max_programs": 5,
        "max_pred_len": 3,
        "max_transform_len": 3,
    },
    "hard": {
        "path": os.path.join(REPO_ROOT, "data", "pbebench", "tasks_full_og.jsonl"),
        "label": "PBEBench-Hard",
        "kind": "pbe",
        "max_programs": 20,
        "max_pred_len": 3,
        "max_transform_len": 3,
    },
    "slr": {
        "path": os.path.join(REPO_ROOT, "data", "slr_bench", "v1_All_full.jsonl"),
        "label": "SLR-Bench",
        "kind": "slr",
    },
}


# ---------------------------------------------------------------------------
# Worker function (must be top-level for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _solve_in_child(solver_path, cfg, rec, queue):
    """Runs inside a child process spawned by _run_task for timeout enforcement."""
    import inspect
    solve_fn, _, _ = _load_solver(solver_path)
    try:
        if cfg["kind"] == "slr":
            from rewards.slr_bench import parse_prompt_examples
            parsed = parse_prompt_examples(rec["prompt"])
            examples = list(zip(parsed["inputs"], parsed["outputs"]))
            result = solve_fn(examples)
            score = result.get("score", 1.0 if result["success"] else 0.0)
            queue.put({"success": result["success"], "program": result.get("program"), "score": score})
        else:
            examples = list(zip(rec["inputs"], rec["outputs"]))
            sig_params = set(inspect.signature(solve_fn).parameters)
            kwargs = {"max_programs": cfg["max_programs"]}
            if "max_pred_len" in sig_params:
                kwargs["max_pred_len"] = cfg["max_pred_len"]
            if "max_transform_len" in sig_params:
                kwargs["max_transform_len"] = cfg["max_transform_len"]
            result = solve_fn(examples, **kwargs)
            score = result.get("score", max(result["scores"]) if result.get("scores") else (1.0 if result["success"] else 0.0))
            queue.put({"success": result["success"], "program": result.get("program"), "score": score})
    except Exception as e:
        queue.put({"error": str(e)})


_FAILED = {"success": False, "program": None, "score": 0.0}


def _run_task(args):
    """Called in a worker process. Loads the solver fresh per-process."""
    task_index, rec, solver_path, cfg, task_timeout = args

    queue = multiprocessing.Queue(maxsize=1)
    proc = multiprocessing.Process(
        target=_solve_in_child,
        args=(solver_path, cfg, rec, queue),
        daemon=True,
    )
    t0 = time.time()
    proc.start()
    proc.join(timeout=task_timeout if task_timeout != float("inf") else None)
    elapsed = round(time.time() - t0, 3)

    if proc.is_alive():
        proc.kill()
        proc.join()
        return task_index, rec, _FAILED, elapsed

    result = queue.get_nowait() if not queue.empty() else _FAILED
    if "error" in result:
        result = _FAILED
    return task_index, rec, result, elapsed


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(dataset_key, solver_path, output_dir=None, limit=None, workers=1, task_timeout=None):
    cfg = DATASETS[dataset_key]
    with open(cfg["path"]) as fh:
        records = [json.loads(line) for line in fh]

    if limit is not None:
        records = records[:limit]

    n = len(records)
    t0 = time.time()

    # Resume: load any already-completed results from an existing JSONL
    completed_rows = {}  # task_index -> row
    out_path = None
    out_fh = None
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{dataset_key}.jsonl")
        if os.path.isfile(out_path):
            with open(out_path) as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        completed_rows[row["task_index"]] = row
                    except (json.JSONDecodeError, KeyError):
                        pass
            if completed_rows:
                print(f"  Resuming: {len(completed_rows)} tasks already done, skipping them.")
        out_fh = open(out_path, "a" if completed_rows else "w")
        print(f"  Writing results to: {out_path}")

    # Running totals start at zero; completed_rows are merged into them after the run
    solved = 0
    total_score = 0.0

    effective_timeout = task_timeout if task_timeout is not None else float("inf")
    task_args = [
        (i, rec, solver_path, cfg, effective_timeout)
        for i, rec in enumerate(records)
        if i not in completed_rows
    ]

    def _normalise_answer(program):
        """Convert [[A,B],...] pairs to ["replace('A','B')",...] strings if needed."""
        if not isinstance(program, list):
            return program
        if program and isinstance(program[0], (list, tuple)):
            return [f"replace('{a}', '{b}')" for a, b in program]
        return program

    def _write_row(task_index, rec, result, elapsed_task):
        reward = result["score"]
        row = {
            # quick_eval-compatible fields
            "task_index": task_index,
            "solved": result["success"],
            "answer": _normalise_answer(result["program"]),
            "best_reward": reward,
            "reward_history": [{"iteration": 0, "reward": reward}],
            "token_usage": {"input": 0, "output": 0, "reasoning": 0},
            "cost_summary": {"elapsed_s": elapsed_task},
            # extra context fields
            "dataset": cfg["label"],
            # PBEBench-specific (absent for SLR)
            "cascade_length": rec.get("cascade_length"),
            "bfcc_string": rec.get("bfcc_string"),
            # SLR-Bench-specific (absent for PBEBench)
            "rule_complexity": rec.get("rule complexity"),
            "curriculum_level": rec.get("curriculum level"),
            "curriculum_tier": rec.get("curriculum tier"),
            "ground_truth_rule": rec.get("ground-truth rule"),
        }
        if out_fh is not None:
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()

    done = len(completed_rows)  # start counter from already-finished tasks
    try:
        if workers <= 1:
            for args in task_args:
                task_index, rec, result, elapsed_task = _run_task(args)
                solved += int(result["success"])
                total_score += result["score"]
                _write_row(task_index, rec, result, elapsed_task)
                done += 1
                if done % 50 == 0 or done == n:
                    print(
                        f"  [{done:4d}/{n}] solved={solved} "
                        f"pass_rate={solved/done:.1%} "
                        f"mean_score={total_score/done:.3f} "
                        f"elapsed={time.time()-t0:.0f}s"
                    )
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_task, a): a[0] for a in task_args}
                for fut in as_completed(futures):
                    task_index, rec, result, elapsed_task = fut.result()
                    solved += int(result["success"])
                    total_score += result["score"]
                    _write_row(task_index, rec, result, elapsed_task)
                    done += 1
                    if done % 50 == 0 or done == n:
                        print(
                            f"  [{done:4d}/{n}] solved={solved} "
                            f"pass_rate={solved/done:.1%} "
                            f"mean_score={total_score/done:.3f} "
                            f"elapsed={time.time()-t0:.0f}s"
                        )
    finally:
        if out_fh is not None:
            out_fh.close()

    # Merge prior rows into final totals for summary
    all_solved = solved + sum(int(r.get("success", r.get("solved", False))) for r in completed_rows.values())
    all_score = total_score + sum(r.get("score", r.get("best_reward", 0.0)) for r in completed_rows.values())

    elapsed = time.time() - t0
    summary = {
        "dataset": cfg["label"],
        "n": n,
        "solved": all_solved,
        "pass_rate": all_solved / n,
        "mean_score": all_score / n,
        "elapsed_s": elapsed,
        "avg_s_per_task": elapsed / n,
    }

    if output_dir is not None:
        summary_path = os.path.join(output_dir, f"{dataset_key}_summary.json")
        with open(summary_path, "w") as fh:
            json.dump({**summary, "solver": solver_path}, fh, indent=2)
        print(f"  Summary written to:  {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a SOLVER.py on PBEBench or SLR-Bench datasets."
    )
    default_solver_abs = _default_solver_path()
    default_solver_rel = os.path.relpath(default_solver_abs, REPO_ROOT)
    parser.add_argument(
        "--solver",
        default=default_solver_abs,
        help=f"Path to SOLVER.py to evaluate (default: {default_solver_rel})",
    )
    parser.add_argument(
        "--dataset", choices=["lite", "hard", "both", "slr"], default="both",
        help="Which dataset to evaluate (default: both PBEBench splits)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N tasks (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Directory to write per-task JSONL and summary JSON files. "
            "Files: <dataset>.jsonl and <dataset>_summary.json. "
            "(default: no output files)"
        ),
    )
    parser.add_argument(
        "--task-timeout", type=int, default=None,
        help="Per-task timeout in seconds. Tasks exceeding this are marked failed. (default: no limit)",
    )
    args = parser.parse_args()

    _, solver_path, _ = _load_solver(args.solver)  # validate path; workers load it themselves

    keys = ["lite", "hard"] if args.dataset == "both" else [args.dataset]
    summaries = []

    for key in keys:
        cfg = DATASETS[key]
        print(f"\n{'='*60}")
        print(f"Solver  : {solver_path}")
        print(f"Dataset : {cfg['label']}")
        print(f"File    : {cfg['path']}")
        if cfg["kind"] == "pbe":
            print(
                f"Limits  : max_programs={cfg['max_programs']}  "
                f"max_pred_len={cfg['max_pred_len']}  "
                f"max_transform_len={cfg['max_transform_len']}"
            )
        if args.limit:
            print(f"Limit   : first {args.limit} tasks")
        if args.workers > 1:
            print(f"Workers : {args.workers}")
        print()
        if args.task_timeout:
            print(f"Timeout : {args.task_timeout}s per task")
        summary = evaluate(
            key, solver_path,
            output_dir=args.output_dir,
            limit=args.limit,
            workers=args.workers,
            task_timeout=args.task_timeout,
        )
        summaries.append(summary)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"Solver  : {solver_path}")
    print(f"{'Dataset':<20}  {'N':>6}  {'Solved':>6}  {'Pass%':>7}  {'MeanScore':>9}  {'AvgTime':>8}")
    print("-" * 65)
    for s in summaries:
        print(
            f"{s['dataset']:<20}  {s['n']:>6}  {s['solved']:>6}  "
            f"{s['pass_rate']:>6.1%}  {s['mean_score']:>9.3f}  "
            f"{s['avg_s_per_task']:>7.2f}s"
        )


if __name__ == "__main__":
    main()
