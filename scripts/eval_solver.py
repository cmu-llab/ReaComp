"""
Evaluate a SOLVER.py file on PBEBench datasets.

Usage:
    python scripts/eval_solver.py --solver built_libraries/claude_code/Thu_Apr_23_807_PM/SOLVER.py
    python scripts/eval_solver.py --solver SOLVER.py --dataset lite
    python scripts/eval_solver.py --solver SOLVER.py --dataset hard
    python scripts/eval_solver.py --solver SOLVER.py --limit 100
    python scripts/eval_solver.py --output-dir evals/solver_results/
    python scripts/eval_solver.py --output-dir evals/solver_results/ --workers 8
"""
import argparse
import json
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
            if "SOLVER.py" in filenames:
                paths.append(os.path.join(dirpath, "SOLVER.py"))
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
    return mod.solve_pbe, path


DATASETS = {
    "lite": {
        "path": os.path.join(REPO_ROOT, "data", "pbebench", "lite_tasks_full_og.jsonl"),
        "label": "PBEBench-Lite",
        "max_programs": 5,
        "max_pred_len": 3,
        "max_transform_len": 3,
    },
    "hard": {
        "path": os.path.join(REPO_ROOT, "data", "pbebench", "tasks_full_og.jsonl"),
        "label": "PBEBench-Hard",
        "max_programs": 20,
        "max_pred_len": 3,
        "max_transform_len": 3,
    },
}


# ---------------------------------------------------------------------------
# Worker function (must be top-level for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_task(args):
    """Called in a worker process. Loads the solver fresh per-process."""
    import inspect
    task_index, rec, solver_path, max_programs, max_pred_len, max_transform_len = args
    # Each worker imports the solver independently (no shared state)
    solve_pbe, _ = _load_solver(solver_path)
    examples = list(zip(rec["inputs"], rec["outputs"]))
    # Pass only kwargs the solver actually accepts
    sig_params = set(inspect.signature(solve_pbe).parameters)
    kwargs = {"max_programs": max_programs}
    if "max_pred_len" in sig_params:
        kwargs["max_pred_len"] = max_pred_len
    if "max_transform_len" in sig_params:
        kwargs["max_transform_len"] = max_transform_len
    t0 = time.time()
    result = solve_pbe(examples, **kwargs)
    return task_index, rec, result, round(time.time() - t0, 3)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(dataset_key, solver_path, output_dir=None, limit=None, workers=1):
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

    # Only run tasks not yet completed
    task_args = [
        (i, rec, solver_path,
         cfg["max_programs"], cfg["max_pred_len"], cfg["max_transform_len"])
        for i, rec in enumerate(records)
        if i not in completed_rows
    ]

    task_args = [
        (i, rec, solver_path,
         cfg["max_programs"], cfg["max_pred_len"], cfg["max_transform_len"])
        for i, rec in enumerate(records)
    ]

    def _write_row(task_index, rec, result, elapsed_task):
        reward = result["score"]
        row = {
            # quick_eval-compatible fields
            "task_index": task_index,
            "solved": result["success"],
            "answer": result["program"],
            "best_reward": reward,
            "reward_history": [{"iteration": 0, "reward": reward}],
            "token_usage": {"input": 0, "output": 0, "reasoning": 0},
            "cost_summary": {"elapsed_s": elapsed_task},
            # extra context fields (ignored by quick_eval)
            "dataset": cfg["label"],
            "cascade_length": rec.get("cascade_length"),
            "bfcc_string": rec.get("bfcc_string"),
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
    all_solved = solved + sum(int(r["success"]) for r in completed_rows.values())
    all_score = total_score + sum(r["score"] for r in completed_rows.values())

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
        description="Evaluate a SOLVER.py on PBEBench datasets."
    )
    default_solver_abs = _default_solver_path()
    default_solver_rel = os.path.relpath(default_solver_abs, REPO_ROOT)
    parser.add_argument(
        "--solver",
        default=default_solver_abs,
        help=f"Path to SOLVER.py to evaluate (default: {default_solver_rel})",
    )
    parser.add_argument(
        "--dataset", choices=["lite", "hard", "both"], default="both",
        help="Which dataset to evaluate (default: both)",
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
    args = parser.parse_args()

    _, solver_path = _load_solver(args.solver)  # validate path; workers load it themselves

    keys = ["lite", "hard"] if args.dataset == "both" else [args.dataset]
    summaries = []

    for key in keys:
        cfg = DATASETS[key]
        print(f"\n{'='*60}")
        print(f"Solver  : {solver_path}")
        print(f"Dataset : {cfg['label']}")
        print(f"File    : {cfg['path']}")
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
        summary = evaluate(
            key, solver_path,
            output_dir=args.output_dir,
            limit=args.limit,
            workers=args.workers,
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
