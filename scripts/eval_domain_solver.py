"""
Evaluate an induced solver on any registered domain.

    python scripts/eval_domain_solver.py --domain pbebench --split lite \
        --solver built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py \
        --output-dir evals/solver_results/claude_code/Thu_Apr_23_807_PM --workers 8

    python scripts/eval_domain_solver.py --domain envsetup --split setupbench \
        --solver built_solvers/envsetup_baseline/SOLVER_ENVSETUP.py --workers 2 --task-timeout 1800

Writes <output-dir>/<split>.jsonl (one quick_eval-compatible row per task, resumable)
and <output-dir>/<split>_summary.json.  Row schema matches scripts/eval_solver.py:
    task_index, solved, answer, best_reward, reward_history, token_usage, cost_summary,
    dataset, plus domain-specific context columns.

Each task runs in a child process so a hung solver can be killed after --task-timeout
seconds; domains that hold external resources (containers) clean them up via
Domain.cleanup_after_timeout.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from domains import get_domain, list_domains  # noqa: E402
from domains.base import SplitSpec, wait_for_child_result  # noqa: E402

_FAILED = {"success": False, "program": None, "score": 0.0, "extras": {}}


def _parse_params(items):
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _solve_in_child(domain_name, split_dict, solver_path, rec, queue):
    try:
        dom = get_domain(domain_name)
        split = SplitSpec(**split_dict)
        solve_fn = dom.load_solver(solver_path)
        out = dom.solve(solve_fn, rec, split)
        queue.put({"success": out.success, "program": out.program, "score": out.score, "extras": out.extras})
    except Exception as exc:  # noqa: BLE001
        import traceback
        queue.put({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def _run_task(args):
    task_index, rec, domain_name, split_dict, solver_path, task_timeout = args
    queue = multiprocessing.Queue(maxsize=1)
    proc = multiprocessing.Process(target=_solve_in_child,
                                   args=(domain_name, split_dict, solver_path, rec, queue), daemon=True)
    t0 = time.time()
    proc.start()
    result, status = wait_for_child_result(proc, queue, task_timeout)
    if proc.is_alive():
        proc.kill()
    proc.join()
    elapsed = round(time.time() - t0, 3)
    if status == "timeout":
        try:
            get_domain(domain_name).cleanup_after_timeout(rec, SplitSpec(**split_dict))
        except Exception:  # noqa: BLE001
            pass
        result = dict(_FAILED, extras={"error": f"timeout after {task_timeout}s"})
    elif status == "died":
        result = dict(_FAILED, extras={"error": f"solver process exited with code {proc.exitcode} without a result"})
    else:
        error = result.pop("error", None)
        result.pop("traceback", None)
        if error:
            result = dict(_FAILED, extras={"error": error})
    return task_index, rec, result, elapsed


def evaluate(domain_name, split_name, solver_path, output_dir=None, limit=None, workers=1,
             task_timeout=None, params=None, tag=None):
    dom = get_domain(domain_name)
    split = dom.split(split_name)
    if params:
        split = SplitSpec(split.name, split.path, split.label, {**split.params, **params})
    split_dict = {"name": split.name, "path": split.path, "label": split.label, "params": split.params}
    records = dom.load_tasks(split_name, limit=limit)
    n = len(records)
    solver_path = os.path.abspath(solver_path)
    stem = tag or split.name

    completed = {}
    out_fh = None
    out_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{stem}.jsonl")
        if os.path.isfile(out_path):
            with open(out_path) as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                        completed[row["task_index"]] = row
                    except (json.JSONDecodeError, KeyError):
                        pass
            if completed:
                print(f"  Resuming: {len(completed)} tasks already done.")
        out_fh = open(out_path, "a" if completed else "w")
        print(f"  Writing results to: {out_path}")

    timeout = task_timeout if task_timeout is not None else float("inf")
    task_args = [(rec["task_index"], rec, domain_name, split_dict, solver_path, timeout)
                 for rec in records if rec["task_index"] not in completed]

    solved = 0
    total = 0.0
    done = len(completed)
    t0 = time.time()

    def write_row(task_index, rec, result, elapsed):
        row = {
            "task_index": task_index,
            "solved": bool(result["success"]),
            "answer": dom.normalise_answer(result["program"]),
            "best_reward": float(result["score"]),
            "reward_history": [{"iteration": 0, "reward": float(result["score"])}],
            "token_usage": {"input": 0, "output": 0, "reasoning": 0},
            "cost_summary": {"elapsed_s": elapsed},
            "dataset": split.label,
            "domain": domain_name,
        }
        row.update(dom.row_extras(rec))
        row.update(result.get("extras") or {})
        if out_fh:
            out_fh.write(json.dumps(row, default=str) + "\n")
            out_fh.flush()

    def progress():
        if done % 25 == 0 or done == n:
            print(f"  [{done:4d}/{n}] solved={solved} pass_rate={solved/max(done,1):.1%} "
                  f"mean_score={total/max(done,1):.3f} elapsed={time.time()-t0:.0f}s")

    try:
        if workers <= 1:
            for a in task_args:
                ti, rec, result, elapsed = _run_task(a)
                solved += int(result["success"]); total += result["score"]
                write_row(ti, rec, result, elapsed); done += 1; progress()
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_run_task, a): a[0] for a in task_args}
                for fut in as_completed(futs):
                    ti, rec, result, elapsed = fut.result()
                    solved += int(result["success"]); total += result["score"]
                    write_row(ti, rec, result, elapsed); done += 1; progress()
    finally:
        if out_fh:
            out_fh.close()

    all_solved = solved + sum(int(r.get("solved", False)) for r in completed.values())
    all_score = total + sum(float(r.get("best_reward", 0.0)) for r in completed.values())
    summary = {"domain": domain_name, "dataset": split.label, "split": split.name, "n": n,
               "solved": all_solved, "pass_rate": all_solved / max(n, 1), "mean_score": all_score / max(n, 1),
               "elapsed_s": time.time() - t0, "solver": solver_path, "params": split.params}
    if output_dir:
        with open(os.path.join(output_dir, f"{stem}_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True, choices=list_domains())
    ap.add_argument("--split", required=True)
    ap.add_argument("--solver", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--task-timeout", type=float, default=None, help="seconds per task (default: none)")
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                    help="override a split parameter, e.g. --param max_programs=50")
    ap.add_argument("--tag", default=None, help="output file stem (default: split name)")
    args = ap.parse_args()

    summary = evaluate(args.domain, args.split, args.solver, args.output_dir, args.limit, args.workers,
                       args.task_timeout, _parse_params(args.param), args.tag)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
