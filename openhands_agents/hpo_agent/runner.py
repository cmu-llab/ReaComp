"""Run a policy over autoresearch HPO tasks: one episode per task record.

Shared by the trace runner (LLM agent), the solver evaluation and the classical
baselines, so all of them go through the same episode loop, the same objective
construction and the same output layout:

    <run_dir>/<task_id>.jsonl         episode log (header row + one row per trial)
    <run_dir>/<task_id>.result.json   summary of the finished episode

Re-running the same command resumes unfinished episodes. Every task has its own files,
so the tasks of one run directory can be split over several processes (one per GPU,
selected with ``--task-id``) without them overwriting each other;
``scripts/summarize_hpo_results.py`` collects the result files.

Real tasks train on whatever GPU the training environment sees. When that is not the
GPU the task record states, the run prints a warning, and its results carry the
device and a label that keeps them apart from the benchmark's in every summary.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from typing import Any, Callable, Dict, List, Optional

from rewards.autoresearch_hpo import (
    STATUS_OK, BaselineFailed, EpisodeInterrupted, EpisodeLogMismatch, ObjectiveUnavailable,
    make_objective, run_episode,
)

RESULT_SUFFIX = ".result.json"

SYNTHETIC_BANNER = ("NOTE: synthetic objective — these numbers exercise the pipeline and "
                    "are NOT benchmark results.")


def add_task_args(parser) -> None:
    """Command-line options every runner shares."""
    parser.add_argument("--tasks-file", default="data/autoresearch_hpo/pilot_tasks.jsonl")
    parser.add_argument("--task-id", action="append", default=None,
                        help="Run only this task (repeatable; default: every task in the file)")
    parser.add_argument("--autoresearch-dir", default=None,
                        help="Checkout of the pinned autoresearch repo (default: "
                             "$AUTORESEARCH_DIR or original_baseline_repos/autoresearch)")
    parser.add_argument("--train-command", default=None,
                        help="Launcher train.py is run with (default: "
                             "$AUTORESEARCH_TRAIN_COMMAND or 'uv run python')")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="Run even if train.py / prepare.py differ from the pinned "
                             "sources (results are labelled 'unpinned sources') or the "
                             "GPU cannot be identified ('hardware not verified'). Not "
                             "needed for a GPU other than the task's: that is accepted, "
                             "recorded and labelled")


def load_tasks(path: str, task_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    if task_ids:
        known = {entry["task_id"] for entry in entries}
        missing = [task_id for task_id in task_ids if task_id not in known]
        if missing:
            raise ValueError(f"task id(s) {missing} not in {path}; available: {sorted(known)}")
        entries = [entry for entry in entries if entry["task_id"] in task_ids]
    if not entries:
        raise ValueError(f"no tasks in {path}")
    return entries


def objective_options(args) -> Dict[str, Any]:
    """``make_objective`` keyword arguments from the options of ``add_task_args``."""
    return {
        "autoresearch_dir": args.autoresearch_dir,
        "run_command": shlex.split(args.train_command) if args.train_command else None,
        "verify_pin": not args.allow_unpinned,
    }


def _progress_printer(entry: Dict[str, Any]) -> Callable[[Dict[str, Any]], None]:
    budget_h = entry["budget"]["time_budget_s"] / 3600.0

    def on_trial(row: Dict[str, Any]) -> None:
        outcome = f"val_bpb={row['val_bpb']:.6f}" if row["status"] == STATUS_OK else str(row["error"])
        best = row["best_val_bpb"]
        best_text = f"{best:.6f}" if best is not None else "-"
        print(f"[{entry['task_id']}] trial {row['trial_id']:>4} {row['status']:<8} {outcome} | "
              f"best {best_text} | {row['cumulative_time_s'] / 3600.0:.2f}/{budget_h:g} h",
              flush=True)

    return on_trial


def summarize(entry: Dict[str, Any], episode: Dict[str, Any], policy_name: str,
              objective, log_path: str) -> Dict[str, Any]:
    """The result record of a finished episode."""
    usage = {"input": 0, "output": 0, "reasoning": 0}
    sources: Dict[str, int] = {}
    for trial in episode["trials"]:
        meta = trial.get("policy_meta") or {}
        # Policy metadata is whatever the policy returned: count only what has the
        # expected shape.
        reported = meta.get("token_usage")
        for key in usage:
            count = reported.get(key) if isinstance(reported, dict) else None
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                usage[key] += count
        source = meta.get("source")
        if isinstance(source, str):
            sources[source] = sources.get(source, 0) + 1
    score = dict(episode["score"])
    score.pop("curve")
    return {
        "task_id": entry["task_id"], "task": entry["task"], "seed": entry["seed"],
        "policy": policy_name, "objective": objective.describe(),
        "time_budget_s": entry["budget"]["time_budget_s"],
        "stop_reason": episode["stop_reason"], **score,
        "suggestion_sources": sources,
        "policy_time_s": round(sum(t["policy_time_s"] for t in episode["trials"]), 3),
        "token_usage": usage,
        "episode_log": log_path,
    }


def result_path(run_dir: str, task_id: str) -> str:
    return os.path.join(run_dir, f"{task_id}{RESULT_SUFFIX}")


def _write_json_atomic(path: str, value: Any) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def run_tasks(entries: List[Dict[str, Any]], make_policy: Callable[[Dict[str, Any]], Callable],
              run_dir: str, policy_name: str,
              objective_kwargs: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Run one episode per task and return the result records (also written to disk).

    ``make_policy(entry)`` builds a fresh policy for each task. ``EpisodeInterrupted``
    propagates; the tasks finished before it keep their result files.
    """
    os.makedirs(run_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        objective = make_objective(entry, **(objective_kwargs or {}))
        if objective.kind == "synthetic":
            print(SYNTHETIC_BANNER, flush=True)
        if objective.hardware_warning:
            print(f"WARNING: {objective.hardware_warning}", flush=True)
        log_path = os.path.join(run_dir, f"{entry['task_id']}.jsonl")
        episode = run_episode(make_policy(entry), entry, objective=objective,
                              log_path=log_path, policy_name=policy_name,
                              on_trial=_progress_printer(entry))
        row = summarize(entry, episode, policy_name, objective, log_path)
        _write_json_atomic(result_path(run_dir, entry["task_id"]), row)
        rows.append(row)
        best = row["best_val_bpb"]
        print(f"[{entry['task_id']}] done: {row['stop_reason']}, {row['n_trials']} trials, "
              f"best val_bpb {best if best is None else format(best, '.6f')}, "
              f"failure rate {row['failure_rate']:.1%}", flush=True)
    print(f"wrote {len(rows)} result file(s) to {run_dir}", flush=True)
    return rows


def run_tasks_cli(entries: List[Dict[str, Any]], make_policy: Callable[[Dict[str, Any]], Callable],
                  run_dir: str, policy_name: str,
                  objective_kwargs: Optional[Dict[str, Any]] = None) -> int:
    """``run_tasks`` for a command-line entry point: returns the process exit status.

    The failures a user can act on (no checkout, an environment that does not train
    the default configuration, an unreachable LLM server, an output directory that
    holds the log of another run) are reported as one line instead of a traceback.
    """
    try:
        run_tasks(entries, make_policy, run_dir, policy_name, objective_kwargs)
    except EpisodeInterrupted as exc:
        print(f"interrupted: {exc}\nRe-run the same command to resume.", file=sys.stderr)
        return 1
    except (ObjectiveUnavailable, BaselineFailed, EpisodeLogMismatch) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
