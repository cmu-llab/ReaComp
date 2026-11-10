#!/usr/bin/env python3
"""Evaluate an induced SOLVER_HPO.py, or a classical HPO baseline, on autoresearch HPO tasks.

One episode per task record, through the same episode loop as the LLM agent
(``rewards/autoresearch_hpo.py: run_episode``): trial 0 trains the default
configuration, then the policy proposes one configuration per trial until the task's
training-time budget is spent. The score of an episode is the lowest val_bpb of any
trial, computed from the trials the harness ran.

Policies:
    --solver PATH      a SOLVER_HPO.py defining solve_hpo(task, history); every decision
                       runs in a fresh process with a time limit
    --policy NAME      random | tpe | cmaes (tpe and cmaes need: pip install optuna cmaes)

Output (resumable: re-run the same command):
    <output-dir>/<task_id>.jsonl         episode log, one row per trial
    <output-dir>/<task_id>.result.json   summary of the finished episode

Usage (from the project root; real tasks train on the local GPU):
    python scripts/eval_hpo_solver.py \\
        --solver built_solvers/claude_code/<timestamp>/SOLVER_HPO.py \\
        --tasks-file data/autoresearch_hpo/pilot_tasks.jsonl \\
        --output-dir evals/solver_results/claude_code/<timestamp>/autoresearch_hpo_pilot_tasks

    python scripts/eval_hpo_solver.py --policy tpe \\
        --tasks-file data/autoresearch_hpo/pilot_tasks.jsonl \\
        --output-dir outputs/autoresearch_hpo_pilot_tasks_tpe

    # one seed per GPU, sharing an output directory
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_hpo_solver.py --solver ... \\
        --tasks-file data/autoresearch_hpo/tasks_full.jsonl \\
        --task-id autoresearch_hpo_24h_seed0 --output-dir ...

Dry run without a GPU (synthetic objective, not benchmark numbers):
    python scripts/eval_hpo_solver.py --policy random \\
        --tasks-file data/autoresearch_hpo/synthetic_tasks.jsonl \\
        --output-dir outputs/autoresearch_hpo_synthetic_tasks_random
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openhands_agents.hpo_agent.baselines import BASELINES, make_baseline  # noqa: E402
from openhands_agents.hpo_agent.runner import (  # noqa: E402
    add_task_args, load_tasks, objective_options, run_tasks_cli,
)
from openhands_agents.hpo_agent.solver_policy import (  # noqa: E402
    DEFAULT_DECISION_TIMEOUT_S, SolverPolicy,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a SOLVER_HPO.py or a classical baseline on autoresearch HPO tasks")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--solver", help="Path to a SOLVER_HPO.py")
    which.add_argument("--policy", choices=BASELINES, help="Classical baseline")
    add_task_args(parser)
    parser.add_argument("--output-dir", required=True,
                        help="Directory for the episode logs and result files")
    parser.add_argument("--decision-timeout", type=float, default=DEFAULT_DECISION_TIMEOUT_S,
                        help="Seconds one solver decision may take before it counts as "
                             "an invalid suggestion (--solver only)")
    args = parser.parse_args(argv)

    try:
        entries = load_tasks(args.tasks_file, args.task_id)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.solver:
        try:
            policy = SolverPolicy(args.solver, decision_timeout_s=args.decision_timeout)
        except OSError as exc:
            parser.error(str(exc))
        policy_name = policy.name

        def make_policy(entry):
            return policy
    else:
        policy_name = args.policy
        try:
            make_baseline(args.policy, entries[0])      # fail now if Optuna is missing
        except ImportError as exc:
            parser.error(str(exc))

        def make_policy(entry):
            return make_baseline(args.policy, entry)

    print(f"policy: {policy_name}\ntasks : {args.tasks_file} ({len(entries)})\n"
          f"output: {args.output_dir}", flush=True)
    return run_tasks_cli(entries, make_policy, args.output_dir, policy_name,
                         objective_options(args))


if __name__ == "__main__":
    sys.exit(main())
