"""
Entry point for the Symbolic Library Agent.

Usage:
    python main.py                  # run all example tasks in batch
    python main.py --task 0         # run a single task by index
    python main.py --list           # list available example tasks
    python main.py --stats          # print library stats after a batch run
"""

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

from examples.tasks import TASKS
from symbolic_agent import Controller

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def _print_result(result: dict, task_index: int) -> None:
    print(f"\n{'='*60}")
    print(f"Task {task_index}  type={result.get('task_type', '?')}")
    print(f"{'='*60}")
    print(f"Solved : {result['solved']}")

    final = result.get("final_output", {})
    if "error" in final:
        print(f"Error  : {final['error']}")
    else:
        print(f"Answer : {final.get('answer', 'N/A')}")
        print(f"Explain: {final.get('explanation', '')}")
        print(f"Conf.  : {final.get('confidence', '?')}")
        if final.get("execution_result") is not None:
            print(f"Exec   : {final['execution_result']}")

    cost = result.get("cost_summary", {})
    print(f"\nCost summary:")
    for k, v in cost.items():
        print(f"  {k:28s}: {v}")

    trace = result.get("trace", [])
    print(f"\nTrace ({len(trace)} steps):")
    for t in trace:
        agent = t.get("agent", "?")
        action = t.get("action") or t.get("actions", [])
        print(f"  step={t.get('step',0)}  agent={agent}  {action}")


def _print_library_stats(controller: Controller) -> None:
    stats = controller.library_stats()
    print(f"\n{'='*60}")
    print(f"Library stats  ({stats['num_functions']} functions)")
    print(f"{'='*60}")
    for f in stats["functions"]:
        print(
            f"  {f['name']:30s}  uses={f['usage_count']:2d}  "
            f"cost={f['creation_cost']:.3f}  useful={f['usefulness']:.2f}"
        )
    print("\nCost summary:")
    for k, v in stats["cost_summary"].items():
        print(f"  {k:28s}: {v}")
    print("\nCost log (last 20 entries):")
    for entry in stats["cost_log"][-20:]:
        print(f"  {entry}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Symbolic Library Agent")
    parser.add_argument("--task", type=int, default=None, help="Run a single task by index")
    parser.add_argument("--list", action="store_true", help="List example tasks")
    parser.add_argument("--stats", action="store_true", help="Print library stats after run")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Claude model to use")
    parser.add_argument("--budget", type=float, default=15.0, help="Step budget per task")
    args = parser.parse_args()

    if args.list:
        print("Available example tasks:")
        for i, t in enumerate(TASKS):
            desc = t["input"].get("description", str(t["input"])[:60])
            print(f"  [{i}]  type={t['type']:20s}  {desc}")
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.  Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    controller = Controller(api_key=api_key, model=args.model)

    if args.task is not None:
        if args.task >= len(TASKS):
            print(f"Task index {args.task} out of range (0–{len(TASKS)-1}).", file=sys.stderr)
            sys.exit(1)
        task = TASKS[args.task]
        result = controller.solve(task["input"], task_type=task["type"], budget=args.budget)
        _print_result(result, args.task)
        if args.stats:
            _print_library_stats(controller)
    else:
        # Batch run: all tasks share the same library
        logger.info("Running %d tasks in batch mode", len(TASKS))
        results = controller.solve_batch(TASKS)
        for i, result in enumerate(results):
            _print_result(result, i)
        if args.stats:
            _print_library_stats(controller)

    print("\nDone.")


if __name__ == "__main__":
    main()
