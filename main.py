"""
Entry point for the Symbolic Library Agent.

Usage:
    python main.py                             # run all built-in example tasks in batch
    python main.py --task 0                    # run a single built-in task by index
    python main.py --list                      # list built-in example tasks
    python main.py --stats                     # print library stats after a batch run
    python main.py --tasks-file tasks.jsonl    # run tasks from a JSON/JSONL file
    python main.py --output-dir results/       # write per-task output to a directory
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict

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


def _load_tasks_file(path: str) -> List[Dict]:
    """
    Load tasks from a JSON or JSONL file.

    Supported formats:
    - JSON:  a single array of task objects  ([ {...}, {...} ])
    - JSONL: one JSON object per line        ({ ... }\\n{ ... }\\n...)

    Each record must have a "prompt" key.  Optional keys:
    - "type"  : task category label (default: "symbolic")
    - any other fields are passed through to the agents as context
    """
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = p.read_text(encoding="utf-8").strip()
    records: List[Dict] = []

    # Try JSON array first, then fall back to JSONL
    if text.startswith("["):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"ERROR: could not parse {path} as JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR: line {lineno} in {path} is not valid JSON: {e}", file=sys.stderr)
                sys.exit(1)

    if not records:
        print(f"ERROR: no records found in {path}", file=sys.stderr)
        sys.exit(1)

    # Validate and normalise
    tasks = []
    for i, rec in enumerate(records):
        if "prompt" not in rec:
            print(f"ERROR: record {i} in {path} is missing required key 'prompt'", file=sys.stderr)
            sys.exit(1)
        # Build a task_input dict the agents can work with
        task_input = {k: v for k, v in rec.items() if k != "type"}
        task_input.setdefault("description", rec["prompt"])
        tasks.append({"input": task_input, "type": rec.get("type", "symbolic")})

    return tasks


def _save_task_output(result: dict, task_index: int, output_dir: str) -> None:
    """
    Write two files to {output_dir}/task_{task_index:04d}/:

    trajectory.json — full agent trace for downstream analysis:
        original_prompt, task_spec, solved, steps_taken,
        trace (per-step agent actions), solution code, library snapshot, cost summary.

    response.json — final reporting-agent output for evaluation:
        original_prompt, solved, answer, explanation, confidence, execution_result.
    """
    task_dir = Path(output_dir) / f"task_{task_index:04d}"
    task_dir.mkdir(parents=True, exist_ok=True)

    # ---- trajectory.json ------------------------------------------------
    trajectory = {
        "task_index": task_index,
        "task_type": result.get("task_type", ""),
        "original_prompt": result.get("original_prompt", ""),
        "task_spec": result.get("task_spec"),
        "solved": result.get("solved", False),
        "steps_taken": result.get("steps", 0),
        "trace": result.get("trace", []),
        "solution": result.get("solution"),
        "library_snapshot": result.get("library_snapshot", []),
        "cost_summary": result.get("cost_summary", {}),
    }
    (task_dir / "trajectory.json").write_text(
        json.dumps(trajectory, indent=2, default=str), encoding="utf-8"
    )

    # ---- response.json --------------------------------------------------
    final = result.get("final_output", {})
    response = {
        "task_index": task_index,
        "task_type": result.get("task_type", ""),
        "original_prompt": result.get("original_prompt", ""),
        "solved": result.get("solved", False),
        "answer": final.get("answer"),
        "explanation": final.get("explanation"),
        "confidence": final.get("confidence"),
        "execution_result": final.get("execution_result"),
        "error": final.get("error"),
    }
    (task_dir / "response.json").write_text(
        json.dumps(response, indent=2, default=str), encoding="utf-8"
    )

    logger.info("Saved task %d output → %s", task_index, task_dir)


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
    parser.add_argument("--task", type=int, default=None, help="Run a single built-in task by index")
    parser.add_argument("--list", action="store_true", help="List built-in example tasks")
    parser.add_argument("--stats", action="store_true", help="Print library stats after run")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Model name to use")
    parser.add_argument("--budget", type=float, default=15.0, help="Step budget per task")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL for local vLLM (e.g. http://localhost:8000/v1). "
             "When set, ANTHROPIC_API_KEY is not required.",
    )
    parser.add_argument(
        "--tasks-file",
        default=None,
        metavar="FILE",
        help="Path to a JSON or JSONL file of tasks.  Each record must have a 'prompt' key. "
             "Optional keys: 'type' (task category, default 'symbolic'). "
             "Runs all records in batch mode, sharing one library across tasks.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory to write per-task output files.  For each task a subdirectory "
             "task_NNNN/ is created containing trajectory.json (full agent trace) and "
             "response.json (final answer for evaluation).",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        metavar="DIR",
        help="Directory to write per-call LLM debug logs (request, raw response including "
             "chain-of-thought, extracted tool calls).  Each run creates a timestamped "
             "subdirectory so logs from multiple runs are never overwritten.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available built-in example tasks:")
        for i, t in enumerate(TASKS):
            desc = t["input"].get("description", str(t["input"])[:60])
            print(f"  [{i}]  type={t['type']:20s}  {desc}")
        return

    if args.base_url:
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
        logger.info("Using vLLM backend at %s with model %s", args.base_url, args.model)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY is not set.  Add it to .env or export it.", file=sys.stderr)
            sys.exit(1)

    controller = Controller(api_key=api_key, model=args.model, base_url=args.base_url, debug_dir=args.debug_dir)

    if args.tasks_file:
        tasks = _load_tasks_file(args.tasks_file)
        logger.info("Loaded %d tasks from %s", len(tasks), args.tasks_file)
        results = controller.solve_batch(tasks)
        for i, result in enumerate(results):
            _print_result(result, i)
            if args.output_dir:
                _save_task_output(result, i, args.output_dir)
        if args.stats:
            _print_library_stats(controller)
    elif args.task is not None:
        if args.task >= len(TASKS):
            print(f"Task index {args.task} out of range (0–{len(TASKS)-1}).", file=sys.stderr)
            sys.exit(1)
        task = TASKS[args.task]
        result = controller.solve(task["input"], task_type=task["type"], budget=args.budget)
        _print_result(result, args.task)
        if args.output_dir:
            _save_task_output(result, args.task, args.output_dir)
        if args.stats:
            _print_library_stats(controller)
    else:
        # Batch run: all built-in tasks share the same library
        logger.info("Running %d built-in tasks in batch mode", len(TASKS))
        results = controller.solve_batch(TASKS)
        for i, result in enumerate(results):
            _print_result(result, i)
            if args.output_dir:
                _save_task_output(result, i, args.output_dir)
        if args.stats:
            _print_library_stats(controller)

    print("\nDone.")


if __name__ == "__main__":
    main()
