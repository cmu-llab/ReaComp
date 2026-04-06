"""
Entry point for the Symbolic Library Agent.

Usage:
    python main.py                               # run all built-in example tasks in batch
    python main.py --task 0                      # run a single built-in task by index
    python main.py --list                        # list built-in example tasks
    python main.py --stats                       # print library stats after a batch run
    python main.py --tasks-file tasks.jsonl      # run tasks from a JSON/JSONL file
    python main.py --output-file results.jsonl   # append each task result live to a JSONL file
    python main.py --tasks-file tasks.jsonl --output-file results.jsonl  # auto-resumes if crashed
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
from rewards import load_reward
from symbolic_agent import Controller
from symbolic_agent.models import Function
from symbolic_agent.baselines.trove import TroVEController
from symbolic_agent.baselines.regal import ReGALController

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

    # Reward loop summary (only present when solve_with_reward was used)
    if result.get("reward_history"):
        best = result.get("best_reward", 0.0)
        iters = len(result["reward_history"])
        print(f"Reward : best={best:.3f}  iters={iters}")
        for h in result["reward_history"]:
            print(f"  iter={h['iteration']}  r={h['reward']:.3f}  blame={h.get('blame','?')}  {h.get('message','')[:80]}")

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
        # Build a task_input dict the agents can work with.
        # Use an inclusion list so only safe NL fields are visible to agents —
        # oracle answer, metadata, and other dataset internals are never leaked.
        # The full record is preserved in "entry" for the reward function and output.
        _AGENT_KEYS = {"question", "prompt", "task"}
        task_input = {k: v for k, v in rec.items() if k in _AGENT_KEYS}
        tasks.append({
            "input": task_input,
            "type": rec.get("type", "symbolic"),
            "reward": rec.get("reward"),   # name of rewards/{name}.py, or None
            "entry": rec,                  # full original record for reward_fn
        })

    return tasks


def _append_task_output(result: dict, task_index: int, output_file: str) -> None:
    """
    Append one JSON line to {output_file} immediately after a task completes.

    Each line is a self-contained record combining:
      - response fields  (answer, explanation, confidence, execution_result)
      - trajectory fields (task_spec, trace, solution, library_snapshot, cost_summary)
      - agent_messages   (every LLM call's request + response, suitable for agentic training)
    """
    final = result.get("final_output", {})
    record = {
        "task_index": task_index,
        "task_type": result.get("task_type", ""),
        "original_prompt": result.get("original_prompt", ""),
        # trajectory
        "task_spec": result.get("task_spec"),
        "solved": result.get("solved", False),
        "steps_taken": result.get("steps", 0),
        "trace": result.get("trace", []),
        "solution": result.get("solution"),
        "library_snapshot": result.get("library_snapshot", []),
        "cost_summary": result.get("cost_summary", {}),
        # response
        "answer": final.get("answer"),
        "explanation": final.get("explanation"),
        "confidence": final.get("confidence"),
        "execution_result": final.get("execution_result"),
        "error": final.get("error"),
        # all component-agent LLM calls (request + response) for training data
        "agent_messages": result.get("agent_messages", []),
        # reward-loop fields (populated only when solve_with_reward is used)
        "reward_history": result.get("reward_history", []),
        "best_reward": result.get("best_reward"),
        "final_reward": result.get("final_reward"),
    }
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    logger.info("Appended task %d → %s", task_index, output_file)


def _save_checkpoint(controller: Controller, last_completed_index: int, checkpoint_file: str) -> None:
    """
    Overwrite {checkpoint_file} with the current controller state so a crashed run
    can be resumed.  Called after every successfully completed task.

    Stores:
      - last_completed_index : last task index that was written to the output file
      - library              : all Function objects (serialised via to_dict())
      - cost_tracker         : cumulative counters and log
    """
    ckpt = {
        "last_completed_index": last_completed_index,
        "library": [f.to_dict() for f in controller.library.functions],
        "cost_tracker": {
            "num_new_functions": controller.cost_tracker.num_new_functions,
            "total_function_length": controller.cost_tracker.total_function_length,
            "reuse_count": controller.cost_tracker.reuse_count,
            "task_loss": controller.cost_tracker.task_loss,
            "log": controller.cost_tracker.log,
        },
    }
    try:
        Path(checkpoint_file).write_text(
            json.dumps(ckpt, indent=2, default=str), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not write checkpoint %s: %s", checkpoint_file, exc)


def _save_trove_checkpoint(controller, last_completed_index: int, checkpoint_file: str) -> None:
    """
    Save TroVE controller state so a crashed run can be resumed.
    Stores the toolbox and _n_processed counter alongside last_completed_index.
    """
    ckpt = {
        "last_completed_index": last_completed_index,
        "n_processed": controller._n_processed,
        "toolbox": controller.toolbox.to_dict(),
    }
    try:
        Path(checkpoint_file).write_text(
            json.dumps(ckpt, indent=2, default=str), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("Could not write TroVE checkpoint %s: %s", checkpoint_file, exc)


def _load_trove_checkpoint(controller, checkpoint_file: str) -> int:
    """
    Restore TroVE controller state from {checkpoint_file}.
    Returns the next task index to run (last_completed_index + 1).
    Returns 0 on any error.
    """
    from symbolic_agent.baselines.trove.toolbox import TroVEToolbox
    try:
        ckpt = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))
        controller.toolbox = TroVEToolbox.from_dict(ckpt.get("toolbox", {}))
        controller._n_processed = ckpt.get("n_processed", 0)
        last_index = ckpt.get("last_completed_index", -1)
        next_index = last_index + 1
        logger.info(
            "TroVE checkpoint loaded: last_completed=%d, toolbox=%d fns, n_processed=%d → resuming from task %d",
            last_index, len(controller.toolbox), controller._n_processed, next_index,
        )
        return next_index
    except Exception as exc:
        logger.warning("Could not load TroVE checkpoint %s: %s — starting from scratch.", checkpoint_file, exc)
        return 0


def _load_checkpoint(controller: Controller, checkpoint_file: str) -> int:
    """
    Restore controller state from {checkpoint_file}.
    Returns the next task index to run (last_completed_index + 1).
    Returns 0 and leaves controller untouched on any error.
    """
    try:
        ckpt = json.loads(Path(checkpoint_file).read_text(encoding="utf-8"))

        # Restore library
        controller.library.functions = []
        for fd in ckpt.get("library", []):
            controller.library.functions.append(Function(
                name=fd["name"],
                code=fd["code"],
                description=fd.get("description", ""),
                domain=fd.get("domain", "general"),
                input_types=fd.get("input_types", []),
                output_type=fd.get("output_type", ""),
                usage_count=fd.get("usage_count", 0),
                creation_cost=fd.get("creation_cost", 0.0),
            ))

        # Restore cost tracker
        ct = ckpt.get("cost_tracker", {})
        controller.cost_tracker.num_new_functions = ct.get("num_new_functions", 0)
        controller.cost_tracker.total_function_length = ct.get("total_function_length", 0)
        controller.cost_tracker.reuse_count = ct.get("reuse_count", 0)
        controller.cost_tracker.task_loss = ct.get("task_loss", 0.0)
        controller.cost_tracker.log = ct.get("log", [])

        # Recompute embeddings for restored functions if semantic retrieval is active.
        controller.library.recompute_embeddings()

        last_index = ckpt.get("last_completed_index", -1)
        next_index = last_index + 1
        logger.info(
            "Checkpoint loaded: last_completed=%d, library=%d functions → resuming from task %d",
            last_index, len(controller.library), next_index,
        )
        return next_index
    except Exception as exc:
        logger.warning("Could not load checkpoint %s: %s — starting from scratch.", checkpoint_file, exc)
        return 0


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
    parser.add_argument(
        "--framework",
        default="ssl_bcr",
        choices=["ssl_bcr", "trove", "regal"],
        help="Solution framework to use. 'ssl_bcr': symbolic library agent (default). "
             "'trove': TroVE online function induction baseline. "
             "'regal': ReGAL offline refactoring baseline. (default: ssl_bcr)",
    )
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Model name to use")
    parser.add_argument("--budget", type=float, default=15.0, help="Step budget per task")
    parser.add_argument("--lam", type=float, default=0.3, help="λ: regularisation weight for library cost in Objective = TaskLoss + λ·TotalCost (default: 0.3)")
    parser.add_argument(
        "--redundancy-mode",
        default="ast_jaccard",
        choices=["ast_jaccard", "edit_distance"],
        help=(
            "Algorithm used to compute the redundancy penalty between library functions. "
            "'ast_jaccard': max(Jaccard on AST node-type sets, Jaccard on callee-name sets) — "
            "fast, captures structural shape and shared dependencies. "
            "'edit_distance': 1 − normalised edit distance on DFS-linearised AST node sequences — "
            "more precise, O(m·n) per pair. (default: ast_jaccard)"
        ),
    )
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
        "--output-file",
        default=None,
        metavar="FILE",
        help="Path to a JSONL file where each completed task is appended immediately as "
             "a single JSON line.  Each record contains the response, full trajectory, "
             "and all component-agent LLM messages (request + response) for training data.",
    )
    parser.add_argument(
        "--max-reward-iters",
        type=int,
        default=3,
        metavar="N",
        help="Maximum reward-feedback iterations per task when the task record has a 'reward' field. "
             "The agent retries until it achieves reward=1.0 or exhausts N iterations. (default: 3)",
    )
    parser.add_argument(
        "--default-reward",
        default=None,
        metavar="NAME",
        help="Reward module name to use for tasks that do not have a 'reward' field "
             "(e.g. 'reasoning_gym'). Enables the reward loop for the whole run.",
    )
    parser.add_argument(
        "--semantic-retrieval",
        action="store_true",
        default=False,
        help="Use sentence_transformers for library retrieval (cosine similarity on "
             "name+description embeddings) instead of token Jaccard. Requires: "
             "pip install sentence_transformers. (default: off)",
    )
    parser.add_argument(
        "--semantic-model",
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help="Sentence transformer model for --semantic-retrieval. "
             "(default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        metavar="DIR",
        help="Directory to write per-call LLM debug logs (request, raw response including "
             "chain-of-thought, extracted tool calls).  Each run creates a timestamped "
             "subdirectory so logs from multiple runs are never overwritten.",
    )
    # TroVE-specific flags
    parser.add_argument(
        "--trove-k",
        type=int,
        default=5,
        metavar="K",
        help="[TroVE] Number of samples per mode (IMPORT/CREATE/SKIP). Paper default: 5. "
             "Set to 1 for fast/cheap runs. (default: 5)",
    )
    parser.add_argument(
        "--trove-trim-every",
        type=int,
        default=500,
        metavar="N",
        help="[TroVE] Trim low-frequency toolbox functions every N tasks. "
             "Paper default: 500. Set to 9999 to disable for small datasets. (default: 500)",
    )
    # ReGAL-specific flags
    parser.add_argument(
        "--regal-train-file",
        default=None,
        metavar="FILE",
        help="[ReGAL] Path to JSONL training file with 'program' key in each record. "
             "Triggers offline training before the test run. (default: None — test-only mode)",
    )
    parser.add_argument(
        "--regal-retrieval",
        default="sentence_transformers",
        choices=["sentence_transformers", "chromadb"],
        help="[ReGAL] Vector retrieval backend for CodeBank/DemoBank. "
             "'sentence_transformers': local cosine similarity (default). "
             "'chromadb': chromadb PersistentClient with sentence_transformer embeddings.",
    )
    parser.add_argument(
        "--regal-embedding-model",
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help="[ReGAL] Sentence transformer model for query embedding (clustering + retrieval). "
             "(default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--regal-codebank-dir",
        default=None,
        metavar="DIR",
        help="[ReGAL] Directory to save/load trained CodeBank and DemoBank. "
             "If set and the directory contains regal_codebank.json, it is loaded before the test run. "
             "After training (--regal-train-file), the banks are saved here.",
    )
    parser.add_argument(
        "--regal-batch-size",
        type=int,
        default=4,
        metavar="N",
        help="[ReGAL] Examples per refactoring batch during training. Paper: 3–5. (default: 4)",
    )
    parser.add_argument(
        "--regal-edit-codebank",
        action="store_true",
        default=False,
        help="[ReGAL] Enable Stage 3a editCodeBank: periodically prompt the LLM to improve "
             "failing helper functions. Off by default. (default: off)",
    )
    parser.add_argument(
        "--regal-edit-every",
        type=int,
        default=5,
        metavar="N",
        help="[ReGAL] Run editCodeBank every N training batches. (default: 5)",
    )
    parser.add_argument(
        "--regal-prune-every",
        type=int,
        default=5,
        metavar="N",
        help="[ReGAL] Run pruneCodeBank every N training batches. (default: 5)",
    )
    parser.add_argument(
        "--regal-icl-budget",
        type=int,
        default=10,
        metavar="N",
        help="[ReGAL] Total number of ICL examples in the test-time agent prompt. (default: 10)",
    )
    parser.add_argument(
        "--regal-icl-split",
        type=float,
        default=0.5,
        metavar="R",
        help="[ReGAL] Fraction of ICL budget drawn from DemoBank (refactored demos) vs "
             "primitive training examples. Paper default: 0.5. (default: 0.5)",
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

    if args.framework == "trove":
        controller = TroVEController(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            debug_dir=args.debug_dir,
            k=args.trove_k,
            trim_every=args.trove_trim_every,
        )
        logger.info("Framework: TroVE (k=%d, trim_every=%d)", args.trove_k, args.trove_trim_every)
    elif args.framework == "regal":
        from pathlib import Path as _Path
        controller = ReGALController(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            debug_dir=args.debug_dir,
            retrieval=args.regal_retrieval,
            embedding_model=args.regal_embedding_model,
            chroma_path=(
                str(_Path(args.regal_codebank_dir) / "chroma")
                if args.regal_codebank_dir and args.regal_retrieval == "chromadb"
                else None
            ),
            edit_codebank=args.regal_edit_codebank,
            edit_every=args.regal_edit_every,
            prune_every=args.regal_prune_every,
            icl_budget=args.regal_icl_budget,
            icl_split=args.regal_icl_split,
        )
        # Load pre-trained banks if available
        if args.regal_codebank_dir:
            cb_path = str(_Path(args.regal_codebank_dir) / "regal_codebank.json")
            db_path = str(_Path(args.regal_codebank_dir) / "regal_demobank.json")
            if _Path(cb_path).exists() and _Path(db_path).exists():
                controller.load(cb_path, db_path)
                logger.info("ReGAL: loaded pre-trained banks from %s", args.regal_codebank_dir)
        # Offline training
        if args.regal_train_file:
            train_tasks = _load_tasks_file(args.regal_train_file)
            logger.info("ReGAL: training on %d examples (batch_size=%d)", len(train_tasks), args.regal_batch_size)
            controller.train(train_tasks, batch_size=args.regal_batch_size)
            if args.regal_codebank_dir:
                _Path(args.regal_codebank_dir).mkdir(parents=True, exist_ok=True)
                controller.save(
                    str(_Path(args.regal_codebank_dir) / "regal_codebank.json"),
                    str(_Path(args.regal_codebank_dir) / "regal_demobank.json"),
                )
        logger.info(
            "Framework: ReGAL (retrieval=%s, codebank=%d fns, demobank=%d demos)",
            args.regal_retrieval, len(controller.codebank), len(controller.demobank),
        )
    else:
        controller = Controller(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url,
            debug_dir=args.debug_dir,
            lam=args.lam,
            redundancy_mode=args.redundancy_mode,
            semantic_retrieval=args.semantic_retrieval,
            semantic_model=args.semantic_model,
        )
        logger.info("Framework: ssl_bcr")

    if args.tasks_file:
        tasks = _load_tasks_file(args.tasks_file)
        logger.info("Loaded %d tasks from %s", len(tasks), args.tasks_file)

        # Checkpoint file lives alongside the output file: results.jsonl → results.ckpt.json
        ckpt_file = (
            str(Path(args.output_file).with_suffix(".ckpt.json"))
            if args.output_file else None
        )

        # Auto-resume: if both output file and checkpoint exist, restore state and skip done tasks.
        # ssl_bcr and trove both support per-task checkpointing.
        # ReGAL is trained offline before the test run and has no per-task state to checkpoint.
        start_index = 0
        if args.framework in ("ssl_bcr", "trove"):
            if (
                ckpt_file
                and Path(ckpt_file).exists()
                and args.output_file
                and Path(args.output_file).exists()
            ):
                if args.framework == "trove":
                    start_index = _load_trove_checkpoint(controller, ckpt_file)
                else:
                    start_index = _load_checkpoint(controller, ckpt_file)
            elif ckpt_file and Path(ckpt_file).exists():
                logger.warning(
                    "Checkpoint found but output file %s is missing — ignoring checkpoint and starting fresh.",
                    args.output_file,
                )

        for i, task in enumerate(tasks):
            if i < start_index:
                continue
            task_input = task.get("input", task)
            task_type = task.get("type", "symbolic")
            reward_name = task.get("reward") or args.default_reward
            logger.info("--- Task %d/%d ---", i + 1, len(tasks))

            if reward_name:
                reward_fn = load_reward(reward_name)
                result = controller.solve_with_reward(
                    task_input=task_input,
                    task_type=task_type,
                    budget=args.budget,
                    reward_fn=reward_fn,
                    entry=task.get("entry", {}),
                    max_reward_iters=args.max_reward_iters,
                )
            else:
                result = controller.solve(task_input, task_type, budget=args.budget)
            _print_result(result, i)
            if args.output_file:
                _append_task_output(result, i, args.output_file)
            if ckpt_file:
                if args.framework == "ssl_bcr":
                    _save_checkpoint(controller, i, ckpt_file)
                elif args.framework == "trove":
                    _save_trove_checkpoint(controller, i, ckpt_file)
            # Library size: ssl_bcr uses controller.library, TroVE uses controller.toolbox,
            # ReGAL uses controller.codebank
            if args.framework == "trove":
                lib_size = len(controller.toolbox)
            elif args.framework == "regal":
                lib_size = len(controller.codebank)
            else:
                lib_size = len(controller.library)
            logger.info("Library size after task %d: %d functions", i + 1, lib_size)
        if args.stats:
            _print_library_stats(controller)
    elif args.task is not None:
        if args.task >= len(TASKS):
            print(f"Task index {args.task} out of range (0–{len(TASKS)-1}).", file=sys.stderr)
            sys.exit(1)
        task = TASKS[args.task]
        result = controller.solve(task["input"], task_type=task["type"], budget=args.budget)
        _print_result(result, args.task)
        if args.output_file:
            _append_task_output(result, args.task, args.output_file)
        if args.stats:
            _print_library_stats(controller)
    else:
        # Batch run: all built-in tasks share the same library
        logger.info("Running %d built-in tasks in batch mode", len(TASKS))
        for i, task in enumerate(TASKS):
            result = controller.solve(task["input"], task_type=task["type"], budget=args.budget)
            _print_result(result, i)
            if args.output_file:
                _append_task_output(result, i, args.output_file)
        if args.stats:
            _print_library_stats(controller)

    print("\nDone.")


if __name__ == "__main__":
    main()
