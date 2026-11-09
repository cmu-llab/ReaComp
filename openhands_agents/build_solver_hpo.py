"""
openhands_agents/build_solver_hpo.py — entry point for the autoresearch HPO SolverBuilder.

Parallel of build_solver.py / build_solver_listfn.py. Runs a single OpenHands
conversation that reads the HPO demos (reasoning traces of the LLM HPO agent, built by
build_hpo_balanced_demos.py) and the HPO building prompt, then writes SOLVER_HPO.py and
SOLVER_HPO_ALGORITHM.md to --output-dir. Afterwards the written solver is run on the
synthetic practice tasks inside the sandbox as a sanity check.

Usage:
    python -m openhands_agents.build_solver_hpo \
        --building-prompt building_prompts/SOLVER_BUILDING_PROMPT_HPO.md \
        --demos-path demos/DEMOS_AUTORESEARCH_HPO_seed_42_100_examples_with_CoT.json \
        --rewards-dir rewards \
        --output-dir built_solvers/qwen3.6_35b_a3b/<timestamp>_hpo \
        --sif-path /scratch/$USER/sif_images/sandbox.sif \
        --base-url http://localhost:8000/v1 \
        --model openai/Qwen/Qwen3.6-35B-A3B

Exit status: 0 when both files were written and the sanity check passed, 1 otherwise.
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Autoresearch HPO SolverBuilder — OpenHands agent that writes SOLVER_HPO.py")

    parser.add_argument("--building-prompt",
                        default="building_prompts/SOLVER_BUILDING_PROMPT_HPO.md",
                        help="Path to SOLVER_BUILDING_PROMPT_HPO.md")
    parser.add_argument("--demos-path", required=True,
                        help="Demos file written by openhands_agents/build_hpo_balanced_demos.py")
    parser.add_argument("--rewards-dir", default="rewards",
                        help="Path to rewards/ directory (contains autoresearch_hpo.py)")
    parser.add_argument("--smoke-tasks", default="data/autoresearch_hpo/synthetic_tasks.jsonl",
                        help="Synthetic task file mounted into the sandbox for practice episodes")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where SOLVER_HPO.py and SOLVER_HPO_ALGORITHM.md are written")

    # Sandbox
    parser.add_argument("--sif-path", default=os.environ.get("SANDBOX_SIF", ""),
                        help="Path to sandbox.sif (or set SANDBOX_SIF env var)")
    parser.add_argument("--sandbox-timeout", type=int, default=120,
                        help="Seconds one execute call of the agent may run")

    # LLM
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="openai/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max agent steps in the conversation")
    parser.add_argument("--debug-dir", default="",
                        help="Directory for debug JSON (trajectory, summary)")

    args = parser.parse_args(argv)

    if not args.sif_path:
        parser.error("--sif-path required (or set SANDBOX_SIF env var)")

    for label, path in [
        ("building-prompt", args.building_prompt),
        ("demos-path", args.demos_path),
        ("rewards-dir", args.rewards_dir),
        ("smoke-tasks", args.smoke_tasks),
    ]:
        if not os.path.exists(path):
            parser.error(f"--{label} path does not exist: {path}")

    from .solver_builder.hpo_tools import ALGORITHM_FILENAME, ALLOWED_FILENAMES, SOLVER_FILENAME
    # Success is judged by the files being present afterwards, so files left over from
    # an earlier run must not be there.
    stale = [name for name in ALLOWED_FILENAMES
             if os.path.exists(os.path.join(args.output_dir, name))]
    if stale:
        parser.error(f"--output-dir {args.output_dir} already contains {stale}; "
                     f"use a fresh directory")

    with open(args.building_prompt, encoding="utf-8") as f:
        building_prompt = f.read()

    from .sandbox import ApptainerSandbox
    sandbox = ApptainerSandbox(sif_path=args.sif_path, timeout=args.sandbox_timeout)

    from .solver_builder.hpo_controller import HPOSolverBuilderController
    controller = HPOSolverBuilderController(
        base_url=args.base_url,
        model=args.model,
        sandbox=sandbox,
        demos_path=os.path.abspath(args.demos_path),
        rewards_dir=os.path.abspath(args.rewards_dir),
        output_dir=os.path.abspath(args.output_dir),
        api_key=args.api_key,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        smoke_tasks_path=os.path.abspath(args.smoke_tasks),
    )

    logger.info("Starting autoresearch HPO SolverBuilder conversation...")
    logger.info("  building_prompt : %s", args.building_prompt)
    logger.info("  demos_path      : %s", args.demos_path)
    logger.info("  rewards_dir     : %s", args.rewards_dir)
    logger.info("  smoke_tasks     : %s", args.smoke_tasks)
    logger.info("  output_dir      : %s", args.output_dir)
    logger.info("  model           : %s", args.model)
    logger.info("  max_steps       : %d", args.max_steps)

    result = controller.build(building_prompt)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        debug_path = os.path.join(args.debug_dir, "solver_builder_hpo_run.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in result.items() if k != "_trajectory"},
                f, indent=2, default=str,
            )
        trajectory_path = os.path.join(args.debug_dir, "solver_builder_hpo_trajectory.json")
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(result.get("_trajectory", []), f, indent=2, default=str)
        logger.info("Debug output written to %s", args.debug_dir)

    if not result["success"]:
        logger.warning("Build incomplete. Check output_dir: %s", args.output_dir)
        missing = []
        if not result.get("solver_path"):
            missing.append(SOLVER_FILENAME)
        if not result.get("algorithm_path"):
            missing.append(ALGORITHM_FILENAME)
        logger.warning("Missing files: %s", missing)
        return 1

    logger.info("Files written:")
    logger.info("  %s           → %s", SOLVER_FILENAME, result["solver_path"])
    logger.info("  %s → %s", ALGORITHM_FILENAME, result["algorithm_path"])
    logger.info("Summary: %s", result["summary"])

    smoke = result["smoke_test"]
    for task in smoke["tasks"]:
        logger.info("  practice %s: %s, %d trials, %s, val_bpb %s → %s",
                    task["task_id"], task["stop_reason"], task["n_trials"],
                    {k: v for k, v in task["status_counts"].items() if v},
                    task["baseline_val_bpb"], task["best_val_bpb"])
    if not smoke["ok"]:
        errors = [smoke["error"]] if smoke["error"] else [
            f"{task['task_id']}: {task['first_invalid_error']}"
            for task in smoke["tasks"] if task["first_invalid_error"]]
        logger.warning("Sanity check on the synthetic practice tasks FAILED: %s",
                       " | ".join(errors) or "no practice tasks")
        return 1
    logger.info("Sanity check on the synthetic practice tasks passed (synthetic objective: "
                "this says the solver runs, not how good it is on nanochat).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
