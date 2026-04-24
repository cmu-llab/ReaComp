"""
openhands_agents/build_solver.py — entry point for the SolverBuilder agent.

Runs a single OpenHands conversation that reads DEMOS.json + the building prompt,
then writes SOLVER.py and SOLVER_ALGORITHM.md to --output-dir.

Usage:
    python -m openhands_agents.build_solver \
        --building-prompt building_prompts/SOLVER_BUILDING_PROMPT.md \
        --demos-path DEMOS.json \
        --rewards-dir rewards \
        --output-dir built_solvers/<run_name> \
        --sif-path /scratch/$USER/sif_images/sandbox.sif \
        --base-url http://localhost:8000/v1 \
        --model openai/Qwen/Qwen3-Coder-30B-A3B-Instruct
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="SolverBuilder — OpenHands agent that writes SOLVER.py")

    # Required
    parser.add_argument("--building-prompt", required=True,
                        help="Path to SOLVER_BUILDING_PROMPT.md")
    parser.add_argument("--demos-path", required=True,
                        help="Path to DEMOS.json")
    parser.add_argument("--rewards-dir", required=True,
                        help="Path to rewards/ directory (contains pbebench.py)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory where SOLVER.py and SOLVER_ALGORITHM.md will be written")

    # Sandbox
    parser.add_argument("--sif-path", default=os.environ.get("SANDBOX_SIF", ""),
                        help="Path to sandbox.sif (or set SANDBOX_SIF env var)")
    parser.add_argument("--sandbox-timeout", type=int, default=60)

    # LLM
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max agent steps in the conversation")

    # Debug
    parser.add_argument("--debug-dir", default="",
                        help="Directory for debug JSON (trajectory, summary)")

    args = parser.parse_args()

    if not args.sif_path:
        parser.error("--sif-path required (or set SANDBOX_SIF env var)")

    # Validate inputs
    for label, path in [
        ("building-prompt", args.building_prompt),
        ("demos-path", args.demos_path),
        ("rewards-dir", args.rewards_dir),
    ]:
        if not os.path.exists(path):
            parser.error(f"--{label} path does not exist: {path}")

    building_prompt = open(args.building_prompt).read()

    from .sandbox import ApptainerSandbox
    sandbox = ApptainerSandbox(sif_path=args.sif_path, timeout=args.sandbox_timeout)

    from .solver_builder.controller import SolverBuilderController
    controller = SolverBuilderController(
        base_url=args.base_url,
        model=args.model,
        sandbox=sandbox,
        demos_path=os.path.abspath(args.demos_path),
        rewards_dir=os.path.abspath(args.rewards_dir),
        output_dir=os.path.abspath(args.output_dir),
        api_key=args.api_key,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
    )

    logger.info("Starting SolverBuilder conversation...")
    logger.info("  building_prompt : %s", args.building_prompt)
    logger.info("  demos_path      : %s", args.demos_path)
    logger.info("  rewards_dir     : %s", args.rewards_dir)
    logger.info("  output_dir      : %s", args.output_dir)
    logger.info("  model           : %s", args.model)
    logger.info("  max_steps       : %d", args.max_steps)

    result = controller.build(building_prompt)

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        debug_path = os.path.join(args.debug_dir, "solver_builder_run.json")
        with open(debug_path, "w") as f:
            json.dump(
                {k: v for k, v in result.items() if k != "_trajectory"},
                f, indent=2, default=str,
            )
        trajectory_path = os.path.join(args.debug_dir, "solver_builder_trajectory.json")
        with open(trajectory_path, "w") as f:
            json.dump(result.get("_trajectory", []), f, indent=2, default=str)
        logger.info("Debug output written to %s", args.debug_dir)

    if result["success"]:
        logger.info("Success! Files written:")
        logger.info("  SOLVER.py          → %s", result["solver_path"])
        logger.info("  SOLVER_ALGORITHM.md → %s", result["algorithm_path"])
        logger.info("Summary: %s", result["summary"])
    else:
        logger.warning("Build incomplete. Check output_dir: %s", args.output_dir)
        missing = []
        if not result.get("solver_path"):
            missing.append("SOLVER.py")
        if not result.get("algorithm_path"):
            missing.append("SOLVER_ALGORITHM.md")
        logger.warning("Missing files: %s", missing)
        sys.exit(1)


if __name__ == "__main__":
    main()
