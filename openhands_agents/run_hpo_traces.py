#!/usr/bin/env python3
"""Run the LLM HPO agent on autoresearch HPO tasks and record its reasoning traces.

One episode per task record: trial 0 trains the default configuration, then the model
proposes one configuration per trial until the task's training-time budget is spent
(see ``openhands_agents/hpo_agent/controller.py`` for the agent and
``rewards/autoresearch_hpo.py`` for the episode protocol). Serves two purposes:

  * the LLM baseline of the domain (the paper's "Karpathy Agent (14 HPs)"; pass
    ``--prompt-style paper --max-tokens 2048 --parse-retries 0`` for the upstream
    settings), and
  * the reasoning traces for solver induction: every trial row of the episode log
    holds the prompt, the model's reasoning and answer, and what the configuration
    did. ``openhands_agents/build_hpo_balanced_demos.py`` turns the logs into a demos
    file.

Output:
    <run-dir>/<task_id>.jsonl         episode log (resumable: re-run the same command)
    <run-dir>/<task_id>.result.json   summary of the finished episode

Usage (model served by vLLM on port 8000; real tasks train on the local GPU):
    python -m openhands_agents.run_hpo_traces \\
        --tasks-file data/autoresearch_hpo/pilot_tasks.jsonl \\
        --base-url http://localhost:8000/v1 --model openai/Qwen/Qwen3.6-35B-A3B \\
        --run-dir outputs/autoresearch_hpo_pilot_tasks_hpo_agent_qwen3.6_35b_a3b

Dry run without a GPU (synthetic objective, not benchmark numbers):
    python -m openhands_agents.run_hpo_traces \\
        --tasks-file data/autoresearch_hpo/synthetic_tasks.jsonl ...
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openhands_agents.hpo_agent.controller import (  # noqa: E402
    PROMPT_STYLES, HPOAgentController, agent_policy_name,
)
from openhands_agents.hpo_agent.runner import (  # noqa: E402
    add_task_args, load_tasks, objective_options, run_tasks_cli,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="LLM HPO agent on autoresearch HPO tasks (baseline + reasoning traces)")
    add_task_args(parser)
    parser.add_argument("--run-dir", required=True,
                        help="Output directory, e.g. outputs/autoresearch_hpo_<tasks>_hpo_agent_<model>")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="openai/Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="cot")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--parse-retries", type=int, default=2,
                        help="Extra requests when an answer has no usable configuration, "
                             "before falling back to a random one")
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    args = parser.parse_args(argv)

    try:
        entries = load_tasks(args.tasks_file, args.task_id)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    def make_policy(entry):
        return HPOAgentController(
            base_url=args.base_url, model=args.model, api_key=args.api_key,
            prompt_style=args.prompt_style, temperature=args.temperature,
            max_tokens=args.max_tokens, parse_retries=args.parse_retries,
            request_timeout=args.request_timeout)

    policy_name = agent_policy_name(args.model, args.prompt_style, args.temperature,
                                    args.max_tokens, args.parse_retries)
    return run_tasks_cli(entries, make_policy, args.run_dir, policy_name,
                         objective_options(args))


if __name__ == "__main__":
    sys.exit(main())
