#!/usr/bin/env python3
"""Build a balanced demos file from LLM HPO agent episode logs.

A demo is one decision of the agent: the prompt it saw (search space, recent history,
incumbent), its reasoning, the configuration it proposed, and what that configuration
did when it was trained. ``success`` means the trial finished and improved on the best
val_bpb so far; everything else (no improvement, out of memory, divergence, other
errors) is a failure. The coding agent that induces the solver therefore sees both
reasoning that paid off and reasoning that did not.

Selection (reproducible via --seed):
  * only decisions the model made itself (trial 0 and random fallbacks are excluded);
  * equal numbers of successes and failures, as many as --n allows;
  * within each half, round-robin over strata so no slice dominates: failures cycle
    over the outcomes and, within an outcome, over the episode phases; successes cycle
    over the episode phases. Phase is the third of the episode the trial falls into
    (early / mid / late).

Output keys follow the other demos files where they apply (prompt, cot,
final_response, success, task_id) plus the HPO-specific outcome fields.

Usage:
    python -m openhands_agents.build_hpo_balanced_demos \\
        --episodes outputs/autoresearch_hpo_pilot_tasks_hpo_agent_qwen3.6_35b_a3b \\
        --n 100 --seed 42
    # -> demos/DEMOS_AUTORESEARCH_HPO_seed_42_<count>_examples_with_CoT.json
"""
import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.autoresearch_hpo import BENCHMARK, STATUS_OK  # noqa: E402

PHASES = ("early", "mid", "late")


def load_episode(path):
    """(header, trial rows) of an episode log, or None if the file is not one."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or rows[0].get("type") != "header" or rows[0].get("benchmark") != BENCHMARK:
        return None
    return rows[0], rows[1:]


def episode_demos(header, trials):
    """One candidate demo per decision the LLM made in this episode."""
    demos = []
    for index, trial in enumerate(trials):
        meta = trial.get("policy_meta") or {}
        if index == 0 or meta.get("source") != "llm":
            continue
        incumbent = trials[index - 1]["best_val_bpb"]
        if trial["status"] != STATUS_OK:
            outcome = trial["status"]
        elif incumbent is None or trial["val_bpb"] < incumbent:
            outcome = "improved"
        else:
            outcome = "no_improvement"
        reasoning = (meta.get("reasoning") or "").strip()
        response = (meta.get("response") or "").strip()
        demos.append({
            "task_id": header["task_id"],
            "step": trial["trial_id"],
            "prompt": meta.get("prompt", ""),
            "cot": "\n\n".join(part for part in (reasoning, response) if part),
            "final_response": json.dumps(trial["raw_suggestion"]),
            "success": outcome == "improved",
            "outcome": outcome,
            "config": trial["config"],
            "val_bpb": trial["val_bpb"],
            "incumbent_val_bpb": incumbent,
            "error": trial["error"],
            "wall_time_s": trial["wall_time_s"],
            "peak_vram_mb": trial["peak_vram_mb"],
            "phase": PHASES[min(2, index * 3 // len(trials))],
            "model": meta.get("model"),
            "objective": header["objective"]["kind"],
        })
    return demos


def interleave(items, keys, rng):
    """All ``items``, ordered so that every prefix is spread evenly over the strata.

    ``keys`` are the stratification levels, outermost first: the result cycles over
    the values of the first key and, within one value, over the values of the next.
    The order of the strata within a cycle and of the items within a stratum is random,
    so a prefix shorter than one cycle does not always favour the same strata.
    """
    if not keys:
        items = list(items)
        rng.shuffle(items)
        return items
    strata = defaultdict(list)
    for item in items:
        strata[keys[0](item)].append(item)
    names = sorted(strata)
    rng.shuffle(names)
    queues = [interleave(strata[name], keys[1:], rng) for name in names]
    ordered = []
    for rank in range(max(map(len, queues), default=0)):
        ordered.extend(queue[rank] for queue in queues if rank < len(queue))
    return ordered


def build_demos(candidates, n, seed):
    """Up to ``n`` demos, half successes and half failures (see the module docstring)."""
    rng = random.Random(seed)
    order = lambda d: (d["task_id"], d["step"])
    candidates = sorted(candidates, key=order)        # independent of the input order
    successes = [d for d in candidates if d["success"]]
    failures = [d for d in candidates if not d["success"]]
    per_side = min(n // 2, len(successes), len(failures))
    phase, outcome = (lambda d: d["phase"]), (lambda d: d["outcome"])
    picked = (interleave(successes, [phase], rng)[:per_side]
              + interleave(failures, [outcome, phase], rng)[:per_side])
    return sorted(picked, key=order)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Balanced demos from LLM HPO agent episode logs")
    parser.add_argument("--episodes", nargs="+", required=True,
                        help="Episode logs, or run directories containing them")
    parser.add_argument("--n", type=int, default=100, help="Maximum number of demos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None,
                        help="Output file (default: demos/DEMOS_AUTORESEARCH_HPO_seed_<seed>_"
                             "<count>_examples_with_CoT.json)")
    args = parser.parse_args(argv)

    paths = []
    for item in args.episodes:
        paths.extend(sorted(glob.glob(os.path.join(item, "*.jsonl"))) if os.path.isdir(item) else [item])
    candidates, kinds = [], set()
    for path in paths:
        episode = load_episode(path)
        if episode is None:      # not an episode log
            continue
        kinds.add(episode[0]["objective"]["kind"])
        candidates.extend(episode_demos(*episode))
    if len(kinds) > 1:
        parser.error(f"episodes mix objective kinds {sorted(kinds)}; build demos from one kind")

    demos = build_demos(candidates, args.n, args.seed)
    if not demos:
        parser.error(f"no balanced demos available: {len(candidates)} LLM decisions, "
                     f"{sum(d['success'] for d in candidates)} of them improvements")
    without_cot = sum(1 for d in demos if not d["cot"])
    out = args.out or (f"demos/DEMOS_AUTORESEARCH_HPO_seed_{args.seed}_{len(demos)}"
                       f"_examples_with_CoT.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(demos, f, indent=2)

    by_outcome = defaultdict(int)
    for demo in demos:
        by_outcome[demo["outcome"]] += 1
    print(f"{len(candidates)} LLM decisions in {len(paths)} file(s) -> {len(demos)} demos "
          f"({dict(sorted(by_outcome.items()))}) -> {out}")
    if without_cot:
        print(f"WARNING: {without_cot} demo(s) have no reasoning text", file=sys.stderr)
    if kinds == {"synthetic"}:
        print("NOTE: these demos come from the synthetic objective and say nothing about "
              "nanochat training.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
