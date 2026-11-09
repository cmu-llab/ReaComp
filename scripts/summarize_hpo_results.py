#!/usr/bin/env python3
"""Summarise autoresearch HPO runs: best val_bpb per policy, mean ± std over runs.

Reads the ``<task_id>.result.json`` files written by ``scripts/eval_hpo_solver.py`` and
``openhands_agents/run_hpo_traces.py`` (searched recursively under every path given)
and prints one row per (objective, hardware, time budget, policy), in the format of
the paper's results table: mean ± sample standard deviation of the best val_bpb over
the runs of the group (the seeds of a task file).

Rows are only comparable within one block (same objective, hardware and budget).
Results measured on the synthetic objective, on unpinned sources or on a GPU other
than the one the task record states are labelled as such and never share a block
with benchmark results; runs on different GPUs never share a block either.

Usage (from the project root):
    python scripts/summarize_hpo_results.py outputs/autoresearch_hpo_tasks_full_* \\
        evals/solver_results/claude_code/<timestamp>/autoresearch_hpo_tasks_full
    python scripts/summarize_hpo_results.py outputs/ --json outputs/autoresearch_hpo_summary.json
"""
import argparse
import glob
import json
import math
import os
import sys

RESULT_GLOB = "*.result.json"
BENCHMARK_KIND = "nanochat"
OTHER_HARDWARE = "hardware differs from the task"
UNKNOWN_FILE = "<not read from a file>"    # rows passed to summarize() directly


def find_results(paths):
    """Result files under ``paths`` (files are taken as they are), without duplicates."""
    found = []
    for path in paths:
        if os.path.isdir(path):
            found.extend(sorted(glob.glob(os.path.join(path, "**", RESULT_GLOB), recursive=True)))
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    unique = {}
    for path in found:
        unique.setdefault(os.path.realpath(path), path)
    return list(unique.values())


def load_results(files):
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            row = json.load(f)
        row["_file"] = path
        rows.append(row)
    return rows


def objective_label(objective):
    """What a result was measured on; anything but 'nanochat' is not the benchmark."""
    kind = objective.get("kind")
    if kind != BENCHMARK_KIND:
        return f"{kind} (NOT benchmark)"
    caveats = []
    if not objective.get("pinned"):
        caveats.append("unpinned sources")
    if objective.get("hardware_matches") is False and objective.get("device"):
        caveats.append(OTHER_HARDWARE)
    elif objective.get("hardware_matches") is not True:
        caveats.append("hardware not verified")
    return kind if not caveats else f"{kind} ({', '.join(caveats)})"


def device_label(objective):
    device = objective.get("device")
    if objective.get("kind") != BENCHMARK_KIND:
        return "-"
    if not device:
        return "unknown GPU"
    text = f"{device['gpu']} {device['total_vram_gb']:g} GB"
    fraction = objective.get("cuda_mem_fraction")
    return text if fraction is None else f"{text} x {fraction:g}"


def mean_std(values):
    """(mean, sample standard deviation); the deviation is None for a single value."""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def summarize(rows):
    """Group result rows; returns a list of group dicts sorted for display."""
    groups = {}
    for row in rows:
        objective = row["objective"]
        key = (objective_label(objective), device_label(objective), row["task"],
               row["time_budget_s"], row["policy"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (objective, device, task, budget_s, policy), members in groups.items():
        seen = {}
        for row in members:
            if row["task_id"] in seen:
                raise ValueError(
                    f"two results for policy {policy!r} on task {row['task_id']!r}: "
                    f"{seen[row['task_id']]} and {row.get('_file', UNKNOWN_FILE)}")
            seen[row["task_id"]] = row.get("_file", UNKNOWN_FILE)
        finished = [r["best_val_bpb"] for r in members if r["best_val_bpb"] is not None]
        mean, std = mean_std(finished) if finished else (None, None)
        defaults = [r["baseline_val_bpb"] for r in members if r["baseline_val_bpb"] is not None]
        usage = [sum(r["token_usage"].values()) for r in members]
        summary.append({
            "objective": objective, "device": device, "task": task,
            "time_budget_h": budget_s / 3600.0, "policy": policy,
            "n_runs": len(members), "task_ids": sorted(seen),
            "best_val_bpb_mean": mean, "best_val_bpb_std": std,
            "best_val_bpb_min": min(finished) if finished else None,
            "baseline_val_bpb_mean": mean_std(defaults)[0] if defaults else None,
            "n_trials_mean": sum(r["n_trials"] for r in members) / len(members),
            "failure_rate_mean": sum(r["failure_rate"] for r in members) / len(members),
            "tokens_mean": sum(usage) / len(usage),
            "stop_reasons": sorted({r["stop_reason"] for r in members}),
        })
    summary.sort(key=lambda g: (g["objective"], g["device"], g["task"], g["time_budget_h"],
                                g["best_val_bpb_mean"] if g["best_val_bpb_mean"] is not None
                                else math.inf, g["policy"]))
    return summary


def format_table(summary):
    lines = []
    block = None
    width = max([len("policy")] + [len(group["policy"]) for group in summary])
    for group in summary:
        key = (group["objective"], group["device"], group["task"], group["time_budget_h"])
        if key != block:
            block = key
            if lines:
                lines.append("")
            lines.append(f"## {group['task']} | objective: {group['objective']} | "
                         f"device: {group['device']} | budget: {group['time_budget_h']:g} h")
            lines.append(f"{'policy':<{width}} {'runs':>5} {'best val_bpb':>19} {'min':>9} "
                         f"{'default':>9} {'trials':>7} {'failed':>7} {'tokens':>10}")
        mean, std = group["best_val_bpb_mean"], group["best_val_bpb_std"]
        if mean is None:
            best = "no finished trial"
        elif std is None:
            best = f"{mean:.4f}"
        else:
            best = f"{mean:.4f} ± {std:.4f}"
        minimum = "-" if group["best_val_bpb_min"] is None else f"{group['best_val_bpb_min']:.4f}"
        default = ("-" if group["baseline_val_bpb_mean"] is None
                   else f"{group['baseline_val_bpb_mean']:.4f}")
        notes = [r for r in group["stop_reasons"] if r != "time_budget"]
        lines.append(
            f"{group['policy']:<{width}} {group['n_runs']:>5} {best:>19} {minimum:>9} "
            f"{default:>9} {group['n_trials_mean']:>7.1f} {group['failure_rate_mean']:>7.1%} "
            f"{group['tokens_mean']:>10.0f}" + (f"  [stopped: {', '.join(notes)}]" if notes else ""))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarise autoresearch HPO results")
    parser.add_argument("paths", nargs="+",
                        help="Run directories (searched recursively) or .result.json files")
    parser.add_argument("--json", default=None, help="Also write the summary to this file")
    args = parser.parse_args(argv)

    try:
        files = find_results(args.paths)
        if not files:
            parser.error(f"no {RESULT_GLOB} files under {args.paths}")
        summary = summarize(load_results(files))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(format_table(summary))
    print(f"\n{len(files)} episode(s). 'best val_bpb' is mean ± sample std over runs "
          f"(lower is better); 'default' is trial 0; 'failed' is the share of trials "
          f"without a val_bpb.")
    if any(group["objective"] != BENCHMARK_KIND for group in summary):
        print("Blocks whose objective is not plain 'nanochat' are not benchmark results.")
    if any(OTHER_HARDWARE in group["objective"] for group in summary):
        print(f"'{OTHER_HARDWARE}': measured on another GPU than the task record states. "
              f"A trial is a fixed training time, so these numbers are comparable with "
              f"runs on the same GPU only (README, 'Running on another GPU').")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
