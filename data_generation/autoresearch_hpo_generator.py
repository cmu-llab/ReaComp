"""Generate task records for the autoresearch HPO domain (arXiv:2603.24647).

One record = one HPO run: a policy seed, a training-time budget and the hardware the
trials run on. The search space, starting configuration and objective come from
rewards/autoresearch_hpo.py, which is the single source of truth.

Presets (written to data/autoresearch_hpo/):

  full       tasks_full.jsonl       the paper's protocol: 24 h of training time,
                                    seeds 0-2, H200 capped to ~76 GB, real training
  pilot      pilot_tasks.jsonl      same, 1 h, seed 0 (a short end-to-end run)
  synthetic  synthetic_tasks.jsonl  SyntheticObjective landscapes for CPU tests and
                                    dry runs; NOT benchmark tasks

Usage (from the project root):
    python data_generation/autoresearch_hpo_generator.py                 # all presets
    python data_generation/autoresearch_hpo_generator.py --preset pilot \
        --time-budget-hours 2 --seeds 0 1 --out data/autoresearch_hpo/pilot_2h.jsonl

Running on a different GPU than the paper's changes the benchmark (5 minutes of
training buy a different number of steps, and the memory limit moves), so
--gpu-name / --total-vram-gb / --training-vram-gb exist to describe such a run
truthfully, not to reproduce the paper's numbers.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.autoresearch_hpo import (  # noqa: E402
    BENCHMARK, FAILURE_PENALTY, METRIC, STARTING_CONFIG, TRIAL_TIME_BUDGET_S, UPSTREAM,
    SyntheticObjective, describe_search_space, search_space_spec,
)

OUT_DIR = "data/autoresearch_hpo"

# Paper setup: one H200 (141 GB), training capped to ~76 GB for every method.
PAPER_GPU = "H200"
PAPER_TOTAL_VRAM_GB = 141.0
PAPER_TRAINING_VRAM_GB = 76.0

# Reference values used only to normalise scores (never shown to a policy).
# start: val_bpb of the default configuration; target: the best mean the paper
# reports (Centaur [Claude Opus 4.6]). Both are H200 numbers.
PAPER_START_VAL_BPB = 0.991
PAPER_TARGET_VAL_BPB = 0.9739

# Safety cap on the number of trials. The time budget is what ends a run (24 h / 300 s
# = 288 full-length trials, more when some fail early); the cap only bounds a policy
# that produces nothing but near-instant failures.
TRIALS_PER_BUDGET_HOUR = 120

PRESETS = {
    "full": {"file": "tasks_full.jsonl", "kind": "nanochat", "hours": 24.0, "seeds": [0, 1, 2]},
    "pilot": {"file": "pilot_tasks.jsonl", "kind": "nanochat", "hours": 1.0, "seeds": [0]},
    "synthetic": {"file": "synthetic_tasks.jsonl", "kind": "synthetic", "hours": 6.0,
                  "seeds": [0], "landscapes": [0, 1, 2, 3, 4]},
}

PROMPT_TEMPLATE = """\
Tune the hyperparameters of nanochat, a GPT-2 scale transformer trained on \
climbmix-400b-shuffle by the train.py of Karpathy's autoresearch. The goal is to \
minimise {metric} (validation bits per byte).

Protocol:
- Every trial trains one configuration for a fixed {trial_minutes:g} minutes on a single \
{gpu} GPU ({total_vram:g} GB VRAM, about {training_vram:g} GB usable for training). \
Configurations that are too large for the GPU crash with an out-of-memory error, and \
unstable ones can diverge; a failed trial yields no {metric} (it is scored as \
{failure_penalty:g}) but still consumes the time it ran.
- Trial 0 is the default configuration. After each trial you see its outcome and \
propose the next configuration.
- The run ends once the cumulative training time reaches {budget_hours:g} h. The \
result of the run is the lowest {metric} of any trial.

Search space (14 hyperparameters):
{space}

Values outside the bounds are clamped, integer hyperparameters are truncated, and \
HEAD_DIM, DEVICE_BATCH_SIZE and TOTAL_BATCH_SIZE are snapped to the nearest power of \
two; TOTAL_BATCH_SIZE is raised to DEVICE_BATCH_SIZE * 2048 when below it. A \
hyperparameter you leave out keeps its default.

Answer with a JSON object mapping hyperparameter names to values."""


def build_task(kind, seed, time_budget_hours, gpu_name=PAPER_GPU,
               total_vram_gb=PAPER_TOTAL_VRAM_GB, training_vram_gb=PAPER_TRAINING_VRAM_GB,
               landscape_seed=None):
    """Build one task record."""
    if training_vram_gb > total_vram_gb:
        raise ValueError("training VRAM cannot exceed the GPU's total VRAM")
    time_budget_s = time_budget_hours * 3600.0
    hardware = {
        "gpu": gpu_name,
        "total_vram_gb": total_vram_gb,
        "training_vram_gb": training_vram_gb,
        # train.py caps its process with torch.cuda.set_per_process_memory_fraction
        # when CUDA_MEM_FRACTION is set. Two decimals: the pinned train.py gives
        # "0.54 for ~76GB on H200" as the example.
        "cuda_mem_fraction": (round(training_vram_gb / total_vram_gb, 2)
                              if training_vram_gb < total_vram_gb else None),
    }
    paper_hardware = (gpu_name, total_vram_gb, training_vram_gb) == (
        PAPER_GPU, PAPER_TOTAL_VRAM_GB, PAPER_TRAINING_VRAM_GB)

    if kind == "synthetic":
        objective = SyntheticObjective(landscape_seed, training_vram_gb, TRIAL_TIME_BUDGET_S)
        objective_spec = {"kind": "synthetic", "landscape_seed": landscape_seed}
        reference = {
            "start_val_bpb": objective.evaluate(STARTING_CONFIG)["val_bpb"],
            "target_val_bpb": objective.optimum_val_bpb,
            "source": "exact optimum of the synthetic landscape (not a benchmark number)",
        }
        task_id = f"{BENCHMARK}_synthetic_L{landscape_seed}_{time_budget_hours:g}h_seed{seed}"
        task = "autoresearch_hpo_synthetic"
    elif kind == "nanochat":
        objective_spec = {
            "kind": "nanochat",
            "repo": UPSTREAM["train_repo"],
            "commit": UPSTREAM["train_commit"],
            "train_py_sha256": UPSTREAM["train_py_sha256"],
            "prepare_py_sha256": UPSTREAM["prepare_py_sha256"],
        }
        reference = {
            "start_val_bpb": PAPER_START_VAL_BPB,
            "target_val_bpb": PAPER_TARGET_VAL_BPB,
            "source": f"{UPSTREAM['paper']}: default configuration and best reported mean "
                      f"(Centaur [Claude Opus 4.6]) on an H200 capped to ~76 GB"
                      + ("" if paper_hardware else "; this task uses different hardware, "
                         "so the normalised score is not comparable to the paper"),
        }
        task_id = f"{BENCHMARK}_{time_budget_hours:g}h_seed{seed}"
        if not paper_hardware:
            task_id += f"_{gpu_name.lower()}_{training_vram_gb:g}gb"
        task = "autoresearch_hpo"
    else:
        raise ValueError(f"unknown objective kind {kind!r}")

    prompt = PROMPT_TEMPLATE.format(
        metric=METRIC, trial_minutes=TRIAL_TIME_BUDGET_S / 60.0, gpu=gpu_name,
        total_vram=total_vram_gb, training_vram=training_vram_gb,
        failure_penalty=FAILURE_PENALTY, budget_hours=time_budget_hours,
        space=describe_search_space())
    return {
        "task_id": task_id,
        "prompt": prompt,
        "type": "hpo",
        "task": task,
        "reward": BENCHMARK,
        "metric": METRIC,
        "seed": seed,
        "search_space": search_space_spec(),
        "starting_config": dict(STARTING_CONFIG),
        "budget": {
            "time_budget_s": time_budget_s,
            "trial_time_budget_s": TRIAL_TIME_BUDGET_S,
            "max_trials": max(2, int(round(time_budget_hours * TRIALS_PER_BUDGET_HOUR))),
        },
        "hardware": hardware,
        "failure_penalty": FAILURE_PENALTY,
        "objective": objective_spec,
        "reference": reference,
    }


def build_preset(name, seeds=None, time_budget_hours=None, landscapes=None, **hardware):
    preset = PRESETS[name]
    seeds = preset["seeds"] if seeds is None else seeds
    hours = preset["hours"] if time_budget_hours is None else time_budget_hours
    if preset["kind"] == "synthetic":
        landscapes = preset["landscapes"] if landscapes is None else landscapes
        return [build_task("synthetic", seed, hours, landscape_seed=landscape, **hardware)
                for landscape in landscapes for seed in seeds]
    return [build_task("nanochat", seed, hours, **hardware) for seed in seeds]


def write_jsonl(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", choices=["all", *PRESETS], default="all")
    parser.add_argument("--out", help="Output file (single preset only; default: the "
                                      f"preset's file under {OUT_DIR}/)")
    parser.add_argument("--seeds", type=int, nargs="+", help="Policy seeds (default: the preset's)")
    parser.add_argument("--time-budget-hours", type=float,
                        help="Cumulative training-time budget (default: the preset's)")
    parser.add_argument("--landscapes", type=int, nargs="+",
                        help="Synthetic landscape seeds (synthetic preset only)")
    parser.add_argument("--gpu-name", default=PAPER_GPU)
    parser.add_argument("--total-vram-gb", type=float, default=PAPER_TOTAL_VRAM_GB)
    parser.add_argument("--training-vram-gb", type=float, default=PAPER_TRAINING_VRAM_GB)
    args = parser.parse_args()

    names = list(PRESETS) if args.preset == "all" else [args.preset]
    if args.out and len(names) != 1:
        parser.error("--out needs a single --preset")
    for name in names:
        records = build_preset(
            name, seeds=args.seeds, time_budget_hours=args.time_budget_hours,
            landscapes=args.landscapes, gpu_name=args.gpu_name,
            total_vram_gb=args.total_vram_gb, training_vram_gb=args.training_vram_gb)
        path = args.out or os.path.join(OUT_DIR, PRESETS[name]["file"])
        write_jsonl(records, path)
        print(f"{name}: wrote {len(records)} task(s) to {path}")


if __name__ == "__main__":
    main()
