You are an expert in hyperparameter optimization and neural network training dynamics.

Write a single Python file implementing a hyperparameter-optimization policy for the autoresearch HPO benchmark.

## Task

Write a Python-based symbolic search policy that tunes the hyperparameters of a small GPT-style language model (nanochat `train.py` from autoresearch, trained on climbmix-400b-shuffle). Given the task description and every trial evaluated so far, the policy proposes the next configuration to train. The goal is the lowest `val_bpb` (validation bits per byte) found within a fixed budget of training time.

You will be shown decisions of an LLM doing this task in @demos/DEMOS_AUTORESEARCH_HPO_seed_42_100_examples_with_CoT.json. Each demo is one decision: the prompt the LLM saw (search space, recent trials, best configuration so far), its reasoning trace, the configuration it proposed, and what happened when that configuration was trained. Half of the demos are successes (the proposal improved on the best `val_bpb` so far) and half are failures (no improvement, out of memory, divergence, other errors), taken from early, middle and late stages of the search. Use them to understand the problem structure and take inspiration from the search strategies, especially where the LLM wastes trials.

Demo fields: `prompt`, `cot` (reasoning trace and answer), `final_response` (the proposed configuration as JSON), `success`, `outcome` (`improved`, `no_improvement`, `oom`, `diverged`, `error`, `timeout`), `config` (the configuration that was trained after clamping and snapping), `val_bpb`, `incumbent_val_bpb` (best value before this decision), `error`, `wall_time_s`, `peak_vram_mb`, `phase` (`early`, `mid`, `late`), `step`, `task_id`.

Output:
- a Python solver file @SOLVER_HPO.py
- a markdown file @SOLVER_HPO_ALGORITHM.md explaining the algorithm

The solver is driven and scored by the episode loop `run_episode` defined in @rewards/autoresearch_hpo.py. It should use (not reimplement) that module for the search-space definition and for config normalization (`SEARCH_SPACE`, `STARTING_CONFIG`, `normalize_config`, `derive_architecture`), importing it as `from rewards.autoresearch_hpo import ...`.

## Requirements

* Output exactly one Python file and one markdown file.
* Use only the Python standard library (plus `rewards.autoresearch_hpo`).
* No LLM calls, no network access, no external data, no reading or writing of files.
* The decision must depend only on the arguments `(task, history)`: no state kept between calls, and any randomness seeded from `task["seed"]` and the history. An interrupted episode is resumed by calling the solver again with the stored history, and must continue exactly as an uninterrupted one would.
* The solver must generalize: to other seeds, other time budgets and other GPUs (memory limits) than those in the demos. Do not replay configurations from the demos as a fixed list and do not hard-code a "known best" configuration; encode the strategy.
* A decision must take at most one second of CPU time, whatever the length of the history.

## Interface

Implement:

```python
def solve_hpo(task, history):
    """
    task: dict describing the run
        - "search_space": list of hyperparameters, each
              {"name", "type": "int" | "float", "low", "high", "log", "power_of_2", "default"}
              or {"name", "type": "cat", "choices", "default"}
        - "starting_config": the configuration of trial 0
        - "budget": {"time_budget_s", "trial_time_budget_s", "max_trials"}
        - "hardware": {"gpu", "total_vram_gb", "training_vram_gb", ...}
        - "seed": int, "metric": "val_bpb", "failure_penalty": 100.0, "task_id", "prompt"
    history: list of finished trials, oldest first; history[0] is the starting config
        each: {"trial_id", "config", "status", "val_bpb", "error",
               "wall_time_s", "peak_vram_mb", "metrics"}
        - "status": "ok" | "oom" | "diverged" | "error" | "timeout" | "invalid"
        - "val_bpb", "peak_vram_mb": numbers for "ok" trials, otherwise None
        - "config": the configuration that was trained (None for "invalid")
        - "metrics": extra numbers train.py reports for "ok" trials
                     (e.g. "num_params_M", "num_steps", "total_tokens_M")
    returns: dict with at least:
        - "config": {hyperparameter name: value} — the next configuration to train
      Any further keys (e.g. "strategy", "rationale") are stored in the episode log
      (under "solver_meta"); they are never used for scoring.
    """
```

## Search Space

Fourteen hyperparameters (bounds and defaults are in `task["search_space"]`):

- `DEPTH` — number of transformer layers
- `ASPECT_RATIO` — width per layer: the model dimension is `DEPTH * ASPECT_RATIO` rounded up to a multiple of `HEAD_DIM`
- `HEAD_DIM` — attention head dimension (power of 2)
- `DEVICE_BATCH_SIZE` — sequences per forward pass (power of 2); each sequence has 2048 tokens
- `TOTAL_BATCH_SIZE` — tokens per optimizer step (power of 2); gradient accumulation steps = `TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * 2048)`
- `EMBEDDING_LR`, `UNEMBEDDING_LR` — Adam learning rates of the token embeddings and the output head
- `MATRIX_LR` — Muon learning rate of the matrix parameters
- `SCALAR_LR` — learning rate of the scalar parameters
- `WEIGHT_DECAY` — weight decay of the matrix parameters
- `WARMUP_RATIO`, `WARMDOWN_RATIO` — fractions of the training time used for learning-rate warmup and warmdown
- `FINAL_LR_FRAC` — final learning rate as a fraction of the initial one
- `WINDOW_PATTERN` — per-layer attention window pattern (`L` = full context, `S` = half context)

Before training, every proposed configuration goes through `normalize_config`: missing hyperparameters take their starting value, unknown keys are dropped, numbers are cast to the hyperparameter's type and clamped to its bounds, power-of-2 hyperparameters are snapped to the nearest power of 2, and `TOTAL_BATCH_SIZE` is raised to at least `DEVICE_BATCH_SIZE * 2048`. A value that cannot be read (non-numeric, NaN, a `WINDOW_PATTERN` outside the choices) makes the suggestion `invalid`: nothing is trained and a trial slot is lost. `derive_architecture(config)` returns the resulting `model_dim`, `num_heads` and `grad_accum_steps`.

## Episode Protocol

* Trial 0 trains `task["starting_config"]`; the solver is first called with that one trial in the history.
* Every trial trains for a fixed `trial_time_budget_s` (300 s) of wall-clock time, so a larger model or a larger batch means fewer optimizer steps, not a longer trial.
* The episode ends when the summed `wall_time_s` of all trials reaches `time_budget_s`, or after `max_trials` trials.
* A failed trial (`oom`, `diverged`, `error`, `timeout`) has no `val_bpb`. It usually ends early and so costs less time than a finished trial, but it teaches nothing about `val_bpb` and still uses a trial slot.
* Training memory is limited to about `task["hardware"]["training_vram_gb"]` GB. Memory grows with `DEPTH`, the model dimension and `DEVICE_BATCH_SIZE`; `peak_vram_mb` of finished trials shows how close a configuration came.
* The result of an episode is the lowest `val_bpb` over all finished trials. Measurements are noisy: the number of optimizer steps that fit into the fixed training time varies from run to run, so training the same configuration twice can give slightly different values.

## Performance

A full trial costs 300 s on a GPU, so a 24 h budget buys about 288 finished trials in a 14-dimensional space. Sample efficiency decides the result: every proposal should be informative, and repeated out-of-memory or diverging proposals are the main way the LLM in the demos loses ground. The solver's own computation is negligible by comparison, but it has to stay within one second per decision.

The real objective cannot be run while building the solver. @data/autoresearch_hpo/synthetic_tasks.jsonl holds synthetic practice tasks whose objective is a cheap stand-in (a memory limit, divergence at aggressive learning rates, a smooth optimum) evaluated in milliseconds:

```python
import json
from rewards.autoresearch_hpo import run_episode
from SOLVER_HPO import solve_hpo

for line in open("data/autoresearch_hpo/synthetic_tasks.jsonl"):
    entry = json.loads(line)
    episode = run_episode(solve_hpo, entry)
    print(entry["task_id"], episode["stop_reason"], episode["score"]["status_counts"],
          episode["score"]["baseline_val_bpb"], episode["score"]["best_val_bpb"])
```

Use these episodes to test behaviour (no `invalid` trials, few failures, steady improvement, sensible use of the budget). The synthetic landscape is not nanochat: do not tune constants to it and do not read the hidden fields of the task records (`objective`, `reference`).

## Behavior

The solver should:
* propose only configurations inside the search space, and vary what it proposes rather than re-evaluating the same configuration without a reason
* learn the memory limit from the history (failed and finished trials, `peak_vram_mb`) and keep proposals on the safe side of it
* recognise divergence and back off from the learning-rate region that caused it
* balance exploration of untried regions against refinement around the best configurations found so far, shifting towards refinement as the budget runs out
* treat hyperparameters on a log scale where the search space marks them so, and respect the power-of-2 and batch-size constraints
* account for noise in `val_bpb` when comparing configurations
* adapt to the task: the budget, the memory limit and the seed come from `task`, not from constants
* return a short description of the reason for each proposal alongside the config (e.g. a `"strategy"` key), which is useful for analysing the episode log
