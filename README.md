# REACOMP: Compiling LLM Reasoning into Symbolic Solvers for Efficient Program Synthesis

A coding agent reads LLM reasoning traces and induces a reusable symbolic solver — at zero per-task LLM cost. The solver is then ensembled with LLM baselines (Best-of-K, Direct Feedback) to push performance beyond what either approach achieves alone.

---

## Overview

| Stage | What happens |
|-------|-------------|
| **Solver induction** | A coding agent (Claude Code or Qwen3.6-35B via OpenHands) reads 100 reasoning traces (`DEMOS.json`) and writes a standalone `SOLVER.py` in a single session |
| **Symbolic evaluation** | `SOLVER.py` is run on PBEBench-Lite, PBEBench-Hard, or SLR-Bench — no LLM calls at inference time |
| **LLM baselines** | Best-of-K (K=32) and Direct Feedback (up to 32 sequential attempts) run via vLLM with `gpt-oss-120b` |
| **Ensembling** | Symbolic solver + LLM outputs are merged; the *effi* strategy uses solver outputs at zero token cost, falling back to LLM only when the solver fails |

---

## Benchmarks

| Benchmark | Tasks | Difficulty axis | Task type |
|-----------|------:|-----------------|-----------|
| **PBEBench-Lite** | 1,008 | Cascade length 2–5 | Induce `replace(A,B)` cascade programs from I/O examples |
| **PBEBench-Hard** | 1,216 | Cascade length 2–20 | Same, harder cascades |
| **SLR-Bench** | 1,000 | Curriculum tier (basic/easy/medium/hard) | Induce Prolog rules `eastbound(T) :- Body.` |
| **Autoresearch HPO** | 3 runs (24 h each, seeds 0–2) + 1 pilot run (1 h) | — | Tune 14 hyperparameters of nanochat pre-training to minimise `val_bpb` within a training-time budget (sequential; needs a GPU). See [Autoresearch HPO](#autoresearch-hpo-nanochat-pre-training) |

---

## Induced Symbolic Solvers

| Solver | Inducing agent | Path |
|--------|---------------|------|
| **CC Solver** | claude-sonnet-4-6 (Claude Code CLI) | `built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py` |
| **OH Qwen Solver** | Qwen3.6-35B-A3B inside OpenHands | `built_solvers/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/SOLVER.py` |
| **CC Solver (SLR)** | claude-sonnet-4-6 (Claude Code CLI) | `built_solvers/claude_code/Sat_Apr_25_251_AM/SOLVER_SLR.py` |
| **OH Qwen Solver (SLR)** | Qwen3.6-35B-A3B inside OpenHands | `built_solvers/qwen3.6_35b_a3b/Sat_Apr_25_643_AM/SOLVER_SLR.py` |

---

## Solver Construction Ablations

To study the effect of the demos file on solver quality, we ablate over the number of examples and the presence of chain-of-thought (CoT) reasoning traces. All ablations use the Qwen solver (cost reasons) and are run via `run_solver_builder_openhands.sh` with the `DEMOS_PATH` env var.

| Ablation | `DEMOS_PATH` |
|----------|-------------|
| 100 examples, with CoT *(main/default)* | `demos/DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json` |
| 100 examples, with CoT | `demos/DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json` |
| 48 examples, with CoT | `demos/DEMOS_PBEBENCH_seed_42_48_examples_with_CoT.json` |
| 12 examples, with CoT | `demos/DEMOS_PBEBENCH_seed_42_12_examples_with_CoT.json` |

All demo files use the same seed (42) and are balanced across success/failure × easy/hard quadrants.

To run an ablation:

```bash
SOLVER_TYPE=pbe DEMOS_PATH=demos/DEMOS_PBEBENCH_seed_42_48_examples_with_CoT.json \
    bash scripts/run_solver_builder_openhands.sh <PORT>
```

The output directory is automatically tagged with the demos filename (e.g. `built_solvers/qwen3.6_35b_a3b/<TIMESTAMP>_DEMOS_PBEBENCH_seed_42_48_examples_with_CoT/`), so runs with different demo files never collide. Evaluate each resulting `SOLVER.py` with `scripts/eval_solver.py --dataset lite`.

---

## Results

See `evals/findings.md` for the full results table and `figures/` for comparison plots.

Key result on **PBEBench-Hard** (cascade length 2–20):

| System | Pass rate |
|--------|----------:|
| BoK (gpt-oss-120b, K=32) | — |
| CC Solver (zero LLM cost) | 69.7% |
| OH Qwen Solver (zero LLM cost) | 57.8% |
| BoK + CC Solver ensemble | best combined |

Solver variance (run1 vs run2): CC solver 99.6% per-task agreement (Δ=0.09pp), Qwen 97.0% (Δ=1.0pp) — essentially deterministic.

---

## Autoresearch HPO (nanochat pre-training)

A ReaComp setup for the benchmark of *Can LLMs Beat Classical Hyperparameter Optimization Algorithms? A Study on autoresearch* ([arXiv:2603.24647](https://arxiv.org/abs/2603.24647)): tune 14 hyperparameters of the nanochat `train.py` from Karpathy's autoresearch to minimise `val_bpb` (validation bits per byte). Every trial trains for a fixed 300 s on one GPU, a run has a cumulative training-time budget (24 h in the paper), a failed trial (out of memory, divergence) is recorded with `val_bpb = 100`, and every method starts from the default configuration as trial 0.

Unlike the other benchmarks, a task is a sequential decision problem, so the induced solver is a search *policy* rather than a one-shot program synthesiser:

```python
def solve_hpo(task, history):      # SOLVER_HPO.py — standard library only, no LLM calls
    return {"config": {...}}       # the next configuration to train
```

| Stage | Autoresearch HPO |
|-------|------------------|
| **Reasoning traces** | An LLM HPO agent (one call per trial; the paper's "Karpathy Agent (14 HPs)") runs episodes; every decision is logged with its prompt, reasoning and outcome (`scripts/run_hpo_traces_vllm.sh`) |
| **Demos** | `openhands_agents/build_hpo_balanced_demos.py` selects decisions balanced between successes (improved the best `val_bpb` so far) and failures (no improvement, OOM, divergence, …) across early/mid/late phases of the search |
| **Solver induction** | A coding agent reads the demos and `building_prompts/SOLVER_BUILDING_PROMPT_HPO.md` and writes `SOLVER_HPO.py` + `SOLVER_HPO_ALGORITHM.md` (Claude Code, or Qwen via `scripts/run_solver_builder_hpo_openhands.sh`) |
| **Evaluation** | `scripts/run_hpo_eval.sh` runs the solver as the policy of an episode: the harness trains every proposed configuration itself and reads `val_bpb` from the training output. Nothing a policy reports about its own results is used (see *What the grading relies on*) |
| **Baselines** | The LLM agent above, plus random search, TPE and CMA-ES (`POLICY=random|tpe|cmaes`; TPE and CMA-ES use Optuna as upstream does) |

Commands for every step are in [COMMANDS.md](COMMANDS.md#7-autoresearch-hpo-nanochat-pre-training).

**Status.** The domain is implemented and tested on CPU (see *Tests* below). No episode on the real objective has been run in this repository yet, so there are no demos, induced solvers or results for it: `demos/`, `built_solvers/` and `evals/` contain no HPO files. For orientation, the paper reports (24 h, 3 seeds, one H200; numbers copied from the paper, not reproduced here) a default-configuration `val_bpb` of about 0.991 and best values of 0.9739 ± 0.0012 for its best method (Centaur), 0.9768 ± 0.0019 for TPE, 0.9873 for random search and 0.9904 for the LLM agent with a 27B open model.

### Fidelity to the original benchmark

`data/autoresearch_hpo/check_upstream_fidelity.py` compares this implementation with the upstream sources at pinned commits (`ferreirafabio/autoresearch-automl@150cd14`, `ferreirafabio/autoresearch@b4180f8`): names, order, types, bounds, log scales and choices of the 14 hyperparameters, the power-of-two set, the starting configuration, the sha256 of `train.py` / `prepare.py`, the 300 s trial budget, and that writing a configuration into `train.py` changes the hyperparameter lines only and round-trips exactly. It passes at the pinned commits. What it cannot check is that training reproduces the paper's numbers; that needs the GPU run in COMMANDS.md §7 (the default configuration should give `val_bpb` ≈ 0.991 on an H200).

Caveats and deliberate differences:

* **Hardware is part of the benchmark.** A trial is 300 s of whatever the GPU computes, and `train.py` caps memory to a fraction of the device (`CUDA_MEM_FRACTION`; the pinned `train.py` documents 0.54 of an H200 for ~76 GB, which is what the task files set). The task files describe the paper's hardware; before the first trial the harness checks that the GPU seen by the training environment has that memory and that `train.py` / `prepare.py` are the pinned files (the pin is a constant of `rewards/autoresearch_hpo.py`, not something a task file can change), and it hashes both files again before every trial, so sources that change during a run stop the run. `--allow-unpinned` turns the hardware and pin checks into labels: the results are then marked `nanochat (unpinned sources)` / `nanochat (hardware not verified)` in every summary and are not benchmark results. To run on another GPU on purpose, generate task files that describe it (`data_generation/autoresearch_hpo_generator.py --gpu-name … --total-vram-gb … --training-vram-gb …`).
* **Training data.** `scripts/setup_autoresearch_hpo.sh` runs the pinned `prepare.py` with its defaults (10 training shards of climbmix-400b-shuffle plus the pinned validation shard). The upstream README does not state how many shards its runs used; `PREPARE_ARGS` overrides the default.
* **LLM agent.** `PROMPT_STYLE=paper MAX_TOKENS=2048` with `--parse-retries 0` gives the upstream agent's prompt and settings. The default style (`cot`) shows the same context but asks for step-by-step reasoning before the JSON answer, so that the traces contain reasoning for models without a separate reasoning channel. The two styles read answers differently: `paper` uses upstream's rule (the first fenced block, or the whole answer, must be the JSON object), `cot` takes the last JSON object that names a hyperparameter and does not use an answer that was cut off by the token limit. In both styles the search-space lines are formatted by this repository, a `WINDOW_PATTERN` outside the listed choices or a non-numeric value is rejected (upstream passes the former to `train.py` and casts strings), and requests carry a seed. As upstream, an unusable answer falls back to a random configuration; the result file counts these (`suggestion_sources`). Model, prompt style, temperature, token limit and retries are part of the policy name (`hpo_agent[<model>|<style>|T=…|max_tokens=…|retries=…]`), so runs with different settings are separate rows of a results table and cannot continue each other's logs.
* **Baselines.** Random search, TPE and CMA-ES are implemented. SMAC, LLAMBO and Centaur from the paper are not. A resumed TPE / CMA-ES episode replays its log through the seeded sampler and continues exactly as an uninterrupted run would; a log the sampler does not reproduce (e.g. written with another Optuna version) is refused.
* **Budget accounting.** As upstream, a finished trial costs the training seconds `train.py` reports and a failed one the wall time until it ended. The time a policy takes to decide (LLM inference, a solver's computation) is recorded per trial (`policy_time_s`) but not charged to the budget, as in the paper, whose budget excludes LLM inference overhead. Two stopping rules are additions of this repository and only matter for a broken policy: an episode ends after `max_trials` trials (120 per budget hour, ten times the number of full trials that fit), and after 10 consecutive suggestions that cannot be read as a configuration (these are logged as `invalid` trials and train nothing).
* **Synthetic objective.** `data/autoresearch_hpo/synthetic_tasks.jsonl` uses `SyntheticObjective`, a deterministic stand-in landscape (a memory limit, divergence at aggressive learning rates, a smooth optimum) evaluated in milliseconds. It exists for CPU tests, dry runs of every launcher and as practice episodes while a solver is being built. It is not nanochat: its numbers are labelled `synthetic (NOT benchmark)` and are never mixed with real results in a summary.

### What the grading relies on

The score of an episode comes from the harness alone: it trains every configuration itself, parses `val_bpb` from the output of `train.py`, and charges the budget from the same output. A policy returns a configuration; whatever else a solver returns is stored in the episode log under `solver_meta` and is never read by the scoring or the summary, and the fields the harness writes about a suggestion (`source`, `token_usage`) cannot be set by a solver.

Two things are trusted and are worth knowing when reading a results table:

* **Solvers are not sandboxed during evaluation.** `scripts/eval_hpo_solver.py` runs every decision of `SOLVER_HPO.py` in a fresh process (so a hang, a crash or leaked state costs the solver a trial instead of the episode), but with the caller's permissions, like the solvers of the other benchmarks. A solver written to cheat could write to the output directory or the training checkout. The per-trial hash of `train.py` / `prepare.py` catches a modified checkout, not everything; read an induced solver before running it.
* **Files on disk.** A resumed episode continues from its log, and `scripts/summarize_hpo_results.py` reports what the `*.result.json` files say. Both are written by the harness, and a log is only continued by the run that wrote it (same task record, objective and policy, compared through the log's header), but neither is signed.

### Tests

```bash
python -m unittest discover -s tests -t .
```

The tests run on a CPU in under a minute and need neither a GPU, a model server nor Apptainer: episodes use the synthetic objective or a stub `train.py` checkout, the LLM is a local fake OpenAI-compatible server, and the SolverBuilder runs a scripted conversation through its real tools with a local stand-in for the sandbox. Tests of the Optuna baselines are skipped when `optuna` / `cmaes` are not installed, and the SolverBuilder tests when the OpenHands SDK is not.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with the API key(s) you need:

```
ANTHROPIC_API_KEY=sk-ant-...   # for Claude Code solver induction
OPENAI_API_KEY=sk-...          # optional, for OpenAI-compatible endpoints
VLLM_API_KEY=EMPTY             # for local vLLM (BoK / DF baselines)
```

---

## Reproduction

Full step-by-step commands are also in **[COMMANDS.md](COMMANDS.md)**. Quick reference below.

### 1. Solver induction (coding agent writes SOLVER.py)

**Claude Code (interactive session — run from project root):**
```bash
# The agent reads demos/ and building_prompts/, then writes SOLVER.py + SOLVER_ALGORITHM.md
claude  # open Claude Code, then @ the building prompt and demos file manually
```

**OpenHands + Qwen (fully automated):**
```bash
# PBEBench solver (default: 100 examples + CoT)
SOLVER_TYPE=pbe bash scripts/run_solver_builder_openhands.sh <PORT>

# SLR-Bench solver (default: 92 examples + CoT)
SOLVER_TYPE=slr bash scripts/run_solver_builder_openhands.sh <PORT>

# Ablation: override the demos file
SOLVER_TYPE=pbe DEMOS_PATH=demos/DEMOS_PBEBENCH_seed_42_48_examples_with_CoT.json \
    bash scripts/run_solver_builder_openhands.sh <PORT>
```
Output lands in `built_solvers/qwen3.6_35b_a3b/<TIMESTAMP>[_<demos_tag>]/`.

### 2. Evaluating symbolic solvers

```bash
# PBEBench-Lite + Hard (both splits)
python scripts/eval_solver.py \
    --solver built_solvers/<agent>/<timestamp>/SOLVER.py \
    --dataset both \
    --workers 32 \
    --task-timeout 60 \
    --output-dir evals/solver_results/<agent>/<timestamp>

# SLR-Bench only
python scripts/eval_solver.py \
    --solver built_solvers/<agent>/<timestamp>/SOLVER_SLR.py \
    --dataset slr \
    --workers 32 \
    --task-timeout 3600 \
    --output-dir evals/solver_results/slr_<agent>/<timestamp>
```
Writes `lite.jsonl`, `hard.jsonl`, or `slr.jsonl` + `*_summary.json` to `--output-dir`.

### 3. Running LLM baselines (requires local vLLM serving `gpt-oss-120b`)

```bash
# Best-of-K
DATASET=lite bash scripts/run_best_of_k_vllm.sh    # PBEBench-Lite
DATASET=hard bash scripts/run_best_of_k_vllm.sh    # PBEBench-Hard
DATASET=slr  bash scripts/run_best_of_k_vllm.sh    # SLR-Bench
# Override port: PORT=8004 DATASET=lite bash scripts/run_best_of_k_vllm.sh

# Direct Feedback
DATASET=lite bash scripts/run_direct_feedback_vllm.sh
DATASET=hard bash scripts/run_direct_feedback_vllm.sh
DATASET=slr  bash scripts/run_direct_feedback_vllm.sh
# Override port/workers: PORT=8002 WORKERS=16 bash scripts/run_direct_feedback_vllm.sh
```
Both scripts checkpoint after every task — safe to kill and relaunch.

**DirectSolve (OpenHands + Qwen3.6-35B-A3B):**
```bash
# Single job (all tasks)
DATASET=data/pbebench/tasks_full_og.jsonl PORT=8002 bash scripts/run_direct_solve_openhands.sh

# Sharded (two parallel jobs covering non-overlapping index ranges)
START_INDEX=0   END_INDEX=800  DATASET=data/pbebench/tasks_full_og.jsonl PORT=8002 bash scripts/run_direct_solve_openhands.sh
START_INDEX=800 END_INDEX=1216 DATASET=data/pbebench/tasks_full_og.jsonl PORT=8002 bash scripts/run_direct_solve_openhands.sh
# Output is auto-suffixed: outputs/hard_direct_solve_openhands_0_800.jsonl etc.

# Merge shards (deduplicates by task_id, sorts by task_index)
python scripts/merge_direct_solve_shards.py \
    outputs/hard_direct_solve_openhands_0_800.jsonl \
    outputs/hard_direct_solve_openhands_800_1216.jsonl \
    --output outputs/hard_direct_solve_openhands.jsonl
```
Indices are **0-based, exclusive end** (Python slice semantics). Global `task_id`/`task_index` are stamped before slicing, so shards always use correct dataset-level IDs regardless of split point.

### 4. Ensembling, eval, and plots

**PBEBench-Lite** (BoK and DF outputs must be pulled from cluster and stripped first):
```bash
# Strip agent_messages from large cluster output files before pulling locally
python scripts/strip_agent_messages.py outputs/lite_tasks_full_og_best_of_k.jsonl
python scripts/strip_agent_messages.py outputs/lite_tasks_full_og_direct_feedback.jsonl

# Build effi ensembles (solver = CC, Qwen run2, or union)
CC=evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl
QWEN=evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/lite.jsonl
BOK=outputs/lite_tasks_full_og_best_of_k_stripped.jsonl
DF=outputs/lite_tasks_full_og_direct_feedback_stripped.jsonl

python scripts/ensemble_outputs.py --sources $CC $QWEN --out outputs/lite_union_solvers.jsonl
python scripts/ensemble_outputs.py --effi --symbolic $CC --sources $BOK --out outputs/lite_effi_bok_cc.jsonl
python scripts/ensemble_outputs.py --effi --symbolic $QWEN --sources $BOK --out outputs/lite_effi_bok_qwen_run2.jsonl
python scripts/ensemble_outputs.py --effi --symbolic outputs/lite_union_solvers.jsonl --sources $BOK --out outputs/lite_effi_bok_all_solvers.jsonl
python scripts/ensemble_outputs.py --effi --symbolic $CC --sources $DF --out outputs/lite_effi_df_cc.jsonl
python scripts/ensemble_outputs.py --effi --symbolic $QWEN --sources $DF --out outputs/lite_effi_df_qwen_run2.jsonl
python scripts/ensemble_outputs.py --effi --symbolic outputs/lite_union_solvers.jsonl --sources $DF --out outputs/lite_effi_df_all_solvers.jsonl
```

**PBEBench-Hard** (BoK raw format; per-task token costs computed from cluster file with CoT):
```bash
# Step 1: compute per-task token counts from the cluster BoK file (needs chains_of_thought)
# Run this on the cluster where the CoT file lives, then pull metrics/bok_hard_tokens_per_task.jsonl
python scripts/compute_bok_tokens.py \
    outputs/gpt_oss_120b_pbebench_hard_outputs_with_CoT.jsonl \
    --tokenizer openai/gpt-oss-120b \
    --out metrics/bok_hard_tokens_per_task.jsonl \
    --metrics-json metrics/bok_hard_tokens_cluster.json

# Step 2: convert raw BoK format to quick_eval-compatible JSONL with per-task token costs
# --token-jsonl (preferred, per-task) takes precedence over --token-json (uniform avg fallback)
python scripts/eval_bok_hard.py \
    --bok outputs/gpt_oss_120b_pbebench_hard_outputs.jsonl \
    --out-jsonl outputs/hard_bok_converted.jsonl \
    --token-jsonl metrics/bok_hard_tokens_per_task.jsonl \
    --token-json metrics/bok_hard_tokens_cluster.json

# Step 3: build effi ensembles (effi = solver at zero cost, BoK fallback only when solver fails)
CC=evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl
QWEN=evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/hard.jsonl
BOK=outputs/hard_bok_converted.jsonl

python scripts/ensemble_outputs.py --sources $CC $QWEN --out outputs/hard_union_cc_qwen_run2.jsonl
python scripts/ensemble_outputs.py --effi --symbolic $CC --sources $BOK --out outputs/hard_effi_bok_cc.jsonl
python scripts/ensemble_outputs.py --effi --symbolic $QWEN --sources $BOK --out outputs/hard_effi_bok_qwen_run2.jsonl
python scripts/ensemble_outputs.py --effi --symbolic outputs/hard_union_cc_qwen_run2.jsonl --sources $BOK \
    --out outputs/hard_effi_bok_cc_qwen_run2.jsonl
# All solvers union = CC + all Qwen PBE runs
python scripts/ensemble_outputs.py --sources $CC $QWEN \
    evals/solver_results/qwen3.6_35b_a3b/*/hard.jsonl --out outputs/hard_union_all_solvers.jsonl
python scripts/ensemble_outputs.py --effi --symbolic outputs/hard_union_all_solvers.jsonl --sources $BOK \
    --out outputs/hard_effi_bok_all_solvers.jsonl
```

**Quick eval on any output:**
```bash
python scripts/quick_eval.py outputs/my_ensemble.jsonl \
    --tasks-file data/pbebench/lite_tasks_full_og.jsonl \
    --metrics-json metrics/my_ensemble.json
```

### 5. Trajectory token cost (solver construction)

```bash
# Estimate token cost of a solver-builder run (with/without KV caching)
python scripts/compute_trajectory_tokens.py \
    debug_oh_solver_builder/<timestamp>/solver_builder_trajectory.json

# With exact Qwen tokenizer (requires transformers)
python scripts/compute_trajectory_tokens.py \
    debug_oh_solver_builder/<timestamp>/solver_builder_trajectory.json \
    --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --metrics-json metrics/solver_build_tokens.json
```

---

## Project Structure

```
.
├── building_prompts/
│   ├── SOLVER_BUILDING_PROMPT_PBE.md   # spec given to coding agent for PBEBench solver
│   ├── SOLVER_BUILDING_PROMPT_SLR.md   # spec given to coding agent for SLR-Bench solver
│   └── SOLVER_BUILDING_PROMPT_HPO.md   # spec given to coding agent for the autoresearch HPO solver
├── demos/
│   ├── DEMOS_PBEBENCH_seed_42_100_examples.json              # 100 PBEBench traces (no CoT)
│   ├── DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json    # 100 PBEBench traces + CoT — default
│   ├── DEMOS_PBEBENCH_seed_42_48_examples_with_CoT.json     # 48 examples + CoT (ablation)
│   ├── DEMOS_PBEBENCH_seed_42_12_examples_with_CoT.json     # 12 examples + CoT (ablation)
│   └── DEMOS_SLRBENCH_seed_42_92_examples_with_CoT.json     # 92 SLR-Bench reasoning traces
├── built_solvers/
│   ├── claude_code/          # solvers induced by Claude Code
│   └── qwen3.6_35b_a3b/      # solvers induced by Qwen via OpenHands
├── data/
│   ├── pbebench/             # PBEBench task files
│   ├── slr_bench/            # SLR-Bench task files
│   └── autoresearch_hpo/     # autoresearch HPO task files (full, pilot, synthetic) + upstream fidelity check
├── data_generation/
│   └── autoresearch_hpo_generator.py   # writes the autoresearch HPO task files
├── evals/
│   ├── solver_results/       # per-task JSONL from eval_solver.py
│   ├── findings.md           # consolidated results
│   ├── experimental_setup.md # full experimental details
│   └── README.md             # eval directory guide
├── outputs/                  # LLM baseline and ensemble outputs
├── figures/                  # comparison plots (PNG)
├── metrics/                  # quick_eval JSON summaries
├── rewards/
│   ├── pbebench.py           # PBEBench reward + cascade_complexity
│   ├── slrbench.py           # SLR-Bench reward + rule_complexity
│   └── autoresearch_hpo.py   # autoresearch HPO: search space, objectives, episode loop, scoring
├── scripts/
│   ├── eval_solver.py                  # evaluate SOLVER.py; writes quick_eval-compatible JSONL
│   ├── quick_eval.py                   # pass rate, mean reward, token usage, complexity
│   ├── ensemble_outputs.py             # standard + effi ensembling
│   ├── plot_pbebench_comparison.py     # 4-way comparison plots for PBEBench
│   ├── plot_slr_comparison.py          # 4-way comparison plots for SLR-Bench
│   ├── run_all_pbebench_lite_evals.sh  # full Lite pipeline: ensembles + eval + plots
│   ├── run_all_pbebench_hard_evals.sh  # full Hard pipeline: ensembles + eval + plots
│   ├── run_all_slr_evals.sh            # full SLR pipeline: ensembles + eval + plots
│   ├── run_best_of_k_vllm.sh           # run BoK baseline via vLLM
│   ├── run_direct_feedback_vllm.sh     # run Direct Feedback baseline via vLLM
│   ├── run_direct_solve_openhands.sh   # run DirectSolve baseline (one OH conversation/task); supports sharding via START_INDEX/END_INDEX
│   ├── merge_direct_solve_shards.py    # merge shard JSONLs + checkpoints into one file
│   ├── run_solver_builder_openhands.sh # run SolverBuilder agent via OpenHands
│   ├── setup_autoresearch_hpo.sh       # HPO: clone the pinned autoresearch checkout + prepare data
│   ├── run_hpo_traces_vllm.sh          # HPO: LLM agent episodes (baseline + reasoning traces)
│   ├── run_solver_builder_hpo_openhands.sh  # HPO: SolverBuilder agent via OpenHands
│   ├── run_hpo_eval.sh                 # HPO: evaluate SOLVER_HPO.py or a classical baseline
│   ├── eval_hpo_solver.py              # HPO: episode runner behind run_hpo_eval.sh
│   └── summarize_hpo_results.py        # HPO: mean ± std of best val_bpb per policy
├── openhands_agents/
│   ├── solver_builder/       # SolverBuilder agent (controller + tools; hpo_* for the HPO domain)
│   ├── hpo_agent/            # HPO: LLM agent, classical baselines, solver-as-policy, episode runner
│   ├── run_hpo_traces.py     # HPO: run the LLM agent on task files
│   ├── build_hpo_balanced_demos.py     # HPO: episode logs → balanced demos file
│   └── build_solver_hpo.py   # HPO: SolverBuilder entry point
├── tests/                    # CPU tests of the autoresearch HPO domain
├── COMMANDS.md               # full reproduction commands
└── main.py                   # legacy CLI (multi-agent system — not used in paper)
```

---

## Eval Scripts

| Script | Purpose |
|--------|---------|
| `scripts/eval_solver.py` | Evaluate `SOLVER.py` on PBEBench or SLR-Bench; writes standard JSONL |
| `scripts/quick_eval.py` | Pass rate, mean reward, token usage, complexity for any JSONL output |
| `scripts/ensemble_outputs.py` | Build standard or effi ensemble from multiple sources |
| `scripts/plot_pbebench_comparison.py` | 4-way comparison figures for PBEBench-Lite or Hard |
| `scripts/plot_slr_comparison.py` | 4-way comparison figures for SLR-Bench (tier + level) |
| `scripts/eval_hpo_solver.py` | Run `SOLVER_HPO.py` or a classical baseline on autoresearch HPO tasks; writes one episode log + result file per task |
| `scripts/summarize_hpo_results.py` | Best `val_bpb` (mean ± std over seeds), trials, failure rate and tokens per policy for autoresearch HPO runs |

---

## Citation

```bibtex
@article{reacomp2026,
  title   = {REACOMP: Compiling LLM Reasoning into Symbolic Solvers for Efficient Program Synthesis},
  year    = {2026}
}
```
