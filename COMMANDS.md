# Reproduction Commands

End-to-end commands to reproduce all results in the paper. Run from the project root.

Assumes a vLLM server serving **gpt-oss-120b** is reachable (default port 8002 for DF, 8004 for BoK). Set `PORT=` to override. The symbolic solver steps require no LLM server.

---

## 1. Symbolic Solver Induction

### Claude Code solver (PBEBench)

Run interactively in a Claude Code session — the agent reads the building prompt and demos directly:

```bash
# Open a Claude Code session in the project root, then instruct it:
# "Read building_prompts/SOLVER_BUILDING_PROMPT_PBE.md and
#  demos/DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json,
#  then implement the solver as described."
claude  # launches Claude Code CLI
```

Output: `built_solvers/claude_code/<timestamp>/SOLVER.py`

### Qwen3.6-Coder solver via OpenHands (PBEBench)

```bash
# Requires a vLLM server serving Qwen/Qwen3.6-35B-A3B at PORT
SOLVER_TYPE=pbe bash scripts/run_solver_builder_openhands.sh <PORT>
```

Output: `built_solvers/qwen3.6_35b_a3b/<timestamp>/SOLVER.py`

### Claude Code solver (SLR-Bench)

```bash
# Same as above but with the SLR building prompt:
# "Read building_prompts/SOLVER_BUILDING_PROMPT_SLR.md and
#  demos/DEMOS_SLRBENCH_seed_42_92_examples_with_CoT.json,
#  then implement the solver as described."
claude
```

Output: `built_solvers/claude_code/<timestamp>/SOLVER_SLR.py`

### Qwen3.6-Coder solver via OpenHands (SLR-Bench)

```bash
SOLVER_TYPE=slr bash scripts/run_solver_builder_openhands.sh <PORT>
```

Output: `built_solvers/qwen3.6_35b_a3b/<timestamp>/SOLVER_SLR.py`

---

## 2. Evaluating Symbolic Solvers

```bash
# PBEBench-Lite (max 5 programs)
python scripts/eval_solver.py \
    --solver built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py \
    --dataset lite \
    --workers 8 \
    --output-dir evals/solver_results/claude_code/Thu_Apr_23_807_PM/

# PBEBench-Hard (max 20 programs)
python scripts/eval_solver.py \
    --solver built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py \
    --dataset hard \
    --workers 8 \
    --output-dir evals/solver_results/claude_code/Thu_Apr_23_807_PM/

# Qwen solver — same flags, different solver path
python scripts/eval_solver.py \
    --solver built_solvers/qwen3.6_coder/Fri_Apr_24_200_AM/SOLVER.py \
    --dataset lite --workers 8 \
    --output-dir evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/

# SLR-Bench (eval_solver auto-detects solve_slr entry point)
python scripts/eval_solver.py \
    --solver built_solvers/claude_code/Sat_Apr_25_251_AM/SOLVER_SLR.py \
    --dataset slr \
    --workers 8 \
    --output-dir evals/solver_results/slr_claude_code/
```

Outputs: `evals/solver_results/<solver>/<timestamp>/{lite,hard,slr}.jsonl`

---

## 3. LLM Baselines (requires vLLM server with gpt-oss-120b)

### Best-of-K (BoK)

```bash
# PBEBench-Lite
DATASET=lite bash scripts/run_best_of_k_vllm.sh
# → outputs/lite_tasks_full_og_best_of_k.jsonl

# PBEBench-Hard
DATASET=hard bash scripts/run_best_of_k_vllm.sh
# → outputs/tasks_full_og_best_of_k.jsonl  (raw {input, outputs} format)

# SLR-Bench
DATASET=slr bash scripts/run_best_of_k_vllm.sh
# → outputs/slr_bench_best_of_k.jsonl
```

**Note on PBEBench-Hard BoK**: the Hard BoK outputs (`outputs/gpt_oss_120b_pbebench_hard_outputs.jsonl`) are taken from the PBEBench paper's public release and are in a raw `{input: {...}, outputs: [...]}` format. `scripts/plot_pbebench_comparison.py` and `scripts/ensemble_outputs.py` auto-detect and handle this format; no conversion is needed.

### Direct Feedback (DF)

```bash
# PBEBench-Lite
DATASET=lite bash scripts/run_direct_feedback_vllm.sh
# → outputs/lite_tasks_full_og_direct_feedback.jsonl

# PBEBench-Hard  (not run — compute cost too high; Hard results use BoK only)

# SLR-Bench
DATASET=slr bash scripts/run_direct_feedback_vllm.sh
# → outputs/slr_bench_direct_feedback.jsonl
```

Both scripts checkpoint progress (`*.ckpt.json`) and are safe to kill and restart.

---

## 4. Ensembling + Eval + Plots

One script per benchmark runs all ensemble combinations, `quick_eval`, and generates the 3 comparison figures.

```bash
# PBEBench-Lite (all 4 systems: DF, BoK, CC Solver, OH Qwen Solver)
bash scripts/run_all_pbebench_lite_evals.sh \
    --metrics-json metrics/pbebench_lite_all.json

# PBEBench-Hard (3 systems: BoK, CC Solver, OH Qwen Solver — no DF)
bash scripts/run_all_pbebench_hard_evals.sh \
    --metrics-json metrics/pbebench_hard_all.json

# SLR-Bench (all 4 systems)
bash scripts/run_all_slr_evals.sh \
    --metrics-json metrics/slr_all.json
```

Figures written to `figures/pbebench_lite_comparison_*.png`, `figures/pbebench_hard_comparison_*.png`, `figures/slr_comparison_*.png`.

To regenerate plots only (without re-running ensembles/eval):

```bash
# PBEBench-Hard
python scripts/plot_pbebench_comparison.py --split hard --plot

# PBEBench-Lite
python scripts/plot_pbebench_comparison.py --split lite --plot

# SLR-Bench (tier and level plots)
python scripts/plot_slr_comparison.py --plot
```

---

## 5. Quick Eval on Any Output File

```bash
# Single file
python scripts/quick_eval.py outputs/slr_bench_best_of_k.jsonl \
    --tasks-file data/slr_bench/v1_All_full.jsonl

# Multiple files side-by-side, save metrics JSON
python scripts/quick_eval.py \
    evals/solver_results/slr_claude_code/slr.jsonl \
    outputs/slr_bench_best_of_k.jsonl \
    outputs/slr_bench_direct_feedback.jsonl \
    --tasks-file data/slr_bench/v1_All_full.jsonl \
    --metrics-json metrics/slr_quick.json
```

---

## 6. Solver Variance Check (PBEBench-Hard)

Re-run both Hard solvers to measure run-to-run variance:

```bash
python scripts/eval_solver.py \
    --solver built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py \
    --dataset hard --workers 8 \
    --output-dir evals/solver_results/claude_code/Thu_Apr_23_807_PM/run2/

python scripts/eval_solver.py \
    --solver built_solvers/qwen3.6_coder/Fri_Apr_24_200_AM/SOLVER.py \
    --dataset hard --workers 8 \
    --output-dir evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/run2/
```

Compare run1 vs run2:

```bash
python scripts/quick_eval.py \
    evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl \
    evals/solver_results/claude_code/Thu_Apr_23_807_PM/run2/hard.jsonl \
    --tasks-file data/pbebench/tasks_full_og.jsonl
```

---

## 7. Autoresearch HPO (nanochat pre-training)

Hyperparameter optimisation of nanochat pre-training, the benchmark of [arXiv:2603.24647](https://arxiv.org/abs/2603.24647). Background, status and caveats are in the [README](README.md#autoresearch-hpo-nanochat-pre-training). Every trial on the real objective trains for 300 s on one GPU: the pilot task (1 h budget) takes a bit over an hour per policy, a task of `tasks_full.jsonl` (the paper's 24 h budget) 24 h per seed. The task files describe the paper's hardware (one H200, 141 GB, training capped to ~76 GB); the runners refuse another GPU unless `--allow-unpinned` is passed, which labels the results as not verified.

### CPU checks (no GPU, no model server)

```bash
# Unit and end-to-end tests (synthetic objective, fake LLM server, stand-in sandbox)
python -m unittest discover -s tests -t .

# Search space, starting configuration and train.py handling vs. the pinned upstream sources
python data/autoresearch_hpo/check_upstream_fidelity.py
# → PASS   (fetches three files from GitHub; --offline-dir <dir> reads local copies)

# Regenerate the task files (deterministic; the fidelity check compares them with the generator)
python data_generation/autoresearch_hpo_generator.py
# → data/autoresearch_hpo/{tasks_full,pilot_tasks,synthetic_tasks}.jsonl
```

Every launcher below also runs on the synthetic objective when `TASKS=data/autoresearch_hpo/synthetic_tasks.jsonl` is set. This is a stand-in landscape for dry runs, not nanochat: its results are labelled `synthetic (NOT benchmark)`.

```bash
POLICY=random TASKS=data/autoresearch_hpo/synthetic_tasks.jsonl bash scripts/run_hpo_eval.sh
python scripts/summarize_hpo_results.py outputs/autoresearch_hpo_synthetic_tasks_random
```

### Training environment (one CUDA GPU)

```bash
# Clones ferreirafabio/autoresearch at the pinned commit into original_baseline_repos/autoresearch,
# checks train.py / prepare.py against the pinned hashes, runs `uv sync` and prepare.py
bash scripts/setup_autoresearch_hpo.sh

# Other checkout location (export it for the runners too) / other prepare.py arguments
AUTORESEARCH_DIR=/scratch/$USER/autoresearch PREPARE_ARGS="--num-shards 10" \
    bash scripts/setup_autoresearch_hpo.sh
```

Sanity check of the objective: trial 0 of every episode trains the default configuration, which the paper reports at `val_bpb` ≈ 0.991 on an H200. The random-search pilot is the cheapest episode to see it:

```bash
POLICY=random bash scripts/run_hpo_eval.sh
head -2 outputs/autoresearch_hpo_pilot_tasks_random/autoresearch_hpo_1h_seed0.jsonl   # header row, then trial 0
```

### LLM HPO agent: baseline and reasoning traces

```bash
# Model server and training need separate GPUs
bash scripts/launch_vllm_qwen3.6_35b_a3b.sh <PORT> <NUM_GPUS>

# Pilot (1 h budget)
PORT=<PORT> TRAIN_GPU=<GPU> bash scripts/run_hpo_traces_vllm.sh

# Paper protocol (24 h budget, seeds 0-2), one seed per GPU
for s in 0 1 2; do
  PORT=<PORT> TRAIN_GPU=$s TASKS=data/autoresearch_hpo/tasks_full.jsonl \
    bash scripts/run_hpo_traces_vllm.sh --task-id autoresearch_hpo_24h_seed$s &
done; wait
```

Output: `outputs/autoresearch_hpo_<tasks>_hpo_agent_qwen3.6_35b_a3b/<task_id>.jsonl` (episode log: prompt, reasoning, configuration and outcome of every trial) and `<task_id>.result.json`. Re-running a command resumes unfinished episodes.

The default prompt asks for step-by-step reasoning before the answer (the traces the solver is induced from). The prompt and settings of the paper's "Karpathy Agent (14 HPs)" baseline:

```bash
PORT=<PORT> TRAIN_GPU=<GPU> TASKS=data/autoresearch_hpo/tasks_full.jsonl \
PROMPT_STYLE=paper MAX_TOKENS=2048 \
    bash scripts/run_hpo_traces_vllm.sh --parse-retries 0
```

Output: `outputs/autoresearch_hpo_tasks_full_hpo_agent_qwen3.6_35b_a3b_paper/`

### Demos

```bash
python -m openhands_agents.build_hpo_balanced_demos \
    --episodes outputs/autoresearch_hpo_tasks_full_hpo_agent_qwen3.6_35b_a3b \
    --n 100 --seed 42
```

Output: `demos/DEMOS_AUTORESEARCH_HPO_seed_42_<count>_examples_with_CoT.json`, balanced between decisions that improved the best `val_bpb` so far and decisions that did not. `<count>` is below 100 when the episodes contain fewer than 50 decisions of either kind (a 1 h pilot episode has about 12 trials); the building prompt and the OpenHands launcher name the 100-example file, so adjust the file name in the instruction or set `DEMOS_PATH` otherwise.

### Solver induction

Claude Code:

```bash
# Open a Claude Code session in the project root, then instruct it:
# "Read building_prompts/SOLVER_BUILDING_PROMPT_HPO.md and
#  demos/DEMOS_AUTORESEARCH_HPO_seed_42_100_examples_with_CoT.json,
#  then implement the solver as described."
claude
```

Output: `built_solvers/claude_code/<timestamp>/SOLVER_HPO.py` and `SOLVER_HPO_ALGORITHM.md`

Qwen3.6 via OpenHands (needs the model server and the Apptainer sandbox from `scripts/build_sandbox_openhands.sh`; no training GPU):

```bash
bash scripts/run_solver_builder_hpo_openhands.sh <PORT>

# Other demos file / sandbox image
DEMOS_PATH=demos/DEMOS_AUTORESEARCH_HPO_seed_42_24_examples_with_CoT.json \
SIF_PATH=/scratch/$USER/sif_images/sandbox.sif \
    bash scripts/run_solver_builder_hpo_openhands.sh <PORT>
```

Output: `built_solvers/qwen3.6_35b_a3b/<timestamp>_hpo/SOLVER_HPO.py` and `SOLVER_HPO_ALGORITHM.md` (the directory name also carries the demos file name when `DEMOS_PATH` is set). The launcher exits non-zero unless both files were written and the solver completes the synthetic practice episodes.

### Evaluating solvers and classical baselines

```bash
# Pilot (1 h budget)
SOLVER=built_solvers/claude_code/<timestamp>/SOLVER_HPO.py bash scripts/run_hpo_eval.sh
POLICY=random bash scripts/run_hpo_eval.sh
POLICY=tpe    bash scripts/run_hpo_eval.sh     # tpe, cmaes: pip install "optuna>=4.0" "cmaes>=0.12"
POLICY=cmaes  bash scripts/run_hpo_eval.sh

# Paper protocol (24 h budget, seeds 0-2), one seed per GPU
for s in 0 1 2; do
  TRAIN_GPU=$s TASKS=data/autoresearch_hpo/tasks_full.jsonl \
  SOLVER=built_solvers/claude_code/<timestamp>/SOLVER_HPO.py \
    bash scripts/run_hpo_eval.sh --task-id autoresearch_hpo_24h_seed$s &
done; wait
```

Output: `evals/solver_results/<model>/<timestamp>/autoresearch_hpo_<tasks>/` for a solver (`<model>/<timestamp>` taken from the solver's path) and `outputs/autoresearch_hpo_<tasks>_<policy>/` for a baseline; `OUTPUT_DIR=` overrides both. An episode log is tied to the run that wrote it (task record, objective and policy): resuming with a solver file whose content changed, with other LLM-agent settings or with an edited task file is refused with a message naming what differs, so a new solver or setting needs a new output directory. Solvers run with the caller's permissions (one process per decision, no sandbox); read an induced solver before evaluating it.

### Results table

```bash
python scripts/summarize_hpo_results.py \
    evals/solver_results/claude_code/<timestamp>/autoresearch_hpo_tasks_full \
    evals/solver_results/qwen3.6_35b_a3b/<timestamp>_hpo/autoresearch_hpo_tasks_full \
    outputs/autoresearch_hpo_tasks_full_hpo_agent_qwen3.6_35b_a3b \
    outputs/autoresearch_hpo_tasks_full_random \
    outputs/autoresearch_hpo_tasks_full_tpe \
    outputs/autoresearch_hpo_tasks_full_cmaes \
    --json metrics/autoresearch_hpo_tasks_full.json
```

One block per objective, hardware and budget, one row per policy: mean ± std over seeds of the best `val_bpb`, the number of trials and the share of failed trials. Only blocks whose objective is plain `nanochat` are benchmark results.
