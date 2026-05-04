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
│   └── SOLVER_BUILDING_PROMPT_SLR.md   # spec given to coding agent for SLR-Bench solver
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
│   └── slr_bench/            # SLR-Bench task files
├── evals/
│   ├── solver_results/       # per-task JSONL from eval_solver.py
│   ├── experimental_setup.md # full experimental details
│   └── README.md             # eval directory guide
├── outputs/                  # LLM baseline and ensemble outputs
├── figures/                  # comparison plots (PNG)
├── metrics/                  # quick_eval JSON summaries
├── rewards/
│   ├── pbebench.py           # PBEBench reward + cascade_complexity
│   └── slrbench.py           # SLR-Bench reward + rule_complexity
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
│   └── run_solver_builder_openhands.sh # run SolverBuilder agent via OpenHands
├── openhands_agents/
│   └── solver_builder/       # SolverBuilder agent (controller + tools)
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

---

## Citation

```bibtex
@article{reacomp2026,
  title   = {REACOMP: Compiling LLM Reasoning into Symbolic Solvers for Efficient Program Synthesis},
  year    = {2026}
}
```
