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

## Domains (reusable layer)

Task families are packaged as *domains* (`domains/`): tasks, agent-visible fields, verifier,
solver entry point, building prompt and demo traces. The historical scripts below are
unchanged; two generic entry points work for every registered domain:

```bash
python -m openhands_agents.run --framework direct_solve --domain pbebench --split lite --output-path outputs/traces.jsonl ...   # traces
python scripts/build_demos.py --domain pbebench --split hard --traces outputs/traces.jsonl --n 100 --out demos/DEMOS_new.json     # demos
python -m openhands_agents.build_solver --domain pbebench --output-dir built_solvers/<run> ...                                     # induction
python scripts/eval_domain_solver.py --domain pbebench --split lite --solver built_solvers/<run>/SOLVER.py --output-dir evals/...  # evaluation
python scripts/hybrid_gate.py --domain pbebench --solver <solver results.jsonl> --fallback <LLM results.jsonl> --out <hybrid.jsonl>  # hybrid
```

Registered domains: `pbebench`, `slr_bench`, `list_functions`, `envsetup` (environment
bootstrapping on SetupBench and EnvBench-Python, where the solver acts in a Docker container),
and five KinDER robotics domains `kinder_<family>` (below).
See `domains/README.md` for the contract and `domains/envsetup/README.md`, `domains/kinder/README.md`
for the new domains.

---

## KinDER Robotics Domains

ReaComp on robot task and motion planning. A per-instance LLM agent solves single scenes of a
[KinDER](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden) environment family by
writing and testing a policy against a simulator of that scene; a coding agent compiles those reasoning
traces into **one policy program for the whole family**; the program is evaluated on 100 held-out scenes
with zero LLM calls, alone and as the first stage of a solver-first hybrid with the agent as fallback.
The protocol is AgenticGenTAMP's ([tomsilver/robocode](https://github.com/tomsilver/robocode)): same
environments and instance generator, same object-count schedule (test counts go beyond the training
counts: the hard tier), step budgets, 60 s policy-time limit, `GeneratedApproach` interface and strict
black-box access (standard library, numpy, scipy; no simulator inside the program).

| Family (domain) | Goal | Train counts | Test counts (extrapolation) |
|-----------------|------|--------------|-----------------------------|
| `kinder_obstruction2d` | place the target block on an obstructed surface | 0–2 | 0–4 (3, 4) |
| `kinder_clutteredretrieval2d` | retrieve the target block from clutter | 1, 3, 5, 10 | up to 25 (15, 20, 25) |
| `kinder_clutteredstorage2d` | put all blocks inside the shelf | 1, 3, 7 | up to 15 (11, 15) |
| `kinder_motion2d` | reach the target through narrow passages | 0–3 | 0–5 (4, 5) |
| `kinder_stickbutton2d` | touch every button, using a stick for far ones | 1–3 | up to 10 (5, 10) |

Implemented: task files (100 train + 100 test instances per family, `data/kinder/`, verified against
KinDER's registered environments), exact replay verifier (`rewards/kinder.py`), the per-task agent
(`openhands_agents/kinder_direct_solve/`, traces + LLM baseline + hybrid fallback), solver induction with
the OpenHands builder or with Claude Code (`scripts/kinder_workspace.py`), solver evaluation and hybrid
through the generic domain scripts, and a report by regime and object count (`scripts/kinder_report.py`).
Everything except the LLM runs on CPU; no Docker or Apptainer is needed.

```bash
python -m domains.tests.test_kinder        # local checks (mock LLM, CPU), ~30 s
```

GPU runs, step by step: **[COMMANDS.md §7](COMMANDS.md#7-kinder-robotics-domains)**. Protocol details,
verification and caveats (different test seeds from AgenticGenTAMP's private suite, unregistered counts,
harness differences, scope): **[domains/kinder/README.md](domains/kinder/README.md)**.

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
│   └── SOLVER_BUILDING_PROMPT_KINDER.md # spec for a KinDER family policy (+ the family's env description)
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
│   └── kinder/               # KinDER task files (<family>_{train,test}.jsonl) + env descriptions
├── data_generation/
│   └── kinder_generator.py   # build / check / verify the KinDER task files
├── domains/                  # domain registry; domains/kinder/ = KinDER simulator, sandbox, domain
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
│   └── kinder.py             # KinDER reward: exact replay of an action sequence
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
│   ├── run_kinder_direct_solve_openhands.sh    # KinDER per-task agent (traces / baseline)
│   ├── run_kinder_solver_builder_openhands.sh  # KinDER SolverBuilder (no Apptainer needed)
│   ├── run_all_kinder_evals.sh         # KinDER solver eval + hybrid + report
│   ├── kinder_workspace.py             # working directory + simulator for Claude Code induction
│   └── kinder_report.py                # pass rate by regime / object count, tokens
├── openhands_agents/
│   ├── solver_builder/       # SolverBuilder agent (controller + tools)
│   ├── kinder_direct_solve/  # KinDER per-task agent (controller + tools)
│   └── run_kinder_direct_solve.py  # its entry point
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
