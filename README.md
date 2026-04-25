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

Full step-by-step commands for all experiments are in **[COMMANDS.md](COMMANDS.md)**:

1. Solver induction (Claude Code CLI or OpenHands + Qwen)
2. Evaluating symbolic solvers on PBEBench / SLR-Bench
3. Running LLM baselines (BoK, Direct Feedback) via vLLM
4. Ensembling + eval + figure generation
5. Quick eval on any output file
6. Solver variance check

---

## Project Structure

```
.
├── building_prompts/
│   ├── SOLVER_BUILDING_PROMPT_PBE.md   # spec given to coding agent for PBEBench solver
│   └── SOLVER_BUILDING_PROMPT_SLR.md   # spec given to coding agent for SLR-Bench solver
├── demos/
│   ├── DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json   # 100 PBEBench reasoning traces
│   └── DEMOS_SLRBENCH_seed_42_92_examples_with_CoT.json    # 92 SLR-Bench reasoning traces
├── built_solvers/
│   ├── claude_code/          # solvers induced by Claude Code
│   └── qwen3.6_35b_a3b/      # solvers induced by Qwen via OpenHands
├── data/
│   ├── pbebench/             # PBEBench task files
│   └── slr_bench/            # SLR-Bench task files
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
