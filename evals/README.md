# Evaluation Outputs

This directory contains evaluation artifacts for two benchmarks: **PBEBench** (Lite + Hard) and **SLR-Bench**.

The paper story in brief: a coding agent reads reasoning traces (`DEMOS.json`) and induces a symbolic solver (`SOLVER.py`) in a single session — at zero per-task LLM cost. That solver is then ensembled with LLM baselines (Best-of-K, Direct Feedback) to push performance beyond what either approach achieves alone.

Full experimental setup: `experimental_setup.md`. Consolidated results: `findings.md`.

---

## Benchmarks

| Benchmark | Tasks | Difficulty axis | Notes |
|-----------|------:|-----------------|-------|
| **PBEBench-Lite** | 1,008 | Cascade length 2–5 | Up to 5 `replace()` programs per solution |
| **PBEBench-Hard** | 1,216 | Cascade length 2–20 | Up to 20 `replace()` programs per solution |
| **SLR-Bench** | 1,000 | Curriculum tier (basic/easy/medium/hard) | Prolog rule induction; `eastbound(T) :- Body.` |

---

## Induced Symbolic Solvers

Two solvers are induced from `DEMOS.json` (100 reasoning traces, 25 per quadrant of difficulty × success):

| Solver | Inducing agent | Path |
|--------|---------------|------|
| **Claude Code solver** | claude-sonnet-4-6 (Claude Code session) | `built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py` |
| **Qwen3.6-Coder solver** | Qwen3.6-35B-A3B inside OpenHands | `built_solvers/qwen3.6_coder/Fri_Apr_24_200_AM/SOLVER.py` |
| **SLR CC solver** | claude-sonnet-4-6 (Claude Code session) | `built_solvers/claude_code/Sat_Apr_25_251_AM/SOLVER_SLR.py` |
| **SLR Qwen solver** | Qwen3.6-35B-A3B inside OpenHands | `built_solvers/qwen3.6_35b_a3b/Sat_Apr_25_643_AM/SOLVER_SLR.py` |

Solver results live under `solver_results/<solver_name>/<timestamp>/`.

---

## LLM Baselines

All LLM baselines use **gpt-oss-120b** served via vLLM. Two strategies:

- **Best-of-K (BoK)**: K=32 parallel samples, pick highest-reward candidate (ties broken by complexity).
- **Direct Feedback (DF)**: up to k=32 sequential attempts with verifier feedback; early exit on reward=1.0.

---

## Ensembling

Two strategies in `scripts/ensemble_outputs.py`:

- **Standard**: best reward → lowest complexity → source order. All LLM tokens counted.
- **Effi**: if the symbolic solver is perfect (reward=1.0) use it at zero LLM token cost; otherwise fall back to best LLM candidate.

---

## Eval Scripts

| Script | Purpose |
|--------|---------|
| `scripts/eval_solver.py` | Evaluate `SOLVER.py` on PBEBench or SLR-Bench; writes standard JSONL |
| `scripts/quick_eval.py` | Accuracy, mean reward, token usage, complexity for any JSONL output |
| `scripts/ensemble_outputs.py` | Build standard or effi ensemble from multiple sources |
| `scripts/run_all_pbebench_lite_evals.sh` | Full PBEBench-Lite pipeline: ensembles + eval + plots |
| `scripts/run_all_pbebench_hard_evals.sh` | Full PBEBench-Hard pipeline: ensembles + eval + plots |
| `scripts/run_all_slr_evals.sh` | Full SLR-Bench pipeline: ensembles + eval + plots |

---

## Figures

Comparison figures (4-way: DF, BoK, CC Solver, OH Qwen Solver) live in `figures/`:

| Figure | Description |
|--------|-------------|
| `pbebench_lite_comparison_passrate.png` | Accuracy vs cascade length (Lite) |
| `pbebench_lite_comparison_meanreward.png` | Mean reward vs cascade length (Lite) |
| `pbebench_lite_comparison_complexity.png` | Cascade complexity vs cascade length (Lite) |
| `pbebench_hard_comparison_passrate.png` | Accuracy vs cascade length (Hard) |
| `pbebench_hard_comparison_meanreward.png` | Mean reward vs cascade length (Hard) |
| `pbebench_hard_comparison_complexity.png` | Cascade complexity vs cascade length (Hard) |
| `slr_comparison_passrate.png` | Accuracy vs curriculum tier (SLR) |
| `slr_comparison_meanreward.png` | Mean reward vs curriculum tier (SLR) |
| `slr_comparison_complexity.png` | Rule complexity vs curriculum tier (SLR) |
| `slr_comparison_level_*.png` | Same three metrics vs curriculum level 1–20 (SLR) |

---

## Metric Definitions

| Metric | Description |
|--------|-------------|
| `best_reward` | Continuous score in [0,1]: fraction of I/O pairs correctly transformed. 1.0 = fully solved. |
| `solved` | `best_reward >= 1.0` |
| `pass_rate` | Fraction of tasks with `solved=True` |
| `mean_reward` | Mean `best_reward` across tasks (includes partial credit) |
| `cascade_complexity` | Sum of `len(A) + len(B)` over all `replace(A,B)` programs in the solution |
| `rule_complexity` | SLR-Bench: number of literals in the rule body (excluding `has_car/2`) |
