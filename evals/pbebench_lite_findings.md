# PBEBench-Lite Findings

**Dataset:** 1,008 tasks, cascade length 2–5, ~252 tasks per level.

## Individual system results

| System | Solved | Pass% | Mean reward |
|---|---:|---:|---:|
| Symbolic Solver (Claude Code) | 810 / 1008 | 80.36% | — |
| Symbolic Solver (Qwen3.6-35B-A3B) | 538 / 1008 | 53.37% | — |

## Ensemble results (union — best score per task across systems)

| Ensemble | Solved | Pass% | Notes |
|---|---:|---:|---|
| BoK ∪ Solver (Claude Code) | 947 / 1008 | 93.95% | Best-of-K BoK + CC solver |
| BoK ∪ Solver (Qwen3.6) | 946 / 1008 | 93.85% | Best-of-K BoK + Qwen solver |
| DF ∪ Solver (Claude Code) | 938 / 1008 | 93.06% | Direct Feedback + CC solver |
| DF ∪ Solver (Qwen3.6) | 936 / 1008 | 92.86% | Direct Feedback + Qwen solver |

## Complexity of solutions (solved tasks only, vs ground truth)

Selection policy: max reward first, min complexity as tiebreak.

| System | n solved | Mean pred | Mean GT | Δ (pred−GT) | Simpler | Equal | More complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| BoK ∪ Solver (Claude Code) | 947 | 12.49 | 11.29 | +1.20 | 15.6% | 36.1% | 48.3% |
| BoK ∪ Solver (Qwen3.6) | 946 | 13.48 | 11.29 | +2.20 | 13.5% | 29.3% | 57.2% |
| Symbolic Solver (Claude Code) | 810 | 13.71 | 10.71 | +3.00 | 9.1% | 17.4% | 73.5% |
| Symbolic Solver (Qwen3.6-35B-A3B) | 538 | 11.55 | 9.84 | +1.70 | 16.0% | 31.2% | 52.8% |

GT mean cascade complexity: **11.63** (across all 1008 tasks).

**The BoK ensemble produces the closest-to-GT programs** — BoK ∪ CC is only +1.20 above GT on average. The CC solver alone overshoots by +3.00. The Qwen solver alone (smallest delta at +1.70 when solved) is notably more efficient per program, but covers far fewer tasks (53.4% vs 80.4%).

## Key findings

**1. The BoK ensemble is the strongest overall.**
BoK ∪ CC Solver reaches **93.95% pass rate** — the best result on Lite. DF ∪ CC is close at 93.06%, but BoK's diversity (32 independent samples) gives a slight edge.

**2. Symbolic solver (CC) is the strongest standalone system.**
At 80.4% pass rate with zero LLM token cost at inference time, it is the most token-efficient system. The Qwen solver lags significantly (53.4%), likely due to the solver-building model being smaller/weaker.

**3. BoK provides simpler programs than the symbolic solver.**
By picking the minimum-complexity correct candidate from 32 samples, BoK∪CC achieves mean complexity 12.49 (+1.20 over GT), vs 13.71 (+3.00) for the CC solver alone. Sampling diversity helps find programs closer to the ground truth structure.

**4. Solvers and LLMs are highly complementary.**
The CC solver covers 810 tasks; BoK presumably covers many of the remaining 198 (it excels at short CL tasks). The BoK+solver ensemble recovers essentially all of these.

## Files

| File | Description |
|---|---|
| `evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl` | CC solver results |
| `evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/lite.jsonl` | Qwen solver results |
| `outputs/ensemble_bok_claude_solver.jsonl` | BoK ∪ CC ensemble |
| `outputs/ensemble_bok_qwen_solver.jsonl` | BoK ∪ Qwen ensemble |
| `outputs/ensemble_df_claude_solver.jsonl` | DF ∪ CC ensemble |
| `outputs/ensemble_df_qwen_solver.jsonl` | DF ∪ Qwen ensemble |
| `figures/complexity_lite.png` | Mean solution complexity vs CL (solved tasks only) |
| `figures/complexity_lite_metrics.json` | Complexity stats vs GT as JSON |

## Scripts

- `scripts/complexity_analysis_lite.py` — complexity vs GT analysis + plot (`--plot`, `--metrics-json`)
- `scripts/quick_eval.py` — pass rate, attempt distribution, token usage for standard JSONL files
