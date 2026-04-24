# PBEBench-Hard Findings

**Dataset:** 1216 tasks, cascade length 2–20, 64 tasks per level.

## Individual system results

| System | Solved | Pass% | Mean reward |
|---|---:|---:|---:|
| BoK-32 (gpt-oss-120b) | 832 / 1216 | 68.42% | 0.9428 |
| Symbolic Solver (Claude Code) | 847 / 1216 | 69.65% | 0.9873 |
| Symbolic Solver (Qwen3.6-35B-A3B) | 716 / 1216 | 58.88% | 0.9742 |

## Ensemble results (union — best score per task across systems)

| Ensemble | Solved | Pass% | Mean reward | Δ vs best individual |
|---|---:|---:|---:|---:|
| BoK-32 ∪ Solver (Claude Code) | 966 / 1216 | 79.44% | 0.9901 | +9.79pp |
| BoK-32 ∪ Solver (Qwen3.6) | 926 / 1216 | 76.15% | 0.9836 | +6.50pp |
| **BoK-32 ∪ CC Solver ∪ Qwen Solver** | **999 / 1216** | **82.15%** | **0.9910** | **+12.50pp** |

## Complexity of solutions (solved tasks only, vs ground truth)

Selection policy: max reward first, min complexity as tiebreak.

| System | n solved | Mean pred | Mean GT | Δ (pred−GT) | Simpler | Equal | More complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| BoK-32 (gpt-oss-120b) | 832 | 36.00 | 33.69 | +2.31 | 10.5% | 49.5% | 40.0% |
| Symbolic Solver (Claude Code) | 847 | 44.21 | 36.16 | +8.06 | 0.5% | 10.7% | 88.8% |
| Symbolic Solver (Qwen3.6-35B-A3B) | 716 | 38.91 | 35.50 | +3.41 | 2.5% | 24.6% | 72.9% |

**BoK-32 finds the simplest solutions** — 49.5% equal to GT and only +2.31 mean delta, versus +8.06 for the CC solver. This is the "simpler programs" benefit of sampling 32 candidates and picking the best. Both symbolic solvers consistently overshoot GT complexity (88.8% and 72.9% more complex), likely because the induction approach finds valid but redundant programs.

## Key findings

**1. BoK-32 dominates at short cascades, solvers dominate at long cascades.**
The crossover is at CL 13 (pass rate) / CL 12 (mean reward). BoK-32 hits near 100% for CL 2–8 while both solvers are at 85–95%. At CL 16+ BoK-32 collapses to <41% while the Claude Code solver holds 55–68%.

**2. Mean reward tells a different story than pass rate.**
Qwen solver has *lower* pass rate (58.9%) than CC solver (69.7%) but *higher* mean reward on unsolved tasks — it produces near-correct partial solutions rather than failing outright. The two solvers are complementary.

**3. Ensembling is very effective.**
All three union ensembles beat every individual system by a large margin. BoK-32 ∪ CC ∪ Qwen reaches **82.15% pass rate (999/1216)** — a +12.5pp gain over the best individual (CC solver at 69.65%). The mean reward plot shows the ensemble hugging 1.0 all the way to CL 18.

**4. BoK-32 and the solvers are highly complementary.**
BoK-32 covers the easy end (low CL, high diversity from 32 samples) while the solvers bring structured induction for long cascades. The union captures both strengths.

## Files

| File | Description |
|---|---|
| `outputs/gpt_oss_120b_pbebench_outputs.jsonl` | BoK-32 raw outputs (32 candidates/task) |
| `evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl` | CC solver results |
| `evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl` | Qwen solver results |
| `figures/ensemble_hard_metrics.json` | All metrics as JSON |
| `figures/ensemble_hard_passrate.png` | Pass rate vs CL (individual + ensembles) |
| `figures/ensemble_hard_meanreward.png` | Mean reward vs CL (individual + ensembles) |
| `figures/solver_cascade_passrate_with_bok.png` | Pass rate comparison (solvers vs BoK, with crossover) |
| `figures/solver_cascade_meanreward_with_bok.png` | Mean reward comparison (solvers vs BoK) |
| `figures/complexity_hard.png` | Mean solution complexity vs CL (solved tasks only) |
| `figures/complexity_hard_metrics.json` | Complexity stats vs GT as JSON |

## Scripts

- `scripts/eval_bok_hard.py` — per-system eval + side-by-side CL breakdown
- `scripts/eval_ensemble_hard.py` — ensemble eval + plots (`--plot`, `--plot-mr`, `--metrics-json`)
- `scripts/plot_solver_cascade.py` — extended with `--bok`/`--metric` flags; auto-annotates crossover
- `scripts/complexity_analysis_hard.py` — complexity vs GT analysis + plot (`--plot`, `--metrics-json`)
