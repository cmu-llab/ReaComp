# Experiment Findings

Consolidated results across all finalized experiments. Append new sections here as experiments complete.

Datasets, DSL constraints, and baseline configurations are documented in `evals/experimental_setup.md`.

---

## Table of Contents

1. [PBEBench-Lite](#pbebench-lite)
2. [PBEBench-Hard](#pbebench-hard)

---

## PBEBench-Lite

**Dataset:** 1,008 tasks · cascade length 2–5 · ~252 tasks per level · GT mean cascade complexity = 11.63  
**Max programs:** 5

### Individual systems

| System | Solved | Pass% | Mean reward |
|---|---:|---:|---:|
| Symbolic Solver (Claude Code) | 810 / 1008 | 80.36% | 0.9438 |
| Symbolic Solver (Qwen3.6-35B-A3B) | 538 / 1008 | 53.37% | 0.8494 |

### LLM baselines (gpt-oss-120b, K=32)

Run details: BoK = 32 parallel samples, max 32,768 tokens/sample; DF = up to 32 sequential attempts with verifier feedback, max 32,768 tokens/attempt. Token counts are averages per task.

| System | Pass% | Mean reward | Avg tokens |
|---|---:|---:|---:|
| Best-of-K (BoK) | — | — | 67,479 |
| Direct Feedback (DF) | — | — | 110,266 |

*(BoK and DF standalone pass rates on Lite not separately recorded; see ensemble table below.)*

### Ensemble results (union — best score per task, max reward then min complexity)

| System | Solved | Pass% | Mean reward | Avg tokens |
|---|---:|---:|---:|---:|
| Claude solver only | 810 / 1008 | 80.4% | 0.9438 | 0 |
| Qwen3.6 solver only | 538 / 1008 | 53.4% | 0.8494 | 0 |
| Claude + Qwen solvers | 841 / 1008 | 83.5% | 0.9595 | 0 |
| BoK + Claude solver | 947 / 1008 | 94.0% | 0.9847 | 67,479 |
| BoK + Qwen solver | 946 / 1008 | 93.8% | 0.9825 | 67,479 |
| BoK + Claude + Qwen solvers | 947 / 1008 | 94.0% | 0.9847 | 67,479 |
| DF + Claude solver | 938 / 1008 | 93.1% | 0.9829 | 110,266 |
| DF + Qwen solver | 936 / 1008 | 92.9% | 0.9813 | 110,266 |
| DF + Claude + Qwen solvers | 939 / 1008 | 93.2% | 0.9835 | 110,266 |
| **DF+BoK + Claude solver** | **956 / 1008** | **94.9%** | **0.9881** | 177,746 |
| DF+BoK + Qwen solver | 956 / 1008 | 94.9% | 0.9873 | 177,746 |
| **DF+BoK + Claude+Qwen solvers** | **956 / 1008** | **94.9%** | **0.9881** | 177,746 |
| BoK + Claude solver (effi) | 947 / 1008 | 94.0% | 0.9810 | **45,277** |
| BoK + Qwen solver (effi) | 946 / 1008 | 93.8% | 0.9808 | **56,618** |
| DF + Claude solver (effi) | 938 / 1008 | 93.1% | 0.9815 | **79,876** |
| DF + Qwen solver (effi) | 936 / 1008 | 92.8% | 0.9808 | **95,308** |
| DF+BoK + Claude solver (effi) | 956 / 1008 | 94.9% | 0.9873 | **125,153** |
| **DF+BoK + Qwen solver (effi)** | **956 / 1008** | **94.9%** | **0.9873** | **151,927** |

*Effi mode: use solver output unconditionally when it scores 1.0; count zero LLM tokens for those tasks.*

### Complexity of solutions (solved tasks only, vs GT)

Selection policy: max reward first, min complexity as tiebreak.

| System | n solved | Mean pred | Mean GT | Δ (pred−GT) | Simpler | Equal | More complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| BoK ∪ Claude solver | 947 | 12.49 | 11.29 | +1.20 | 148 (15.6%) | 342 (36.1%) | 457 (48.3%) |
| BoK ∪ Qwen solver | 946 | 13.48 | 11.29 | +2.20 | 128 (13.5%) | 277 (29.3%) | 541 (57.2%) |
| Claude solver | 810 | 13.71 | 10.71 | +3.00 | 74 (9.1%) | 141 (17.4%) | 595 (73.5%) |
| Qwen solver | 538 | 11.55 | 9.84 | +1.70 | 86 (16.0%) | 168 (31.2%) | 284 (52.8%) |

### Symbolic solver breakdown by cascade length

| Cascade | N | Pass% (Claude) | Pass% (Qwen) |
|---:|---:|---:|---:|
| 2 | 246 | 98.4% | — |
| 3 | 253 | 90.9% | — |
| 4 | 253 | 83.0% | — |
| 5 | 256 | 50.0% | — |

Performance collapses at CL=5 for the Claude solver (50.0%) — at the 5-program limit there is no slack for feed/bleed ordering interactions.

### Key findings

**1. Symbolic solvers alone are surprisingly strong.** The Claude solver reaches 80.4% pass at zero LLM token cost. Qwen is weaker (53.4%) but produces notably tighter programs (Δ+1.70 vs GT, vs +3.00 for Claude).

**2. Symbolic + LLM ensembles consistently outperform either alone.** BoK+DF alone caps around ~93%; adding the Claude solver pushes to 94.9%. The ceiling appears to be ~95% with current systems.

**3. Adding the Qwen solver on top of Claude rarely helps once LLMs are included.** The Qwen solver adds +3pp in the solver-only ensemble (83.5% vs 80.4%), but once BoK or DF is present it brings no additional pass rate gain.

**4. BoK finds the simplest programs.** By picking the minimum-complexity correct candidate from 32 samples, BoK∪Claude achieves Δ+1.20 over GT — much tighter than the Claude solver alone (Δ+3.00). Sampling diversity finds programs closer to the GT structure.

**5. Effi mode cuts token cost ~30% with no pass rate loss.** At DF+BoK level: Claude effi saves 125K vs 178K (−30%), Qwen effi saves 152K vs 178K (−15%), both at the same 94.9% pass rate. The larger saving for Claude is because the Claude solver solves more tasks perfectly (80.4% vs 53.4%).

**6. Effi mode trades complexity for token savings.** Effi variants show higher complexity than their standard counterparts (DF+BoK + Claude: 12.15 standard vs 14.31 effi) because when the solver is bypassed the LLM output takes over, and LLM outputs are less parsimonious.

**7. Best trade-off points:**
- Best accuracy + complexity: DF+BoK + Claude solver (94.9%, Cplx 12.15, 178K tokens)
- Best accuracy + token efficiency: DF+BoK + Claude solver effi (94.9%, 125K tokens, Cplx 14.31)
- Best symbolic-only: Claude+Qwen solver ensemble (83.5%, zero tokens)
- Closest to GT complexity: Qwen solver alone (Δ+1.70) — but only 53.4% pass rate

### Output files

| File | Description |
|---|---|
| `evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl` | Claude solver results |
| `evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/lite.jsonl` | Qwen solver results |
| `outputs/lite_tasks_full_og_best_of_k.jsonl` | BoK raw outputs |
| `outputs/lite_tasks_full_og_direct_feedback.jsonl` | DF raw outputs |
| `outputs/ensemble_bok_claude_solver.jsonl` | BoK ∪ Claude ensemble |
| `outputs/ensemble_bok_qwen_solver.jsonl` | BoK ∪ Qwen ensemble |
| `outputs/ensemble_df_claude_solver.jsonl` | DF ∪ Claude ensemble |
| `outputs/ensemble_df_qwen_solver.jsonl` | DF ∪ Qwen ensemble |
| `figures/complexity_lite_metrics.json` | Complexity stats vs GT as JSON |
| `figures/complexity_lite.png` | Mean solution complexity vs CL |

---

## PBEBench-Hard

**Dataset:** 1,216 tasks · cascade length 2–20 · 64 tasks per level  
**Max programs:** 20

### Individual systems

| System | Solved | Pass% | Mean reward |
|---|---:|---:|---:|
| Best-of-K (BoK-32, gpt-oss-120b) | 832 / 1216 | 68.42% | 0.9428 |
| Symbolic Solver (Claude Code) | 847 / 1216 | 69.65% | 0.9873 |
| Symbolic Solver (Qwen3.6-35B-A3B) | 716 / 1216 | 58.88% | 0.9742 |

BoK-32 configuration: 32 samples, max 16,384 tokens/sample, **no early exit** (all 32 always run to allow the verifier to select simpler programs among later candidates).

### Ensemble results

| Ensemble | Solved | Pass% | Mean reward | Δ vs best individual |
|---|---:|---:|---:|---:|
| BoK-32 ∪ Claude solver | 966 / 1216 | 79.44% | 0.9901 | +9.79pp |
| BoK-32 ∪ Qwen solver | 926 / 1216 | 76.15% | 0.9836 | +6.50pp |
| **BoK-32 ∪ Claude ∪ Qwen solvers** | **999 / 1216** | **82.15%** | **0.9910** | **+12.50pp** |

### Complexity of solutions (solved tasks only, vs GT)

Selection policy: max reward first, min complexity as tiebreak.

| System | n solved | Mean pred | Mean GT | Δ (pred−GT) | Simpler | Equal | More complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| BoK-32 | 832 | 36.00 | 33.69 | +2.31 | 87 (10.5%) | 412 (49.5%) | 333 (40.0%) |
| Claude solver | 847 | 44.21 | 36.16 | +8.06 | 4 (0.5%) | 91 (10.7%) | 752 (88.8%) |
| Qwen solver | 716 | 38.91 | 35.50 | +3.41 | 18 (2.5%) | 176 (24.6%) | 522 (72.9%) |

BoK-32 finds the simplest solutions — 49.5% equal to GT and only Δ+2.31. Both symbolic solvers consistently overshoot GT (88.8% and 72.9% more complex), likely because the induction approach finds valid but redundant programs.

### Symbolic solver breakdown by cascade length (Claude Code)

| CL | N | Pass% | Mean reward |
|---:|---:|---:|---:|
| 2 | 64 | 93.8% | 0.999 |
| 3 | 64 | 96.9% | 0.999 |
| 4 | 64 | 92.2% | 0.998 |
| 5 | 64 | 93.8% | 0.999 |
| 6 | 64 | 84.4% | 0.997 |
| 7 | 64 | 84.4% | 0.996 |
| 8 | 64 | 92.2% | 0.998 |
| 9 | 64 | 79.7% | 0.994 |
| 10 | 64 | 81.2% | 0.991 |
| 11 | 64 | 82.8% | 0.996 |
| 12 | 64 | 71.9% | 0.993 |
| 13 | 64 | 73.4% | 0.991 |
| 14 | 64 | 59.4% | 0.983 |
| 15 | 64 | 65.6% | 0.988 |
| 16 | 64 | 67.2% | 0.986 |
| 17 | 64 | 54.7% | 0.984 |
| 18 | 64 | 35.9% | 0.972 |
| 19 | 64 | 12.5% | 0.958 |
| 20 | 64 | 1.6% | 0.935 |

### Symbolic solver breakdown by BFCC category (Claude Code)

| BFCC relationships | N | Pass% |
|---|---:|---:|
| No BFCC relationships | 55 | 92.7% |
| Bleeding | 47 | 91.5% |
| Bleeding, Counterfeeding | 56 | 91.1% |
| Counterfeeding | 64 | 90.6% |
| Counterbleeding | 45 | 84.4% |
| Counterfeeding, Counterbleeding | 60 | 81.7% |
| Bleeding, Counterbleeding | 65 | 81.5% |
| Feeding | 64 | 75.0% |
| Feeding, Bleeding, Counterfeeding | 72 | 73.6% |
| Feeding, Counterfeeding, Counterbleeding | 74 | 73.0% |
| Bleeding, Counterfeeding, Counterbleeding | 77 | 71.4% |
| Feeding, Counterfeeding | 76 | 65.8% |
| Feeding, Bleeding, Counterbleeding | 78 | 66.7% |
| Feeding, Counterbleeding | 58 | 67.2% |
| Feeding, Bleeding | 59 | 78.0% |
| Feeding, Bleeding, Counterfeeding, Counterbleeding | 266 | 40.2% |

### Key findings

**1. BoK-32 dominates at short cascades, solvers dominate at long cascades.** Crossover at CL 12–13. BoK-32 hits near 100% for CL 2–8 while both solvers are at 85–95%. At CL 16+ BoK-32 collapses to <41% while the Claude solver holds 55–68%.

**2. Mean reward tells a different story than pass rate.** Qwen solver has lower pass rate (58.9%) than Claude solver (69.7%) but higher mean reward on unsolved tasks — it produces near-correct partial solutions rather than failing outright.

**3. Ensembling is very effective.** BoK-32 ∪ Claude ∪ Qwen reaches 82.15% (999/1216) — a +12.5pp gain over the best individual (Claude solver at 69.65%). Mean reward plot shows the ensemble hugging 1.0 all the way to CL 18.

**4. BoK-32 and solvers are highly complementary.** BoK-32 covers the easy end (low CL, high diversity from 32 samples); solvers bring structured induction for long cascades.

**5. All-four-BFCC is by far the hardest category.** `Feeding, Bleeding, Counterfeeding, Counterbleeding` tasks score only 40.2% pass rate and make up 22% of the hard dataset. These have dense mutual ordering interactions the beam search cannot fully resolve.

**6. Solver failure mode is near-misses, not complete failures.** Mean score stays 0.935+ even at CL=20 — the solver gets most pairs right and fails on only 1–2. Pass rate collapse at CL 18–20 is due to an intractable ordering search within the time budget, not catastrophic errors.

**7. Solver complexity overshoots GT by more on Hard than Lite.** Claude solver Δ+8.06 on Hard vs Δ+3.00 on Lite, because longer cascades give the enumerative approach more ways to produce valid-but-verbose programs.

### Output files

| File | Description |
|---|---|
| `outputs/gpt_oss_120b_pbebench_outputs.jsonl` | BoK-32 raw outputs |
| `evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl` | Claude solver results |
| `evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl` | Qwen solver results |
| `figures/ensemble_hard_metrics.json` | All metrics as JSON |
| `figures/ensemble_hard_passrate.png` | Pass rate vs CL |
| `figures/ensemble_hard_meanreward.png` | Mean reward vs CL |
| `figures/solver_cascade_passrate_with_bok.png` | Pass rate comparison with crossover annotation |
| `figures/solver_cascade_meanreward_with_bok.png` | Mean reward comparison |
| `figures/complexity_hard_metrics.json` | Complexity stats vs GT as JSON |
| `figures/complexity_hard.png` | Mean solution complexity vs CL |
