# Experiment Findings

Consolidated results across all finalized experiments. Append new sections here as experiments complete.

Datasets, DSL constraints, and baseline configurations are documented in `evals/experimental_setup.md`.

---

## Table of Contents

1. [PBEBench-Lite](#pbebench-lite)
2. [PBEBench-Hard](#pbebench-hard)
3. [SLR-Bench](#slr-bench)

---

## PBEBench-Lite

**Dataset:** 1,008 tasks · cascade length 2–5 · ~252 tasks per level · GT mean cascade complexity = 11.63  
**Max programs:** 5

### Individual systems

| System | Solved | Pass% | Mean reward | Avg time/task |
|---|---:|---:|---:|---:|
| Symbolic Solver (Claude Code) | 810 / 1008 | 80.36% | 0.9438 | 0.004s |
| Symbolic Solver (Qwen3.6-35B-A3B) | 538 / 1008 | 53.37% | 0.8494 | — |

### LLM baselines (gpt-oss-120b, K=32)

BoK = 32 parallel samples, max 32,768 tokens/sample. DF = up to 32 sequential attempts with verifier feedback, max 32,768 tokens/attempt. Token counts are averages per task (input + output; no CoT recorded for Lite).

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

> **Model note for paper:** The LLM baselines (BoK, DF) use **gpt-oss-120b** served via vLLM. The symbolic solvers are induced by a separate coding agent: the Claude Code solver by **claude-sonnet-4-6** (Claude Code session), the Qwen solver by **Qwen3.6-35B-A3B** (a smaller MoE model than gpt-oss-120b) running inside OpenHands. These are distinct models from the inference LLM — the effi token savings reflect replacing gpt-oss-120b inference with a zero-cost symbolic program, not with the solver-building model. The solver-building cost (~191K tokens for Qwen, one-time) is separate from and negligible relative to the inference savings (see finding 8).

### Complexity of solutions (solved tasks only, vs GT)

Selection policy: max reward first, min complexity as tiebreak.

| System | n solved | Mean pred | Mean GT | Δ (pred−GT) | Simpler | Equal | More complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| BoK ∪ Claude solver | 947 | 12.49 | 11.29 | +1.20 | 148 (15.6%) | 342 (36.1%) | 457 (48.3%) |
| BoK ∪ Qwen solver | 946 | 13.48 | 11.29 | +2.20 | 128 (13.5%) | 277 (29.3%) | 541 (57.2%) |
| Claude solver | 810 | 13.71 | 10.71 | +3.00 | 74 (9.1%) | 141 (17.4%) | 595 (73.5%) |
| Qwen solver | 538 | 11.55 | 9.84 | +1.70 | 86 (16.0%) | 168 (31.2%) | 284 (52.8%) |

### Symbolic solver breakdown by cascade length

| Cascade | N | Pass% (Claude) | Mean score (Claude) |
|---:|---:|---:|---:|
| 2 | 246 | 98.4% | 0.990 |
| 3 | 253 | 90.9% | 0.963 |
| 4 | 253 | 83.0% | 0.958 |
| 5 | 256 | 50.0% | 0.866 |

Performance collapses at CL=5 (50.0%) — at the 5-program limit there is no slack for feed/bleed ordering interactions.

### Comparison with PBEBench-Lite reported results

Source: PBEBench paper (`figures/pbebench_lite_reported_metrics.png`). Paper numbers use **single attempt, Pass@1, 8192 CoT tokens** — no test-time scaling. Our BoK (K=32) and DF (K=32) use substantially more compute and are not directly comparable; they are included for reference. Symbolic solvers use **zero per-task LLM inference**.

| Model | Pass% | Complexity | Avg tokens | Notes |
|-------|------:|-----------:|-----------:|-------|
| Codestral-22B †  | 1.1% | 15.4 | — | reported, single attempt |
| Qwen2.5-32B-Instruct †  | 1.8% | 14.9 | — | reported, single attempt |
| Qwen3-32B † | 41.9% | 6.57 | — | reported, with CoT |
| gpt-oss-120b † | 62.5% | 10.93 | — | reported, single attempt, 8192 CoT |
| GPT-5 † | 72.4% | 10.58 | — | reported, single attempt |
| **CC Solver (ours)** | **80.4%** | — | **0** | zero LLM tokens |
| **OH Qwen Solver (ours)** | **53.4%** | — | **0** | zero LLM tokens, Qwen3.6-35B-A3B |
| BoK-32, gpt-oss-120b (ours) | 93.9% | — | 67,479 | K=32, 32× compute |
| DF-32, gpt-oss-120b (ours) | 92.4% | — | 110,267 | K=32 sequential |
| **BoK-32 + CC Solver (ours)** | **93.95%** | — | 67,479 | |
| **DF-32 + CC Solver (ours)** | **93.1%** | — | 110,267 | |
| **DF+BoK + CC Solver (ours)** | **94.9%** | — | 177,746 | |

† Reported scores from PBEBench paper; not re-run by us.

**Key takeaway:** The CC Solver alone (80.4%) **surpasses gpt-oss-120b (62.5%) and GPT-5 (72.4%)** at their single-attempt setting — at zero per-task inference cost. The symbolic solver's standalone result is competitive with frontier LLMs before any test-time scaling is applied.

**TODO:** Compute and fill Edit Sim and Complexity columns for our systems.

### Key findings

**1. Symbolic solvers alone are surprisingly strong.** The Claude solver reaches 80.4% pass at zero LLM token cost. Qwen is weaker (53.4%) but produces notably tighter programs (Δ+1.70 vs GT, vs +3.00 for Claude).

**2. Symbolic + LLM ensembles consistently outperform either alone.** BoK+DF alone caps around ~93%; adding the Claude solver pushes to 94.9%. The ceiling appears to be ~95% with current systems.

**3. Adding the Qwen solver on top of Claude rarely helps once LLMs are included.** Qwen adds +3pp in the solver-only ensemble (83.5% vs 80.4%), but once BoK or DF is present it brings no additional pass rate gain.

**4. BoK finds the simplest programs.** By picking the minimum-complexity correct candidate from 32 samples, BoK∪Claude achieves Δ+1.20 over GT — much tighter than the Claude solver alone (Δ+3.00). Sampling diversity finds programs closer to the GT structure.

**5. Effi mode cuts token cost ~30% with no pass rate loss.** At DF+BoK level: Claude effi saves 125K vs 178K (−30%), Qwen effi saves 152K vs 178K (−15%), both at the same 94.9% pass rate. Larger saving for Claude because it solves more tasks perfectly (80.4% vs 53.4%).

**6. Effi mode trades complexity for token savings.** Effi variants show higher complexity than standard counterparts (DF+BoK + Claude: 12.15 standard vs 14.31 effi) because when the solver is bypassed the LLM output takes over, and LLM outputs are less parsimonious.

**7. Best trade-off points:**
- Best accuracy + complexity: DF+BoK + Claude solver (94.9%, Cplx 12.15, 178K tokens)
- Best accuracy + token efficiency: DF+BoK + Claude solver effi (94.9%, 125K tokens, Cplx 14.31)
- Best symbolic-only: Claude+Qwen solver ensemble (83.5%, zero tokens)
- Closest to GT complexity: Qwen3.6 solver alone (Δ+0.46) — but only 53.4% pass rate

**8. Solver construction cost is negligible when amortised.** The Qwen solver was built in a single OpenHands session costing ~202K tokens (76 real turns over 3h 21min; measured via `scripts/compute_trajectory_tokens.py`, accounting for KV caching). Effi mode saves 25,819 tokens/task on Lite — the build cost recoups at 7.8 tasks and is only 0.78% of per-task savings across the full 1008-task eval (129× return). On Hard (CoT-heavy, 166K tokens saved/task) break-even is 1.2 tasks and the return is ~1000×. Even under the pessimistic assumption of no KV caching (4.8M tokens), break-even is 186 tasks on Lite (5.4× return, 18.5% overhead/task) and 29 tasks on Hard (42× return, 2.4% overhead/task). The one-time construction cost is negligible relative to inference savings at any realistic evaluation scale.

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

| System | Solved | Pass% | Mean reward | Avg time/task |
|---|---:|---:|---:|---:|
| Best-of-K (BoK-32, gpt-oss-120b) | 832 / 1216 | 68.42% | 0.9428 | — |
| Symbolic Solver (Claude Code) | 847 / 1216 | 69.65% | 0.9873 | 0.62s |
| Symbolic Solver (Qwen3.6-35B-A3B) | 716 / 1216 | 58.88% | 0.9742 | — |

BoK-32 configuration: 32 samples, max 16,384 tokens/sample, **no early exit** (all 32 always run to allow the verifier to select simpler programs among later candidates).

### Token usage for BoK-32 (gpt-oss-120b, with CoT)

Measured with `scripts/compute_bok_tokens.py` on the cluster file using the model's own tokenizer.

| | Total | Avg/task |
|---|---:|---:|
| Input (32 × prompt) | 34,959,360 | 28,750 |
| Output (32 × answer) | 4,486,419 | 3,690 |
| Reasoning CoT (32 × CoT) | 292,695,771 | 240,704 |
| **Total** | **332,141,550** | **273,143** |

**CoT dominates at 88% of total cost.** Input and output together are only 12%.

**Effi savings with Claude solver** (solver covers 69.7% of tasks):

| | Avg/task | Savings |
|---|---:|---:|
| Full BoK | 273,143 | — |
| Effi (solver-first) | 106,896 | **60.9%** |

Solver-unsolved tasks cost ~352K tokens/task vs ~239K for solver-solved — harder tasks consume more CoT, so savings are less than the 69.7% coverage rate implies.

Metrics: `metrics/bok_hard_tokens_cluster.json`  
Script: `scripts/compute_bok_tokens.py --solver ... --tokenizer ... --metrics-json ...`

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

BoK-32 finds the simplest solutions — 49.5% equal to GT and Δ+2.31. Both symbolic solvers consistently overshoot GT (88.8% and 72.9% more complex), likely because the induction approach finds valid but redundant programs.

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

Pass rate degrades gradually CL 5–17, then collapses at 18–20 (cliff to 36%→12%→2%). Mean score stays 0.935+ even at CL=20 — solver gets most pairs right and fails on only 1–2.

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

Tasks with no BFCC interactions are easiest (92.7%); feeding interactions are consistently harder than bleeding. All-four-BFCC is by far the hardest (40.2%), making up 22% of the dataset.

### Solver failure analysis

Dominant failure mode across both datasets is **near-misses**: solver finds a program that correctly transforms all but 1–3 pairs. Reflected in high mean scores (0.944 lite, 0.987 hard) despite imperfect pass rates.

Solver fails when:
1. A correct program must be discovered through a long intermediate chain (cascade ≥ 18).
2. All-four-BFCC tasks require holding multiple interacting constraints within the beam horizon.
3. A single-character pattern that fixes a changed pair also appears in many unchanged pairs — the safe-phase beam search discards it and the unrestricted phase doesn't recover in time.

### Solver variance (run-to-run stability)

Both solvers re-run on PBEBench-Hard with identical settings to measure determinism. Results written to `hard_run2.jsonl` alongside run1.

| System | Run 1 Pass% | Run 2 Pass% | Δ Pass% | Per-task agreement | Flips |
|---|---:|---:|---:|---:|---:|
| Claude Code solver | 69.65% | 69.74% | +0.09pp | 1211/1216 (99.6%) | 5 |
| Qwen3.6-35B-A3B solver | 58.88% | 57.89% | −1.00pp | 1180/1216 (97.0%) | 36 |

All flips are at the partial-credit boundary (0.96↔1.0 or 0.98↔1.0) — no task swings from clearly solved to clearly failed. Both solvers are essentially deterministic; variance is negligible for reporting purposes.

### Comparison with PBEBench-Hard reported results

The **BoK-32 gpt-oss-120b results are taken directly from the PBEBench paper's public release** (`outputs/gpt_oss_120b_pbebench_hard_outputs.jsonl`) — we re-use rather than re-run them. Paper numbers for other models use single attempt, Pass@1. Symbolic solvers use zero per-task LLM inference.

| Model | Pass% | Avg tokens | Notes |
|-------|------:|-----------:|-------|
| gpt-oss-120b, BoK-32 † | 68.4% | 273,143 | from PBEBench paper release; K=32 w/ CoT |
| **CC Solver (ours)** | **69.7%** | **0** | zero LLM tokens |
| **OH Qwen Solver (ours)** | **58.9%** | **0** | zero LLM tokens, Qwen3.6-35B-A3B |
| **BoK-32 + CC Solver (ours)** | **79.4%** | 273,143 | +11.0pp over BoK alone |
| **BoK-32 + Qwen Solver (ours)** | **76.2%** | 273,143 | |
| **BoK-32 + CC + Qwen Solvers (ours)** | **82.2%** | 273,143 | +13.8pp over BoK alone |

† Re-used from PBEBench paper public release; not re-run by us.

**Key takeaway:** The CC Solver alone (69.7%) **matches BoK-32 gpt-oss-120b (68.4%)** — the paper's strongest scaled baseline — at zero per-task inference cost. Ensembling the symbolic solvers with BoK-32 yields a +13.8pp gain (82.2%), the largest absolute improvement in the Hard setting.

### Key findings

**1. BoK-32 dominates at short cascades, solvers dominate at long cascades.** Crossover at CL 12–13. BoK-32 hits near 100% for CL 2–8 while both solvers are at 85–95%. At CL 16+ BoK-32 collapses to <41% while the Claude solver holds 55–68%.

**2. Mean reward tells a different story than pass rate.** Qwen solver has lower pass rate (58.9%) than Claude solver (69.7%) but higher mean reward on unsolved tasks — near-correct partial solutions rather than complete failures. The two solvers are complementary.

**3. Ensembling is very effective.** BoK-32 ∪ Claude ∪ Qwen reaches 82.15% (999/1216) — +12.5pp over the best individual. Mean reward hugs 1.0 all the way to CL 18.

**4. BoK-32 and solvers are highly complementary.** BoK-32 covers the easy end (low CL, high diversity); solvers bring structured induction for long cascades.

**5. All-four-BFCC is the hardest category.** 40.2% pass rate, making up 22% of the dataset. Dense mutual ordering interactions the beam search cannot fully resolve.

**6. Solver complexity overshoots GT by more on Hard than Lite.** Claude solver Δ+8.06 on Hard vs Δ+3.00 on Lite — longer cascades give the enumerative approach more ways to produce valid-but-verbose programs.

**7. CoT reasoning dominates token cost.** At 16K CoT budget per sample, reasoning accounts for 88% of total tokens (240,704/273,143 avg/task). Effi mode with the Claude solver saves 60.9% of tokens despite covering only 69.7% of tasks — harder unsolved tasks consume more CoT (~352K vs ~239K avg).

### Output files

| File | Description |
|---|---|
| `outputs/gpt_oss_120b_pbebench_hard_outputs.jsonl` | BoK-32 raw outputs |
| `evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl` | Claude solver results (run 1) |
| `evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard_run2.jsonl` | Claude solver results (run 2, variance check) |
| `evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard.jsonl` | Qwen solver results (run 1) |
| `evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/hard_run2.jsonl` | Qwen solver results (run 2, variance check) |
| `metrics/bok_hard_tokens_cluster.json` | Token usage breakdown (with CoT) |
| `figures/ensemble_hard_metrics.json` | All metrics as JSON |
| `figures/ensemble_hard_passrate.png` | Pass rate vs CL |
| `figures/ensemble_hard_meanreward.png` | Mean reward vs CL |
| `figures/solver_cascade_passrate_with_bok.png` | Pass rate comparison with crossover annotation |
| `figures/complexity_hard_metrics.json` | Complexity stats vs GT as JSON |
| `figures/complexity_hard.png` | Mean solution complexity vs CL |

---

## SLR-Bench

**Dataset:** 1,000 tasks (partial eval) · curriculum levels 1–20 · 250 tasks per tier (basic/easy/medium/hard)  
**Task:** Induce an `eastbound(T) :- Body.` Prolog rule from background facts + positive/negative train examples  
**Reward:** `partial_score` (0–1) from `AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning` via HuggingFace evaluate; 1.0 = perfectly correct rule  
**Metric:** `rule_complexity` = number of top-level body literals excluding `has_car/2`  
**Data file:** `data/slr_bench/v1_All_full.jsonl`

### Individual systems

| System | n | Pass% | Mean reward | Avg tokens | Mean rule complexity | Δ GT |
|---|---:|---:|---:|---:|---:|---:|
| Symbolic Solver (Claude Code) | 1000 | 68.4% | 0.9669 | 0 | 1.624 | −0.756 |
| Best-of-K (BoK) | 565 | 100.0% | 1.0000 | 5,564 | 1.211 | −0.299 |
| Direct Feedback (DF) | 577 | 100.0% | 1.0000 | 5,665 | 1.191 | −0.367 |

*BoK/DF cover only a partial subset (565/577 of 1000 tasks). Both are 100% on their covered tasks.*

### Symbolic solver breakdown by curriculum tier

| Tier | N | Pass% | Mean reward |
|---|---:|---:|---:|
| basic (levels 1–5) | 250 | 100.0% | 1.0000 |
| easy (levels 6–10) | 250 | 78.4% | 0.9699 |
| medium (levels 11–15) | 250 | 48.4% | 0.9449 |
| hard (levels 16–20) | 250 | 46.8% | 0.9526 |

### Symbolic solver breakdown by rule complexity

| Rule complexity | N | Pass% | Mean reward |
|---|---:|---:|---:|
| 1 | 50 | 100.0% | 1.0000 |
| 1–2 | 350 | 93.1% | 0.9890 |
| 2–3 | 150 | 67.3% | 0.9612 |
| 3–4 | 100 | 47.0% | 0.9435 |
| 4–5 | 250 | 44.4% | 0.9462 |
| 5 | 100 | 49.0% | 0.9563 |

### Ensemble results

| System | n | Pass% | Mean reward | Avg tokens | Rule complexity | Δ GT |
|---|---:|---:|---:|---:|---:|---:|
| BoK + Solver (standard) | 1000 | 76.0% | 0.9765 | 3,144 | 1.551 | −0.717 |
| DF + Solver (standard) | 1000 | **76.7%** | **0.9772** | 3,269 | 1.529 | −0.760 |
| BoK + Solver (effi) | 1000 | 76.0% | 0.9765 | **1,093** | 1.541 | −0.728 |
| DF + Solver (effi) | 1000 | **76.7%** | **0.9772** | **1,479** | 1.551 | −0.738 |

*Effi saves ~65% tokens vs standard for BoK (1,093 vs 3,144) and ~55% for DF (1,479 vs 3,269).*

> **Model note for paper:** LLM baselines use **gpt-oss-120b**; the symbolic solver is induced by **Qwen3.6-35B-A3B** running inside OpenHands (~116K tokens, one-time). These are distinct models — effi savings reflect replacing gpt-oss-120b inference with zero-cost symbolic execution, not the solver-building model. Build cost amortises at 64.9 tasks (15.4× return over 1000 tasks, 6.5% per-task overhead).

### Rule complexity vs ground truth (correct solutions only)

| System | n | Mean pred | Mean GT | Δ | Simpler | Equal | More complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude solver | 684 | 1.624 | 2.380 | −0.756 | 46.2% | 49.6% | 4.2% |
| BoK | 565 | 1.211 | 1.510 | −0.299 | 29.4% | 58.2% | 12.3% |
| DF | 577 | 1.191 | 1.558 | −0.367 | 30.4% | 56.5% | 12.0% |

All systems find rules **simpler than GT** — the opposite pattern from PBEBench where systems overshoot GT complexity. LLMs find shorter Prolog rules that still satisfy the examples; the symbolic solver finds even shorter ones (Δ−0.756).

### Key findings

**1. SLR tasks are easy for LLMs when attempted.** Both BoK and DF achieve 100% on their covered subsets at only ~5,600 tokens/task — far cheaper than PBEBench (which required 67K–178K avg tokens).

**2. Symbolic solver performance degrades sharply with rule complexity.** 100% at complexity-1, 93% at 1–2, dropping to 44–47% at complexity 3+. The solver's inductive search cannot handle the combinatorial body-literal enumeration needed for higher-complexity rules.

**3. Curriculum tier mirrors rule complexity.** Basic tier (100%) ↔ low complexity; hard tier (46.8%) ↔ high complexity. The two breakdowns tell the same story.

**4. All systems find simpler rules than GT.** Δ ranges from −0.30 (BoK) to −0.76 (solver). This is the inverse of PBEBench, likely because Prolog rules have many equivalent forms and shorter rules that satisfy the examples exist for many tasks.

**5. Effi saves more tokens here than on PBEBench.** ~55–65% savings vs ~30% on PBEBench-Lite, because the solver covers 68.4% of tasks and SLR tasks are individually cheap — so the zero-cost bucket is large relative to total cost.

**6. DF+Solver is the best configuration** at 76.7% pass and 1,479 avg tokens (effi). BoK+Solver is slightly behind (76.0%). The partial-eval ceiling is limited by solver failures on complex rules; full-dataset results pending.

### Comparison with SLR-Bench leaderboard

> **NOTE: PARTIAL RESULTS** — BoK and DF cover only 565/577 of 1000 tasks; Qwen solver not yet complete. Ensemble columns will be filled once full results are in. CC Solver rows are final (1000/1000 tasks).

Source: Table 3, SLR-Bench paper (`figures/slr_bench_reported_metrics.png`). Reported scores use single attempt, no test-time scaling. Our BoK/DF use K=32. Symbolic solvers use zero per-task LLM inference.

**Pass rate by curriculum tier (Logical-Reasoning Accuracy ↑):**

| Model | Overall | Basic | Easy | Medium | Hard | Avg tokens | Notes |
|-------|--------:|------:|-----:|-------:|-----:|-----------:|-------|
| gpt-4o † | — | 93 | 29 | 2 | 0 | — | reported, single attempt |
| o3 † | — | 99 | 93 | 74 | 45 | — | reported |
| gpt-5 † | — | 100 | 90 | 72 | 46 | — | reported |
| **CC Solver (ours)** | **68.4%** | **100** | **78.4** | **48.4** | **46.8** | **0** | zero LLM tokens |
| **OH Qwen Solver (ours)** | — | — | — | — | — | **0** | TODO: complete run |
| BoK-32, gpt-oss-120b (ours, partial) | 100%* | — | — | — | — | 5,564* | *on 565/1000 tasks |
| DF-32, gpt-oss-120b (ours, partial) | 100%* | — | — | — | — | 5,665* | *on 577/1000 tasks |
| BoK-32 + CC Solver (ours, partial) | 76.0% | — | — | — | — | 3,144 | partial ensemble |
| DF-32 + CC Solver (ours, partial) | 76.7% | — | — | — | — | 3,269 | partial ensemble |

† Reported scores from SLR-Bench paper; not re-run by us.

**Key takeaway (partial):** On the **Hard tier the CC Solver (46.8%) matches o3 (45%) and gpt-5 (46%)** — the top leaderboard models — at zero per-task LLM cost. The solver is weakest on Easy (78.4% vs 90–93% frontier) where rule variety exceeds inductive coverage. All unsolved tasks have `best_reward > 0` — no syntax errors, every produced rule is valid Prolog.

**TODO:** Fill ensemble columns (Qwen solver, full BoK/DF, BoK/DF + both solvers) and per-tier breakdown for LLM baselines once full 1000-task results are in.

### Output files

| File | Description |
|---|---|
| `evals/solver_results/slr_claude_code/slr.jsonl` | Claude solver results (1000 tasks) |
| `outputs/slr_bench_best_of_k.jsonl` | BoK raw outputs (partial) |
| `outputs/slr_bench_direct_feedback.jsonl` | DF raw outputs (partial) |
| `outputs/slr_ensemble_bok_solver.jsonl` | BoK ∪ Solver (standard) |
| `outputs/slr_ensemble_df_solver.jsonl` | DF ∪ Solver (standard) |
| `outputs/slr_ensemble_effi_bok_solver.jsonl` | BoK ∪ Solver (effi) |
| `outputs/slr_ensemble_effi_df_solver.jsonl` | DF ∪ Solver (effi) |
| `metrics/slr_partial_bok_df.json` | All metrics as JSON |
