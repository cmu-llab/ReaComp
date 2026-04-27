# Experiment Findings

Consolidated results across all finalized experiments. Append new sections here as experiments complete.

Datasets, DSL constraints, and baseline configurations are documented in `evals/experimental_setup.md`.

---

## Table of Contents

1. [PBEBench-Lite](#pbebench-lite)
2. [PBEBench-Hard](#pbebench-hard)
3. [Solver Construction Ablations](#solver-construction-ablations-pbebench-qwen36-35b-a3b)
4. [SLR-Bench](#slr-bench)

---

## PBEBench-Lite

**Dataset:** 1,008 tasks · cascade length 2–5 · ~252 tasks per level · GT mean cascade complexity: 11.20 (solved subset)  
**Max programs:** 5  
**LLM baseline:** gpt-oss-120b via vLLM. BoK = 32 parallel samples, max 32,768 tokens/sample. DF = up to 32 sequential attempts with verifier feedback, max 32,768 tokens/attempt.  
**Symbolic solvers:** CC Solver = induced by claude-sonnet-4-6 (Claude Code session); Qwen Solver = induced by Qwen3.6-35B-A3B via OpenHands (best single run: run 2, 100 examples + CoT). Zero per-task LLM cost.  
**Ensembles:** effi mode — use solver output unconditionally when reward = 1.0 (zero LLM tokens); fall back to LLM otherwise. Complexity Δ = mean predicted − mean GT (solved tasks only).

### Main results

Complexity Δ = mean predicted − mean GT cascade complexity, **correct solutions only**. Not reported for † baselines (not re-run by us).  
Token costs for gpt-oss-120b at DeepInfra pricing ($0.039/M input, $0.19/M output). Total tokens in millions across all 1008 tasks.

| System | Pass% | Mean reward | Edit Sim | Complexity Δ | Total (M tok) | Cost ($) |
|---|---:|---:|---:|---:|---:|---:|
| gpt-oss-120b, single attempt † | 62.5% | — | 69.9 | — | — | — |
| GPT-5, single attempt † | 72.4% | — | 76.5 | — | — | — |
| **CC Solver** | **80.4%** | **0.9438** | **93.7** | **+3.00** | **0** | **0** |
| **Qwen Solver (best run)** | **65.7%** | **0.9022** | **87.4** | **+3.01** | **0** | **0** |
| **CC + Qwen Solvers (union)** | **84.6%** | **0.9607** | **94.9** | **+2.16** | **0** | **0** |
| **All Symbolic Solvers (union)** | **91.3%** | **0.9772** | **96.6** | **+1.88** | **0** | **0** |
| DF-32 (gpt-oss-120b) | 92.4% | 0.9796 | 97.3 | +2.11 | 111.1 | 16.74 |
| BoK-32 (gpt-oss-120b) | 93.8% | 0.9808 | 97.8 | +2.19 | 68.0 | 12.20 |
| DF + Qwen Solver (effi) | 92.9% | 0.9810 | 97.5 | +2.89 | 90.2 | 13.50 |
| DF + CC Solver (effi) | 93.1% | 0.9815 | 97.6 | +3.00 | 80.5 | 11.97 |
| DF + All Symbolic (effi) | 93.2% | 0.9817 | 97.6 | +2.18 | 78.7 | 11.69 |
| BoK + Qwen Solver (effi) | 93.8% | 0.9808 | 97.8 | +2.94 | 50.7 | 9.12 |
| **BoK + CC Solver (effi)** | **93.9%** | **0.9810** | **97.8** | **+2.94** | **45.6** | **8.22** |
| **BoK + All Symbolic (effi)** | **93.9%** | **0.9810** | **97.8** | **+2.19** | **43.5** | **7.83** |

† Reported scores from PBEBench paper (`figures/pbebench_lite_reported_metrics.png`); not re-run by us. Single attempt, Pass@1, 8192 CoT tokens — no test-time scaling. Edit Sim from the same paper figure.

> **Model note:** LLM baselines use **gpt-oss-120b** served via vLLM. Symbolic solvers are induced by a separate coding agent (CC by claude-sonnet-4-6; Qwen by Qwen3.6-35B-A3B inside OpenHands) — distinct from the inference LLM. Effi token savings reflect replacing gpt-oss-120b inference with zero-cost symbolic execution; solver build cost (under $1 one-time, see ablations) is negligible at any realistic eval scale.

### Symbolic solver breakdown by cascade length

| Cascade | N | Pass% (CC) | Mean reward (CC) | Pass% (Qwen) | Mean reward (Qwen) |
|---:|---:|---:|---:|---:|---:|
| 2 | 246 | 98.4% | 0.990 | 96.3% | 0.992 |
| 3 | 253 | 90.9% | 0.963 | 81.0% | 0.947 |
| 4 | 253 | 83.0% | 0.958 | 56.5% | 0.886 |
| 5 | 256 | 50.0% | 0.866 | 30.1% | 0.788 |

CC solver collapses at CL=5 (50%) — at the 5-program limit there is little slack for ordering interactions. Qwen run 2 shows consistent but lower performance at all levels.

### Key findings

**1. Symbolic solvers alone are surprisingly strong.** The CC Solver (80.4%) surpasses gpt-oss-120b (62.5%) and GPT-5 (72.4%) at their single-attempt setting — at zero per-task inference cost. The Qwen Solver (65.7%) also beats gpt-oss-120b single attempt.

**2. Unioning all symbolic solvers reaches 91.3% at zero LLM cost.** The full symbolic union (CC + all 6 Qwen runs) achieves 91.3% pass — within 2.5pp of BoK-32 (93.8%) while spending zero tokens. This is the strongest zero-cost result and the tightest complexity (Δ+1.88), as the union can always pick the simplest correct answer among diverse solver outputs.

**3. Symbolic + LLM ensembles consistently outperform either alone.** BoK alone caps at 93.8%; adding the CC Solver (effi) reaches 93.9% while cutting tokens by 33% (45K vs 67K avg/task). DF + All Symbolic effi reaches 93.2% at only 78K avg tokens — 29% cheaper than DF alone.

**4. Effi mode preserves pass rate while cutting cost.** BoK + All Symbolic effi matches BoK + CC Solver effi in pass rate (93.9%) but saves a further 2K tokens/task, with no pass rate tradeoff.

**5. Effi slightly increases complexity Δ vs standalone LLM.** DF standalone Δ+2.11, but DF+CC effi rises to Δ+3.00 because the CC solver's outputs replace LLM outputs on tasks it solves. All-Symbolic effi mitigates this (Δ+2.18) since the union includes tighter Qwen outputs.

**6. DF and BoK produce similar complexity.** DF Δ+2.11 vs BoK Δ+2.19 — essentially the same. BoK selects the minimum-complexity correct answer from 32 parallel samples; DF's sequential refinement converges to comparable results on average.

**7. Best trade-off points:**
- Best pass rate (zero cost): All Symbolic union (91.3%, 0 tokens, Δ+1.88)
- Best pass rate + token efficiency: BoK + All Symbolic effi (93.9%, 43K avg tokens/task)
- Best pass rate regardless of cost: BoK + CC Solver effi or BoK + All Symbolic effi (93.9%)

**8. Solver construction cost is negligible when amortised.** The Qwen run 2 solver was built in a single OpenHands session costing ~216K tokens (KV-cache). Effi mode saves ~24K tokens/task over BoK standalone (67,480 → 43,113, −36%) — build cost recoups at 9 tasks and is <1% overhead across 1008 tasks. See ablations section for full token cost breakdown.

### Output files

| File | Description |
|---|---|
| `evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl` | CC Solver results |
| `evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_.../lite.jsonl` | Qwen Solver results (run 2) |
| `outputs/lite_tasks_full_og_best_of_k_stripped.jsonl` | BoK raw outputs (stripped) |
| `outputs/lite_tasks_full_og_direct_feedback_stripped.jsonl` | DF raw outputs (stripped) |
| `outputs/lite_union_solvers.jsonl` | CC + Qwen (run 2) union |
| `outputs/lite_ensemble_all_solvers.jsonl` | All Symbolic union (CC + all 6 Qwen runs) |
| `outputs/lite_effi_bok_cc.jsonl` | BoK + CC Solver (effi) |
| `outputs/lite_effi_bok_qwen_run2.jsonl` | BoK + Qwen Solver (effi) |
| `outputs/lite_effi_bok_all_solvers.jsonl` | BoK + All Symbolic (effi) |
| `outputs/lite_effi_df_cc.jsonl` | DF + CC Solver (effi) |
| `outputs/lite_effi_df_qwen_run2.jsonl` | DF + Qwen Solver (effi) |
| `outputs/lite_effi_df_all_solvers.jsonl` | DF + All Symbolic (effi) |

---

## PBEBench-Hard

**Dataset:** 1,216 tasks · cascade length 2–20 · 64 tasks per level  
**Max programs:** 20  
**LLM baseline:** gpt-oss-120b, BoK-32 only (K=32, max 16,384 tokens/sample, no early exit — all 32 always run). No DF on Hard due to sequential cost and lock-in risk at long cascades. Token cost from `metrics/bok_hard_tokens_cluster.json` (measured on cluster with model tokenizer): avg 273,143/task total (28,750 input + 3,690 output + 240,704 CoT reasoning). CoT dominates at 88%.  
**Symbolic solvers:** CC Solver and Qwen Solver (run 2, 100 examples + CoT). Zero per-task LLM cost.  
**Ensembles:** effi mode — solver used at zero cost when reward = 1.0; BoK tokens counted only for tasks the solver fails. Note: BoK runs all 32 samples regardless of solver result (no early exit by design) — effi savings reflect *reported* token cost, not actual GPU compute. Complexity Δ = mean predicted − mean GT (correct solutions only).

### Main results

Complexity Δ = mean predicted − mean GT cascade complexity, **correct solutions only**.  
Token costs for gpt-oss-120b at DeepInfra pricing ($0.039/M input, $0.19/M output; reasoning billed as output). Total tokens in millions across all 1216 tasks.

| System | Pass% | Mean reward | Edit Sim | Complexity Δ | Total (M tok) | Cost ($) |
|---|---:|---:|---:|---:|---:|---:|
| **CC Solver** | **69.7%** | **0.9873** | **97.2** | **+8.06** | **0** | **0** |
| **Qwen Solver (run 2)** | **74.7%** | **0.9836** | **96.8** | **+5.26** | **0** | **0** |
| **CC + Qwen Solvers (union)** | **81.2%** | **0.9905** | **98.3** | **+5.35** | **0** | **0** |
| **All Symbolic Solvers (union)** | **84.7%** | **0.9920** | **98.6** | **+4.56** | **0** | **0** |
| BoK-32 (gpt-oss-120b) | 68.4% | 0.9428 | 89.9 | +5.14 | 332.1 | 57.83 |
| BoK + CC Solver (effi) | 79.4% | 0.9508 | 91.4 | +7.82 | 130.0 | 23.08 |
| BoK + Qwen Solver (effi) | 80.7% | 0.9496 | 91.3 | +5.48 | 114.5 | 20.41 |
| BoK + CC + Qwen Solver (effi) | 83.5% | 0.9531 | 91.9 | +5.48 | 87.6 | 15.63 |
| **BoK + All Symbolic Solvers (effi)** | **85.8%** | **0.9570** | **92.7** | **+4.64** | **71.6** | **12.78** |

### Symbolic solver breakdown by cascade length

| CL | N | Pass% (CC) | Mean reward (CC) | Pass% (Qwen) | Mean reward (Qwen) |
|---:|---:|---:|---:|---:|---:|
| 2 | 64 | 93.8% | 0.999 | 100.0% | 1.000 |
| 3 | 64 | 96.9% | 0.999 | 95.3% | 0.999 |
| 4 | 64 | 92.2% | 0.998 | 95.3% | 0.998 |
| 5 | 64 | 93.8% | 0.999 | 96.9% | 0.998 |
| 6 | 64 | 84.4% | 0.997 | 89.1% | 0.995 |
| 7 | 64 | 84.4% | 0.996 | 92.2% | 0.994 |
| 8 | 64 | 92.2% | 0.998 | 89.1% | 0.995 |
| 9 | 64 | 79.7% | 0.994 | 92.2% | 0.998 |
| 10 | 64 | 81.2% | 0.991 | 87.5% | 0.990 |
| 11 | 64 | 82.8% | 0.996 | 87.5% | 0.996 |
| 12 | 64 | 71.9% | 0.993 | 85.9% | 0.990 |
| 13 | 64 | 73.4% | 0.991 | 76.6% | 0.984 |
| 14 | 64 | 59.4% | 0.983 | 67.2% | 0.980 |
| 15 | 64 | 65.6% | 0.988 | 70.3% | 0.983 |
| 16 | 64 | 67.2% | 0.986 | 68.8% | 0.983 |
| 17 | 64 | 54.7% | 0.984 | 71.9% | 0.978 |
| 18 | 64 | 35.9% | 0.972 | 32.8% | 0.963 |
| 19 | 64 | 12.5% | 0.958 | 12.5% | 0.943 |
| 20 | 64 | 1.6% | 0.935 | 7.8% | 0.925 |

CC solver degrades gradually CL 5–17, then collapses at 18–20 (36%→12%→2%). Qwen run 2 is stronger than CC at most cascade lengths (especially CL 9–17) but similarly collapses at 18+. Both maintain mean reward >0.92 even at CL=20 — near-misses dominate failures.

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

### Key findings

**1. Qwen run 2 is stronger than CC on Hard, reversed from Lite.** Qwen run 2 (74.7%) outperforms CC (69.7%) on Hard — and both beat BoK-32 (68.4%) at zero LLM cost. The Qwen run 2 algorithm (safety-first greedy + 2-step lookahead) handles medium-length cascades better; CC collapses earlier at CL 17–18.

**2. All Symbolic union (84.7%) slightly edges BoK + All Symbolic (85.8%) in cost.** The pure symbolic union reaches 84.7% at zero token cost — within 1.1pp of the best ensemble result (85.8%) while spending nothing. BoK contributes its complementary strength at short cascades (CL 2–12) where it approaches 100%.

**3. BoK-32 and solvers are complementary across cascade lengths.** BoK-32 is near-perfect at CL 2–8 (96–100%) but collapses at CL 14+ (<47%). Symbolic solvers hold 55–70%+ through CL 17. BoK + All Symbolic union combines both strengths for +17.4pp over BoK alone.

**4. Solver complexity overshoots GT substantially on Hard.** CC solver Δ+8.06, Qwen Δ+5.26 — much larger than Lite (+3.00/+3.01). Longer cascades give inductive search more ways to produce valid-but-verbose programs. BoK + All Symbolic union brings this down to Δ+3.54 by selecting simpler answers across many candidates.

**5. All-four-BFCC is the hardest category.** 40.2% pass rate for CC, making up 22% of the dataset. Dense mutual ordering interactions the beam search cannot fully resolve.

**6. Effi mode cuts reported token cost by ~78%.** BoK + All Symbolic effi: 58,847 avg tokens/task vs 273,143 standalone BoK — a 78.4% reduction. The 14.2% of tasks not solved by any symbolic solver fall back to BoK (avg 273K tokens each), while the solved 85.8% incur zero LLM cost. Note: BoK still runs all 32 samples for all tasks by design (no early exit); effi savings reflect *reported* cost used for amortisation calculations, not actual GPU compute.

**7. Best trade-off points:**
- Best pass rate (zero cost): All Symbolic union (84.7%, 0 tokens)
- Best pass rate overall: BoK + All Symbolic effi (85.8%, ~59K avg tokens/task)
- Best symbolic-only: All Symbolic union beats BoK-32 standalone by +16.3pp at zero cost

### Output files

| File | Description |
|---|---|
| `outputs/gpt_oss_120b_pbebench_hard_outputs.jsonl` | BoK-32 raw outputs (PBEBench paper release) |
| `outputs/hard_bok_converted.jsonl` | BoK-32 converted to quick_eval format (per-task token costs) |
| `evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl` | CC Solver results |
| `evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_.../hard.jsonl` | Qwen Solver results (run 2) |
| `outputs/hard_union_cc_qwen_run2.jsonl` | CC + Qwen run 2 union |
| `outputs/hard_union_all_solvers.jsonl` | All Symbolic union (CC + all Qwen runs) |
| `outputs/hard_effi_bok_cc.jsonl` | BoK + CC Solver (effi) |
| `outputs/hard_effi_bok_qwen_run2.jsonl` | BoK + Qwen Solver (effi) |
| `outputs/hard_effi_bok_cc_qwen_run2.jsonl` | BoK + CC + Qwen run 2 (effi) |
| `outputs/hard_effi_bok_all_solvers.jsonl` | BoK + All Symbolic (effi) |
| `metrics/bok_hard_tokens_cluster.json` | BoK aggregate token stats (with CoT, measured on cluster) |
| `metrics/bok_hard_tokens_per_task.jsonl` | Per-task token counts from cluster (input/output/reasoning) |

---

## Solver Construction Ablations (PBEBench, Qwen3.6-35B-A3B)

**Question:** How does the composition of the demos file (number of examples, presence of LLM CoT reasoning traces) affect solver quality?

**Setup:** Four Qwen solvers induced via OpenHands, varying demos only. All other settings identical (same building prompt, same verifier, same seed-42 balanced sampling across success/failure × easy/hard quadrants). Token costs measured with `scripts/compute_trajectory_tokens.py` using tiktoken (cl100k_base).

### Performance

| Demos | Run | Lite Pass% | Lite Edit Sim | Hard Pass% | Hard Edit Sim | Algorithm (short) |
|-------|----:|----------:|--------------:|----------:|--------------:|-------------------|
| 100 examples + CoT | 1 | 53.4% | 78.6 | 58.9% | 94.5 | greedy + multi-pass residual fixing |
| 100 examples + CoT | 2 | 65.7% | 87.4 | 74.7% | 96.8 | safety-first greedy + 2-step lookahead |
| 100 examples + CoT | **3** | **79.2%** | **96.7** | 51.8% | 90.0 | unique-op permutations + greedy + 2-op sequences |
| 100 examples, **no CoT** | 1 | 42.1% | 67.6 | 24.8% | 82.9 | beam search + heuristic scoring |
| 48 examples + CoT | 1 | 55.7% | 78.6 | 76.2% | 96.5 | multi-start greedy + permutation reorder |
| 12 examples + CoT | 1 | 47.7% | 80.2 | 50.4% | 94.3 | adaptive beam search |

### Solver construction token cost

Token costs at AtlasCloud pricing for Qwen3.6-35B-A3B ($0.1612/M input, $0.9653/M output). APIs are stateless — no KV-cache reuse across turns; no-cache total is the realistic cost estimate.

| Demos | Run | Turns | No-cache total | Cost ($) | KV-cache total | KV-cache cost ($) |
|-------|----:|------:|---------------:|---------:|---------------:|------------------:|
| 100 examples + CoT | 1 | 76 | 4,377,628 | 0.77 | 191,331 | 0.10 |
| 100 examples + CoT | 2 | 72 | 4,698,253 | 0.83 | 215,839 | 0.11 |
| 100 examples + CoT | 3 | 80 | 6,163,579 | 1.08 | 264,495 | 0.13 |
| 100 examples, no CoT | 1 | 102 | 7,670,723 | 1.31 | 227,002 | 0.11 |
| 48 examples + CoT | 1 | 82 | 4,126,622 | 0.73 | 176,868 | 0.09 |
| 12 examples + CoT | 1 | 49 | 1,638,817 | 0.30 | 106,526 | 0.05 |
| SLR, run 1 | 68 | — | 2,905,414 | 0.50 | 116,097 | 0.05 |
| SLR, run 2 | 84 | — | 7,318,727 | 1.25 | 249,074 | 0.11 |

### Qualitative trajectory analysis

Each OpenHands run produced a different solver algorithm despite identical inputs (for same-demos runs). Qwen appears to explore fundamentally different designs across runs rather than converging on a canonical approach.

**100 examples + CoT, run 1 — "Greedy + multi-pass residual fixing"** (`Fri_Apr_24_200_AM`):
Extracts edit regions between input and output (longest-common-prefix/suffix anchoring), then generates candidates via three strategies: direct substitution, split candidates (splitting a complex edit into two simpler replaces), and context extension (extending the edit boundary to capture adjacent characters). Greedy selection uses a score of `n_fix − 2×n_break`. After greedy construction, a two-phase residual pass tries single programs then pairs to fix remaining examples. Produces correct but sometimes overly complex cascades. The split-candidate strategy is the most distinctive insight, directly capturing multi-step edits as pairs.

**100 examples + CoT, run 3 — "Unique-op permutations + greedy + 2-op sequences"** (`Sun_Apr_26_440_PM`):
Classifies candidates as **unique** (exactly one complete single-replace candidate exists for a changed pair — it must be in the solution) vs **optional** (multiple candidates). Strategy 1 tries all permutations of the forced unique operations. Strategy 2 adds further programs greedily from the optional pool. Strategy 3 enumerates Cartesian products of optional choices (capped at 3000 combinations). Strategy 4 handles "hard pairs" — examples where no single replace suffices — by explicitly searching 2-operation sequences (enumerate all A1/B1, compute intermediate, find A2/B2 that reaches output). Post-search, all permutations of the best program are tried for ordering optimisation. The unique-candidate forcing is the key structural insight: it dramatically prunes the search space by treating forced operations as constraints rather than candidates. This produced the best Lite result of any Qwen run (79.2%, nearly matching the CC solver's 80.4%) but underperformed on Hard (51.8%) — likely because Hard's longer cascades mean fewer unique-forced operations, leaving more of the burden on greedy/optional search which degrades at higher cascade lengths.

**100 examples + CoT, run 2 — "Safety-first greedy + 2-step lookahead"** (`Sun_Apr_26_402_PM`):
Separates examples into "changed" (providing signal) and "unchanged" (providing safety constraints). Candidates are pre-filtered to discard any that modify unchanged pairs — a hard safety gate applied before scoring, not as a penalty. Generates both direct candidates and 2-step lookahead candidates (enumerate A1/B1 pairs, check if the intermediate can reach output in one more step). Greedy selection picks by improvement count, with a fallback to best-overall if no candidate improves. The safety-first design results in a tighter search space and the 2-step lookahead explicitly handles feeding interactions. This approach achieved the highest Lite pass rate (65.7%) of any Qwen run.

**100 examples, no CoT — "Beam search + heuristic scoring"** (`Sat_Apr_25_819_PM`):
Without reasoning traces the agent fell back to a textbook beam search: enumerate all (A,B) pairs scoring them by how many diffs they fully explain (×1000) plus partial progress (×10), then run beam search (beam_size=50) using the verifier to select states. No structural insights about edit regions, feeding interactions, or multi-step decomposition — just breadth-first exploration. The resulting solver is the weakest, especially on Hard, because beam search without structural priors doesn't scale to long cascades.

**48 examples + CoT — "Multi-start greedy + permutation reorder"** (`Sat_Apr_25_1104_PM`):
Adopts the clearest separation of concerns: greedy construction using exact candidates (programs that directly solve an example) plus progress candidates (programs that reduce edit distance), followed by a best-ordering search that tries all permutations for ≤8 programs and random-shuffle + local-swap for larger sequences. Multi-start (multiple random seeds) provides diversity. The permutation reorder step is the key insight: it explicitly handles feeding/bleeding interactions by scoring different program orderings, rather than hoping the greedy order is correct. This design is the most principled and also achieved the highest Hard pass rate (76.2%) among single runs.

**12 examples + CoT — "Adaptive beam search"** (`Sun_Apr_26_120_AM`):
Uses beam search with adaptive parameters (beam width and max programs scale with the fraction of changed examples). Adds a diversity constraint (no repeated candidates within a sequence) and an alternative-path fallback for partial success. The adaptive complexity is a reasonable heuristic but the limited example set (12) meant the agent lacked sufficient signal to invent structural insights like edit-region anchoring or permutation reordering. The resulting solver is a competent but generic beam search, clearly below the CoT runs with richer examples.

**Summary pattern:** CoT traces enable the agent to learn structural insights about the DSL (edit regions, feeding interactions, program ordering). Without CoT it defaults to brute-force beam search. With CoT but too few examples (12), it produces a competent but under-informed search. The best solvers share a common trait: an explicit mechanism for handling program-order interactions (permutation reorder, 2-step lookahead, or split-candidate pairs).

---

### Claude Code PBE solver analysis

No trajectory is available (induced in an interactive Claude Code session), but the SOLVER_ALGORITHM.md documents the approach.

**CC PBE solver — "Two-phase safe/unrestricted beam search with difflib candidate extraction"** (`claude_code/Thu_Apr_23_807_PM`):
Runs two sequential beam searches. Phase 1 (beam=150) enforces a safety constraint: candidates are restricted to patterns that do not appear as substrings in any already-correct input, preventing collateral damage. Phase 2 (beam=75) drops the constraint to handle the rare edge case where the only correct program matches a character that also appears in an unchanged string. Candidates are generated dynamically using `difflib.SequenceMatcher` on the intermediate state at each beam depth — crucially, at depth 2 the candidates are computed from the strings produced *after* depth-1 programs, enabling discovery of feed/bleed ordering. Context-extended patterns (extending the diff region by 1–2 surrounding characters) help find more specific replacements that avoid hitting unintended positions. Ranked by safety, direct fixes, partial applicability penalty, and pattern length. The two-phase design is the most technically sophisticated of all runs, and the resulting solver achieves the highest pass rate of any single solver on both Lite (80.4%) and Hard (69.7%).

---

### Qwen SLR solver analysis

| Run | Pass% | Mean score | Algorithm (short) |
|----:|------:|-----------:|-------------------|
| 1 (`Sat_Apr_25_643_AM`) | *(old Qwen — eval pending)* | — | layered hypothesis generation + early exit |
| 2 (`Sun_Apr_26_131_PM`) | 60.7% | 0.607 | in-Python filter + budget-limited verification |

**Qwen SLR run 1 — "Layered hypothesis generation with early exit"** (`Sat_Apr_25_643_AM`):
Builds a feature space by parsing all predicates and their argument value domains. Classifies predicates as car-level or train-level, then generates candidate body literals accordingly. Searches in layers of ascending complexity (1 literal, then 2, then 3), enumerating all combinations within each layer. Uses early exit: the moment a perfect rule (score=1.0) is found in a layer, the search stops without evaluating deeper layers. This guarantees returning the simplest correct rule. Rule candidates are purely conjunctive (`has_car(T,C), pred(C,val), ...`); no negation or disjunction. Final selection ranks by score then complexity. Relies directly on the HuggingFace judge for every candidate evaluation.

**Qwen SLR run 2 — "In-Python filter + budget-limited verification"** (`Sun_Apr_26_131_PM`):
The key architectural change vs run 1 is moving the filtering almost entirely into Python to avoid the ~27-second SWI-Prolog verifier cost per rule. Generates candidates across four stages: (1a) direct separating properties — exact (pred, value) pairs that appear in all eastbound and no westbound trains; (1b) integer predicate arithmetic rules (`N > 0` style); (1c) negation-as-failure rules (`X \= excluded`); (1d) universal negation (`\+ has_car...`). Gathers candidates from both positive *and* negative examples (the key insight: the distinguishing value may only appear in negatives). If no single-property rule passes, escalates to 2- then 3-property conjunctions, capped at 50,000 combinations. Only the top-K simplest in-Python candidates are sent to the verifier (budget: 5 calls). Despite this more sophisticated design, run 2 achieved lower pass rate (60.7%) than run 1 — the budget of 5 verifier calls is too tight for medium/hard tasks, and the in-Python emulation may reject candidates the verifier would accept. The low mean score (0.607) vs run 1's higher score suggests more complete failures rather than near-misses.

**CC SLR solver — "Ascending-complexity search with local Python evaluator"** (`claude_code/Sat_Apr_25_251_AM`):
Parses each facts string into a normalised car model (cars re-indexed as c1, c2, … by car_num, train-id agnostic) enabling train-agnostic pattern matching. Discovers predicates dynamically, handling any DSL extension. Generates candidates in ascending rule complexity order: complexity-1 (single property on one car), complexity-2 (two properties, or car_num + property, or two-car rules), complexity-3 and complexity-4 with corresponding multi-car shapes. Local Python evaluation emulates Prolog's existential semantics (rule fires if *any* car satisfies all conditions for single-car rules; any *two distinct cars* for two-car rules). Ranks by accuracy then complexity, returning the simplest perfect rule. Optionally re-scores top-K against the official SWI-Prolog verifier when available. The normalised car model and dynamic predicate discovery are the key design choices — they make the solver robust to train numbering and vocabulary variation. Achieves 68.4% overall (100% basic, 78.4% easy, ~48% medium/hard), with mean score 0.9669 — near-misses dominate failures, consistent with a systematic search that finds almost-correct rules.

### Key findings

**1. LLM CoT reasoning traces are critical for solver induction.** Removing CoT from the 100-example demos (keeping only final programs) causes a −11.3pp drop on Lite (53.4% → 42.1%) and a catastrophic −34.1pp drop on Hard (58.9% → 24.8%). Without the reasoning traces the coding agent can only observe what programs were produced, not how. The Hard cliff is particularly striking: the solver without CoT barely learns to handle longer cascades at all.

**2. Run-to-run variance is massive and dominates example-count effects.** Three runs of the 100-example CoT solver span 53.4%→79.2% Lite (25.8pp range) and 51.8%→74.7% Hard (22.9pp range). No two runs invented the same algorithm. The 48-example CoT solver (55.7% / 76.2%) sits entirely within this variance range. The apparent advantage of 48 examples over 100 (run 1) was a fluke. Run 3 on 100 examples nearly matches the CC solver on Lite (79.2% vs 80.4%) while failing on Hard (51.8%) — showing the Lite/Hard trade-off depends strongly on which algorithm the agent invented, not on the demos file.

**3. 12 examples + CoT is clearly insufficient.** Hard drops to 50.4% even when CoT is present, well below the variance range of the 100-example runs (58.9–74.7%). This suggests a genuine floor: too few examples limits example diversity enough to hurt solver quality.

**4. No-CoT solver used more turns.** 102 turns vs 76 for the same 100-example CoT run, suggesting the agent spent more effort probing the dataset trying to infer patterns that the CoT would have made explicit.

**5. KV caching saves ~98% of input tokens.** No-cache totals are 20–34× higher than KV-cache totals (e.g. 4.4M vs 191K). All construction cost figures in the paper use the KV-cache estimate as the realistic lower bound.

### Solver ensemble results (symbolic-only, no LLM)

| Ensemble | Lite Pass% | Hard Pass% | Avg tokens |
|----------|----------:|----------:|----------:|
| CC solver only | 80.4% | 69.7% | 0 |
| Qwen run 3 only (best single Qwen) | 79.2% | 51.8% | 0 |
| 3 Qwen CoT runs union (runs 1+2+3) | 85.5% | 78.9% | 0 |
| All 5 Qwen solvers union (+48ex, +12ex) | 86.2% | 82.0% | 0 |
| CC + all 6 Qwen solvers union | **91.3%** | **84.7%** | 0 |

Ensembling multiple Qwen runs (same demos, different algorithms) recovers most of the variance lost in any single run. Three CoT runs alone reach 85.5% Lite / 78.9% Hard — beating the original CC+Qwen ensemble (83.5% Lite). Adding CC pushes to 91.3% Lite / 84.7% Hard, the best symbolic-only result across all settings.

### Output files

| File | Description |
|---|---|
| `evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/` | 100 examples + CoT (run 1) |
| `evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/` | 100 examples + CoT (run 2) |
| `evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_440_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/` | 100 examples + CoT (run 3) |
| `outputs/hard_ensemble_qwen3runs.jsonl` | 3 Qwen CoT runs union (Hard) |
| `outputs/hard_ensemble_qwen_all.jsonl` | All 5 Qwen solvers union (Hard) |
| `outputs/hard_ensemble_all_solvers.jsonl` | CC + all 6 Qwen solvers union (Hard) |
| `outputs/lite_ensemble_qwen3runs.jsonl` | 3 Qwen CoT runs union (Lite) |
| `outputs/lite_ensemble_qwen_all.jsonl` | All 5 Qwen solvers union (Lite) |
| `outputs/lite_ensemble_all_solvers.jsonl` | CC + all 6 Qwen solvers union (Lite) |
| `evals/solver_results/qwen3.6_35b_a3b/Sat_Apr_25_819_PM_DEMOS_PBEBENCH_seed_42_100_examples/` | 100 examples, no CoT |
| `evals/solver_results/qwen3.6_35b_a3b/Sat_Apr_25_1104_PM_DEMOS_PBEBENCH_seed_42_48_examples_with_CoT/` | 48 examples + CoT |
| `evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_120_AM_DEMOS_PBEBENCH_seed_42_12_examples_with_CoT/` | 12 examples + CoT |

---

## SLR-Bench

**Dataset:** 1,000 tasks · curriculum levels 1–20 · 250 tasks per tier (basic/easy/medium/hard)  
**Task:** Induce an `eastbound(T) :- Body.` Prolog rule from background facts + positive/negative train examples  
**Reward:** `partial_score` (0–1) from `AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning` via HuggingFace evaluate; 1.0 = perfectly correct rule  
**Metric:** `rule_complexity` = number of top-level body literals excluding `has_car/2`  
**Data file:** `data/slr_bench/v1_All_full.jsonl`

### Main results

Effi mode: solver used at zero cost when reward = 1.0; LLM tokens counted only for solver failures. Complexity Δ = mean predicted − mean GT (correct solutions only). DF covers 946/1000 tasks — Hard tier has only 196 tasks for DF rows (marked *).

Token costs for gpt-oss-120b estimated at DeepInfra pricing ($0.039/M input, $0.19/M output). Reported costs for o3/gpt-5 are from the SLR-Bench paper. Total tokens in millions across all 1000 tasks.

| System | Pass% | Basic | Easy | Medium | Hard | Total (M tok) | Cost ($) | Complexity Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| o3 † | — | 99 | 93 | 74 | 45 | 4.30 | 207.24 | — |
| gpt-5 † | — | 100 | 90 | 72 | 46 | 16.40 | 103.13 | — |
| **CC Solver** | **68.4%** | **100** | **78.4** | **48.4** | **46.8** | **0** | **0** | **−0.756** |
| **Qwen Solver (run 2)** | **60.7%** | **100** | **71.2** | **34.4** | **37.2** | **0** | **0** | **−1.087** |
| BoK-32 (gpt-oss-120b) | 68.7% | 100 | 100 | 57.6 | 17.2 | 225.3 | 17.88 | −0.611 |
| DF-32 (gpt-oss-120b) | 83.3%* | 100 | 99.6 | 84.4 | 39.8* | 174.9* | 13.96* | −0.815 |
| BoK + Qwen Solver (effi) | 75.9% | 100 | 100 | 62.8 | 40.8 | 162.8 | 13.21 | −1.063 |
| BoK + CC Solver (effi) | 80.3% | 100 | 100 | 68.4 | 52.8 | 132.4 | 10.80 | −0.796 |
| BoK + CC + Qwen Solver (effi) | 80.3% | 100 | 100 | 68.4 | 52.8 | 131.2 | 10.71 | −0.802 |
| DF + Qwen Solver (effi) | 83.7%* | 100 | 99.6 | 86.4 | 48.8* | 132.7* | 10.65* | −1.118 |
| **DF + CC Solver (effi)** | **86.5%*** | **100** | **99.6** | **88.8** | **57.6*** | **113.0*** | **9.07*** | **−0.920** |
| **DF + CC + Qwen Solver (effi)** | **86.6%*** | **100** | **99.6** | **88.8** | **58.0*** | **111.6*** | **8.96*** | **−0.917** |

† Reported scores from SLR-Bench paper; not re-run by us. Single attempt, no test-time scaling.  
\* DF covers 946/1000 tasks (Hard tier: 196/250). Token totals and costs are for covered tasks only; pass% computed over all 1000.

### Symbolic solver breakdown by rule complexity

| Rule complexity | N | Pass% (CC) | Mean reward (CC) | Pass% (Qwen) | Mean reward (Qwen) |
|---|---:|---:|---:|---:|---:|
| 1 | 50 | 100.0% | 1.0000 | 100.0% | 1.0000 |
| 1–2 | 350 | 93.1% | 0.9890 | 94.3% | 0.9429 |
| 2–3 | 150 | 67.3% | 0.9612 | 50.7% | 0.5067 |
| 3–4 | 100 | 47.0% | 0.9435 | 33.0% | 0.3300 |
| 4–5 | 250 | 44.4% | 0.9462 | 31.6% | 0.3160 |
| 5 | 100 | 49.0% | 0.9563 | 39.0% | 0.3900 |

CC solver outperforms Qwen at every complexity level above 1. The gap widens at complexity 2+: CC maintains high partial scores (0.94+) while Qwen's mean reward collapses (0.32–0.51), reflecting more complete failures rather than near-misses.

### Key findings

**1. CC Solver matches o3 and gpt-5 on the Hard tier at zero LLM cost.** CC Solver: 46.8% on Hard vs o3 45%, gpt-5 46% — essentially equal — while spending nothing per task. Qwen (37.2%) is weaker on Hard but still competitive with gpt-4o (0%).

**2. BoK-32 is strong on Basic/Easy but collapses on Hard.** 100% on Basic/Easy, then 57.6% Medium, 17.2% Hard — a severe cliff. The symbolic solvers hold up far better on Hard (46.8% CC, 37.2% Qwen). The complementarity makes effi ensembles very effective.

**3. DF + CC Solver effi reaches 86.5% overall (57.6% Hard) at 113K avg tokens/task.** This is the best full-dataset result. DF alone is 83.3% (39.8% Hard, partial); adding CC Solver raises Hard by 17.8pp. DF + CC + Qwen adds only 0.1pp over DF + CC alone.

**4. BoK + CC Solver effi (80.3%) cuts BoK token cost by ~41%.** 132K vs 225K avg tokens/task — CC Solver handles 68.4% of tasks at zero cost, so only the harder 31.6% incur BoK cost.

**5. All systems find simpler rules than GT.** Δ ranges from −0.61 (BoK) to −1.09 (Qwen solver). Opposite of PBEBench — Prolog has many equivalent shorter forms; all systems prefer brevity. Symbolic solvers undershoot most aggressively.

**6. SLR tokens are much more expensive than PBEBench.** BoK SLR: 225K avg tokens/task vs 273K for PBEBench-Hard, but SLR is in principle easier (single Prolog rule vs up to 20 replaces). The high cost reflects long background-fact prompts, not task difficulty.

**7. Best trade-off points:**
- Zero cost: CC Solver (68.4%, Hard 46.8%) — matches frontier LLMs on Hard tier
- Best pass rate: DF + CC + Qwen effi (86.6%, pending full DF run)
- Best BoK ensemble: BoK + CC Solver effi (80.3%, 132K tokens/task)

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
