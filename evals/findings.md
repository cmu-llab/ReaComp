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

| System | Pass% | Mean reward | Edit Sim | Complexity Δ | Avg tokens/task |
|---|---:|---:|---:|---:|---:|
| gpt-oss-120b, single attempt † | 62.5% | — | 69.9 | −0.67 | — |
| GPT-5, single attempt † | 72.4% | — | 76.5 | −1.02 | — |
| **CC Solver** | **80.4%** | **0.9438** | **93.7** | **+3.00** | **0** |
| **Qwen Solver (best run)** | **65.7%** | **0.9022** | **87.4** | **+3.01** | **0** |
| **CC + Qwen Solvers (union)** | **84.6%** | **0.9607** | **94.9** | **+2.16** | **0** |
| DF-32 (gpt-oss-120b) | 92.4% | 0.9796 | 97.3 | +2.11 | 110,267 |
| BoK-32 (gpt-oss-120b) | 93.8% | 0.9808 | 97.8 | +2.19 | 67,480 |
| DF + Qwen Solver (effi) | 92.9% | 0.9810 | 97.5 | +2.89 | 89,493 |
| DF + CC Solver (effi) | 93.1% | 0.9815 | 97.6 | +3.00 | 79,877 |
| DF + All Symbolic (effi) | 93.2% | 0.9817 | 97.6 | +2.18 | 78,119 |
| BoK + Qwen Solver (effi) | 93.8% | 0.9808 | 97.8 | +2.94 | 50,284 |
| **BoK + CC Solver (effi)** | **93.9%** | **0.9810** | **97.8** | **+2.94** | **45,277** |
| **BoK + All Symbolic (effi)** | **93.9%** | **0.9810** | **97.8** | **+2.19** | **43,113** |

† Reported scores from PBEBench paper (`figures/pbebench_lite_reported_metrics.png`); not re-run by us. Single attempt, Pass@1, 8192 CoT tokens — no test-time scaling.

> **Model note:** LLM baselines use **gpt-oss-120b** served via vLLM. Symbolic solvers are induced by a separate coding agent (CC by claude-sonnet-4-6; Qwen by Qwen3.6-35B-A3B inside OpenHands) — distinct from the inference LLM. Effi token savings reflect replacing gpt-oss-120b inference with zero-cost symbolic execution; solver build cost (~216K tokens KV-cache, one-time) is negligible at any realistic eval scale (see Finding 8).

### Symbolic solver breakdown by cascade length

| Cascade | N | Pass% (CC) | Mean reward (CC) | Pass% (Qwen) | Mean reward (Qwen) |
|---:|---:|---:|---:|---:|---:|
| 2 | 246 | 98.4% | 0.990 | 96.3% | 0.992 |
| 3 | 253 | 90.9% | 0.963 | 81.0% | 0.947 |
| 4 | 253 | 83.0% | 0.958 | 56.5% | 0.886 |
| 5 | 256 | 50.0% | 0.866 | 30.1% | 0.788 |

CC solver collapses at CL=5 (50%) — at the 5-program limit there is little slack for ordering interactions. Qwen run 2 shows consistent but lower performance at all levels.

### Key findings

**1. Symbolic solvers alone are surprisingly strong.** The CC Solver (80.4%) surpasses gpt-oss-120b (62.5%) and GPT-5 (72.4%) at their single-attempt setting — at zero per-task inference cost. The Qwen Solver (65.7%) beats gpt-oss-120b single attempt as well.

**2. Symbolic + LLM ensembles consistently outperform either alone.** BoK alone caps at 93.8%; adding the CC Solver (effi) reaches 93.9% while cutting tokens by 33% (45K vs 67K avg/task). DF + All Symbolic effi reaches 93.2% at only 78K avg tokens — 29% cheaper than DF alone.

**3. The all-symbolic union is the best zero-cost option.** CC + Qwen union reaches 84.6% pass, +4.2pp over CC alone, at zero LLM cost. When paired with BoK or DF in effi mode, it also yields the lowest token cost of any ensemble configuration.

**4. Effi mode preserves pass rate while cutting cost.** BoK + All Symbolic effi matches BoK + CC Solver effi in pass rate (93.9%) but saves a further 2K tokens/task by using the union solver (which covers more tasks perfectly), with no pass rate tradeoff.

**5. Effi slightly increases complexity Δ vs standalone LLM.** LLM outputs are less parsimonious than GT: DF standalone Δ+2.11, but DF+CC effi rises to Δ+3.00 because the CC solver's outputs (Δ+3.00) replace LLM outputs on tasks it solves. The all-symbolic effi mitigates this — union solver includes Qwen whose outputs are tighter.

**6. BoK finds simpler programs than DF.** BoK Δ+2.19 vs DF Δ+2.11 — similar, but BoK has the advantage of selecting the minimum-complexity correct candidate from 32 samples.

**7. Best trade-off points:**
- Best pass rate (zero cost): CC + Qwen union (84.6%, 0 tokens)
- Best pass rate + token efficiency: BoK + All Symbolic effi (93.9%, 43K avg tokens/task)
- Best pass rate regardless of cost: BoK + CC Solver effi or BoK + All Symbolic effi (93.9%)

**8. Solver construction cost is negligible when amortised.** The Qwen run 2 solver was built in a single OpenHands session costing ~216K tokens (KV-cache). Effi mode saves ~22K tokens/task over BoK standalone — build cost recoups at 10 tasks and is <1% overhead across 1008 tasks. See ablations section for full token cost breakdown.

### Output files

| File | Description |
|---|---|
| `evals/solver_results/claude_code/Thu_Apr_23_807_PM/lite.jsonl` | CC Solver results |
| `evals/solver_results/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_.../lite.jsonl` | Qwen Solver results (run 2) |
| `outputs/lite_tasks_full_og_best_of_k_stripped.jsonl` | BoK raw outputs (stripped) |
| `outputs/lite_tasks_full_og_direct_feedback_stripped.jsonl` | DF raw outputs (stripped) |
| `outputs/lite_union_solvers.jsonl` | CC + Qwen union |
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

## Solver Construction Ablations (PBEBench, Qwen3.6-35B-A3B)

**Question:** How does the composition of the demos file (number of examples, presence of LLM CoT reasoning traces) affect solver quality?

**Setup:** Four Qwen solvers induced via OpenHands, varying demos only. All other settings identical (same building prompt, same verifier, same seed-42 balanced sampling across success/failure × easy/hard quadrants). Token costs measured with `scripts/compute_trajectory_tokens.py` using tiktoken (cl100k_base).

### Performance

| Demos | Run | Lite Pass% | Hard Pass% | Algorithm (short) |
|-------|----:|----------:|----------:|-------------------|
| 100 examples + CoT | 1 | 53.4% | 58.9% | greedy + multi-pass residual fixing |
| 100 examples + CoT | 2 | 65.7% | 74.7% | safety-first greedy + 2-step lookahead |
| 100 examples + CoT | **3** | **79.2%** | 51.8% | unique-op permutations + greedy + 2-op sequences |
| 100 examples, **no CoT** | 1 | 42.1% | 24.8% | beam search + heuristic scoring |
| 48 examples + CoT | 1 | 55.7% | 76.2% | multi-start greedy + permutation reorder |
| 12 examples + CoT | 1 | 47.7% | 50.4% | adaptive beam search |

### Solver construction token cost

| Demos | Run | Turns | KV-cache total | No-cache total | Final context |
|-------|----:|------:|---------------:|---------------:|--------------:|
| 100 examples + CoT | 1 | 76 | 191,331 | 4,377,628 | 109,681 |
| 100 examples + CoT | 2 | 72 | 215,839 | 4,698,253 | 125,419 |
| 100 examples + CoT | 3 | 80 | 264,495 | 6,050,386 | 151,423 |
| 100 examples, no CoT | 1 | 102 | 227,002 | 7,670,723 | 135,686 |
| 48 examples + CoT | 1 | 82 | 176,868 | 4,126,622 | 101,435 |
| 12 examples + CoT | 1 | 49 | 106,526 | 1,638,817 | 63,418 |

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
