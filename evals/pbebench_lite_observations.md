# PBEBench-Lite Ensemble Observations

All results on PBEBench-Lite (n=1008). GT mean cascade complexity = 11.63.

## Full Results Table

| System | Pass% | MeanRew | Cplx | ΔGT | AvgTok |
|--------|-------|---------|------|-----|--------|
| Claude solver | 80.4% | 0.9438 | 13.43 | +1.80 | 0 |
| Qwen3.6 solver | 53.4% | 0.8494 | 12.09 | +0.46 | 0 |
| Claude+Qwen solvers | 83.5% | 0.9595 | 12.65 | +1.02 | 0 |
| BoK + Claude solver | 94.0% | 0.9847 | 12.60 | +0.97 | 67,479 |
| BoK + Qwen solver | 93.8% | 0.9825 | 13.70 | +2.07 | 67,479 |
| BoK + Claude+Qwen solvers | 94.0% | 0.9847 | 12.60 | +0.97 | 67,479 |
| DF + Claude solver | 93.1% | 0.9829 | 12.60 | +0.97 | 110,266 |
| DF + Qwen solver | 92.9% | 0.9813 | 13.67 | +2.04 | 110,266 |
| DF + Claude+Qwen solvers | 93.2% | 0.9835 | 12.60 | +0.97 | 110,266 |
| **DF+BoK + Claude solver** | **94.9%** | **0.9881** | **12.15** | **+0.52** | 177,746 |
| DF+BoK + Qwen solver | 94.9% | 0.9873 | 12.74 | +1.11 | 177,746 |
| **DF+BoK + Claude+Qwen solvers** | **94.9%** | **0.9881** | **12.15** | **+0.52** | 177,746 |
| BoK + Claude solver (effi) | 94.0% | 0.9810 | 14.43 | +2.80 | **45,277** |
| BoK + Qwen solver (effi) | 93.8% | 0.9808 | 13.72 | +2.09 | **56,618** |
| DF + Claude solver (effi) | 93.1% | 0.9815 | 14.49 | +2.86 | **79,876** |
| DF + Qwen solver (effi) | 92.8% | 0.9808 | 13.75 | +2.12 | **95,308** |
| DF+BoK + Claude solver (effi) | 94.9% | 0.9873 | 14.31 | +2.68 | **125,153** |
| **DF+BoK + Qwen solver (effi)** | **94.9%** | **0.9873** | **13.31** | **+1.68** | **151,927** |
| GT (ground truth) | — | — | 11.63 | +0.00 | — |

Solvers: `evals/solver_results/claude_code/Thu_Apr_23_807_PM/` and `evals/solver_results/qwen3.6_coder/Fri_Apr_24_200_AM/`.
LLM outputs: `outputs/lite_tasks_full_og_best_of_k.jsonl` (BoK) and `outputs/lite_tasks_full_og_direct_feedback.jsonl` (DF).
Ensemble scripts: `scripts/run_ensembles.sh`, `scripts/ensemble_outputs.py`.

---

## Observations

### 1. Symbolic solvers alone are surprisingly strong

The Claude symbolic solver alone hits **80.4% pass rate** with zero LLM token cost — a strong baseline purely from enumerative search over the replace(A,B) DSL. Qwen3.6 solver reaches 53.4%, weaker on accuracy but generating notably tighter programs (complexity 12.09 vs 13.43 for Claude, vs GT 11.63).

### 2. Symbolic + LLM ensembles consistently outperform either alone

Adding a symbolic solver to any LLM system improves pass rate with no token overhead. DF+BoK alone would cap around ~93%; adding the Claude solver pushes it to **94.9%**. The ceiling appears to be ~95% with current systems.

### 3. Adding the Qwen solver on top of Claude rarely helps once LLMs are in the mix

The Qwen solver adds +3pp in symbolic-only ensemble (83.5% vs 80.4%), showing genuine complementarity with Claude solver. However, once BoK or DF is included, adding Qwen on top of Claude brings no pass rate gain — the LLM systems already cover the cases Qwen uniquely solves.

### 4. Claude solver dominates on complexity; Qwen solver is surprisingly parsimonious

When Claude solver is used in an ensemble, complexity stays low (~12.15 at best, ΔGT +0.52). Qwen solver produces programs almost as tight as GT (+0.46) despite lower pass rate — it appears to favour shorter cascades. Qwen LLM outputs are the worst for complexity (ΔGT +2.07 for BoK+Qwen solver), suggesting the LLM and symbolic solver have opposite complexity tendencies for Qwen.

### 5. Efficiency (effi) mode cuts token cost ~30% with no pass rate drop

Effi mode zeros out LLM tokens whenever the symbolic solver is perfect (80.4% of tasks for Claude solver, 53.4% for Qwen solver). At the DF+BoK level:
- Claude effi: 125K avg tokens vs 178K standard (−30%), same 94.9% pass rate
- Qwen effi: 152K avg tokens vs 178K standard (−15%), same 94.9% pass rate

The cost reduction is larger for Claude effi because the Claude solver solves more tasks perfectly.

### 6. Effi mode trades complexity for token savings

The effi variants consistently show higher complexity than their standard counterparts (e.g. DF+BoK + Claude solver: 12.15 standard vs 14.31 effi). When the symbolic solver is bypassed in effi mode, the LLM output takes over — and LLM outputs are less parsimonious than the solver on average. This is the fundamental tradeoff: effi optimises for token cost, not solution simplicity.

### 7. Best trade-off points

- **Best accuracy + complexity**: DF+BoK + Claude solver (94.9%, Cplx 12.15, 178K tokens)
- **Best accuracy + token efficiency**: DF+BoK + Claude solver effi (94.9%, 125K tokens, but Cplx 14.31)
- **Best symbolic-only**: Claude+Qwen solver ensemble (83.5%, zero tokens)
- **Closest to GT complexity**: Qwen3.6 solver alone (12.09, ΔGT +0.46) — but only 53.4% pass rate