# Computational Environment

This document records the hardware, software, and per-experiment compute budgets needed to reproduce all results in the paper. It is intended to satisfy the NeurIPS reproducibility checklist (§ Experiments compute resources).

---

## Hardware

| Role | Hardware | Notes |
|---|---|---|
| Open-source LLM inference (vLLM) | 2 × NVIDIA A100 80 GB | Self-hosted on an internal compute cluster; serves both `gpt-oss-120b` and `Qwen3.6-35B-A3B` |
| Claude Code solver induction (CC Solver) | Anthropic API (claude-sonnet-4-6) | Interactive Claude Code CLI session; no local GPU required |
| SWI-Prolog verifier (SLR-Bench eval) | CPU | Called as a subprocess; ~300 ms per rule check |
| All other eval/analysis scripts | CPU (laptop/workstation) | Python stdlib, Pandas, Matplotlib only |

All self-hosted LLM inference was run on an internal university compute cluster. The **dollar costs reported in this document and in the paper are reference costs computed using public API pricing** (DeepInfra for `gpt-oss-120b`, AtlasCloud for `Qwen3.6-35B-A3B`) **for comparison purposes only** — the actual experiments were run on the cluster at no direct monetary cost.

---

## Software

| Component | Version |
|---|---|
| Python | 3.11 |
| vLLM | 0.8.x (serves `gpt-oss-120b` and `Qwen3.6-35B-A3B`) |
| openhands-sdk | 1.16.1 (QO Agent / DirectSolve and Qwen solver induction) |
| Apptainer sandbox image | `python:3.11-slim` + numpy / scipy / sympy / evaluate / datasets |
| SWI-Prolog | 9.x (SLR-Bench verifier subprocess) |
| anthropic Python SDK | latest at time of experiments (claude-sonnet-4-6) |

---

## Per-experiment compute

Reference pricing used for cost estimates:
- **gpt-oss-120b**: $0.039 / M input tokens, $0.190 / M output tokens (reasoning billed as output) — DeepInfra public pricing
- **Qwen3.6-35B-A3B**: $0.1612 / M input tokens, $0.9653 / M output tokens — AtlasCloud public pricing

Token counts are exact, taken from provider usage fields logged in every output JSONL. "M tok" = millions of tokens (input + output combined).

### PBEBench-Lite (1008 tasks, max\_programs = 5)

| Experiment | Model | n tasks | Total M tok | Ref. cost | Notes |
|---|---|---:|---:|---:|---|
| DF-32 (gpt-oss-120b) | gpt-oss-120b | 1008 | 111.1 | $16.74 | 32 attempts max, single-turn per attempt |
| BoK-32 (gpt-oss-120b) | gpt-oss-120b | 1008 | 68.0 | $12.20 | 32 independent samples |
| QO Agent / DirectSolve | Qwen3.6-35B-A3B | 1008 | 395.3 | $85.38 | Complete. openhands-sdk; avg 11.1 steps/task (max 100) |
| CC Solver (build) | claude-sonnet-4-6 (CC CLI) | 1 session | — | ~$2 | One-time; interactive session ~30 min, exact cost from tracked run |
| QO Solver run 1 (build) | Qwen3.6-35B-A3B | 1 session | 4.46 | $0.79 | 76 turns, 201 min |
| QO Solver run 2 (build) | Qwen3.6-35B-A3B | 1 session | 4.82 | $0.85 | 72 turns, 34 min |
| QO Solver run 3 (build) | Qwen3.6-35B-A3B | 1 session | 6.24 | $1.10 | 80 turns, 44 min |
| QO Solver (100ex no CoT, build) | Qwen3.6-35B-A3B | 1 session | 7.81 | $1.34 | 102 turns, 44 min (ablation) |
| QO Solver (48ex + CoT, build) | Qwen3.6-35B-A3B | 1 session | 4.18 | $0.74 | 82 turns, 26 min (ablation) |
| QO Solver (12ex + CoT, build) | Qwen3.6-35B-A3B | 1 session | 1.65 | $0.30 | 49 turns, 15 min (ablation) |

Solver inference (CC Solver, QO Solver) has zero per-task LLM cost — pure Python.  
Ensemble outputs are computed deterministically from individual outputs; no additional LLM calls.

### PBEBench-Hard (1216 tasks, max\_programs = 20)

| Experiment | Model | n tasks | Total M tok | Ref. cost | Notes |
|---|---|---:|---:|---:|---|
| BoK-32 (gpt-oss-120b) | gpt-oss-120b | 1216 | 332.1 | $57.83 | Complete. |
| QO Agent / DirectSolve | Qwen3.6-35B-A3B | 1216 | ~1054 proj. | ~$225 proj. | 1162/1216 done at $215 actual; $0.185/task avg → ~$225 projected. avg 18.7 steps/task |
| CC Solver (inference only) | — | 1216 | 0 | ~$2 build | same solver as Lite; build ~30 min |
| QO Solver run 2 (inference only) | — | 1216 | 0 | $0.85 build | same solver as Lite run 2 |

### SLR-Bench (1000 tasks)

| Experiment | Model | n tasks | Total M tok | Ref. cost | Notes |
|---|---|---:|---:|---:|---|
| DF-32 (gpt-oss-120b) | gpt-oss-120b | 1000 | 224.2 | $17.43 | Complete. |
| BoK-32 (gpt-oss-120b) | gpt-oss-120b | 1000 | 225.3 | $17.88 | Complete. |
| QO Agent / DirectSolve | Qwen3.6-35B-A3B | 1000 | ~754 proj. | ~$157 proj. | 783/1000 done at $123 actual; avg 754K tokens/task → ~$157 projected. avg 8.9 steps/task |
| CC Solver (build) | claude-sonnet-4-6 (CC CLI) | 1 session | — | $4.01 | One-time; interactive session ~30 min, exact cost from tracked run |
| QO Solver run 1 (build) | Qwen3.6-35B-A3B | 1 session | 2.93 | $0.51 | 68 turns, 25 min |
| QO Solver run 2 (build) | Qwen3.6-35B-A3B | 1 session | 7.49 | $1.28 | 84 turns, 47 min |

---

## Total compute summary

| Category | Total ref. cost (est.) |
|---|---|
| gpt-oss-120b inference (Lite DF + BoK, Hard BoK, SLR DF + BoK) | ~$122 |
| Qwen3.6-35B-A3B — QO Agent / DirectSolve (Lite + Hard\* + SLR\*, partial) | ~$353 |
| Qwen3.6-35B-A3B — QO Solver induction (6 PBE + 2 SLR runs) | ~$7 |
| claude-sonnet-4-6 — CC Solver induction (1 PBE + 1 SLR session) | ~$6 (~$2 PBE + $4.01 SLR) |
| **Total reported experiments** | **~$488** |

All costs are reference figures using public API pricing. Actual compute was run on an internal cluster (2 × A100 80 GB) at no direct monetary cost.

Additional compute not reported in the paper (preliminary runs, failed solver-induction experiments, prompt engineering, debugging): estimated ~$100–200 equivalent in additional Qwen DirectSolve / solver-builder sessions and gpt-oss-120b exploration runs.

---

## Execution time

| Experiment | Wall-clock time |
|---|---|
| QO Solver build (Qwen, one run) | 15–201 min (median ~40 min) |
| CC Solver build (Claude Code CLI, one session) | ~30 min per session (PBE and SLR) |
| DF / BoK inference (gpt-oss-120b, ~1000 tasks, 8 workers) | ~2 days per benchmark |
| QO Agent / DirectSolve (Qwen, 1000 tasks, 8 workers) | ~24–72 hours (hard-tier tasks ~3.5M tokens/task) |
| Symbolic solver inference (CC/QO, all tasks) | < 5 min (pure Python, no LLM calls) |
| SLR-Bench SWI-Prolog verifier | ~300 ms per rule; ~5 min for 1000 tasks at 1 worker |

---

## Parallelism

- DF, BoK, and QO Agent baselines run with **8 parallel workers** via Python `ThreadPoolExecutor`, one task per thread.
- Solver induction (QO / CC) is a single sequential agent session per run; multiple runs were launched in parallel across separate sessions.
- Ensemble outputs are computed single-threaded in post-processing; no LLM calls.

---

## NeurIPS checklist entry

```latex
\item {\bf Experiments compute resources}
    \item[] Question: For each experiment, does the paper provide sufficient information
    on the computer resources (type of compute workers, memory, time of execution)
    needed to reproduce the experiments?
    \item[] Answer: \answerYes{}
    \item[] Justification: All experiments were run on an internal compute cluster
    using 2$\times$NVIDIA A100 80\,GB GPUs to self-host \texttt{gpt-oss-120b} and
    \texttt{Qwen3.6-35B-A3B} via vLLM\@. The Anthropic API
    (\texttt{claude-sonnet-4-6}) was used for CC Solver induction via Claude Code.
    Per-experiment token counts (exact, logged in output files), reference costs
    (public API pricing for comparison), wall-clock times, and parallelism settings
    are reported in Appendix~X (Computational Environment). The full research project
    consumed an estimated \$100--200 additional compute equivalent in preliminary and
    ablation runs not included in the paper.
```
