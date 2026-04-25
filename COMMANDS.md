# Reproduction Commands

End-to-end commands to reproduce all results in the paper. Run from the project root.

Assumes a vLLM server serving **gpt-oss-120b** is reachable (default port 8002 for DF, 8004 for BoK). Set `PORT=` to override. The symbolic solver steps require no LLM server.

---

## 1. Symbolic Solver Induction

### Claude Code solver (PBEBench)

Run interactively in a Claude Code session — the agent reads the building prompt and demos directly:

```bash
# Open a Claude Code session in the project root, then instruct it:
# "Read building_prompts/SOLVER_BUILDING_PROMPT_PBE.md and
#  demos/DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json,
#  then implement the solver as described."
claude  # launches Claude Code CLI
```

Output: `built_solvers/claude_code/<timestamp>/SOLVER.py`

### Qwen3.6-Coder solver via OpenHands (PBEBench)

```bash
# Requires a vLLM server serving Qwen/Qwen3.6-35B-A3B at PORT
SOLVER_TYPE=pbe bash scripts/run_solver_builder_openhands.sh <PORT>
```

Output: `built_solvers/qwen3.6_35b_a3b/<timestamp>/SOLVER.py`

### Claude Code solver (SLR-Bench)

```bash
# Same as above but with the SLR building prompt:
# "Read building_prompts/SOLVER_BUILDING_PROMPT_SLR.md and
#  demos/DEMOS_SLRBENCH_seed_42_92_examples_with_CoT.json,
#  then implement the solver as described."
claude
```

Output: `built_solvers/claude_code/<timestamp>/SOLVER_SLR.py`

### Qwen3.6-Coder solver via OpenHands (SLR-Bench)

```bash
SOLVER_TYPE=slr bash scripts/run_solver_builder_openhands.sh <PORT>
```

Output: `built_solvers/qwen3.6_35b_a3b/<timestamp>/SOLVER_SLR.py`

---

## 2. Evaluating Symbolic Solvers

```bash
# PBEBench-Lite (max 5 programs)
python scripts/eval_solver.py \
    --solver built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py \
    --dataset lite \
    --workers 8 \
    --output-dir evals/solver_results/claude_code/Thu_Apr_23_807_PM/

# PBEBench-Hard (max 20 programs)
python scripts/eval_solver.py \
    --solver built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py \
    --dataset hard \
    --workers 8 \
    --output-dir evals/solver_results/claude_code/Thu_Apr_23_807_PM/

# Qwen solver — same flags, different solver path
python scripts/eval_solver.py \
    --solver built_solvers/qwen3.6_coder/Fri_Apr_24_200_AM/SOLVER.py \
    --dataset lite --workers 8 \
    --output-dir evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/

# SLR-Bench (eval_solver auto-detects solve_slr entry point)
python scripts/eval_solver.py \
    --solver built_solvers/claude_code/Sat_Apr_25_251_AM/SOLVER_SLR.py \
    --dataset slr \
    --workers 8 \
    --output-dir evals/solver_results/slr_claude_code/
```

Outputs: `evals/solver_results/<solver>/<timestamp>/{lite,hard,slr}.jsonl`

---

## 3. LLM Baselines (requires vLLM server with gpt-oss-120b)

### Best-of-K (BoK)

```bash
# PBEBench-Lite
DATASET=lite bash scripts/run_best_of_k_vllm.sh
# → outputs/lite_tasks_full_og_best_of_k.jsonl

# PBEBench-Hard
DATASET=hard bash scripts/run_best_of_k_vllm.sh
# → outputs/tasks_full_og_best_of_k.jsonl  (raw {input, outputs} format)

# SLR-Bench
DATASET=slr bash scripts/run_best_of_k_vllm.sh
# → outputs/slr_bench_best_of_k.jsonl
```

**Note on PBEBench-Hard BoK**: the Hard BoK outputs (`outputs/gpt_oss_120b_pbebench_hard_outputs.jsonl`) are taken from the PBEBench paper's public release and are in a raw `{input: {...}, outputs: [...]}` format. `scripts/plot_pbebench_comparison.py` and `scripts/ensemble_outputs.py` auto-detect and handle this format; no conversion is needed.

### Direct Feedback (DF)

```bash
# PBEBench-Lite
DATASET=lite bash scripts/run_direct_feedback_vllm.sh
# → outputs/lite_tasks_full_og_direct_feedback.jsonl

# PBEBench-Hard  (not run — compute cost too high; Hard results use BoK only)

# SLR-Bench
DATASET=slr bash scripts/run_direct_feedback_vllm.sh
# → outputs/slr_bench_direct_feedback.jsonl
```

Both scripts checkpoint progress (`*.ckpt.json`) and are safe to kill and restart.

---

## 4. Ensembling + Eval + Plots

One script per benchmark runs all ensemble combinations, `quick_eval`, and generates the 3 comparison figures.

```bash
# PBEBench-Lite (all 4 systems: DF, BoK, CC Solver, OH Qwen Solver)
bash scripts/run_all_pbebench_lite_evals.sh \
    --metrics-json metrics/pbebench_lite_all.json

# PBEBench-Hard (3 systems: BoK, CC Solver, OH Qwen Solver — no DF)
bash scripts/run_all_pbebench_hard_evals.sh \
    --metrics-json metrics/pbebench_hard_all.json

# SLR-Bench (all 4 systems)
bash scripts/run_all_slr_evals.sh \
    --metrics-json metrics/slr_all.json
```

Figures written to `figures/pbebench_lite_comparison_*.png`, `figures/pbebench_hard_comparison_*.png`, `figures/slr_comparison_*.png`.

To regenerate plots only (without re-running ensembles/eval):

```bash
# PBEBench-Hard
python scripts/plot_pbebench_comparison.py --split hard --plot

# PBEBench-Lite
python scripts/plot_pbebench_comparison.py --split lite --plot

# SLR-Bench (tier and level plots)
python scripts/plot_slr_comparison.py --plot
```

---

## 5. Quick Eval on Any Output File

```bash
# Single file
python scripts/quick_eval.py outputs/slr_bench_best_of_k.jsonl \
    --tasks-file data/slr_bench/v1_All_full.jsonl

# Multiple files side-by-side, save metrics JSON
python scripts/quick_eval.py \
    evals/solver_results/slr_claude_code/slr.jsonl \
    outputs/slr_bench_best_of_k.jsonl \
    outputs/slr_bench_direct_feedback.jsonl \
    --tasks-file data/slr_bench/v1_All_full.jsonl \
    --metrics-json metrics/slr_quick.json
```

---

## 6. Solver Variance Check (PBEBench-Hard)

Re-run both Hard solvers to measure run-to-run variance:

```bash
python scripts/eval_solver.py \
    --solver built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py \
    --dataset hard --workers 8 \
    --output-dir evals/solver_results/claude_code/Thu_Apr_23_807_PM/run2/

python scripts/eval_solver.py \
    --solver built_solvers/qwen3.6_coder/Fri_Apr_24_200_AM/SOLVER.py \
    --dataset hard --workers 8 \
    --output-dir evals/solver_results/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/run2/
```

Compare run1 vs run2:

```bash
python scripts/quick_eval.py \
    evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl \
    evals/solver_results/claude_code/Thu_Apr_23_807_PM/run2/hard.jsonl \
    --tasks-file data/pbebench/tasks_full_og.jsonl
```
