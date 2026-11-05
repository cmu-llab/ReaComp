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

---

## 7. KinDER Robotics Domains

Five KinDER environment families, one domain each: `kinder_obstruction2d`, `kinder_clutteredretrieval2d`,
`kinder_clutteredstorage2d`, `kinder_motion2d`, `kinder_stickbutton2d`. One solver (a `GeneratedApproach`
policy, `SOLVER_KINDER.py`) is induced per family. Details, protocol and caveats: `domains/kinder/README.md`.

Only the LLM needs a GPU. The simulators, sandboxes and solver evaluation run on CPU (every DirectSolve worker
runs its own simulator and sandbox processes), on the GPU node or on any machine that reaches the vLLM server
(set `GPU_NODE=<host>`).

### Setup (Python 3.12, no Docker or Apptainer needed)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt openhands-sdk==1.16.1 openhands-tools==1.16.1 kindergarden==0.2.5
python -m domains.tests.test_kinder                       # local checks (mock LLM), ~30 s, CPU only
python -m data_generation.kinder_generator check          # task files reproduce with this KinDER install
```

### Serve the model

```bash
mkdir -p vllm_logs
bash scripts/launch_vllm_qwen3.6_35b_a3b.sh 8000 <TENSOR_PARALLEL_SIZE>
```

The agents read `MODEL` (default `openai/Qwen/Qwen3.6-35B-A3B`), `PORT` and `GPU_NODE`. Reasoning is captured
either way: from `reasoning_content` when the server separates it (vLLM `--reasoning-parser`), otherwise from
the message text.

### One family end to end

```bash
FAMILY=obstruction2d      # or clutteredretrieval2d, clutteredstorage2d, motion2d, stickbutton2d
FAMILY_UC=$(echo $FAMILY | tr a-z A-Z)

# 1. Per-task agent on the 100 training instances → reasoning traces
FAMILY=$FAMILY SPLIT=train PORT=8000 WORKERS=8 bash scripts/run_kinder_direct_solve_openhands.sh
# → outputs/kinder_${FAMILY}_train_direct_solve_openhands.jsonl

# 2. Demos for induction
python scripts/build_demos.py --domain kinder_$FAMILY --split train \
    --traces outputs/kinder_${FAMILY}_train_direct_solve_openhands.jsonl --n 100 \
    --out demos/DEMOS_KINDER_${FAMILY_UC}_seed_42_100_examples_with_CoT.json

# 3a. Solver induction with the open model (OpenHands)
FAMILY=$FAMILY bash scripts/run_kinder_solver_builder_openhands.sh 8000
# → built_solvers/qwen3.6_35b_a3b/<timestamp>_kinder_${FAMILY}/SOLVER_KINDER.py

# 3b. ... or with Claude Code: a working directory with the prompt, demos and a simulator
python scripts/kinder_workspace.py create --domain kinder_$FAMILY --out workspaces/kinder_$FAMILY
python scripts/kinder_workspace.py serve workspaces/kinder_$FAMILY &
(cd workspaces/kinder_$FAMILY && claude)   # "Read BUILDING_PROMPT.md and implement the solver as described."
# then copy SOLVER_KINDER.py and SOLVER_KINDER_ALGORITHM.md to built_solvers/claude_code/<timestamp>_kinder_${FAMILY}/

# 4. LLM baseline on the 100 test instances (also the hybrid's fallback)
FAMILY=$FAMILY SPLIT=test PORT=8000 WORKERS=8 bash scripts/run_kinder_direct_solve_openhands.sh
# → outputs/kinder_${FAMILY}_test_direct_solve_openhands.jsonl

# 5. Solver on the test split (zero LLM calls), solver-first hybrid, report by regime and count
FAMILY=$FAMILY bash scripts/run_all_kinder_evals.sh built_solvers/qwen3.6_35b_a3b/<timestamp>_kinder_${FAMILY}
# → evals/solver_results/qwen3.6_35b_a3b/<run>/test.jsonl, outputs/kinder_${FAMILY}_test_hybrid_<run>.jsonl,
#   metrics/kinder_${FAMILY}_<run>.json
```

Steps 1 and 4 are independent and can run in parallel; both support sharding
(`START_INDEX=0 END_INDEX=50 ...`, merged with `scripts/merge_direct_solve_shards.py`) and resume with
`--skip-existing` (always on in the launcher).

### Task files

```bash
python -m data_generation.kinder_generator build                                  # data/kinder/ (committed)
python -m data_generation.kinder_generator verify                                 # vs KinDER's registered environments
python -m data_generation.kinder_generator verify --robocode-src /tmp/robocode    # + vs AgenticGenTAMP's own code
python -m data_generation.kinder_generator build --test-seed <seed> --out <dir>   # robocode's exact evaluation suite
KINDER_DATA_DIR=<dir> FAMILY=$FAMILY SPLIT=test bash scripts/run_kinder_direct_solve_openhands.sh   # baseline on it
KINDER_DATA_DIR=<dir> BASELINE=outputs/kinder_${FAMILY}_test_<dir name>_direct_solve_openhands.jsonl \
    FAMILY=$FAMILY bash scripts/run_all_kinder_evals.sh <solver_dir>                  # solver + hybrid on it
```
