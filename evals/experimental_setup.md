# Experimental Setup

## Task: Programming by Example (PBE)

Each task is a set of (input string, output string) pairs. The goal is to infer an ordered sequence of string replacement programs of the form `replace(A, B)` that transforms every input into its paired output. Programs are applied in order, and their effects can interact (e.g. a later rule can undo or extend an earlier one), making order non-trivial.

**DSL constraints** (enforced by the verifier in `rewards/pbebench.py`):
- Each program has the form `replace(A, B)` using Python's built-in `str.replace`.
- `1 ≤ len(A) ≤ 3`, `0 ≤ len(B) ≤ 3` (short predicate and transform strings).
- At most 5 programs per sequence (PBEBench-Lite) or 20 (PBEBench-Hard).
- No other Python functions or imports.

**Evaluation metric**: pass rate (fraction of tasks where all input→output pairs are correctly transformed), mean reward (fraction of pairs correct, averaged over tasks), and cascade complexity (sum of `len(A) + len(B)` across all programs in the sequence — lower is simpler and closer to ground truth).

---

## Datasets

| Dataset | Tasks | Max programs | Notes |
|---------|-------|--------------|-------|
| **PBEBench-Lite** | 1,008 | 5 | Primary evaluation set; shorter cascades (length 2–5) |
| **PBEBench-Hard** | 1,216 | 20 | Harder tasks; longer cascades (length 2–20) |

Tasks are characterised by two structural properties: **cascade length** (number of replacement programs in the ground-truth solution) and **BFCC relation count** (number of feeding/bleeding/counterfeeding/counterbleeding interactions between rules, drawn from formal linguistics). Both are used for per-stratum breakdowns in evaluation.

Ground-truth mean cascade complexity on PBEBench-Lite: **11.63**.

---

## Baselines

All baselines use **gpt-oss-120b** served via vLLM on an internal cluster. The model uses a Harmony output format (separate analysis/final channels). The task prompt follows the PBEBench format: inputs and outputs are shown and the model must produce a `replace(...)` program sequence.

### Best-of-K (BoK)

**Approach**: generate K independent samples for each task in parallel (temperature > 0), score each against the verifier, and select the highest-reward candidate. Ties broken by cascade complexity (prefer simpler programs).

**PBEBench-Lite configuration**: K = 32, max 32,768 tokens per sample (full CoT budget).

**PBEBench-Hard configuration**: K = 32, max 16,384 tokens per sample (reduced CoT budget). **All 32 attempts are always run, even if a correct solution is found early** — there is no early exit. This is a deliberate choice: PBEBench-Hard tasks are genuinely hard, and continuing past the first correct answer allows the verifier to find *simpler* programs among later candidates. The token cost is therefore a firm upper bound (K × tokens_per_sample) and overestimates real-world usage; in practice one could stop at the first correct answer.

**Key property**: embarrassingly parallel; no feedback loop. Token budget is K × tokens_per_sample, used regardless of whether an early sample is correct.

**Implementation**: `symbolic_agent/baselines/best_of_k.py`, run via `scripts/run_best_of_k_vllm.sh`.

### Direct Feedback (DF)

**Approach**: sequential single-turn attempts with verifier feedback. Attempt 1 uses the raw task prompt. If the answer is incorrect, the verifier's diagnostic message (which programs failed, step-by-step trace of first failure) is appended and a new attempt is made. Early exit on reward = 1.0.

**Configuration**: up to k = 32 attempts, max 32,768 tokens per attempt.

**Key property**: adaptive — the model sees exactly what went wrong and can fix it. In practice 80%+ of tasks are solved on the first attempt; the feedback loop mainly helps on edge cases (see attempt distribution in `evals/pbebench_lite_observations.md`).

**Implementation**: `symbolic_agent/baselines/direct_feedback.py`, run via `scripts/run_direct_feedback_vllm.sh`.

---

## Symbolic Solver Induction

The core idea is to use a strong model to *induce a symbolic solver* from reasoning traces, then run that solver — at zero LLM token cost — across all tasks.

### Step 1: Collect reasoning traces (DEMOS.json)

`DEMOS.json` contains 100 reasoning traces from an LLM solving PBEBench tasks — 25 from each quadrant of the (difficulty × success) grid:

| | Success | Failure |
|--|---------|---------|
| **Easy** | 25 | 25 |
| **Hard** | 25 | 25 |

Each trace includes the task prompt, input/output examples, cascade length, BFCC relation string, the model's chain-of-thought (`cot`), and final response. The failure traces are deliberately included so the solver-building model can learn what goes wrong and design around it.

### Step 2: Induce the solver

A coding agent reads `DEMOS.json` and the task specification (`building_prompts/SOLVER_BUILDING_PROMPT.md`), which instructs it to:
- Analyse recurring strategies in successful traces and failure modes in unsuccessful ones.
- Implement a Python-based symbolic program synthesiser (`solve_pbe(examples)`) using only the standard library.
- Use the verifier (`rewards/pbebench.py`) to score candidates internally.
- Prefer simple, compositional rules with low description complexity.
- Return top-K highest-scoring programs if no perfect solution is found.

Two solver variants were induced:

**Claude Code solver** (`built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py`): induced by Claude Code (claude-sonnet-4-6) reading `DEMOS.json` directly in a Claude Code session.

**Qwen3.6-Coder solver** (`built_solvers/qwen3.6_coder/Fri_Apr_24_200_AM/SOLVER.py`): induced by Qwen3.6-35B-A3B (Qwen3.6-Coder) running inside an OpenHands agent loop. The agent receives the building prompt as its task, bind-mounts `DEMOS.json` and `rewards/pbebench.py` into a sandboxed environment, and uses `execute_code` to inspect traces and test logic before writing the final solver files. Implemented in `openhands_agents/solver_builder/`; run via `scripts/run_solver_builder_openhands.sh`.

### Step 3: Evaluate

The induced solver is evaluated with `scripts/eval_solver.py` on PBEBench-Lite and PBEBench-Hard. Results are written in a format compatible with `scripts/quick_eval.py` for direct comparison with LLM baselines.

---

## Ensembling

Symbolic solvers and LLM baselines are complementary: the solver never costs tokens and is strong on straightforward tasks; LLMs handle the long tail. Two ensembling strategies are implemented in `scripts/ensemble_outputs.py`:

**Standard**: per task, select the candidate with the highest verifier reward; break ties by cascade complexity (prefer simpler programs), then source order. All LLM tokens are counted regardless.

**Efficiency (effi)**: if the symbolic solver achieves a perfect score (reward = 1.0), use it unconditionally and record zero LLM token cost for that task. Otherwise fall back to the best LLM candidate by reward then complexity, counting token usage only from LLM sources. This rewards the symbolic solver's zero-cost coverage: with the Claude solver solving 80.4% of tasks perfectly, effi mode cuts average token cost by ~30% at DF+BoK level with no pass rate loss.

All ensemble combinations are scripted in `scripts/run_ensembles.sh`.
