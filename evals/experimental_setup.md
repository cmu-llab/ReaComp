# Experimental Setup

## Tasks

### Programming by Example (PBE)

Each task is a set of (input string, output string) pairs. The goal is to infer an ordered sequence of string replacement programs of the form `replace(A, B)` that transforms every input into its paired output. Programs are applied in sequence using Python's built-in `str.replace()` and their effects can interact: an earlier rule can create or destroy material that a later rule acts on, making program order non-trivial (feeding, bleeding, counterfeeding, counterbleeding interactions from formal linguistics).

**DSL constraints** (enforced by `rewards/pbebench.py`):
- Each program has the form `replace(A, B)`.
- `1 ≤ len(A) ≤ 3`, `0 ≤ len(B) ≤ 3`.
- At most 5 programs per sequence (PBEBench-Lite) or 20 (PBEBench-Hard).
- No other Python functions or imports.

**Evaluation metrics**: accuracy (fraction of tasks where all pairs are correctly transformed), mean reward (mean fraction of pairs correct), and cascade complexity (sum of `len(A) + len(B)` across all programs — lower is simpler and closer to ground truth).

### Symbolic Logic Rules (SLR)

Each task is a set of (background facts, direction label) pairs, where background facts are Prolog ground atoms describing train properties and direction is `eastbound` or `westbound`. The goal is to infer a Prolog rule of the form `eastbound(T) :- Body.` that correctly classifies all trains. Rules are evaluated via SWI-Prolog using the HuggingFace `evaluate` library (`AIML-TUDA/VerifiableRewardsForScalableLogicalReasoning`).

**DSL**: Prolog conjunctive rules over car-property predicates (`car_color`, `car_len`, `has_wall`, `has_roof`, `has_payload`, `car_num`, `has_wheel`, `load_num`, `passenger_num`, `car_type`, `has_window`), anchored by `has_car(T, C)`.

**Evaluation metrics**: accuracy (reward = 1.0), mean reward (partial_score 0–1), rule complexity (number of top-level body literals excluding `has_car/2` — lower is simpler), and curriculum tier/level breakdowns.

---

## Datasets

| Dataset | Tasks | Max programs | Difficulty axis |
|---------|------:|--------------|-----------------|
| **PBEBench-Lite** | 1,008 | 5 | Cascade length 2–5 (~252 tasks per length) |
| **PBEBench-Hard** | 1,216 | 20 | Cascade length 2–20 (64 tasks per length) |
| **SLR-Bench** | 1,000 | — | Curriculum level 1–20; 4 tiers of 250 tasks: basic (1–5), easy (6–10), medium (11–15), hard (16–20) |

PBEBench tasks are additionally characterised by BFCC relation count (number of feeding/bleeding/counterfeeding/counterbleeding interactions). Ground-truth mean cascade complexity: 11.63 (Lite), varies up to 20 (Hard). Data files: `data/pbebench/lite_tasks_full_og.jsonl`, `data/pbebench/tasks_full_og.jsonl`, `data/slr_bench/v1_All_full.jsonl`.

---

## LLM Baselines

All baselines use **gpt-oss-120b** served via vLLM on an internal cluster. The model uses a Harmony output format with separate reasoning (CoT) and final-answer channels. Task prompts follow the PBEBench / SLR-Bench format; the model produces a `replace(...)` sequence or Prolog rule as output.

### Best-of-K (BoK)

Generate K independent samples per task in parallel (temperature > 0), score each with the verifier, and select the highest-reward candidate. Ties broken by cascade/rule complexity (prefer simpler programs).

**PBEBench-Lite configuration**: K = 32, max 32,768 tokens per sample (full CoT budget).

**PBEBench-Hard configuration**: K = 32, max 16,384 tokens per sample (reduced CoT budget). All 32 attempts always run with no early exit — continuing past the first correct answer allows selection of *simpler* programs among later candidates. Token cost is therefore a firm upper bound. DF is not run on Hard due to the high sequential cost and risk of strategy lock-in at long cascade lengths.

**SLR-Bench configuration**: K = 32, max 32,768 tokens per sample.

**Implementation**: `symbolic_agent/baselines/best_of_k.py`, run via `scripts/run_best_of_k_vllm.sh`.

### Direct Feedback (DF)

Sequential single-turn attempts with verifier feedback. Attempt 1 uses the raw task prompt. Each subsequent attempt appends the verifier's diagnostic message from the prior attempt (which programs failed, step-by-step trace of first failure). Early exit on reward = 1.0.

**Configuration**: up to K = 32 attempts, max 32,768 tokens per attempt. Run on PBEBench-Lite and SLR-Bench only (not Hard; see above).

**Key property**: adaptive — the model sees exactly what went wrong and can fix it. In practice 80%+ of PBEBench-Lite tasks are solved on the first attempt; feedback mainly helps edge cases.

**Implementation**: `symbolic_agent/baselines/direct_feedback.py`, run via `scripts/run_direct_feedback_vllm.sh`.

---

## Symbolic Solver Induction

The core idea is to use a coding agent to *compile* LLM reasoning traces into a reusable symbolic solver that runs at zero per-task LLM cost.

### Step 1: Collect reasoning traces

`demos/` contains reasoning traces from an LLM solving PBEBench and SLR-Bench tasks, sampled to cover all quadrants of the difficulty × success grid:

**PBEBench demos** (`demos/DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json`): 100 traces, 25 from each quadrant (easy×success, easy×failure, hard×success, hard×failure). Each trace includes the task prompt, input/output examples, cascade length, BFCC relation string, the model's chain-of-thought (`cot`), and final response. Failure traces are deliberately included so the solver-building agent can learn what goes wrong. Additional variants in `demos/`: 48-example and 12-example subsets (both with CoT), and a 100-example no-CoT ablation.

**SLR-Bench demos** (`demos/DEMOS_SLRBENCH_seed_42_92_examples_with_CoT.json`): 92 traces sampled from a 2×3×4 grid (success/failure × effort level: low/medium/high × curriculum tier: basic/easy/medium/hard); one cell was empty yielding 92 instead of 96.

### Step 2: Induce the solver

A coding agent reads the demos and a task specification (`building_prompts/SOLVER_BUILDING_PROMPT_PBE.md` or `SOLVER_BUILDING_PROMPT_SLR.md`), which instructs it to analyse recurring strategies in successful traces and failure modes in unsuccessful ones, implement a Python-based symbolic solver using only the standard library, use the verifier to score candidates internally, prefer simple programs, and return top-K highest-scoring programs if no perfect solution is found.

**Claude Code solver** (PBE): induced by claude-sonnet-4-6 in an interactive Claude Code session reading `DEMOS.json` directly. Path: `built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py`. Algorithm: two-phase safe/unrestricted beam search with dynamic candidate generation via `difflib.SequenceMatcher` on intermediate states.

**Claude Code solver** (SLR): induced by claude-sonnet-4-6 in an interactive Claude Code session. Path: `built_solvers/claude_code/Sat_Apr_25_251_AM/SOLVER_SLR.py`. Algorithm: ascending-complexity search with local Python evaluator emulating Prolog existential semantics; normalised car models for train-agnostic matching.

**OpenHands Qwen solver** (PBE, multiple runs): induced by Qwen3.6-35B-A3B running inside an OpenHands agent loop. The agent receives the building prompt as its task, bind-mounts `DEMOS.json` and `rewards/pbebench.py` into a sandboxed Apptainer environment, and uses `execute_code` / `write_file` tools to inspect traces and test logic before writing the final solver. **Run-to-run variance is large** (Lite 53–79%, Hard 52–75% across three runs on identical demos): each run invents a qualitatively different algorithm. Primary run: `built_solvers/qwen3.6_35b_a3b/Fri_Apr_24_200_AM/SOLVER.py`. Implemented in `openhands_agents/solver_builder/`; run via `scripts/run_solver_builder_openhands.sh`.

**OpenHands Qwen solver** (SLR): same setup. Path: `built_solvers/qwen3.6_35b_a3b/Sun_Apr_26_131_PM/SOLVER.py`. Algorithm: in-Python property filtering with staged candidate generation (direct separating properties → negation-as-failure → multi-property conjunctions) and budget-limited SWI-Prolog verification (≤5 calls/task).

### Step 3: Evaluate

`scripts/eval_solver.py` runs the induced solver on PBEBench-Lite, PBEBench-Hard, or SLR-Bench. Results are written in a format compatible with `scripts/quick_eval.py` for direct comparison with LLM baselines.

Construction cost: $0.30–$1.34 per Qwen run (exact, native Qwen3.6-35B-A3B tokenizer via `transformers.AutoTokenizer`; AtlasCloud pricing $0.1612/M input, $0.9653/M output). Full per-run breakdown in `findings.md` §Solver Construction Ablations. The one-time build cost is negligible relative to inference savings at any realistic evaluation scale (see findings §8).

---

## Ensembling

Symbolic solvers and LLM baselines are complementary: the solver has zero per-task cost and is strong on tractable tasks; LLMs handle the long tail. Two strategies are implemented in `scripts/ensemble_outputs.py`:

**Standard**: per task, select the candidate with the highest verifier reward; break ties by cascade/rule complexity (prefer simpler), then source order. All LLM tokens counted regardless.

**Efficiency (effi)**: if the symbolic solver achieves a perfect score (reward = 1.0), use it unconditionally and record zero LLM token cost. Otherwise fall back to the best LLM candidate. With the CC solver solving 80.4% of PBEBench-Lite tasks perfectly, effi mode cuts average token cost by ~30% at DF+BoK level with no accuracy loss; ~61% savings on Hard; ~65% on SLR-Bench.

Multiple solvers can also be unioned (no LLM involved): each solver's best answer per task is scored and the highest-reward, lowest-complexity answer wins. Ensembling 3 Qwen CoT runs alone reaches 85.5% Lite / 78.9% Hard; adding CC and all Qwen variants reaches 91.3% Lite / 84.7% Hard — all at zero LLM cost.

Full pipeline scripts: `scripts/run_all_pbebench_lite_evals.sh`, `scripts/run_all_pbebench_hard_evals.sh`, `scripts/run_all_slr_evals.sh`.
