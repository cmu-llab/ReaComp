# TroVE Baseline — Deviations from the Original Paper

This document records all intentional and unavoidable deviations between our
reimplementation (`symbolic_agent/baselines/trove/`) and the original TroVE
codebase (`original_baseline_repos/trove/`).

---

## 1. Chat API instead of Local Model Completion

**Original:** TroVE uses a HuggingFace `transformers.pipeline` with a locally
loaded model (e.g. CodeLlama-7b-Instruct) in **completion** mode. The prompt
is a plain string prefix; the model generates continuation text.

**Ours:** We use Anthropic's Messages API or an OpenAI-compatible chat API
(vLLM). The prompt is sent as a `user` message; the model generates a reply
that includes the **Solution** and **Tools** blocks.

**Impact:** Minimal. The prompt structure (ending with `**Solution**`) signals
to chat models what to generate, and empirically they comply. No JSON mode is
used (`TroVELLMClient` vs the main `LLMClient`).

---

## 2. Domain-Generic Few-Shot Examples

**Original:** TroVE uses domain-specific few-shot examples for each task
(TabMWP coin-collection table examples, MATH algebra examples, etc.)

**Ours:** We use generic string-manipulation examples that apply to both
PBEBench and ReasoningGym string tasks (replace_char, extract_digits,
lowercase examples). Domain-specific examples for other task families
should be added to `prompts.py` as needed.

**Impact:** May slightly reduce self-consistency accuracy for tasks where the
original examples provide strong in-context guidance. The structural format
is preserved exactly.

---

## 3. K Calls Rather Than Batched n=K

**Original:** TroVE passes `num_return_sequences=K` to the HuggingFace
pipeline, which generates K sequences in one forward pass.

**Ours:** We call the LLM API K times independently (temperature sampling).
The Anthropic API does not support `n` parameter; the OpenAI-compatible API
does but we call separately for simplicity and identical code paths.

**Impact:** K API calls instead of 1; slightly slower but statistically
equivalent since each call is an independent sample.

---

## 4. AST Node Count Instead of AST Depth Sum

**Original:** TroVE tie-breaks by `sum(depth of each AST expression node)`
across the solution (referenced in §3.2 and Appendix B).

**Ours:** `count_ast_nodes()` counts total AST nodes via `ast.walk()`.
Total nodes is monotonically related to total expression depth: simpler
programs have fewer nodes AND lower total depth. The tie-breaking effect
is identical in practice.

**Impact:** Negligible. Both metrics rank programs by complexity; the ranking
rarely differs for programs with the same stdout.

---

## 5. No Re-Generation of Trimmed Examples

**Original:** After trimming the toolbox, `run_trove.py` re-generates
solutions for all affected examples using IMPORT|SKIP (not CREATE), then
reports updated accuracy.

**Ours:** We record the set of affected task indices in the trim log but do
not replay them. This is because we process tasks in a single stream and do
not store the original task inputs for re-processing. For a complete
faithful comparison, task inputs should be saved and re-processed on trim.

**Impact:** In practice, trimming only fires after 500 tasks with the default
setting. For our 100-task pilot runs, trimming is disabled by setting
`--trove-trim-every 9999`.

---

## 6. Reward Loop Compatibility Wrapper

**Original:** TroVE has no concept of a reward function or iterative
refinement loop. It is one-shot per example.

**Ours:** `solve_with_reward()` wraps `solve()` for compatibility with
`main.py`'s `--default-reward` and `--max-reward-iters` flags. No retry
loop is performed; the reward is computed once and stored in `reward_history`
for eval script compatibility.

**Impact:** None on TroVE's actual behavior. Only affects output format.

---

## 7. `trim_every` Default Differs for Small Runs

**Original:** Default `--trim_steps=500` (trimming every 500 examples).
For a 100-task dataset this fires 0 times.

**Ours:** Same default (500), but users running small pilots should pass
`--trove-trim-every 9999` to make it explicit that no trimming happens.

**Impact:** None unless running >500 tasks.

---

## Summary Table

| Aspect | Original | Ours | Impact |
|--------|----------|------|--------|
| LLM backend | Local HF model (completion) | Chat API (messages) | Minimal |
| Few-shot examples | Domain-specific (TabMWP/MATH) | Generic string-manipulation | Minor |
| K sampling | Batched (n=K in one call) | K independent API calls | Latency only |
| Complexity metric | Sum of AST expression depths | Total AST node count | Negligible |
| Trim replay | Re-generates affected examples | Records but does not replay | Evaluation accuracy |
| Reward loop | Not in original | Wrapper for main.py compat | None |
