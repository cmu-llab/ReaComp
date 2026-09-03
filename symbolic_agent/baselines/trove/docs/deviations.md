# TroVE Implementation: Deviations and Faithful Elements

This document tracks how this port differs from — and where it stays
faithful to — the original TroVE algorithm
([Wang et al., 2024](https://arxiv.org/abs/2401.12869),
[zorazrw/trove](https://github.com/zorazrw/trove)).

## 1. Algorithmic deviations

### 1.1 Native OpenAI tool calling for IMPORT mode
The original TroVE shows the model a `**Toolbox**` markdown block
listing top-k function signatures and asks it to write a `**Solution**`
plus `**Tools**` block referencing those functions by name. We replace
this for the IMPORT mode (when `backend == "openai"` and the toolbox is
non-empty) with **native OpenAI tool calling**: the toolbox is exposed
via the `tools=[...]` parameter of `chat.completions.create`, the model
emits structured `tool_calls` during its reasoning, and `dispatch_tool_call`
runs each one in the sandboxed executor and returns the stdout. This
makes function usage observable and credit-able from the trajectory
itself.

### 1.2 Reward-based candidate selection (default)
The paper uses self-consistency (majority vote on stdout, AST tie-break)
to pick the best of K samples per mode. We default to **reward-based
selection**: every candidate is scored by the per-task reward function,
ties broken by minimum AST node count. This is more reliable on
PBEBench (program-list outputs rarely tie as strings). The original
self-consistency selector remains available via `--trove-selection consistency`.

### 1.3 PBEBench-shaped few-shot examples
For `task_family="pbebench"` we replace the generic CREATE / SKIP / IMPORT
example pairs with PBEBench-shaped pairs that demonstrate `replace()`
chains. CREATE mode also shows signature-only examples of reusable helper
shapes (apply, score, search, prune, debug, end-to-end solve) instead of
full function definitions, to reduce anchoring on a single copied helper.
The legacy default examples remain for `task_family="default"`.

### 1.4 Strict **Solution** parsing for PBEBench
The legacy parser falls back to "first ```python``` block anywhere" when
no `**Solution**` block is present. For `task_family="pbebench"` this
fallback is disabled, preventing CoT scratchpad from being accidentally
promoted to the answer.

## 2. Faithful elements

- 3-mode generation (IMPORT, CREATE, SKIP).
- K samples per mode (default K=5, paper).
- AST-tie-breaking by node count (simplest solution wins).
- Periodic toolbox trimming with threshold `C·log_{20}(n)`, default
  `C=1.0`, matching the original implementation.
- Frequency-based top-k retrieval for the toolbox view.
- Dict-keyed toolbox structure mirroring `utils/code.py`.
- Library updates: IMPORT credits frequency, CREATE adds new functions
  on success, SKIP makes no library changes.

## 3. Infrastructural patches

- **JSONL-per-task checkpointing** via `--output-file`, with crash
  resumption.
- **`reasoning_content` fallback** in `_call_openai` for `gpt-oss` Harmony
  channel splits where the answer text lives in `message.reasoning_content`.
- **Executor timeout 60s** (vs. 10s in earlier versions of this port),
  closer to the original's ~100s.
- **`<|`-truncation sanitizer** in `dispatch_tool_call` and
  `_update_library`. Defensive workaround for the open vLLM
  [PR #35906](https://github.com/vllm-project/vllm/pull/35906) covering
  Harmony control-token leakage into tool names. When that PR lands
  upstream the sanitizer becomes a no-op and is left in place.

## 4. Backend coverage caveat

Anthropic backend code paths exist and are exercised by CREATE / SKIP and
the legacy text-based IMPORT fallback, but **the smoke run and reported
numbers are vLLM-served `gpt-oss` only**. IMPORT-with-tools requires
the OpenAI/vLLM backend and is the only path we test end-to-end.

## 5. vLLM version requirement

- Minimum vLLM: **v0.16.0** (branch-cut 2026-02-08).
- Required upstream change: [PR #28729](https://github.com/vllm-project/vllm/pull/28729)
  ("Multiple fixes for gpt-oss Chat Completion prompting"), merged
  2025-12-12. v0.16.0 is the first stable release branch-cut after the merge.
- Known open caveat: [PR #35906](https://github.com/vllm-project/vllm/pull/35906)
  ("Sanitize leaked Harmony control tokens"), still open as of late
  March 2026 — see §3 for the sanitizer mitigation.
