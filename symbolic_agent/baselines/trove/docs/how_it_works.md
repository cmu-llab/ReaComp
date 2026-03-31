# TroVE Baseline — How It Works

## Overview

TroVE (Tool induction Via Verifiable Examples) is an **online** program
synthesis framework that builds a shared function library incrementally
as it processes tasks one at a time. It uses **3-way generation** with
**self-consistency selection** to decide whether to use existing library
functions, create new ones, or write an inline solution.

## File Structure

```
symbolic_agent/baselines/trove/
├── __init__.py          # Exports TroVEController
├── controller.py        # Main loop: multi-way generation, library updates, trimming
├── executor.py          # Subprocess execution of generated solutions
├── llm.py               # Plain-text LLM client (no JSON mode)
├── parse.py             # Parser for **Solution** / **Tools** blocks
├── prompts.py           # Prompt builders for IMPORT / CREATE / SKIP modes
├── toolbox.py           # In-memory toolbox with frequency tracking + trimming
└── docs/
    ├── how_it_works.md  # This file
    └── deviations.md    # Deviations from the original paper
```

## The Three Modes

### IMPORT mode
**Prompt instruction (verbatim from paper):**
> "You task is to write Python program solutions to the given questions.
> The toolbox section lists all the available functions that can be used in your solution."

- The prompt shows the current toolbox as signature+docstring blocks (no full code body).
- The model writes a **Solution** block that calls library functions by name.
- The **Tools** block contains `from toolbox import <fn_name>` lines.
- If IMPORT wins: the used functions have their frequency incremented.

### CREATE mode
**Prompt instruction (verbatim from paper):**
> "You task is to write Python program solutions to the given questions.
> You should also create Python functions that can be used by your solution,
> if you believe the function can be reused to solve other questions."

- No toolbox shown.
- The model writes a **Solution** block + defines new functions in **Tools**.
- If CREATE wins and execution succeeded: new functions are added to the toolbox.

### SKIP mode
**Prompt instruction (verbatim from paper):**
> "You task is to write Python program solutions to the given questions."

- No toolbox, no encouragement to write new functions.
- The model writes a self-contained solution with only import statements in **Tools**.
- Library is never updated when SKIP wins.

## K-Sample Self-Consistency

For each mode, `K` independent samples are generated (default K=5, matching
the paper). The best sample is selected by:

1. **Discard failures**: remove samples where execution failed (returncode ≠ 0).
2. **Majority vote**: pick the most common `stdout` output among successes.
3. **Tie-break**: among samples with the majority output, pick the one with
   the fewest AST nodes (simplest solution — TroVE §3.2, Appendix B).
4. **Fallback**: if all K samples fail, return the first one (marked as failure).

The same criterion is then applied across the three per-mode winners to pick
the overall best.

## Toolbox Format

Functions are stored as dicts:
```python
{
    "name":      str,   # function name
    "signature": str,   # def fn(args) -> ret:
    "docstr":    str,   # description from docstring
    "function":  str,   # full source code
    "type":      str,   # "function" or "import"
    "frequency": int,   # usage count across tasks
    "indices":   list,  # task indices that used/created this function
}
```

The **Toolbox** section in IMPORT prompts shows only `# {docstr}\n{signature}`
(no full body). The full code is concatenated into the execution namespace so
called functions are resolvable at runtime.

## Periodic Trimming

Every `trim_every` tasks (default 500), functions whose frequency is below
`C × log₂₀(n)` (C=0.5, n=tasks processed) are removed from the toolbox.
The set of task indices that used the trimmed functions is returned; in the
original paper these examples are re-generated with IMPORT|SKIP only
(not CREATE, since the missing function is gone). In our stream setting we
record which indices were affected but do not replay them.

## Execution Model

Generated code is executed in a subprocess:
1. Concatenate: `toolbox_full_code` + `tools_code` + `solution_code`.
2. Write to a temp `.py` file.
3. Run `python tmpfile.py` with a 10-second timeout.
4. Capture stdout; `returncode == 0` → success.

The execution namespace always includes the full toolbox source so any
library function can be called by name from the solution.

## Controller Interface

`TroVEController` exposes the same interface as the ssl_bcr `Controller`:

| Method | Description |
|--------|-------------|
| `solve(task_input, task_type)` | One task, one pass |
| `solve_with_reward(task_input, ..., reward_fn, entry)` | Same as solve + reward evaluation |
| `library_stats()` | Returns toolbox summary for --stats |

The result dict matches `_append_task_output()` expectations in `main.py`.
