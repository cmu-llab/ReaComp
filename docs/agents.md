# Agents

All agents share a single `LLMClient` instance and communicate through the state dict. Each agent makes exactly one LLM call per invocation and expects a JSON object in return.

---

## TaskParser

**File:** `symbolic_agent/task_parser.py`

**Role:** Convert a natural-language task description into a structured `TaskSpec` before the solve loop begins. The spec drives domain-aware retrieval and provides structured hints to SSL and BCR.

**Input:** Raw NL prompt string.

**Output:** `TaskSpec` dataclass (see [data-structures.md](data-structures.md#taskspec)).

**JSON schema the model must output:**

```json
{
  "domain": "list_manipulation",
  "input_types": ["list[int]"],
  "output_type": "list[int]",
  "operation_hints": ["filter", "even_numbers"],
  "symbolic_inputs": "lst = [1, 2, 3, 4]"
}
```

Valid domains: `list_manipulation`, `string_manipulation`, `sequence`, `math`, `logic`, `grid`, `symbolic`, `other`.

**Failure mode:** If the LLM call fails or returns unparseable JSON, `TaskParser.parse()` catches the exception and returns a minimal `TaskSpec(original_prompt=prompt)` with default values. The solve loop continues without structured metadata.

---

## SSL Agent — Symbolic Search and Library

**File:** `symbolic_agent/ssl_agent.py`

**Role:** Maintain the shared function library. Given the current task, decide whether to reuse an existing function, compose two or more existing functions into a new one, or create a brand-new function.

**Preference order:** reuse > compose > create.

**Input:** State dict, `FunctionLibrary`, `CostTracker`, optional `TaskSpec`.

**Output:** Updated state with `working_memory["active_functions"]` populated (list of function names the BCR agent should use) and a new trace entry.

**JSON schema the model must output:**

```json
{
  "action": "reuse" | "compose" | "create",
  "name": "snake_case_function_name",
  "code": "def name(lst: list[int]) -> list[int]:\n    ...",
  "description": "One-line docstring.",
  "uses": ["existing_fn_1", "existing_fn_2"],
  "domain": "list_manipulation",
  "input_types": ["list[int]"],
  "output_type": "list[int]"
}
```

Field relevance by action:

| Field | reuse | compose | create |
|---|---|---|---|
| `name` | existing function name | new function name | new function name |
| `code` | — | required | required |
| `description` | — | recommended | recommended |
| `uses` | — | required (names of composed fns) | — |
| `domain`, `input_types`, `output_type` | — | recommended | recommended |

**Robustness fallbacks (model output is imperfect):**
- If `name` is missing but `function_name` is present, use that (old field name habit)
- If `code` is missing, scan all string-valued fields for a Python `def` statement
- If `name` is still missing after code is found, extract the function name from the `def` line via regex
- If both `name` and `code` remain absent, log a warning and skip

**Side effects:**
- `reuse`: increments `func.usage_count` via `CostTracker.record_reuse()`
- `create` / `compose`: adds function to library via `library.add()`, calls `safe_exec()` to verify the code parses and executes without errors, records cost via `CostTracker.record_new_function()`

---

## BCR Agent — Bottom-up Conceptual Reasoning

**File:** `symbolic_agent/bcr_agent.py`

**Role:** Attempt to solve the current task using library functions. If the task is too complex for a direct solution, decompose it into ordered sub-tasks for SSL to address in the next iteration.

**Input:** State dict, `FunctionLibrary`, `CostTracker`, optional `TaskSpec`.

**Output:** Updated state. On `solve`: sets `state["solved"] = True` and populates `state["solution"]`. On `decompose`: populates `state["working_memory"]["subtasks"]`.

**JSON schema — solve:**

```json
{
  "action": "solve",
  "code": "def solve(lst: list[int]) -> list[int]:\n    return filter_even(lst)",
  "reasoning": "The library already has filter_even which does exactly this.",
  "functions_used": ["filter_even"]
}
```

**JSON schema — decompose:**

```json
{
  "action": "decompose",
  "subtasks": [
    {"description": "Filter odd numbers", "input": "list[int]"},
    {"description": "Square each element", "input": "list[int]"}
  ],
  "composition_plan": "Apply filter_odd, then map_square, then sum."
}
```

**Robustness fallbacks:**
- `code` also accepted as `solution_code` (old field name)
- Entry-point function name is always inferred by regex from the first `def` in `code` — the model is never required to provide it separately
- `solution_function` is accepted if present (for backward compatibility)
- If `code` is absent or no `def` is found, log a warning and return state unchanged (task remains unsolved, loop continues)

---

## Reporting Agent

**File:** `symbolic_agent/reporting_agent.py`

**Role:** Format the final solution into a clean, human-readable answer. Receives the original NL prompt so it can honour any output-format instructions or in-context examples embedded in it (e.g. "return a comma-separated string" or "output as a Python list"). Does **no** new reasoning.

**Input:** State dict (must have `state["solved"] == True`), `FunctionLibrary`.

**Output:** Populates `state["final_output"]`.

**Execution step:** Before calling the LLM, the agent attempts to actually *run* the solution code using `execute_with_library()`. This loads all library functions into a namespace, then calls the solution function with inferred arguments. If execution succeeds, the concrete result is included in the prompt so the LLM can format it accurately.

**JSON schema the model must output:**

```json
{
  "answer": "[2, 4, 6]",
  "explanation": "Filtered the list using filter_even from the library.",
  "confidence": 0.95
}
```

**Fallback:** If the LLM call fails or `answer` is absent from the result, the agent falls back to `str(execution_result)` or `str(solution)` with `confidence=0.5`.
