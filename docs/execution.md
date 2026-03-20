# Code Execution

**File:** `symbolic_agent/executor.py`

All LLM-generated Python code runs in a sandboxed namespace with import restrictions. There are two entry points.

---

## `safe_exec(code, namespace=None)`

Executes arbitrary Python code in an isolated namespace.

```python
ok, namespace, error = safe_exec("def double(x): return x * 2")
# ok=True, namespace={"double": <function>, ...}, error=""

ok, namespace, error = safe_exec("import os")
# ok=False, namespace=None, error="Forbidden import: os"
```

**Safety checks (before exec):**
1. Parses the code with `ast.parse()` — rejects syntax errors immediately
2. Walks the AST for `Import` / `ImportFrom` nodes — rejects any import of a forbidden module

**Forbidden modules:** `os`, `sys`, `subprocess`, `shutil`, `importlib`, `socket`, `pathlib`, `tempfile`, `signal`, `ctypes`, `multiprocessing`.

Standard library modules not on this list (e.g. `math`, `re`, `itertools`, `functools`, `collections`) are allowed. External packages are only available if installed in the runtime environment.

**Namespace:** If no namespace is provided, a fresh one is created with `{"__builtins__": __builtins__}` — so built-ins (`len`, `range`, `map`, `filter`, `sum`, etc.) are available without import.

---

## `execute_with_library(solution_code, function_name, args, library_functions)`

The primary entry point used by the Reporting agent. Loads the full library into a shared namespace, then runs the solution code and calls the entry-point function.

```python
ok, result, error = execute_with_library(
    solution_code="def solve(lst): return filter_even(lst)",
    function_name="solve",
    args=[[1, 2, 3, 4, 5, 6]],
    library_functions=library.functions,
)
# ok=True, result=[2, 4, 6], error=""
```

**Execution order:**
1. Creates a single namespace with builtins
2. Calls `safe_exec(lib_func.code, namespace)` for each library function in order — they accumulate in the namespace so later functions can call earlier ones
3. Calls `safe_exec(solution_code, namespace)` — the solution can call any library function by name
4. Looks up `namespace[function_name]` and calls it with `*args`
5. Returns `(True, result, "")` on success or `(False, None, traceback)` on any exception

**Failures are non-fatal** — the Reporting agent logs execution errors and falls back to formatting the solution code directly without a concrete execution result.

---

## `infer_call_args(task)`

A best-effort helper that extracts call arguments from a task description so the solution function can be executed automatically (used by the Reporting agent and `solve_with_reward`).

```python
args = infer_call_args(task_input)
ok, result, err = execute_with_library(code, func_name, args, library_functions)
```

Extraction priority (first match wins):

| Task shape | Extracted args |
|---|---|
| `{"examples": [{"input": x, ...}]}` | `[x]` (first example input) |
| `{"input": x}` | `[x]` |
| `{"question": "Calculate ..."}` | `["Calculate ..."]` — bare question string, skipping any few-shot exemplar in the full prompt |
| `{"prompt": "..."}` | `["..."]` — full prompt string |
| list | `[task]` |
| anything else | `[]` (no args — execution is optional) |

Returns `[]` when no concrete input can be found; `execute_with_library` and the Reporting agent both handle `None` execution results gracefully.

---

## Where `safe_exec` is also used

The **SSL agent** calls `safe_exec(new_func.code)` after creating or composing a function. A syntax or import error is logged as a warning but does **not** prevent the function from being added to the library. This keeps the loop running even when the LLM produces slightly imperfect code.
