# Data Structures

---

## State

The state dict is the central data object passed between agents. It is created by `make_state()` in `symbolic_agent/models.py` and mutated in-place by each agent.

```python
State = {
    # --- set at creation ---
    "task_input":     Any,        # raw task (dict with "description"/"examples", or string)
    "task_type":      str,        # category label e.g. "list_transform", "sequence"
    "original_prompt": str,       # the raw NL prompt string (passed to Reporting agent)
    "budget":         float,      # remaining step budget (decrements each step)
    "steps":          int,        # current step index
    "solved":         bool,       # True once BCR produces a valid solution.
                                  # In solve_with_reward(), overridden to mean best_reward >= 1.0
    "solution":       None,       # populated by BCR on solve (see below)
    "working_memory": None,       # populated by SSL and BCR (see below)
    "library":        list[str],  # snapshot of library function names at task start
    "trace":          list[dict], # step-by-step agent log (see below)

    # --- added after TaskParser runs ---
    "task_spec": {
        "domain":           str,
        "input_types":      list[str],
        "output_type":      str,
        "operation_hints":  list[str],
        "symbolic_inputs":  str,
    },

    # --- added after Reporting agent runs ---
    "final_output": {
        "answer":           str,
        "explanation":      str,
        "confidence":       float,
        "execution_result": Any,   # actual Python value from running the solution
        # or on failure:
        "error":            str,
    },

    # --- reward loop fields (solve_with_reward only) ---
    "reward_history":  list[dict],  # one entry per iteration (see below)
    "best_reward":     float,       # highest reward value seen across iterations
    "final_reward":    dict,        # {"value": float, "message": str} from last iteration

    # --- added by Controller after solve loop ---
    "cost_summary":      dict,   # from CostTracker.summary()
    "library_snapshot":  list,   # list of Function.to_dict() at end of task
    "agent_messages":    list,   # every LLM call during this task (see below)
}
```

### `state["reward_history"]` entries (solve_with_reward only)

```python
{
    "iteration":        int,    # 0-indexed attempt number
    "reward":           float,  # reward value for this attempt
    "blame":            str,    # "execution" | "library" | "partial" | "logic"
    "message":          str,    # reward function's feedback string
    "solution_summary": str,    # first 200 chars of BCR's reasoning (for BCR fix-mode context)
}
```

---

### `state["solution"]` (set by BCR)

For `action=direct` (question/prompt tasks):
```python
{
    "action":         "direct",
    "answer":         str,        # the answer string derived by BCR
    "reasoning":      str,        # BCR's explanation of how it applied the library function
    "functions_used": list[str],  # library function names conceptually applied
}
```

For `action=solve` (algorithmic tasks):
```python
{
    "action":         "solve",    # field present but optional for backwards compat
    "code":           str,        # complete Python function definition
    "function":       str,        # entry-point function name (inferred from def)
    "reasoning":      str,        # BCR's explanation
    "functions_used": list[str],  # library function names called in code
}
```

### `state["working_memory"]` (set by SSL after library ops)

```python
{
    "active_functions": list[str],  # names of functions SSL flagged as relevant
}
```

After BCR decompose:

```python
{
    "active_functions":  list[str],
    "subtasks":          list[{"description": str, "input": str}],
    "composition_plan":  str,
}
```

### `state["trace"]` entries

SSL step:
```python
{"step": int, "agent": "SSL", "actions": [{"action": "create"|"reuse"|"compose", "function": str, ...}]}
```

BCR solve:
```python
{"step": int, "agent": "BCR", "action": "solve", "reasoning": str}
```

BCR decompose:
```python
{"step": int, "agent": "BCR", "action": "decompose", "subtasks": list[str]}
```

---

## agent_messages

`state["agent_messages"]` is a list of every LLM call made during the task, in call order. It is populated by `Controller.solve()` from `LLMClient.get_task_log()`.

Each entry has the same structure as a `--debug-dir` file:

```python
{
    "tag":          str,   # "task_parser" | "ssl" | "bcr" | "reporting"
    "model":        str,   # model ID used for the call
    "request": {
        "system":   str,             # exact system prompt sent
        "messages": list[dict],      # [{"role": "user", "content": "..."}]
        "max_tokens": int,
    },
    "response": {
        # openai backend:
        "finish_reason":    str,
        "content":          str,     # raw JSON string the model produced
        "reasoning_content": str,    # chain-of-thought (if available)
        # anthropic backend:
        "stop_reason":      str,
        "content":          str,
        "usage":            dict,    # {"input_tokens": int, "output_tokens": int}
    },
    "parsed_result": dict,  # the parsed JSON dict the agent received ({} on parse failure)
}
```

This field is included in every `--output-file` JSONL record and is intended as training data for agentic LLM fine-tuning.

---

## Function

Defined in `symbolic_agent/models.py`.

```python
@dataclass
class Function:
    name:           str             # snake_case identifier
    code:           str             # complete Python function definition
    description:    str  = ""       # one-line docstring
    domain:         str  = "general"  # task domain (see DOMAINS in task_parser.py)
    input_types:    list[str] = []  # Python type annotations for parameters
    output_type:    str  = ""       # Python type annotation for return value
    embedding:      list[float] | None = None  # reserved for future dense retrieval
    usage_count:    int  = 0        # incremented each time BCR uses this function
    creation_cost:  float = 0.0     # set by CostTracker.record_new_function()
```

**`usefulness()`** — higher is better:
```python
def usefulness(self) -> float:
    return self.usage_count / (self.creation_cost + 1e-6)
```

**`to_dict()`** — serialisation for output files, includes `usefulness` as a computed field.

---

## TaskSpec

Defined in `symbolic_agent/task_parser.py`.

```python
@dataclass
class TaskSpec:
    original_prompt:  str
    domain:           str       = "symbolic"  # one of DOMAINS
    input_types:      list[str] = []
    output_type:      str       = ""
    operation_hints:  list[str] = []
    symbolic_inputs:  str       = ""          # e.g. "lst = [1, 2, 3]"
```

`TaskSpec` is created by `TaskParser.parse()` and passed to SSL and BCR on every step. On parse failure, a minimal fallback spec with only `original_prompt` set is returned.

---

## Cost Tracker

Defined in `symbolic_agent/costs.py`.

Tracks cumulative costs across all steps of a single task. In batch mode, a single `CostTracker` persists across all tasks.

### Events

```python
tracker.record_new_function(func)
# → num_new_functions += 1
# → total_function_length += len(func.code.splitlines())
# → func.creation_cost = ALPHA + BETA * num_lines

tracker.record_reuse(func)
# → reuse_count += 1
# → func.usage_count += 1
```

### Cost formula

```
TotalCost = α·num_new_functions
          + β·total_function_length
          + γ·redundancy_penalty(functions)
          − δ·reuse_reward(functions)

Objective = task_loss + λ·TotalCost
```

**Redundancy penalty** — proxy for duplicate functions. For every pair of function names with >70% character-level similarity, adds their similarity ratio to the penalty.

**Reuse reward** — `Σ log(1 + usage_count)` over all library functions. Log-scale prevents a single heavily-used function from dominating.

### Default weights

| Symbol | Name | Default | CLI flag |
|---|---|---|---|
| α | per new function | 1.0 | — |
| β | per line of code | 0.05 | — |
| γ | redundancy penalty | 2.0 | — |
| δ | reuse reward | 0.5 | — |
| λ | regularisation | 0.3 | `--lam` |

`task_loss` is set by `solve_with_reward()` to `1 - best_reward` per task and accumulates across tasks. It remains 0 when using `solve()` without a reward function.
