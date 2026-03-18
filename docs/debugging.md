# Debugging and Diagnostics

---

## Per-Call LLM Debug Logs (`--debug-dir`)

Pass `--debug-dir <DIR>` to write one JSON file per LLM call. Each program run creates a timestamped subdirectory so logs from multiple runs never overwrite each other.

```bash
python main.py --tasks-file scripts/mock_tasks.jsonl \
               --base-url http://localhost:8002/v1 \
               --model openai/gpt-oss-120b \
               --debug-dir debug_logs/
```

Files are written immediately after each call and survive mid-run crashes.

```
debug_logs/
└── run_20260318T060621/
    ├── 0001_task_parser_20260318T060635_517252.json
    ├── 0002_ssl_20260318T060643_455080.json
    ├── 0003_bcr_20260318T060652_379541.json
    └── 0004_reporting_20260318T060658_119875.json
```

### Reading a debug log

```json
{
  "call_index": 2,
  "tag": "ssl",
  "model": "openai/gpt-oss-120b",
  "request": {
    "system": "...",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 1024
  },
  "response": {
    "finish_reason": "stop",
    "content": "{\"action\": \"create\", ...}",
    "reasoning_content": "Let me think through this..."
  },
  "parsed_result": {
    "action": "create",
    "name": "double_elements",
    "code": "def double_elements(lst: list[int]) -> list[int]:\n    return [x * 2 for x in lst]"
  }
}
```

**Key fields to inspect:**
- `response.reasoning_content` — the model's internal chain-of-thought. Check this first when behaviour is unexpected.
- `response.content` — the final committed JSON string (what gets parsed).
- `parsed_result` — what the agent actually received. If `{}`, JSON parsing failed (the `content` field will show why).
- `request.system` — the exact system prompt sent. Useful for verifying the library content and task spec were formatted correctly.

---

## Common Failure Modes

### Agent loops without solving (BCR called many times)

The task is never marked `solved=True`. Causes:

1. **BCR returns `{}`** — JSON parsing failed. Check `response.content` in BCR debug logs for malformed JSON or empty content.
2. **BCR returns a valid dict but without `code`** — check `parsed_result` keys. The model may have used an unexpected field name.
3. **SSL creates no functions** — check SSL debug logs. If `parsed_result` has `action="create"` but `name` or `code` is absent, the fallback inference also failed. Check `response.content` for what the model actually output.

### `solved=True` but execution fails

The Reporting agent runs the solution code via `execute_with_library()`. Execution errors are logged at `WARNING` level:

```
WARNING  symbolic_agent.reporting_agent  Reporting: execution error: ...
```

Check `state["solution"]["code"]` in `trajectory.json` — the code may call a library function that doesn't exist or has incorrect argument types.

### `parsed_result` is `{}` repeatedly

The model is not producing valid JSON. Possible causes:
- The model doesn't follow JSON instructions well (try a stronger model or add few-shot JSON examples to the system prompt)
- `response_format={"type": "json_object"}` is not supported by the vLLM deployment — check server logs
- `max_tokens` is too small and the output is truncated mid-JSON (increase it)

### 400 error from vLLM

The GptOss reasoning backend formerly required exactly 1 tool (causes a 400 if more are provided). This system no longer sends tools at all, so this error should not occur. If you see a 400, check:
- Whether `response_format={"type": "json_object"}` is supported by your vLLM version/model
- Server-side logs for the specific constraint being violated

---

## Per-Task Output Files (`--output-dir`)

```bash
python main.py --tasks-file tasks.jsonl --output-dir results/
```

Creates:

```
results/
├── task_0000/
│   ├── trajectory.json   # full agent trace
│   └── response.json     # final answer for evaluation
├── task_0001/
│   └── ...
```

### `trajectory.json`

Full record for analysis and debugging:

```json
{
  "task_index": 0,
  "task_type": "list_transform",
  "original_prompt": "Given a list of integers, return only the even numbers.",
  "task_spec": {"domain": "list_manipulation", "input_types": ["list[int]"], ...},
  "solved": true,
  "steps_taken": 2,
  "trace": [
    {"step": 0, "agent": "SSL", "actions": [{"action": "create", "function": "filter_even"}]},
    {"step": 1, "agent": "BCR", "action": "solve", "reasoning": "..."}
  ],
  "solution": {
    "code": "def solve(lst): return filter_even(lst)",
    "function": "solve",
    "functions_used": ["filter_even"]
  },
  "library_snapshot": [{"name": "filter_even", "domain": "list_manipulation", ...}],
  "cost_summary": {"num_new_functions": 1, "reuse_count": 1, "total_cost": 1.15, ...}
}
```

### `response.json`

Minimal record for task-level evaluation:

```json
{
  "task_index": 0,
  "original_prompt": "Given a list of integers, return only the even numbers.",
  "solved": true,
  "answer": "[2, 4, 6]",
  "explanation": "Filtered the list using filter_even.",
  "confidence": 0.95,
  "execution_result": [2, 4, 6]
}
```

---

## Logging

All agents use Python's standard `logging` module under their module name:

```
symbolic_agent.controller
symbolic_agent.ssl_agent
symbolic_agent.bcr_agent
symbolic_agent.reporting_agent
symbolic_agent.task_parser
symbolic_agent.llm_client
```

Default level is `INFO`. To see debug-level traces:

```python
import logging
logging.getLogger("symbolic_agent").setLevel(logging.DEBUG)
```

Key log messages to watch:

| Logger | Level | Message | Meaning |
|---|---|---|---|
| `llm_client` | WARNING | `Failed to parse JSON` | Model output couldn't be parsed |
| `ssl_agent` | WARNING | `missing name/code, skipping` | SSL create/compose had no usable code |
| `ssl_agent` | INFO | `inferred name=... from code` | Name was extracted from `def` line |
| `bcr_agent` | WARNING | `solve missing code, skipping` | BCR couldn't produce a solution this step |
| `reporting_agent` | WARNING | `execution error` | Solution code raised an exception |
