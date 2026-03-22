# Output File Format

This document describes the files written to `outputs/` (or any path passed to `--output-file`)
and the companion checkpoint files.  It is the reference for anyone writing analysis scripts,
reward functions, or training-data pipelines against run logs.

---

## JSONL output — one record per task

Each completed task is immediately appended to the output file as a single JSON line by
`_append_task_output()` in `main.py`.  The schema is a **flat** projection of the internal
state dict (see [data-structures.md](data-structures.md) for the nested state).

```json
{
  "task_index":      0,
  "task_type":       "symbolic",
  "original_prompt": "...",

  "task_spec": {
    "domain":          "string_manipulation",
    "input_types":     ["list[str]"],
    "output_type":     "list[str]",
    "operation_hints": ["replace"],
    "symbolic_inputs": "inputs = ['xbu', 'uxwkkg', ...]"
  },

  "solved":      true,
  "steps_taken": 2,
  "trace": [
    {"step": 0, "agent": "SSL", "actions": [{"action": "create", "function": "infer_replace_sequence"}]},
    {"step": 1, "agent": "BCR", "action": "solve", "reasoning": "..."}
  ],
  "solution": {
    "code":           "def solve(prompt): ...",
    "function":       "solve",
    "reasoning":      "...",
    "functions_used": ["infer_replace_sequence"]
  },
  "library_snapshot": [{"name": "infer_replace_sequence", "domain": "string_manipulation", ...}],
  "cost_summary": {
    "num_new_functions":    1,
    "total_function_length": 12,
    "reuse_count":          1,
    "task_loss":            0.0,
    "total_cost":           1.05,
    "objective":            1.37
  },

  "answer":           "[\"replace('b', 'h')\", \"replace('g', 'vzw')\", ...]",
  "explanation":      "Applied the inferred replace sequence to the corpus.",
  "confidence":       0.92,
  "execution_result": ["replace('b', 'h')", "replace('g', 'vzw')", "replace('z', 'xwy')"],
  "error":            null,

  "reward_history": [
    {
      "iteration":        0,
      "reward":           0.6,
      "blame":            "logic",
      "message":          "Score=0.600: 3/5 inputs mapped correctly. ...",
      "solution_summary": "Used infer_replace_sequence to find programs..."
    },
    {
      "iteration":        1,
      "reward":           1.0,
      "blame":            "logic",
      "message":          "",
      "solution_summary": "Fixed ordering of programs..."
    }
  ],
  "best_reward":  1.0,
  "final_reward": {"value": 1.0},

  "agent_messages": [
    {
      "tag":   "task_parser",
      "model": "openai/gpt-oss-120b",
      "request": {
        "system":   "...",
        "messages": [{"role": "user", "content": "..."}],
        "max_tokens": 512
      },
      "response": {
        "finish_reason":    "stop",
        "content":          "{\"domain\": \"string_manipulation\", ...}",
        "reasoning_content": "..."
      },
      "parsed_result": {"domain": "string_manipulation", ...}
    },
    {"tag": "ssl",       ...},
    {"tag": "bcr",       ...},
    {"tag": "reporting", ...}
  ]
}
```

### Field reference

| Field | Type | Always present | Notes |
|---|---|---|---|
| `task_index` | int | yes | 0-based index within the run |
| `task_type` | str | yes | Category label passed via `"type"` key in task file |
| `original_prompt` | str | yes | The raw NL prompt string |
| `task_spec` | dict\|null | yes | Output of TaskParser; null if parse failed |
| `solved` | bool | yes | True if `best_reward >= 1.0` (reward mode) or BCR produced a solution |
| `steps_taken` | int | yes | Number of SSL/BCR steps in the final iteration |
| `trace` | list | yes | High-level agent action log; no raw prompts |
| `solution` | dict\|null | yes | BCR's last solution (see below) |
| `library_snapshot` | list | yes | All library functions at end of task |
| `cost_summary` | dict | yes | From `CostTracker.summary()` |
| `answer` | str\|null | yes | Reporting agent's formatted answer string |
| `explanation` | str\|null | yes | Reporting agent's one-line explanation |
| `confidence` | float\|null | yes | Self-reported confidence from Reporting agent (0–1) |
| `execution_result` | any | yes | Raw Python value from running solution code; see below |
| `error` | str\|null | yes | Error string if Reporting agent's execution failed; otherwise null |
| `reward_history` | list | yes | Empty list when no reward function was used |
| `best_reward` | float\|null | yes | Highest reward achieved; null when no reward function used |
| `final_reward` | dict\|null | yes | `{"value": float, "message": str}` from last reward iteration; null if no reward |
| `agent_messages` | list | yes | Every LLM call in order; identical format to `--debug-dir` files |

---

### `solution` field shapes

For `action=solve` (algorithmic task — BCR wrote Python code):
```json
{
  "code":           "def solve(prompt): ...",
  "function":       "solve",
  "reasoning":      "...",
  "functions_used": ["some_library_fn"]
}
```
Note: older runs (pre-BCR action field) may have `solution` with only `code`/`function` keys and
no `"action"` key.  The action can be inferred: if `"code"` is present → `solve`; if `"answer"`
is present → `direct`.

For `action=direct` (Q&A task — BCR answered without writing code):
```json
{
  "action":         "direct",
  "answer":         "42",
  "reasoning":      "...",
  "functions_used": ["caesar_decrypt"]
}
```

---

### `execution_result` formats by task type

The value is whatever the solution function returned when called by `execute_with_library()`,
or the BCR `direct` answer string, serialised with `json.dumps(..., default=str)`.

| Task type | Typical `execution_result` shape | Notes |
|---|---|---|
| **PBEBench** | `list[str]` — `["replace('b','h')", "replace('g','vzw')"]` | Most common; the solution function returns the program sequence |
| **PBEBench** | `str` — markdown-fenced list string | Happens when BCR uses `direct` action or the function returns a string |
| **PBEBench** | `null` | Execution failed (library function missing, code error, etc.) |
| **reasoning_gym** | `str` — minimal answer e.g. `"42"` | BCR `direct` action |
| **reasoning_gym** | `list[str]` — move sequences | Tasks like `tower_of_hanoi` |
| **list_transform** | `list[int]` | Standard algorithmic tasks |

**Reward functions receive `execution_result` directly as `result`.**  Both `rewards/pbebench.py`
and `rewards/reasoning_gym.py` handle all shape variants listed above.

---

## Schema differences: pre-reward vs current runs

Runs in `outputs/` before the reward loop was added (e.g. `pbebench_lite_pilot_tasks.jsonl`)
are missing several fields and have a slightly different `solution` shape:

| Field | Old runs (no reward) | Current runs |
|---|---|---|
| `reward_history` | `[]` (present but empty) | Populated with per-iteration dicts |
| `best_reward` | `null` | float |
| `final_reward` | `null` | `{"value": float, "message": str}` |
| `solution["action"]` | **absent** | `"solve"` or `"direct"` |
| `agent_messages[*].response` | has `reasoning_content` (vLLM/GptOss) | same |

When writing analysis scripts that iterate over a mix of old and new logs, always use `.get()`
with a fallback:

```python
best_reward = rec.get("best_reward")           # None for old runs
action = (rec.get("solution") or {}).get("action")  # None for old runs → treat as "solve"
reward_iters = len(rec.get("reward_history", []))
```

---

## Checkpoint file

When `--output-file results.jsonl` is passed, a companion checkpoint is written alongside it
as `results.ckpt.json` after every completed task.  It enables crash recovery via `--tasks-file`
auto-resume.

```json
{
  "last_completed_index": 12,
  "library": [
    {
      "name":           "infer_replace_sequence",
      "code":           "def infer_replace_sequence(inputs, outputs): ...",
      "description":    "Infer an ordered replace program sequence from I/O pairs.",
      "domain":         "string_manipulation",
      "input_types":    ["list[str]", "list[str]"],
      "output_type":    "list[str]",
      "usage_count":    7,
      "creation_cost":  1.6
    }
  ],
  "cost_tracker": {
    "num_new_functions":    3,
    "total_function_length": 42,
    "reuse_count":          11,
    "task_loss":            4.2,
    "log":                  [...]
  }
}
```

The checkpoint is a **snapshot of shared state** (library + cost counters) as of the last
successfully written task.  On resume, the controller restores from the checkpoint, then
picks up from `last_completed_index + 1`.  Tasks already present in the output file are
not re-run.

**Invariant:** checkpoint index and output file line count are always in sync.  If the output
file is deleted but the checkpoint exists, the run starts fresh (the checkpoint is ignored with
a warning).

---

## Quick analysis snippets

### Score distribution over a run

```python
import json

with open("outputs/pbebench_lite_pilot_tasks.jsonl") as f:
    records = [json.loads(l) for l in f]

scores = [r.get("best_reward") or (1.0 if r["solved"] else 0.0) for r in records]
print(f"Mean: {sum(scores)/len(scores):.3f}  Perfect: {sum(s==1.0 for s in scores)}/{len(scores)}")
```

### Inspect BCR's LLM call for a specific task

```python
rec = records[4]
bcr_call = next(m for m in rec["agent_messages"] if m["tag"] == "bcr")
print(bcr_call["response"]["content"])       # raw JSON the model produced
print(bcr_call["response"].get("reasoning_content", ""))  # chain-of-thought
```

### Find tasks where the constraint violation penalty triggered

```python
for rec in records:
    for h in rec.get("reward_history", []):
        if "constraints violated" in h.get("message", ""):
            print(rec["task_index"], h["message"])
```
