# Adding Tasks

---

## Task input formats

The system accepts tasks in three ways.

### 1. Built-in tasks (`examples/tasks.py`)

A Python list of dicts, each with `"type"` and `"input"`. The `"input"` dict should have a `"description"` string and optionally `"examples"` (list of `{"input": ..., "output": ...}` pairs):

```python
{
    "type": "list_transform",
    "input": {
        "description": "Given a list of integers, return only the even numbers.",
        "examples": [
            {"input": [1, 2, 3, 4, 5, 6], "output": [2, 4, 6]},
            {"input": [7, 8, 9, 10],       "output": [8, 10]},
        ],
    },
}
```

Run with: `python main.py --task 0` (single) or `python main.py` (all).

### 2. JSON file (`--tasks-file`)

A JSON array of task objects:

```json
[
  {"prompt": "Given a list of integers, return only the even numbers.", "type": "list_transform"},
  {"prompt": "Reverse a list.", "type": "list_transform"},
  {"prompt": "Return the sum of all elements in a list."}
]
```

### 3. JSONL file (`--tasks-file`)

One JSON object per line:

```jsonl
{"prompt": "Given a list of integers, return only the even numbers.", "type": "list_transform"}
{"prompt": "Reverse a list.", "type": "list_transform"}
{"prompt": "Return the sum of all elements in a list."}
```

Run with: `python main.py --tasks-file tasks.jsonl`

### Record schema

| Key | Required | Description |
|---|---|---|
| `prompt` | yes | Natural-language task description passed to the agents |
| `type` | no | Task category label (default: `"symbolic"`). Used for output metadata only |
| any other keys | no | Passed through to agents as additional context |

---

## What makes a good task prompt

The agents work best when the prompt:

- **Describes the transformation clearly** — "Given a list of integers, return only the even numbers" is better than "filter list"
- **Optionally includes examples** — in-context input/output examples help the Reporting agent format the answer correctly and give BCR concrete data types to reason about
- **States the output format** if it differs from the obvious — e.g. "return as a comma-separated string" or "return True/False"

The TaskParser converts the prompt to a `TaskSpec`. If the prompt is clear, the domain and type inference will be accurate, which improves library retrieval.

---

## Task types / domains

The `type` field in a task record is a free-form label used for output metadata. The `domain` used internally for library retrieval is inferred by the TaskParser from the prompt content.

Built-in domains: `list_manipulation`, `string_manipulation`, `sequence`, `math`, `logic`, `grid`, `symbolic`, `other`.

To get good cross-task library reuse, group related tasks together in your task file so functions from earlier tasks are available for later ones.

---

## Batch mode and library sharing

All tasks in a single run share one `FunctionLibrary`. Functions created for task 0 are candidates for tasks 1, 2, 3, etc. Order your tasks from simpler to more compositional to maximise reuse:

```jsonl
{"prompt": "Filter even numbers from a list."}
{"prompt": "Double each element in a list."}
{"prompt": "Double each element and then keep only the even results."}
```

Task 3 should automatically reuse the functions created in tasks 1 and 2.

---

## Evaluating results

Use `--output-file results.jsonl` to write each completed task as a single JSON line immediately after it finishes. The file contains one record per task with response, trajectory, and all agent LLM messages combined.

```bash
python main.py --tasks-file tasks.jsonl --output-file results.jsonl
```

Reading results for evaluation:

```python
import json

with open("results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        print(r["task_index"], r["solved"], r["answer"])
```

Extracting agent messages for training data:

```python
import json

with open("results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        for msg in r["agent_messages"]:
            print(msg["tag"], msg["request"]["messages"], msg["response"]["content"])
```

See [debugging.md](debugging.md#per-task-output-file---output-file) for the full record schema.
