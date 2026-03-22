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

| Key | Visible to agents | Required | Description |
|---|---|---|---|
| `prompt` | yes | yes | Natural-language task description. Always shown to agents. |
| `type` | no | no | Task category label (default: `"symbolic"`). Output metadata only. |
| `question` | yes | no | Concise question string (reasoning_gym style). When present, BCR displays this instead of `prompt` to skip redundant few-shot context (~600 tokens saved per call). |
| `task` | yes | no | Short task-type name (e.g. `"caesar_cipher"`). Passed to agents as context. |
| `reward` | no | no | Reward module name (e.g. `"reasoning_gym"`, `"pbebench"`). Enables the closed-loop reward loop for this task. |
| any other keys | **no** | no | Passed through to the reward function via `entry` but **never shown to agents**. Safe to include oracle fields like `answer`, `programs`, `metadata`. |

Only `question`, `prompt`, and `task` are forwarded to agents (`_AGENT_KEYS` inclusion list in `main.py`). All other keys — including ground-truth answers — are stripped from `task_input` before any agent call.

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

## Reward loop

When a task record has a `reward` field, the controller uses `solve_with_reward()`. After each solve attempt it executes the solution, passes the raw result to the reward function, and if the score is below 1.0, feeds the history back to BCR and retries.

Reward functions live in `rewards/{name}.py`:

```python
def reward(result: Any, execution_ok: bool, entry: dict) -> dict:
    # result       : raw Python value from execute_with_library(), or None on error
    # execution_ok : False if the solution code raised an exception
    # entry        : full task record dict (prompt, answer, metadata, ...)
    return {"value": 0.85, "message": "Score=0.85: wrong sign"}
```

`rewards/reasoning_gym.py` covers all 104 reasoning_gym task types automatically via `get_score_answer_fn(source_dataset)`.

For datasets where the record already has a `reward` field, no extra flags are needed. For datasets where the field is absent, use `--default-reward <name>`:

```bash
python main.py --tasks-file data/reasoning_gym/easy_pilot_tasks.jsonl \
               --default-reward reasoning_gym \
               --max-reward-iters 3 \
               --output-file outputs/rg_easy.jsonl
```

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

See [output-format.md](output-format.md) for the full output record schema, checkpoint format, and analysis snippets.
