# Symbolic Library Agent

An agentic reasoning system that solves symbolic inductive reasoning tasks by:

1. **Constructing and reusing a shared library of functions** across tasks
2. **Minimising complexity via reuse** (Occam's razor)
3. **Iteratively decomposing problems** using learned abstractions

---

## Architecture

```mermaid
flowchart TD
    NL["Natural Language Prompt"]

    NL -->|"parse"| PARSE["Task Parser\ndomain · I/O types · op hints"]
    NL -->|"original prompt"| RPT

    PARSE -->|"task spec"| LOOP

    subgraph LOOP["Solve Loop  (SSL → BCR × MAX_STEPS)"]
        direction LR
        SSL["SSL Agent\nreuse · compose · create"]
        BCR["BCR Agent\nsolve · decompose"]
        SSL -->|"write"| LIB[("Function Library\ndomain-aware · type-matched")]
        LIB -->|"retrieve"| SSL
        LIB -->|"retrieve"| BCR
    end

    LOOP -->|"solution"| RPT
    RPT["Reporting Agent\nformat using original prompt"]
    RPT --> ANS(["Answer"])

    LOOP --> LLM
    RPT --> LLM
    PARSE --> LLM

    subgraph LLM["LLM Backend"]
        direction LR
        ANT["Anthropic claude-*"]
        OAI["vLLM / OpenAI-compat."]
    end
```

### State Object

```python
State = {
    "task_input":     ...,   # task description + examples
    "task_type":      "",    # e.g. "list_transform", "sequence"
    "working_memory": None,  # active functions, sub-tasks
    "library":        [],    # snapshot of library at task start
    "trace":          [],    # step-by-step agent log
    "budget":         15.0,  # remaining step budget
    "steps":          0,
    "solved":         False,
    "solution":       None,
}
```

### Function Representation

```python
@dataclass
class Function:
    name: str
    code: str
    description: str = ""
    usage_count: int = 0
    creation_cost: float = 0.0

    def usefulness(self):
        return usage_count / (creation_cost + 1e-6)
```

### Controller Loop

```python
for step in range(MAX_STEPS):
    if solved(state): break

    if should_call_ssl(state):
        state = SSL_agent(state)   # update library
    else:
        state = BCR_agent(state)   # attempt solution

state = Reporting_agent(state)
```

### Cost Function

```
TotalCost = α·NumNewFunctions + β·TotalFunctionLength
          + γ·RedundancyPenalty − δ·ReuseReward

Objective = TaskLoss + λ·TotalCost
```

| Weight | Symbol | Default |
|--------|--------|---------|
| NumNewFunctions | α | 1.0 |
| TotalFunctionLength | β | 0.05 |
| RedundancyPenalty | γ | 2.0 |
| ReuseReward | δ | 0.5 |
| Regularisation | λ | 0.3 |

---

## Project Structure

```
symbolic-library-agent/
├── symbolic_agent/
│   ├── __init__.py
│   ├── models.py           # Function, make_state
│   ├── library.py          # FunctionLibrary (add, retrieve, format)
│   ├── costs.py            # CostTracker
│   ├── executor.py         # Safe Python code execution
│   ├── llm_client.py       # Unified LLM adapter (Anthropic + OpenAI/vLLM)
│   ├── task_parser.py      # NL → TaskSpec (domain, I/O types, hints)
│   ├── ssl_agent.py        # SSL agent (library management)
│   ├── bcr_agent.py        # BCR agent (solve / decompose)
│   ├── reporting_agent.py  # Reporting agent (format output)
│   └── controller.py       # Main controller loop
├── examples/
│   └── tasks.py            # Example symbolic reasoning tasks
├── main.py                 # CLI entry point
├── requirements.txt
└── .env                    # ANTHROPIC_API_KEY (not committed)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Anthropic API

```bash
# List available example tasks
python main.py --list

# Run a single task (e.g. task 0)
python main.py --task 0

# Run all tasks in batch (library is shared across tasks)
python main.py

# Run all tasks and print final library stats
python main.py --stats

# Write per-task output files for evaluation
python main.py --output-dir results/

# Use a specific model
python main.py --model claude-opus-4-6 --task 0
```

### Local vLLM

The system supports any OpenAI-compatible endpoint via `--base-url`.  No
Anthropic API key is required when using this mode.

```bash
# Start a vLLM server (example)
vllm serve gpt-oss-120b --port 8000

# Run a single task against it
python main.py --base-url http://localhost:8000/v1 --model gpt-oss-120b --task 0

# Batch run with stats
python main.py --base-url http://localhost:8000/v1 --model gpt-oss-120b --stats
```

If the vLLM server requires an API key, set it in your environment:

```bash
export VLLM_API_KEY=your-key
```

**How it works:** `symbolic_agent/llm_client.py` is a thin adapter that
translates the Anthropic tool-use schema (`input_schema`, `tool_choice`) to
the OpenAI function-calling schema (`parameters`, `tool_choice="required"`)
and maps the response back to the same interface the agents expect.  The
agents themselves are backend-agnostic.

#### Test script

`scripts/test_vllm.sh` runs the agent against a local vLLM server using
`scripts/mock_tasks.jsonl` as the task file:

```bash
bash scripts/test_vllm.sh
```

To use a different server or model, edit the arguments at the top of the script directly.

### Running tasks from a file

Pass a JSON or JSONL file with `--tasks-file`.  All tasks in the file share
one library, so functions learned on early tasks can be reused on later ones.

```bash
python main.py --tasks-file tasks.jsonl
python main.py --tasks-file tasks.json --stats
# combine with vLLM
python main.py --tasks-file tasks.jsonl --base-url http://localhost:8000/v1 --model gpt-oss-120b
```

**JSON format** — a single array of task objects:

```json
[
  {"prompt": "Given a list of integers, return only the even numbers.", "type": "list_transform"},
  {"prompt": "Reverse a list.", "type": "list_transform"},
  {"prompt": "Return the sum of all elements in a list."}
]
```

**JSONL format** — one JSON object per line:

```jsonl
{"prompt": "Given a list of integers, return only the even numbers.", "type": "list_transform"}
{"prompt": "Reverse a list.", "type": "list_transform"}
{"prompt": "Return the sum of all elements in a list."}
```

**Record schema:**

| Key | Required | Description |
|-----|----------|-------------|
| `prompt` | yes | The task description passed to the agents |
| `type` | no | Task category label (default: `"symbolic"`) |
| any other keys | no | Passed through to agents as additional context |

### Output files for evaluation

Pass `--output-dir <DIR>` to write two JSON files per task:

```
results/
  task_0000/
    trajectory.json   # full agent trace for analysis
    response.json     # final answer for evaluation
  task_0001/
    ...
```

**`trajectory.json`** — everything needed to analyse the agent's reasoning:

```json
{
  "task_index": 0,
  "task_type": "list_transform",
  "original_prompt": "Given a list of integers, return only the even numbers.",
  "task_spec": { "domain": "list_manipulation", "input_types": ["list[int]"], "output_type": "list[int]", "operation_hints": ["filter"], "symbolic_inputs": "lst = [1, 2, 3, 4]" },
  "solved": true,
  "steps_taken": 2,
  "trace": [ { "step": 0, "agent": "SSL", "actions": [...] }, { "step": 1, "agent": "BCR", "action": "solve", "reasoning": "..." } ],
  "solution": { "code": "def solve(lst): ...", "function": "solve", "functions_used": ["filter_even"] },
  "library_snapshot": [ { "name": "filter_even", "domain": "list_manipulation", "usage_count": 1, ... } ],
  "cost_summary": { "num_new_functions": 1, "reuse_count": 1, "total_cost": 1.15, ... }
}
```

**`response.json`** — minimal record for task-level evaluation:

```json
{
  "task_index": 0,
  "task_type": "list_transform",
  "original_prompt": "Given a list of integers, return only the even numbers.",
  "solved": true,
  "answer": "[2, 4]",
  "explanation": "Filtered the list using the filter_even library function.",
  "confidence": 0.95,
  "execution_result": [2, 4]
}
```

Works with all input modes:

```bash
python main.py --output-dir results/
python main.py --task 0 --output-dir results/
python main.py --tasks-file tasks.jsonl --output-dir results/
python main.py --tasks-file tasks.jsonl --base-url http://localhost:8000/v1 --model gpt-oss-120b --output-dir results/
```

---

## Key Behaviours

- **Reuse over invention**: the SSL agent checks the library before creating anything new.
- **Composition**: new functions are built from existing ones where possible.
- **Shared library**: running tasks in batch mode lets the library grow across tasks, so later tasks can reuse functions from earlier ones.
- **Cost tracking**: every new function and every reuse is logged; total cost and objective are reported after each task.
- **Safe execution**: generated Python code is run in a sandboxed namespace with forbidden-import checks.

---

## Example Output

```
============================================================
Task 0  type=list_transform
============================================================
Solved : True
Answer : [2, 4, 6]
Explain: Doubled each element using the map_double library function.
Conf.  : 0.95
Exec   : [2, 4, 6]

Cost summary:
  num_new_functions            : 1
  total_function_length        : 3
  reuse_count                  : 0
  redundancy_penalty           : 0.0
  reuse_reward                 : 0.0
  total_cost                   : 1.15
  objective                    : 0.345
```
