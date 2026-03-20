# Architecture

## Overview

The Symbolic Library Agent is a multi-agent system for solving symbolic inductive reasoning tasks. It is inspired by DreamCoder: rather than solving each task from scratch, it builds a **shared library of reusable Python functions** across tasks. Simpler tasks create primitive building blocks; later tasks compose and reuse them.

The system has four agents that collaborate in a fixed loop:

```
Natural-language prompt
        │
        ▼
  ┌─────────────┐
  │ TaskParser  │  NL → structured TaskSpec (domain, I/O types, hints)
  └──────┬──────┘
         │ TaskSpec
         ▼
  ┌──────────────────────────────────────────────────────┐
  │                   Solve Loop (≤ MAX_STEPS)           │
  │                                                      │
  │   ┌───────────┐   library ops    ┌───────────────┐   │
  │   │ SSL Agent │◄────────────────►│FunctionLibrary│   │
  │   └─────┬─────┘                  └───────────────┘   │
  │         │ working_memory                ▲             │
  │         ▼                              │             │
  │   ┌───────────┐    solution / decompose│             │
  │   │ BCR Agent │────────────────────────┘             │
  │   └───────────┘                                      │
  └──────────────────────────────────────────────────────┘
         │ state["solution"]
         ▼
  ┌──────────────────┐
  │ Reporting Agent  │  formats final answer using original prompt
  └──────────────────┘
```

## Control Flow

### `solve()` — standard mode

```python
task_spec = TaskParser.parse(prompt)

for step in range(MAX_STEPS):
    if state["solved"]:
        break
    if _should_call_ssl(state):
        state = ssl_agent.run(state, library, cost_tracker, task_spec)
    else:
        state = bcr_agent.run(state, library, cost_tracker, task_spec)

if not state["solved"]:
    state = bcr_agent.run(...)   # one final attempt

if state["solved"]:
    state = reporting_agent.run(state, library)
```

### `solve_with_reward()` — closed-loop reward mode

When a task record has a `reward` field (or `--default-reward` is set), the controller runs a reward-feedback loop. BCR is retried up to `--max-reward-iters` times with the reward signal injected into its prompt ("fix mode"):

```python
task_spec = TaskParser.parse(prompt)

for reward_iter in range(max_reward_iters):
    state = _run_solve_loop(state, task_spec, budget)   # SSL → BCR as above
    result = execute_with_library(solution_code, args)
    reward = reward_fn(result, execution_ok, entry)     # from rewards/{name}.py
    state["reward_history"].append({iter, reward, blame, message, ...})
    if reward["value"] >= 1.0:
        break
    # BCR sees reward_history on next iter → "fix mode" prompt

state["solved"] = best_reward >= 1.0
cost_tracker.task_loss += 1.0 - best_reward
reporting_agent.run(state, library)   # runs once on final solution
```

**Blame assignment** (`_determine_blame`): the controller categorises each failed attempt as `"execution"` (code crashed), `"library"` (used library functions, may need fixing), `"partial"` (non-zero score), or `"logic"` (produced code but scored 0).

### SSL → BCR routing (`_should_call_ssl`)

SSL runs on step 0 (library always needs updating before the first solve attempt) and whenever:
- The trace is empty
- BCR just decomposed a task (library needs new sub-functions)
- BCR has run more recently than SSL in the last 3 steps and the task is unsolved

Otherwise BCR runs.

## Agent Descriptions

### TaskParser
Converts the raw natural-language prompt into a structured `TaskSpec` — domain, I/O types, operation hints, and an example input snippet. This runs once at the start, before the solve loop. If the LLM call fails, a minimal fallback spec is used and the loop continues. The spec is passed to SSL and BCR on every step to guide domain-aware library retrieval.

### SSL Agent (Symbolic Search and Library)
The library manager. On each invocation it looks at the current task and the existing library and chooses one of three actions:
- **reuse** — an existing function already covers the need; mark it active
- **compose** — combine two or more existing functions into a new wrapper
- **create** — write a brand-new function from scratch

Preference is always reuse > compose > create. New and composed functions are immediately added to the shared library and verified with `safe_exec`. The SSL output tells BCR which functions to use via `state["working_memory"]["active_functions"]`.

### BCR Agent (Bottom-up Conceptual Reasoning)
The solver. It sees the current task, the full library, and the active functions suggested by SSL, and produces one of three outputs:

- **direct** — for question/prompt-based tasks (e.g. reasoning_gym): BCR reads the question, extracts the concrete input value with its LLM understanding, applies the library function mentally, and returns the answer string directly. No Python code is written and nothing is added to the library. E.g. for a Caesar cipher question BCR identifies the cipher text and returns `caesar_decrypt` applied to it.
- **solve** — for algorithmic tasks with structured input (lists, grids, graphs): BCR writes a reusable Python function. The entry-point name is inferred from the `def` line by regex.
- **decompose** — if the task is too complex, breaks it into ordered sub-tasks stored in `state["working_memory"]["subtasks"]`; SSL will then handle each sub-task in the next iteration.

The `direct` action keeps the library clean — instance-specific solve wrappers never accumulate. Library functions (e.g. `caesar_decrypt`) are still credited via `functions_used` for cost tracking.

**Fix mode:** when `state["reward_history"]` is non-empty (reward loop), BCR receives the previous attempt scores, blame labels, and feedback messages in its user prompt. It is instructed to try a different approach and return a minimal clean answer string (e.g. `"42"` not `"The answer is 42"`) to avoid partial-credit penalties.

### Reporting Agent
Formats the final answer. It runs exactly once, after `state["solved"] == True`. Before calling the LLM, it actually *executes* the solution code (via `execute_with_library`) so the concrete result is available. The LLM then formats that result according to any output-format instructions in the original prompt (e.g. "return as a comma-separated string").

**How `confidence` works:** The reporting agent asks the LLM to output a `confidence` float (0.0–1.0) as part of its JSON response. This is a self-reported estimate by the model — it is not computed from execution success or any external metric. If execution succeeded, the model typically reports high confidence; if execution failed or produced an unexpected result, it reports lower confidence. The fallback path (when LLM parsing fails) always sets `confidence = 0.5`.

---

## Message Passing

There are two distinct levels of communication in the system.

### Level 1 — Between agents (via state dict)

Agents never call each other directly. They communicate by reading from and writing to a shared **state dict** that is passed through the Controller. Each agent reads what it needs, does its work (one LLM call), and writes its output back to state.

Key fields written by each agent:

| Agent | Reads from state | Writes to state |
|---|---|---|
| TaskParser | `original_prompt` | `task_spec` |
| SSL | `task_input`, `task_type`, `working_memory` | `working_memory["active_functions"]`, `trace` |
| BCR | `task_input`, `task_type`, `working_memory` | `solution`, `solved`, `trace` |
| Reporting | `solution`, `original_prompt` | `final_output` |

The `FunctionLibrary` and `CostTracker` are passed as separate arguments alongside the state dict (they are shared across tasks in a session, not per-task).

### Level 2 — Between agents and the LLM (via JSON prompts)

Each agent constructs a prompt in two parts:
- **System prompt** — describes the agent's role, rules, and the exact JSON schema it must output
- **User message** — assembles the current task context, library contents, task spec, and any working memory into a single text string

The LLM returns a plain JSON object (no tool-calling). The agent parses that JSON into a Python dict and reads the fields it cares about. For example, BCR reads `result["action"]`, `result["code"]`, and `result["functions_used"]` and writes them into the state dict.

```
Agent builds prompt          LLM returns JSON          Agent writes to state
──────────────────           ─────────────────         ──────────────────────
system: role + schema    ─►  {"action": "solve",  ─►  state["solution"] = {...}
user:   task + library       "code": "def ...",        state["solved"] = True
                             "functions_used": [...]}   state["trace"].append(...)
```

All LLM calls pass through `LLMClient.create()`, which handles backend differences (Anthropic vs OpenAI-compatible), strips markdown fences, and returns a parsed dict. The `agent_messages` field in the output JSONL contains the complete request + response for every LLM call during a task.

See [data-structures.md](data-structures.md) for the full State schema and [agents.md](agents.md) for each agent's complete JSON schema.

## LLM Communication — JSON Output Mode

All agents instruct the LLM to respond with a plain **JSON object** via the system prompt. No tool-calling API is used. This was an intentional design choice:

- Reasoning models (e.g. `gpt-oss-120b` with `reasoning_backend='GptOss'`) either ignore the `tool_calls` API field entirely or produce 400 errors when multiple tools are provided
- JSON output is supported by every model that can follow instructions
- The OpenAI backend adds `response_format={"type": "json_object"}` for hard guarantees
- The schema is readable in the system prompt alongside the agent logic

See [llm-client.md](llm-client.md) for the client implementation and [agents.md](agents.md) for each agent's JSON schema.

## Cross-Task Library Sharing

When running in batch mode (`solve_batch` or `--tasks-file`), a single `FunctionLibrary` and `CostTracker` instance are shared across all tasks. Functions created for task N are available to tasks N+1, N+2, etc.

This enables emergent abstraction: a `filter_even` function created for "return even numbers" is automatically retrieved as a candidate when solving "sum of even numbers" later.

## Cost Objective

The system tracks a cost objective that rewards compact, reusable libraries:

```
TotalCost = α·NumNewFunctions + β·TotalFunctionLength
          + γ·RedundancyPenalty − δ·ReuseReward

Objective = TaskLoss + λ·TotalCost
```

Default weights: α=1.0, β=0.05, γ=2.0, δ=0.5, λ=0.3 (λ is configurable via `--lam`). `task_loss` is set by `solve_with_reward()` to `1 - best_reward` per task; it stays 0 when using `solve()`. See [data-structures.md](data-structures.md#cost-tracker).
