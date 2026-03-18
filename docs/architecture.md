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

### SSL → BCR routing (`_should_call_ssl`)

SSL runs on step 0 (library always needs updating before the first solve attempt) and whenever:
- The trace is empty
- BCR just decomposed a task (library needs new sub-functions)
- BCR has run more recently than SSL in the last 3 steps and the task is unsolved

Otherwise BCR runs.

## Agent Communication via State

Agents communicate entirely through a shared **state dict** — no direct agent-to-agent calls.

Key fields written by each agent:

| Agent | Writes |
|---|---|
| TaskParser | `state["task_spec"]` |
| SSL | `state["working_memory"]["active_functions"]`, `state["trace"]` |
| BCR | `state["solution"]`, `state["solved"]`, `state["trace"]` |
| Reporting | `state["final_output"]` |

See [data-structures.md](data-structures.md) for the full State schema.

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

Weights: α=1.0, β=0.05, γ=2.0, δ=0.5, λ=0.3. See [data-structures.md](data-structures.md#cost-tracker).
