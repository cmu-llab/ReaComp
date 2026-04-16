# ReAct + Library baseline

## Architecture

Single-conversation agent using the OpenHands SDK. Four tools are available within one conversation:

| Tool | Purpose |
|------|---------|
| `execute_code(code)` | Run Python in Apptainer sandbox; `from library import fn` works |
| `add_to_library(name, description, code)` | Validate + persist a new function to the shared library |
| `check_reward(answer)` | Call the task verifier inline; returns reward ∈ [0,1] + feedback |
| `finish(answer)` | Submit final answer and stop the conversation |

## Why check_reward instead of an outer loop

The old design restarted the entire conversation after each reward evaluation, which:
- discarded all conversation history (agent forgot what it tried)
- capped attempts at `--max-reward-iters` (default 3) regardless of progress

`check_reward` gives the agent the verifier as a first-class tool. It can call it as many times as needed within the same conversation, retaining full context. The single-conversation budget is controlled by `--max-steps` (default 100).

## Library reuse mechanism

Retrieved functions are shown with **full code** in the task prompt:

```
Library functions available (reuse these if applicable):

  # apply_ordered_string_rules: Apply a list of (old, new) replacement rules in order
  def apply_ordered_string_rules(s, rules):
      ...
```

Previously only `name: description` was shown — the agent couldn't judge whether a function was suitable without seeing the signature and body, so it created a new one every time.

Usage tracking: `ExecuteCodeTool` scans executed code for `from library import fn_name` and calls `library.increment_usage([fn_name])` on success. This powers TroVE-style trim and the library growth metric.

## BM25 retrieval

`PkgLibrary.retrieve(query, k)` — Jaccard on tokenised `name + description` tokens. Top-k results (default k=5) are fetched at conversation start. The agent sees the retrieved functions but can also add new ones; new additions are immediately importable in subsequent `execute_code` calls.

## SDK compatibility notes

All custom Action/Observation classes are prefixed `RL` to avoid name collisions in the SDK's global discriminated union (e.g. `RLFinishAction`, not `FinishAction`).

Tools are injected directly into the agent's private tool map because the SDK's `Agent.tools` only accepts name-reference `Tool(name=str)` objects for server-registered tools:

```python
agent.__pydantic_private__["_tools"] = {t.name: t for t in tool_instances}
agent.__pydantic_private__["_initialized"] = True
```

System prompt must be in a `.j2` file — `Agent(system_prompt=...)` is silently ignored:

```python
agent = Agent(..., system_prompt_filename=os.path.join(_prompts_dir, "system_prompt.j2"))
```

## Token tracking

The OpenHands SDK abstracts away raw API responses, so `usage` tokens are not currently captured for react_library. The `--debug-dir` output records `reward_history` and `library_additions_this_task` as proxies. If exact token counts are needed, the LLM wrapper would need to be patched.

## Parallelism

`--workers N` runs N tasks concurrently in a `ThreadPoolExecutor`. Each thread creates its own `Conversation` and `tempdir`. Library writes are protected by `threading.Lock` in `PkgLibrary`. Note: parallel tasks retrieve the library state at conversation start, so a function added by task A may not be visible to task B if B started before A committed its addition.
