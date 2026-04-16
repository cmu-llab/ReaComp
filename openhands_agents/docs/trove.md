# TroVE baseline

Paper-faithful rewrite of [Wang et al. 2024](https://arxiv.org/abs/2401.12869).

## Algorithm

For each task (online / streaming):

1. Generate **3K candidates** concurrently (K per mode): IMPORT, CREATE, SKIP
2. Execute all candidates in the Apptainer sandbox
3. **Select best** by reward (deviation from paper's self-consistency — we have external reward fns); tiebreak: fewest AST nodes
4. If best used CREATE mode: add its new function to the toolbox
5. Increment usage counts for toolbox functions referenced in the winning solution
6. Every `--trim-every` tasks: remove functions below usage threshold λ = 0.5 × log10(n_processed)

## Prompt modes

| Mode | What the LLM does |
|------|------------------|
| `skip` | Solve directly with Python primitives; no toolbox |
| `import` | Import and use an existing toolbox function |
| `create` | Write a NEW standalone helper, then use it |

`import` mode is skipped when the toolbox is empty.

## Toolbox listing format

Functions are shown with their **def signature** (not just name + description) so the model can judge whether to import:

```
  def apply_ordered_string_rules(s, rules)  # Apply replacement rules in order
  def find_replace_sequence(inputs, outputs)  # Find str.replace ops from examples
```

This is `PkgLibrary.as_listing_with_signatures()`.

## Token tracking

TroVE uses raw `aiohttp` calls (not the OpenHands SDK), so `usage` is read directly from the API response. Each task result includes:

```json
"token_usage": {"prompt_tokens": 1840, "completion_tokens": 612}
```

Aggregate across tasks in `scripts/eval.py` for total token cost.

## Parallelism

`--workers N` processes N tasks concurrently via `ThreadPoolExecutor`. Each task's 3K LLM calls are already concurrent within the task (via `asyncio.gather`). The toolbox `PkgLibrary` is thread-safe (per-write locking).

Note: TroVE's trim runs in the calling thread after each task completes. With multiple workers the `n_processed` counter (used for the trim threshold) is not atomically incremented — minor inaccuracy at trim time is acceptable.
