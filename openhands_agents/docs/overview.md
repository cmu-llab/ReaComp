# openhands_agents — Overview

Three sandboxed baselines for symbolic-library-agent, separate from `main.py`.

## Baselines

| Framework | CLI `--framework` | Key idea |
|-----------|-------------------|----------|
| ReAct+Library | `react_library` | OpenHands SDK agent; inline `check_reward` for iterative refinement; shared function library via BM25 retrieval |
| TroVE | `trove` | Paper-faithful rewrite; 3 modes × K samples per task, reward-based selection |
| Best-of-K | `best_of_k` | Fully async; K independent samples, pick best by reward; no library |

## Directory layout

```
openhands_agents/
  run.py                  # CLI entry point
  pkg_library.py          # Shared PkgLibrary (functions as .py files, thread-safe)
  sandbox.py              # ApptainerSandbox — code execution
  react_library/
    controller.py         # ReActLibraryController
    tools.py              # ExecuteCode, AddToLibrary, CheckReward, Finish tools
    prompts.py            # build_task_prompt (includes full fn code for reuse)
    prompts/system_prompt.j2
  trove/
    controller.py         # TroVEController
    prompts.py            # IMPORT / CREATE / SKIP prompt builders
  best_of_k/
    controller.py         # BestOfKController
  docs/                   # This folder
  scripts/                # Shell run scripts
```

## Quick start

```bash
python -m openhands_agents.run \
  --framework react_library \
  --dataset-path data/interleaved/pbebench_rg_string_pilot.jsonl \
  --output-path outputs/oh_react_library.jsonl \
  --sif-path /scratch/$USER/sif_images/sandbox.sif \
  --pkg-dir /scratch/$USER/oh_packages \
  --base-url http://localhost:8000/v1 \
  --model openai/Qwen/Qwen3-Coder-480B-A22B-Instruct \
  --default-reward pbebench \
  --workers 8 \
  --debug-dir outputs/debug_react_library
```

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--workers N` | 1 | Parallel task threads (react_library / trove) |
| `--max-steps N` | 100 | Max agent steps per conversation (react_library) |
| `--library-k N` | 5 | BM25 top-K functions shown per task (react_library) |
| `--k N` | 5 | Samples per mode (trove) or total samples (best_of_k) |
| `--trim-every N` | 200 | Toolbox trim period in tasks (trove) |
| `--debug-dir DIR` | — | Per-task JSON debug files (prompt, answer, reward) |
| `--skip-existing` | false | Resume: skip tasks already in checkpoint |
| `--clear` | false | Delete output, checkpoint, and pkg_dir before starting |

## Output

JSONL file compatible with `scripts/eval.py`. One record per task:

```json
{
  "task_id": 42,
  "dataset": "pbebench",
  "solved": true,
  "answer": "...",
  "best_reward": 1.0,
  "reward_history": [...],
  "library_size": 12,
  "library_additions_this_task": 1
}
```

Checkpoint: `<output>.ckpt.json` — tracks `completed_ids` (set, order-independent) + controller state.

## See also

- [react_library.md](react_library.md) — tool design, check_reward, reuse mechanism
- [trove.md](trove.md) — TroVE algorithm, token tracking, listing format
- [pkg_library.md](pkg_library.md) — shared library internals, thread safety
- [sandbox.md](sandbox.md) — Apptainer sandbox, bind mounts, constraints
