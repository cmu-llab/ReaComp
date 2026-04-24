# SolverBuilder agent

## Purpose

Whereas the other baselines solve PBE tasks one at a time, the SolverBuilder agent has a single meta-level goal: **read the reasoning traces in `DEMOS.json` and write a symbolic solver** (`SOLVER.py` + `SOLVER_ALGORITHM.md`). The hypothesis is that a strong open-source coding model (Qwen3-Coder) can synthesise a reasonable symbolic solver from the same trace data used to prompt-engineer Claude Code's hand-crafted solver.

## Architecture

Single OpenHands conversation, three tools:

| Tool | Purpose |
|------|---------|
| `execute_code(code)` | Run Python snippets in the sandbox; `DEMOS.json` and the verifier are bind-mounted at `/workspace/` |
| `write_file(filename, content)` | Write `SOLVER.py` or `SOLVER_ALGORITHM.md` to the host output directory |
| `finish(summary)` | Signal completion; stops the conversation |

No per-task loop, no shared library, no reward iteration — the agent runs exactly once and produces two files.

## What the agent sees

The user message is the full content of `building_prompts/SOLVER_BUILDING_PROMPT.md`, prefixed with a file-path resolution note that maps the `@`-references in the spec to their sandbox locations:

```
@DEMOS.json          → /workspace/DEMOS.json        (read via execute_code)
@rewards/pbebench.py → from rewards.pbebench import reward
@SOLVER.py           → write_file(filename='SOLVER.py', ...)
@SOLVER_ALGORITHM.md → write_file(filename='SOLVER_ALGORITHM.md', ...)
```

Inside the sandbox, the bind mounts are:

| Host path | Container path | Mode |
|-----------|---------------|------|
| `DEMOS.json` | `/workspace/DEMOS.json` | ro |
| `rewards/` | `/workspace/rewards` | ro |

`sys.path` is automatically extended to include `/workspace`, so `from rewards.pbebench import reward` works directly without any explicit path manipulation.

## Entry point

```bash
python -m openhands_agents.build_solver \
    --building-prompt building_prompts/SOLVER_BUILDING_PROMPT.md \
    --demos-path DEMOS.json \
    --rewards-dir rewards \
    --output-dir built_solvers/<run_name> \
    --sif-path /scratch/$USER/sif_images/sandbox.sif \
    --base-url http://localhost:8000/v1 \
    --model openai/Qwen/Qwen3-Coder-30B-A3B-Instruct
```

Or via the convenience script:

```bash
bash scripts/run_solver_builder_openhands.sh <PORT>
```

Output lands in `built_solvers/oh_solver_<timestamp>/`. Debug trajectory in `debug_oh_solver_builder/<timestamp>/`.

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--max-steps N` | 200 | Max agent steps in the conversation |
| `--max-tokens N` | 16384 | LLM token budget per step (large: solver code is long) |
| `--sandbox-timeout N` | 60 | Per-`execute_code` timeout (seconds) |
| `--debug-dir DIR` | — | Write `solver_builder_run.json` + `solver_builder_trajectory.json` |

## Downstream evaluation

The produced `SOLVER.py` can be evaluated with `scripts/eval_solver.py` on PBEBench tasks. Compare against the Claude Code-authored solver to measure the open-source gap.

## Design notes

- **`extra_binds` in `ApptainerSandbox.run_code()`** — added to support binding multiple host files/directories into the container at arbitrary paths. The existing `lib_dir` parameter only supported a single package directory; solver_builder needs both `DEMOS.json` (a file) and `rewards/` (a directory) visible simultaneously.
- **`write_file` validates filenames** — only `SOLVER.py` and `SOLVER_ALGORITHM.md` are accepted, preventing the agent from accidentally writing to arbitrary host paths.
- **`SB` prefix on all Action/Observation classes** — required because the OpenHands SDK registers these in a global discriminated union keyed by class name. Collisions with other loaded frameworks would cause silent misrouting.
- **System prompt is written to a temp `.j2` file** — same pattern as `StaticLibraryController`; `Agent(system_prompt=...)` is silently ignored by the SDK.
