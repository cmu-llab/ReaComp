# openhands_agents

Sandboxed reimplementations of three baselines. Code execution runs inside an Apptainer container (minimal `python:3.11-slim + numpy scipy sympy` image). Reward computation runs on the host. Separate entry point from `main.py`.

| Baseline | Framework | Key idea |
|---|---|---|
| `react_library` | OpenHands SDK (`Agent` + `Conversation`) | ReAct loop with shared growing function library |
| `trove` | Pure async vLLM | 3×K concurrent candidates, reward selection, per-function file trimming |
| `best_of_k` | Fully async vLLM | Two-stage: generate all K×N samples first, score+pick second |

---

## Prerequisites

### 1. Build the sandbox SIF (once)

No Docker required. The script tries two strategies in order:

**Strategy A — `--fakeroot` build** (preferred; works if user namespaces are enabled on the cluster):
```bash
bash openhands_agents/scripts/build_sandbox.sh /scratch/$USER/sif_images
# produces: /scratch/$USER/sif_images/sandbox.sif
```
This builds from `openhands_agents/sandbox.def` (`python:3.11-slim + numpy scipy sympy`) using `apptainer build --fakeroot`.

**Strategy B — reuse `openhands.sif`** (automatic fallback if fakeroot fails):
If you already have `openhands.sif` pulled (e.g. from a different project), the script will symlink it as `sandbox.sif` provided it has numpy/scipy/sympy. Pull it first if needed:
```bash
apptainer pull /scratch/$USER/sif_images/openhands.sif \
    docker://ghcr.io/all-hands-ai/openhands:main
bash openhands_agents/scripts/build_sandbox.sh /scratch/$USER/sif_images
```

### 2. Install host dependencies (once per env)

```bash
pip install openhands-sdk openhands-tools   # for react_library
pip install aiohttp                         # for trove and best_of_k
```

Reward computation packages must also be installed on the host:

```bash
pip install reasoning-gym    # for reasoning_gym reward
# PBEBench: no extra packages
# SLR-Bench: SWI-Prolog must be available on the host
```

### 3. Serve the model via vLLM (GPU node)

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --port 8000 \
    --tensor-parallel-size <num_gpus>
```

Note the hostname — pass it as `--base-url http://<gpu-node>:8000/v1`.

---

## Running the baselines

Run scripts live in `scripts/` alongside the other baselines. Edit the config variables at the top of each script, then run from the project root.

### ReAct + Library

```bash
# Edit GPU_NODE, MODEL, DATASET etc. at the top of the script
bash scripts/run_react_library_openhands.sh
```

### TroVE

```bash
bash scripts/run_trove_openhands.sh
```

### Best-of-K

```bash
bash scripts/run_best_of_k_openhands.sh
```

### Build sandbox SIF

```bash
bash scripts/build_sandbox_openhands.sh
```

### Direct CLI (for quick smoke tests)

```bash
python -m openhands_agents.run \
    --framework react_library \
    --dataset-path data/interleaved/pbebench_rg_string_pilot.jsonl \
    --output-path outputs/oh_rl_test.jsonl \
    --sif-path /scratch/$USER/sif_images/sandbox.sif \
    --pkg-dir /scratch/$USER/oh_packages \
    --base-url http://<gpu-node>:8000/v1 \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --default-reward reasoning_gym \
    --max-steps 6 \
    --max-reward-iters 2
```

---

## CLI flags reference

| Flag | Default | Description |
|---|---|---|
| `--framework` | required | `react_library`, `trove`, or `best_of_k` |
| `--dataset-path` | required | Input JSONL |
| `--output-path` | required | Output JSONL (appended live) |
| `--default-reward` | required | Reward module (e.g. `reasoning_gym`, `pbebench`) |
| `--sif-path` | `$SANDBOX_SIF` | Path to `sandbox.sif` |
| `--pkg-dir` | `~/oh_packages` | Host dir for `library/` and `toolbox/` package dirs |
| `--base-url` | `http://localhost:8000/v1` | vLLM endpoint |
| `--model` | `Qwen/Qwen3-Coder-480B-A22B-Instruct` | Model name as served |
| `--api-key` | `$VLLM_API_KEY` or `EMPTY` | API key |
| `--max-tokens` | `4096` | Max tokens per LLM call |
| `--k` | `5` | Candidates per mode (trove) or total samples (best_of_k) |
| `--max-steps` | `8` | react_library: max agent steps per conversation |
| `--max-reward-iters` | `3` | react_library: reward-feedback loop iterations |
| `--library-k` | `5` | react_library: BM25 top-k functions shown per step |
| `--trim-every` | `200` | trove: trim toolbox every N tasks |
| `--max-concurrent` | `64` | best_of_k: async concurrency cap |
| `--temperature` | `0.8` | Sampling temperature (best_of_k) |
| `--skip-existing` | off | Skip tasks already in output JSONL |
| `--checkpoint-path` | `<output>.ckpt.json` | Checkpoint file |

---

## Architecture

```
openhands_agents/
  Dockerfile          python:3.11-slim + numpy scipy sympy
  sandbox.py          ApptainerSandbox — all code execution goes here
  pkg_library.py      PkgLibrary — per-function .py files + BM25 retrieval
                      (shared base for react_library and trove toolbox)
  react_library/
    tools.py          ExecuteCodeTool, AddToLibraryTool, FinishTool (OH SDK)
    controller.py     OpenHands Agent+Conversation; outer reward loop
    prompts.py
  trove/
    controller.py     3×K concurrent generation, reward selection, trim
    prompts.py        IMPORT / CREATE / SKIP mode prompts
  best_of_k/
    controller.py     async Stage 1 (generate) + Stage 2 (score+pick)
  run.py              CLI entry point
  scripts/            (internal helpers — run scripts are in top-level scripts/)
    build_sandbox.sh
```

### Library/toolbox as Python package files

Functions are stored as individual `.py` files in a package directory on the host, bind-mounted read-only into the container:

```
/scratch/$USER/oh_packages/
  library/          # react_library — bound as /exec/library:ro
    __init__.py     # auto-generated: `from .fn_name import fn_name`
    fn_name.py      # one file per function
  toolbox/          # trove — bound as /exec/toolbox:ro
    __init__.py
    fn_name.py
```

Generated code imports normally: `from library import fn_name`. The `__init__.py` is regenerated every time a function is added or trimmed. Functions must be **standalone** — no imports from `library` or `toolbox` inside function bodies, so trimming never silently breaks other functions.

### Reward computation

Runs on the **host**, not inside the container. The sandbox only executes generated code. Install reward-specific dependencies (e.g. `reasoning-gym`, SWI-Prolog) in your host environment.

---

## Tool invocation — function calling, not fine-tuning

The three custom tools (`execute_code`, `add_to_library`, `finish`) are exposed to the model via the standard **OpenAI function-calling API** — their JSON schemas are sent with every LLM request. The model does not need to be fine-tuned or pre-trained on these specific tool names; it generalises from its instruction-tuning on function calling.

`execute_code` is essentially a sandboxed terminal execution step. Switching to a raw bash tool would add complexity (shell escaping, stateful env, directory tracking) with no benefit — the current abstraction is cleaner.

Watch for Qwen models placing tool calls inside `reasoning_content` instead of the structured response field. The SDK already injects a corrective nudge when this happens (see `Agent.step()` in `openhands_docs/`).

## SDK compatibility notes (openhands-sdk 1.16.1)

`Agent.tools` accepts only `Tool(name=str)` name references — these are resolved by the SDK at runtime via `resolve_kind()` for server-registered tools. Custom `ToolDefinition` subclasses must be injected directly into the agent's private runtime map after construction:

```python
agent = Agent(llm=..., tools=[], include_default_tools=[])
# include_default_tools=[] avoids name conflict with the SDK's built-in FinishTool
agent.__pydantic_private__["_tools"] = {t.name: t for t in tool_instances}
agent.__pydantic_private__["_initialized"] = True
```

`ToolDefinition.create()` must return `list[Self]` (not a bare instance) to match the SDK's `Sequence[Self]` contract.

## Verifying the OpenHands SDK version

The `react_library` tools use `Agent(system_prompt=..., max_iterations=...)` and `Conversation.stop()`. Check these against your installed SDK version before running:

```python
from openhands.sdk import Agent, Conversation
help(Agent.__init__)
```

Adjust `react_library/controller.py` and `tools.py` if the SDK API differs.
