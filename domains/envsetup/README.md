# Environment-setup domain

Benchmarks: **SetupBench** (microsoft/SetupBench, 93 tasks: repository setup in 7
ecosystems, dependency resolution, database setup, background services; bare
`ubuntu:22.04` unless the task says otherwise) and **EnvBench-Python**
(JetBrains-Research/EnvBench, 329 repositories; 229 train / 100 test split as published).

| file | role |
|---|---|
| `tasks.py` | normalised task records, repository cache (git / HF tarballs), container provisioning (fixtures, prerunners) |
| `container.py` | `DockerTaskContainer` and the persistent root `bash` session every driver uses |
| `verifier.py` | hidden checks: SetupBench success command in the live container; EnvBench script replay + pyright |
| `domain.py` | `EnvSetupDomain`: runs an induced `solve_envsetup(task, shell)` under budget, verifies, tears down |
| `agent.py` | OpenHands bash agent (trace generation, LLM baseline, warm-start hybrid fallback) |
| `demos.py` | balanced demo traces from our agent runs and EnvBench's published GPT-4o trajectories |
| `solver_interface.md` | the contract an induced solver must follow |
| `vendor/` | EnvBench's evaluation scripts (MIT) used verbatim for replay verification |

Data: `data/envsetup/setupbench/{scenarios,fixtures}` and `data/envsetup/envbench/splits`
are vendored (MIT); repositories, tarballs and trajectories are cached under
`data/envsetup/cache/` (git-ignored). Task files: `python -m domains.envsetup.tasks build`.

Deviation from upstream: SetupBench's 9 npm dependency-resolution tasks reference a
private Azure image; records use `node:16` and carry `image_substituted: true`.

## Typical run

```bash
# 1. LLM agent (traces + baseline), model served by vLLM
python -m domains.envsetup.agent --split setupbench --base-url http://gpu:8000/v1 \
    --model openai/Qwen/Qwen3.8-27B --output outputs/envsetup/agent_setupbench.jsonl --workers 4

# 2. demos for induction: from our traces (generic builder) ...
python scripts/build_demos.py --domain envsetup --split setupbench \
    --traces outputs/envsetup/agent_setupbench.jsonl --n 100 --out demos/DEMOS_ENVSETUP.json
#    ... or mixing in EnvBench's published GPT-4o trajectories for the train split
python -m domains.envsetup.demos --traces outputs/envsetup/agent_envbench_train.jsonl \
    --envbench-published train --n 100 --out demos/DEMOS_ENVSETUP.json

# 3. induce a solver with a coding agent: Claude Code on a Docker host (it can run the dev eval),
#    or the OpenHands solver builder (reads demos, cannot run containers from its sandbox)
python -m openhands_agents.build_solver --domain envsetup --output-dir built_solvers/envsetup_<run> \
    --sif-path ... --base-url ... --model ...

# 4. evaluate the solver, then the solver-first hybrid
python scripts/eval_domain_solver.py --domain envsetup --split setupbench \
    --solver built_solvers/<run>/SOLVER_ENVSETUP.py --workers 4 --task-timeout 2400 \
    --output-dir evals/solver_results/envsetup/<run>
python scripts/hybrid_gate.py --domain envsetup --solver evals/solver_results/envsetup/<run>/setupbench.jsonl \
    --fallback outputs/envsetup/agent_setupbench.jsonl --out outputs/envsetup/hybrid_setupbench.jsonl
# warm handoff variant: the agent continues in the solver's container
python -m domains.envsetup.agent --split setupbench --warm-start-solver built_solvers/<run>/SOLVER_ENVSETUP.py ...
```

Requirements: Docker on the machine running these commands (containers need root
inside), the Python 3.12 environment from the repo root (`uv venv --python 3.12 .venv &&
uv pip install openhands-sdk==1.16.1 openhands-tools==1.16.1`), and network access for
`apt`/`pip` inside containers.
