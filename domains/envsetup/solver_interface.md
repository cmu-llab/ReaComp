# Solver interface: environment setup

An induced solver for this domain is one Python file, `SOLVER_ENVSETUP.py`, that
defines

```python
def solve_envsetup(task: dict, shell) -> dict:
    ...
    return {"success": bool, "notes": str}
```

It is called once per task inside the ReaComp harness (`domains/envsetup/domain.py`)
with a live container already provisioned. Everything the solver does happens
through `shell`.

## `task` (agent-visible fields only)

| key | meaning |
|---|---|
| `problem_statement` | the benchmark's task text (SetupBench statement, or the EnvBench statement) |
| `category` | `reposetup`, `dependency_resolution`, `dbsetup`, `bgsetup`, `envbench_python` |
| `language`, `ecosystem` | hints from the benchmark where available (may be `None`) |
| `base_image` | e.g. `ubuntu:22.04` (bare), `ruby:2.7`, `node:16`, the EnvBench image |
| `workdir` | where the repository or fixtures were placed (`/testbed`, `/data/project`); the shell starts there |
| `initial_commands` | commands the harness already ran (e.g. `eval "$(pyenv init -)"` for EnvBench) |
| `budget_s` | wall-clock budget for the whole solve |

The hidden verifier (`verify` in the raw record) is never passed to the solver.

## `shell`

`shell.run(command: str, timeout: float | None = None) -> ShellResult` with fields
`exit_code`, `output` (stdout+stderr, tail), `timed_out`.

* One persistent `bash` session as root: `cd`, `export`, `source` persist between
  calls; background processes (`nohup ... &`, `service x start`) keep running.
* stdin is `/dev/null`: use non-interactive flags (`apt-get install -y`, `pip --no-input`).
* A command that exceeds its timeout has its whole process group killed and the
  session is reopened (cwd and exported variables are reset; the filesystem is kept).
* Commands issued after `budget_s` is exhausted are refused with exit code 124.
* `shell.log` holds every command with its exit code; the harness's replayable
  script is the ordered list of commands that exited 0.

## Requirements for an induced solver

* Python standard library only; deterministic; no LLM or network calls from the
  solver process itself (network **inside** the container, e.g. `apt-get`, `pip`, is fine).
* Must generalise across repositories and ecosystems: no per-task lookup tables
  keyed on repository names.
* Return `success=True` only when the solver's own checks pass (imports resolve,
  build succeeds, service answers). This self-claim gates the hybrid: a false
  claim skips the LLM fallback, so precision matters more than recall.
* Respect `budget_s`: check elapsed time and fail fast on hopeless tasks.

## Evaluation

```bash
python scripts/eval_domain_solver.py --domain envsetup --split setupbench_dev \
    --solver built_solvers/<run>/SOLVER_ENVSETUP.py --workers 2 --task-timeout 2400 \
    --output-dir evals/solver_results/envsetup/<run>
```

Result rows carry `claimed_success` (the solver's claim) next to `solved` (the
hidden verifier), so self-check precision can be read off directly.
