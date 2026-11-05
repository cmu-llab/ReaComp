You are an expert in developer tooling, build systems and symbolic program induction.

Write a single Python file implementing a deterministic solver for **environment
setup** tasks: given a fresh Linux container containing a repository (or fixture
files) and a short task statement, install toolchains and dependencies, build,
initialise databases or start services so that the project is ready to run.

## Task

You will be shown demonstrations of an LLM agent solving such tasks in
@demos/DEMOS_ENVSETUP.json. Each demo contains the task statement, a snapshot of the
repository, the agent's full command trace with a one-line rationale per command
and the observed output / exit code, the resulting script, and whether the hidden
verifier accepted the final state. Successful and failed traces are included across
task categories (repository setup in several ecosystems, dependency resolution,
database setup, background services, EnvBench Python repositories). Use them to
understand which strategies work, where agents waste steps, and which failure modes
recur.

Output:
- a Python solver file @SOLVER_ENVSETUP.py
- a markdown file @SOLVER_ENVSETUP_ALGORITHM.md explaining the algorithm

## Interface

Implement exactly the contract in @domains/envsetup/solver_interface.md:

```python
def solve_envsetup(task, shell):
    """
    task : dict with problem_statement, category, language, ecosystem, base_image,
           workdir, initial_commands, budget_s  (no verifier is provided)
    shell: shell.run(cmd, timeout=None) -> result with .exit_code, .output, .timed_out
           (persistent root bash session inside the task container)
    returns {"success": bool, "notes": str}
    """
```

## Requirements

* Python standard library only; no LLM calls, no network access from the solver
  process (network inside the container is fine).
* Deterministic and general: probe the container and repository (manifests, lock
  files, READMEs, CI configs, `.python-version`/`.nvmrc`/...), decide an install plan,
  execute it, verify with real checks (imports, builds, test collection, service
  pings), and repair common failures (missing system libraries, wrong Python/Node
  version, peer-dependency conflicts, lock-file mismatches, services needing a
  config change before they accept connections).
* `success` must reflect your own verification; a false positive is worse than a
  false negative because it disables the LLM fallback in the hybrid pipeline.
* Stay within `task["budget_s"]`; give up early on hopeless paths.
* No per-repository hard-coding. Strategies keyed on ecosystem, manifest shape or
  error message patterns are what we want.

## Development loop

Run the solver on the small dev split (Docker required) and inspect the logs:

```bash
python scripts/eval_domain_solver.py --domain envsetup --split setupbench_dev \
    --solver SOLVER_ENVSETUP.py --workers 2 --task-timeout 2400 --output-dir /tmp/envsetup_dev
```

Do not read the hidden `verify` field of task records or the `hidden` block; the
solver must work from the agent-visible fields only.
