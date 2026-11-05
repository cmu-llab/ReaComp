# Domains

A *domain* is the unit of extension in ReaComp. It tells the pipeline where the
tasks are, what an agent or solver is allowed to see, how the verifier scores a
candidate, how an induced `SOLVER.py` is called, and where the solver-building
prompt and demo traces live. Everything else (trace generation, solver induction,
solver evaluation, hybrid inference, metrics) is generic.

| domain | tasks | verifier | solver entry point |
|---|---|---|---|
| `pbebench` | replace-cascade PBE (Lite, Hard, real forward reconstruction) | exact execution | `solve_pbe(examples, max_programs=...)` |
| `slr_bench` | inductive Prolog rules over train descriptions | official partial score | `solve_slr(examples)` |
| `list_functions` | integer-list functions, held-out generalisation | exact list equality on held-out pairs | `solve_listfn(examples)` |
| `envsetup` | SetupBench (93) + EnvBench-Python (329): bootstrap a container | hidden success command / pyright | `solve_envsetup(task, shell)` |

```python
from domains import get_domain
dom = get_domain("pbebench")
tasks = dom.load_tasks("lite")
solve = dom.load_solver("built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py")
outcome = dom.solve(solve, tasks[0], dom.split("lite"))   # SolveOutcome(success, score, program, extras)
```

Generic entry points (every stage of the ReaComp pipeline):

```bash
# 1. traces: per-task LLM agent / Best-of-K through the OpenHands runner ...
python -m openhands_agents.run --framework direct_solve --domain <name> --split <split> --output-path traces.jsonl ...
#    ... or the domain's own trace runner when it declares one (envsetup: python -m domains.envsetup.agent)
# 2. demos: traces -> balanced solver-induction demos in the domain's schema
python scripts/build_demos.py --domain <name> --split <split> --traces traces.jsonl --n 100 --out demos/DEMOS_<NAME>.json
# 3. induction: the solver-building coding agent takes its prompts and output file names from the domain
python -m openhands_agents.build_solver --domain <name> --output-dir built_solvers/<run> ...
# 4. evaluation and 5. solver-first hybrid
python scripts/eval_domain_solver.py --domain <name> --split <split> --solver SOLVER.py --output-dir ...
python scripts/hybrid_gate.py --domain <name> --solver solver.jsonl [solver2.jsonl ...] --fallback llm.jsonl --out hybrid.jsonl
```

The historical per-domain scripts (`scripts/eval_solver.py`, `scripts/ensemble_outputs.py`,
`scripts/eval_listfn_solver.py`, ...) are unchanged; the wrappers here delegate to the
same reward modules and reproduce their results.

## Adding a domain

1. Put the tasks in a JSONL file (one record per task) and a verifier in
   `rewards/<name>.py` following `reward(result, execution_ok, entry, **kw) -> {"value", "message"}`.
2. Subclass `domains.base.Domain`: set `name`, `description`, `reward_module`, `solver_entrypoints`,
   `agent_visible_keys`, `building_prompt`, `demos`, `solver_filename`; implement `splits()` and
   `solve()` (call the solver on one record and return a `SolveOutcome`). Override
   `demo_from_trace` if your traces need task-specific fields (the base handles BoK/DF rows),
   `demo_group_key` for balancing, `row_extras`, `complexity`, `agent_task_text`,
   `direct_solve_spec` / `solver_builder_spec` for custom prompts, `cleanup_after_timeout`,
   and set `trace_runner` if the per-task agent needs its own loop.
3. Register it in `domains/__init__.py` (`"name": "module.path:ClassName"`).
4. Write `building_prompts/SOLVER_BUILDING_PROMPT_<NAME>.md` describing the solver
   contract to the coding agent. Demos then come from `scripts/build_demos.py`.

`domains/tests/test_abstraction.py` checks that the generic layer reproduces the legacy
behaviour (prompts, demo schemas, the List Functions demo file) and is the place to add
a regression test for a new domain.

`domains/envsetup/` is the reference implementation of a domain whose solver acts in
an external environment (a Docker container) rather than returning a program.
