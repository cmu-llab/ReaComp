# KinDER robotics domains

ReaComp on robot task and motion planning: per-instance LLM agents solve single scenes of a
[KinDER](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden) environment
family, a coding agent compiles their reasoning traces into **one policy program for the whole
family**, and that program is evaluated on held-out scenes with zero LLM calls, alone and as the
first stage of a solver-first hybrid.

The protocol follows AgenticGenTAMP ("Coding Agents for Generalized Task and Motion Planning
Problems", [tomsilver/robocode](https://github.com/tomsilver/robocode)), so results are directly
comparable with its programs, planners and agents: same environments, same instance generator,
same object-count schedule, same step budgets, same 60 s policy-time limit, same program
interface (`GeneratedApproach`) and the same strict black-box access (standard library, numpy
and scipy; no simulator inside the program).

## Families

Kinematic 2D environments (no physics engine; a step takes 2–9 ms on one CPU core). Each family
is its own domain, `kinder_<family>`, because one solver is induced per family.

| domain | goal | count = number of | train counts | test counts (extrapolation in bold) | step budget |
|---|---|---|---|---|---|
| `kinder_obstruction2d` | place the target block on the target surface | obstructions on the surface | 0, 1, 2 | 0, 1, 2, **3**, **4** | 300–900 |
| `kinder_clutteredretrieval2d` | retrieve the target block from clutter into the target region | obstructions | 1, 3, 5, 10 | 1, 3, 5, 10, **15**, **20**, **25** | 450–4050 |
| `kinder_clutteredstorage2d` | put all blocks inside the shelf | blocks | 1, 3, 7 | 1, 3, 7, **11**, **15** | 450–2550 |
| `kinder_motion2d` | reach the target region through narrow passages | passages | 0, 1, 2, 3 | 0, 1, 2, 3, **4**, **5** | 300–1050 |
| `kinder_stickbutton2d` | touch every button (a graspable stick reaches far ones) | buttons | 1, 2, 3 | 1, 2, 3, **5**, **10** | 450–1800 |

Splits per family (`data/kinder/<family>_{train,test}.jsonl`):

* `train`, 100 instances cycling through the training counts: per-task agent traces for the
  demos, and instances the solver builder can test on.
* `test`, 100 instances cycling through the evaluation counts (AgenticGenTAMP's suite size): 58–68
  instances at training counts (`regime: design`) and 32–42 at larger counts
  (`regime: extrapolation`), the hard tier.

The step budget of an instance is `300 + 150 × count`. Every record stores `count`, `regime`,
`seed`, `max_steps` and a SHA-256 of its initial state; the seed is never shown to agents or
solvers. `data/kinder/<family>_env.md` is the environment description agents receive (KinDER's
own description, observation, action and reward documentation, as in AgenticGenTAMP's
strict black-box card).

## Solver interface

A solver (and every per-task policy) is one Python file defining AgenticGenTAMP's class:

```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): ...   # primitives == {}
    def reset(self, state, info): ...       # info["object_count"]
    def get_action(self, state): ...        # -> 5 floats (dx, dy, dtheta, darm, vac)
```

* `state` is an object-centric state (`get_objects(type)`, `get_object_names()`,
  `get_object_from_name(name)`, `get(obj, feature)`; types via `observation_space.get_type(name)`).
  It is a pure-Python mirror (`client.py`) with the API of AgenticGenTAMP's strict black-box
  client, and the same object at development time and at evaluation.
* Actions must lie within `[action_space.low, action_space.high]` in float32; an out-of-bounds or
  non-finite action ends the episode as a failure (as it crashes the episode in robocode).
* Imports: standard library, numpy, scipy, sibling files. Checked statically before evaluation
  (`strict.py`, the allowlist of robocode's `strict_blackbox.py`; no `importlib`, `runpy`,
  `__import__` or `sys.modules`).
* 60 s of policy time per instance (reset + all get_action calls), interrupted by SIGALRM; the
  timeout is a `BaseException` and re-fires every second, so a policy cannot catch it and continue
  (as in robocode).

An episode is solved when the environment terminates within the step budget. The executed
actions are stored in the row (`answer`, compact float32 encoding) and verified by exact replay
from the instance's reset (`rewards/kinder.py`); KinDER's kinematic environments are
deterministic, so the replay must agree with the rollout (`replay_agrees` in every row).

## Pipeline

| stage | entry point | output |
|---|---|---|
| traces / LLM baseline | `openhands_agents/run_kinder_direct_solve.py` (`scripts/run_kinder_direct_solve_openhands.sh`) | `outputs/kinder_<family>_<split>_direct_solve_openhands.jsonl` |
| demos | `scripts/build_demos.py --domain kinder_<family> --split train` | `demos/DEMOS_KINDER_<FAMILY>_seed_42_100_examples_with_CoT.json` |
| induction (open model) | `openhands_agents/build_solver.py --domain kinder_<family>` (`scripts/run_kinder_solver_builder_openhands.sh`) | `built_solvers/<model>/<timestamp>_kinder_<family>/SOLVER_KINDER.py` |
| induction (Claude Code, Codex) | `scripts/kinder_workspace.py create` + `serve` | same files, copied into `built_solvers/claude_code/...` |
| evaluation + hybrid | `scripts/eval_domain_solver.py`, `scripts/hybrid_gate.py` (`scripts/run_all_kinder_evals.sh`) | `evals/solver_results/<model>/<run>/test.jsonl`, `outputs/kinder_<family>_test_hybrid_<run>.jsonl` |
| report | `scripts/kinder_report.py` | pass rate by regime and count, tokens, solver coverage |

**Per-task agent (KinDER DirectSolve).** One OpenHands conversation per instance with two tools:
`python(code, reason)` runs a script in a sandbox whose `kinder_client` reaches a simulator of
that instance only (reset/step/render; no state access, no seeds), and `submit(code, notes)`
submits a `GeneratedApproach` for that instance (rejected with a message if it does not define
the class or imports something forbidden, so the agent can fix it). If the model answers without a
tool call before submitting, it is reminded to submit (twice at most, within its turn budget);
`end_reason` records how each run ended. The harness evaluates the submitted policy in a separate
process. Rows record every step (the model's reasoning and
message, the code, the output), the submitted `policy_code` and the verified result; they are
the demos' source, the LLM baseline on `test`, and the hybrid's fallback.

**Solver builder.** The generic OpenHands solver builder with the KinDER sandbox: its
`execute_code` runs in a working directory holding `DEMOS.json`, `ENV.md`, `kinder_client.py` and
the solver it has written, and the simulator serves the training instances plus fresh instances
of any configured count (as AgenticGenTAMP's agents can sample them). No Apptainer image is
needed. External coding agents get the same directory and simulator from `scripts/kinder_workspace.py`.

## Experiments

Per family, with the same open model for every LLM arm (commands in COMMANDS.md §7):

| arm | how | measures |
|---|---|---|
| per-instance LLM agent | DirectSolve on `test` | baseline success and tokens per instance |
| ReaComp solver | builder on the default demos, 5 independent runs (AgenticGenTAMP also uses 5 replicates) | zero-token success by regime and count, run-to-run variance |
| ReaComp solver, Claude Code | `scripts/kinder_workspace.py` | inducing-agent capability |
| no traces | builder with an empty demos file (`echo '[]' > demos/DEMOS_KINDER_<F>_empty.json`, `DEMOS_PATH=...`): the AgenticGenTAMP setting (simulator only) | the value of the traces |
| no reasoning | `scripts/build_demos.py ... --no-cot` (`..._100_examples.json`) | the value of the reasoning in the traces |
| fewer demos | `--n 12`, `--n 48` | data efficiency |
| ensemble | `scripts/hybrid_gate.py --solver run1.jsonl run2.jsonl ...` (best row per task) | union coverage of the runs |
| hybrid | solver first, DirectSolve fallback (`scripts/run_all_kinder_evals.sh`) | success vs tokens (Pareto), solver coverage |

Induction cost (one-time tokens of a builder run) comes from its debug trajectory
(`scripts/compute_trajectory_tokens.py`); with the per-instance cost of the agent it gives the
break-even number of instances. The extrapolation counts are the hard tier: compare the regimes
in `scripts/kinder_report.py`.

## Code

| file | role |
|---|---|
| `families.py` | the five families: KinDER class, count keyword, train/test counts, step budget |
| `client.py` | state/space mirrors, wire codec, action contract, simulator client, `rollout`/`evaluate` helpers (stdlib + numpy; copied into sandboxes as `kinder_client.py`) |
| `sim.py` | KinDER side: instances, policy rollout with the policy clock, exact replay, action encoding, environment description |
| `server.py` | simulator server (Unix socket, JSON lines; reset/step/render) |
| `sandbox.py`, `sandbox_boot.py` | working directory + server + runner with an import guard, scrubbed environment and timeouts; `BuilderSandbox` for the solver builder |
| `strict.py` | static import allowlist for solvers |
| `prompts.py` | policy contract, task message, solver-builder prompts |
| `domain.py` | `KinderDomain`: splits, solve, rows, demos, builder spec and sandbox, isolated policy evaluation |

Elsewhere: `data_generation/kinder_generator.py` (task files and verification), `rewards/kinder.py`
(verifier), `building_prompts/SOLVER_BUILDING_PROMPT_KINDER.md`,
`openhands_agents/kinder_direct_solve/` (agent), `domains/tests/test_kinder.py` (tests).

## Verification of the task files

`python -m data_generation.kinder_generator verify` checks the generated instances against the
benchmark's own code:

1. **Stored instances reproduce**: every record's initial-state digest matches the installed KinDER.
2. **KinDER**: for every instance whose count is a registered KinDER variant
   (`kinder.make("kinder/<Family>-<x><count>-v0")`), the registered environment and ours agree
   bit for bit on the action space, the initial state, and the states and terminations along 50
   random actions.
3. **AgenticGenTAMP** (`--robocode-src <checkout of tomsilver/robocode>`): the family configs equal
   robocode's `experiments/conf/environment/<family>_generalized.yaml`, robocode's own
   `generate_eval_seeds` yields the stored seeds, its round-robin count schedule yields the stored
   counts, and its `VariableObjectCountEnv` regenerates every instance (same initial state, step
   budget and transitions).

Result with kindergarden 0.2.5 (2026-11-05): checks 1 and 2 pass for all 1,000 instances; 873
are compared directly with registered variants, the other 127 have counts that KinDER does not
register (see caveats). Check 3 needs a robocode checkout:

```bash
git clone --depth 1 https://github.com/tomsilver/robocode.git /tmp/robocode
python -m data_generation.kinder_generator verify --robocode-src /tmp/robocode
```

## Caveats

* **Different test instances, same distribution.** AgenticGenTAMP keeps its evaluation seed
  private (available from its authors on request). Our `test` split uses the same generator and
  schedule with a public master seed (20261106), so success rates are comparable in expectation
  over 100 instances, not instance for instance. Build robocode's exact suite into a private
  directory with `python -m data_generation.kinder_generator build --test-seed <their seed> --out <dir>`
  and run on it with `KINDER_DATA_DIR=<dir>` set (the launchers then name their outputs after the
  directory, so results on the two suites never mix).
* **Unregistered counts.** ClutteredRetrieval2D counts 3, 5, 15, 20 and ClutteredStorage2D count 11
  have no registered KinDER variant; they are built from the environment class with the count
  keyword, exactly as robocode's `VariableObjectCountEnv` builds every count (verification 3
  covers them; verification 2 cannot).
* **KinDER version.** We use kindergarden 0.2.5 from PyPI; robocode pins the KinDER repository at
  commit `5b2dbac` (also 0.2.5). Their kinematic 2D sources are identical (they differ only in
  Dynamic3D registration). `python -m data_generation.kinder_generator check` detects a KinDER
  version that no longer reproduces the stored instances.
* **Training instances.** AgenticGenTAMP has no fixed training set (its agents sample instances
  freely). ReaComp needs traces, so `train` fixes 100 instances at the training counts; the solver
  builder can additionally sample fresh instances of any count, as AgenticGenTAMP's agents can.
* **Evaluation harness.** Same episode semantics as robocode, with these differences: the policy
  is constructed afresh for every instance (robocode reuses one object across episodes; both call
  `reset` per episode), the state is the mirror rather than the `relational_structs` object
  (programs written against robocode's strict client run unchanged; methods that only the real
  class has, e.g. `pretty_str`, do not exist), and robocode's determinism re-run is replaced by
  exact replay of the executed actions.
* **Scope.** The five kinematic 2D families with a count axis. PushPullHook2D (fixed size; robocode
  gives it vector observations), the Dynamic2D families (pymunk) and the 3D families are not
  included yet; `families.py` is where they would be added.

## Tests

```bash
.venv/bin/python -m domains.tests.test_kinder            # ~30 s, CPU only
.venv/bin/python -m domains.tests.test_kinder --quick    # ~15 s, without the agent/builder end-to-end runs
```

The end-to-end parts drive the real per-task agent and solver builder (OpenHands, sandboxes,
simulator servers, policy evaluation, demos) against a scripted mock LLM
(`domains/tests/mock_llm.py`) and a scripted Motion2D policy
(`domains/tests/fixtures/kinder_motion2d_policy.py`), which solves Motion2D instances of every count.
