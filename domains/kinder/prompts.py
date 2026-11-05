"""Prompts of the KinDER domain: the policy contract, the per-task agent's task message and the
solver builder's prompts (the per-task agent's system prompt is in
openhands_agents/kinder_direct_solve/controller.py, like direct_solve).

The policy contract is AgenticGenTAMP's ``GeneratedApproach`` interface in its strict
black-box setting, so per-task policies, induced solvers and AgenticGenTAMP programs
are interchangeable.
"""
from __future__ import annotations

POLICY_INTERFACE = '''\
```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        """action_space: Box with .low, .high, .shape (float32 vector; see Action Space).
        observation_space: .types, .type_features, .get_type(name) -> Type.
        primitives: an empty dict (no planners or simulator are provided)."""

    def reset(self, state, info):
        """Start of an episode. state: the initial ObjectCentricState;
        info["object_count"]: the instance's count."""

    def get_action(self, state):
        """Return the next action for the current state (a sequence of floats)."""
```

Rules
- An episode runs `reset(state, info)` once, then `get_action(state)` after every step until the
  goal is reached (success) or the step budget is spent (failure).
- Every action must lie within `[action_space.low, action_space.high]` (compared in float32); an
  out-of-bounds or non-finite action ends the episode as a failure. Clip with
  `np.clip(action, action_space.low, action_space.high)`.
- Imports: the Python standard library, `numpy` and `scipy` only. The policy cannot use the
  simulator, `kinder_client`, or any robotics/geometry package; it must plan from the state.
- Time: 60 seconds of policy computation per episode (reset plus all get_action calls).
- The state passed in is fresh on every call; the policy may keep its own fields (e.g. a plan).
'''

CLIENT_USAGE = '''\
```python
from kinder_client import make_env, rollout, evaluate, load_approach, format_state
env = make_env()                    # connects to the simulator of this sandbox
state, info = env.reset()           # {reset_doc}
print(format_state(state))          # every object with its features
state, reward, terminated, truncated, info = env.step(action)   # terminated: goal reached
env.max_steps, env.count            # step budget and object count of the current instance
env.render("frame.png")             # PNG image of the current state
result = rollout(GeneratedApproach, env{rollout_args})      # one episode -> solved, num_steps, error, final_state
```'''

AGENT_TASK_TEMPLATE = """\
{card}

# Your instance

{instance}
"""

BUILDER_SYSTEM_PROMPT = """\
You are an expert in robot task and motion planning, program induction and Python.

Your task is to write ONE policy program that solves every instance of a KinDER environment
family, including instances with more objects than any shown to you. The detailed
specification is in the user message.

Tools
  execute_code(code)
      Runs `code` as a Python script in a fresh process inside your working directory, which
      contains:
        DEMOS.json                 demonstrations of an LLM agent solving single instances
        ENV.md                     the environment description (also in the user message)
        kinder_client.py           simulator client: import kinder_client
        {solver_file}           your solver, after you write it with write_file
                                   (re-copied into the directory before every run)
      Importable: the standard library, numpy, scipy and kinder_client. Each call has a
      {timeout}s time limit; output is truncated, so print summaries.
  write_file(filename, content)
      Writes {solver_file} or {algorithm_file}. You MUST write both before finishing.
  finish(summary)
      Signals that you are done, after both files are written.

Simulator access (inside execute_code)
{client_usage}
  evaluate(load_approach("{solver_file}"), env, limit=20)   # pinned training instances

Workflow
  1. Read the specification and ENV.md; inspect DEMOS.json (which strategies succeed, where
     agents fail, how behaviour changes with the object count).
  2. Design a general algorithm; write {solver_file}.
  3. Evaluate it on training instances and on fresh instances with the largest counts; fix the
     failures you find; repeat.
  4. Write {algorithm_file} and call finish.

Requirements
  - The solver implements GeneratedApproach exactly as specified, imports only the standard
    library, numpy and scipy, and never imports kinder_client.
  - Do NOT produce placeholder code. Write a complete, working implementation.
"""

BUILDER_FILE_NOTE = """\
File reference guide (for @-references in the specification below):
  @DEMOS.json → DEMOS.json in your working directory (read it with execute_code)
  @ENV.md → ENV.md in your working directory (its content is appended to this message)
  @{solver_file} → write with write_file(filename="{solver_file}", content=...)
  @{algorithm_file} → write with write_file(filename="{algorithm_file}", content=...)

---

"""

ENV_APPENDIX = """

---

# Environment description (ENV.md)

{card}
"""

WORKSPACE_GUIDE = """\
# Working directory

| file | content |
| --- | --- |
| `BUILDING_PROMPT.md` | this specification |
| `DEMOS.json` | demonstrations (see the specification) |
| `ENV.md` | the environment description |
| `kinder_client.py` | simulator client and state classes (standard library + numpy) |

Write `{solver_file}` and `{algorithm_file}` in this directory. Run test scripts from this
directory with an interpreter that has numpy and scipy; `kinder_client.make_env()` connects to the
simulator through `kinder_env.sock` here, which is available while the environment server for this
directory runs. Write test scripts as separate files: the solver itself must not import kinder_client.

Simulator access from a test script:
{client_usage}
  evaluate(load_approach("{solver_file}"), env, limit=20)   # pinned training instances
"""


def client_usage(pinned_only: bool) -> str:
    if pinned_only:
        return CLIENT_USAGE.format(reset_doc="always this task's instance", rollout_args="")
    return CLIENT_USAGE.format(
        reset_doc="training instance 0; reset(instance=i) for i < env.num_instances;\n"
                  "                                    # reset(seed=s, count=k) for a fresh instance, k in env.counts",
        rollout_args=", instance=0")
