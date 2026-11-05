You are an expert in robot task and motion planning and symbolic program induction.

Write a single Python file implementing a policy that solves **every instance** of a KinDER
environment family: a 2D robot (circular base, retractable arm, vacuum gripper) must reach a
goal among a variable number of objects. The family is described in @ENV.md (appended below).

## Task

Write one general policy program for the family. An instance is a random scene with a given
number of count-defining objects (obstructions, blocks, passages, buttons); the policy reads
the scene from the state and must succeed for any count, including counts larger than any in
the demonstrations.

You will be shown an LLM agent solving single instances in @DEMOS.json. Each demo has the
instance (`prompt`: count, step budget and initial state), the agent's steps (`cot`: its
reasoning, the code it ran against a simulator of that instance, and the output it saw), the
policy it submitted for that one instance (`final_response`), whether that policy solved the
instance (`success`) and, for failures, why (`failure`). Successful and failed attempts are
included across object counts. The per-instance policies often hard-code the coordinates of
one scene; use the demos to learn which strategies work, which geometric facts matter
(clearances, grasp poses, collision behaviour), and where agents fail, then compile that into
one general algorithm.

Output:
- a Python solver file @SOLVER_KINDER.py
- a markdown file @SOLVER_KINDER_ALGORITHM.md explaining the algorithm

## Requirements

* Output exactly one Python file and one markdown file.
* Use only the Python standard library, `numpy` and `scipy`. The policy has no simulator, no
  planner and no robotics or geometry package: it plans from the state it receives.
* No per-instance tables or memorised coordinates: the policy is evaluated on 100 held-out
  instances whose counts go beyond the training counts.
* Deterministic behaviour (no wall-clock-dependent decisions).
* At most 60 seconds of computation per instance (reset plus all get_action calls).

## Interface

Implement exactly this class (AgenticGenTAMP's `GeneratedApproach`):

```python
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        """action_space: Box with .low, .high, .shape (float32 vector, see Action Space).
        observation_space: .types, .type_features, .get_type(name) -> Type.
        primitives: an empty dict."""

    def reset(self, state, info):
        """Start of an episode: state is the initial ObjectCentricState,
        info["object_count"] the instance's count."""

    def get_action(self, state):
        """Return the next action (a sequence of floats)."""
```

* The state API: `state.get_objects(observation_space.get_type("rectangle"))`,
  `state.get_object_names()`, `state.get_object_from_name(name)`, `state.get(obj, feature)`,
  `obj.name`, `obj.type.name`, `obj.is_instance(type)`. Objects, types and features are listed
  in @ENV.md.
* An episode runs `reset(state, info)` once and then `get_action(state)` after every step,
  until the goal holds (success) or the step budget `300 + 150 * count` is spent (failure).
* Every action must lie within `[action_space.low, action_space.high]` (compared in float32);
  an out-of-bounds or non-finite action ends the episode as a failure. Clip with
  `np.clip(action, action_space.low, action_space.high)`.
* A motion that would cause a collision is not executed: the robot (and anything it holds)
  stays where it was. A policy must detect lack of progress rather than assume it moved.

## Behavior

The policy should:
* parse the scene into the quantities the task depends on (goal pose, free space, which
  objects block what, grasp poses) from the state alone;
* decompose the task into reusable behaviours (navigate, align, extend arm, grasp with the
  vacuum, carry, place, release, push) and sequence them as the scene requires;
* scale with the object count: decide how many objects to move and in which order, and where
  to put them, instead of handling a fixed number of cases;
* monitor progress and recover (replan, back off, retry from another side) when a behaviour
  stalls;
* stay within the step budget: prefer direct, collision-free motions.

## Testing

Your working directory has `kinder_client.py`, a client for a simulator of this family. Test
scripts (not the solver) may use it:

```python
from kinder_client import make_env, load_approach, rollout, evaluate, format_state
env = make_env()
state, info = env.reset(instance=0)            # training instances 0 .. env.num_instances-1
state, info = env.reset(seed=123, count=4)     # a fresh instance of any count in env.counts
print(format_state(state))
state, reward, terminated, truncated, info = env.step(action)   # terminated: goal reached
policy = load_approach("SOLVER_KINDER.py")
print(rollout(policy, env, seed=7, count=max(env.counts)))       # solved, num_steps, error, ...
evaluate(policy, env, limit=20)                # success by count on training instances
```

Evaluate on the largest counts before finishing; the held-out test instances include them.
