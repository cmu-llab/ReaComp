# ClutteredRetrieval2D (variable object count)

A 2D environment where the goal is to retrieve a target block and place it inside a target region.

The target block may be initially obstructed.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.

## Generalization

This environment contains a VARIABLE number of objects. A program must handle ANY number of them, in principle unbounded. Do not assume a fixed object count, a fixed number of objects of any type, or any fixed index layout; iterate the objects in the state and act on whatever is there.

The count of an instance is the number of obstructions around the target; it is given as `info['object_count']` at reset. Training instances use counts [1, 3, 5, 10]; evaluation instances use counts [1, 3, 5, 10, 15, 20, 25].

## Observation

Each observation is an `ObjectCentricState`: a set of typed objects, each with named features. The number of objects VARIES between episodes, so there is no fixed-length vector and no fixed index layout. Read it with:

- `state.get_objects(type)` / `state.get_object_names()` / `state.get_object_from_name(name)` to enumerate objects (`type = observation_space.get_type("rectangle")`),
- `state.get(obj, feature)` to read a feature; `obj.name`, `obj.type.name`, `obj.is_instance(type)`.

| **Type** | **Features** | **Example objects** |
| --- | --- | --- |
| crv_robot | x, y, theta, base_radius, arm_joint, arm_length, vacuum, gripper_height, gripper_width | robot |
| rectangle | x, y, theta, static, color_r, color_g, color_b, z_order, width, height | obstruction0, obstruction1, ... |
| target_block | x, y, theta, static, color_r, color_g, color_b, z_order, width, height | target_block |
| target_region | x, y, theta, static, color_r, color_g, color_b, z_order, width, height | target_region |

Type hierarchy: `circle` is a `kinematic2d`, `double_rectangle` is a `kinematic2d`, `lobject` is a `kinematic2d`, `rectangle` is a `kinematic2d`, `target_block` is a `rectangle`, `target_region` is a `rectangle`, `target_surface` is a `rectangle`.
The count-defining objects are named `obstruction0`, `obstruction1`, ... and their number changes per episode.

## Action Space

The entries of an array in this Box space correspond to the following action features:
| **Index** | **Feature** | **Description** | **Min** | **Max** |
| --- | --- | --- | --- | --- |
| 0 | dx | Change in robot x position (positive is right) | -0.050 | 0.050 |
| 1 | dy | Change in robot y position (positive is up) | -0.050 | 0.050 |
| 2 | dtheta | Change in robot angle in radians (positive is ccw) | -0.196 | 0.196 |
| 3 | darm | Change in robot arm length (positive is out) | -0.100 | 0.100 |
| 4 | vac | Directly sets the vacuum (0.0 is off, 1.0 is on) | 0.000 | 1.000 |

## Reward

A penalty of -1.0 is given at every time step until termination, which occurs when the target block is inside the target region.

## Episode

An episode succeeds when the environment terminates (the goal above holds) within the step budget of 300 + 150 x count steps (count = number of obstructions around the target). Coordinates are in world units; angles are in radians.
