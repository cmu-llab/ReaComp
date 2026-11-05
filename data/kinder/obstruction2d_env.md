# Obstruction2D (variable object count)

A 2D environment where the goal is to place a target block onto a target surface. The block must be completely contained within the surface boundaries.

The target surface may be initially obstructed.

The robot has a movable circular base and a retractable arm with a rectangular vacuum end effector. Objects can be grasped and ungrasped when the end effector makes contact.

## Generalization

This environment contains a VARIABLE number of objects. A program must handle ANY number of them, in principle unbounded. Do not assume a fixed object count, a fixed number of objects of any type, or any fixed index layout; iterate the objects in the state and act on whatever is there.

The count of an instance is the number of obstructions on the target surface; it is given as `info['object_count']` at reset. Training instances use counts [0, 1, 2]; evaluation instances use counts [0, 1, 2, 3, 4].

## Observation

Each observation is an `ObjectCentricState`: a set of typed objects, each with named features. The number of objects VARIES between episodes, so there is no fixed-length vector and no fixed index layout. Read it with:

- `state.get_objects(type)` / `state.get_object_names()` / `state.get_object_from_name(name)` to enumerate objects (`type = observation_space.get_type("rectangle")`),
- `state.get(obj, feature)` to read a feature; `obj.name`, `obj.type.name`, `obj.is_instance(type)`.

| **Type** | **Features** | **Example objects** |
| --- | --- | --- |
| crv_robot | x, y, theta, base_radius, arm_joint, arm_length, vacuum, gripper_height, gripper_width | robot |
| rectangle | x, y, theta, static, color_r, color_g, color_b, z_order, width, height | obstruction0, obstruction1, ... |
| target_block | x, y, theta, static, color_r, color_g, color_b, z_order, width, height | target_block |
| target_surface | x, y, theta, static, color_r, color_g, color_b, z_order, width, height | target_surface |

Type hierarchy: `circle` is a `kinematic2d`, `double_rectangle` is a `kinematic2d`, `lobject` is a `kinematic2d`, `rectangle` is a `kinematic2d`, `target_block` is a `rectangle`, `target_surface` is a `rectangle`.
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

A penalty of -1.0 is given at every time step until termination, which occurs when the target block is "on" the target surface. The definition of "on" is given below:
```python
def is_on(
    state: ObjectCentricState,
    top: Object,
    bottom: Object,
    static_object_cache: dict[Object, MultiBody2D],
    tol: float = 0.025,
) -> bool:
    """Checks top object is completely on the bottom one.

    Only rectangles are currently supported.

    Assumes that "up" is positive y.
    """
    top_geom = rectangle_object_to_geom(state, top, static_object_cache)
    bottom_geom = rectangle_object_to_geom(state, bottom, static_object_cache)
    # The bottom-most vertices of top_geom should be contained within the bottom
    # geom when those vertices are offset by tol.
    sorted_vertices = sorted(top_geom.vertices, key=lambda v: v[1])
    for x, y in sorted_vertices[:2]:
        offset_y = y - tol
        if not bottom_geom.contains_point(x, offset_y):
            return False
    return True
```

## Episode

An episode succeeds when the environment terminates (the goal above holds) within the step budget of 300 + 150 x count steps (count = number of obstructions on the target surface). Coordinates are in world units; angles are in radians.
