"""A scripted Motion2D policy used by domains/tests/test_kinder.py (GeneratedApproach interface).

Face +x, pass every narrow passage through the centre of its gap (passages sorted by x),
then drive to the centre of the target region.  Standard library + numpy only.
"""
import numpy as np


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low, dtype=np.float64)
        self.high = np.asarray(action_space.high, dtype=np.float64)
        self.waypoints = []

    def reset(self, state, info):
        obstacles = [o for o in state if o.name.startswith("obstacle")]
        walls = {}
        for o in obstacles:
            walls.setdefault(round(state.get(o, "x"), 6), []).append(o)
        self.waypoints = []
        for x in sorted(walls):
            pair = sorted(walls[x], key=lambda o: state.get(o, "y"))
            bottom, top = pair[0], pair[-1]
            gap_lo = state.get(bottom, "y") + state.get(bottom, "height")
            gap_hi = state.get(top, "y")
            yc = 0.5 * (gap_lo + gap_hi)
            width = state.get(bottom, "width")
            self.waypoints += [(x - 0.15, yc), (x + width + 0.15, yc)]
        target = state.get_object_from_name("target_region")
        self.waypoints.append((state.get(target, "x") + 0.5 * state.get(target, "width"),
                               state.get(target, "y") + 0.5 * state.get(target, "height")))

    def get_action(self, state):
        robot = state.get_object_from_name("robot")
        x, y, theta = state.get(robot, "x"), state.get(robot, "y"), state.get(robot, "theta")
        action = np.zeros(5)
        if abs(theta) > 1e-3:                       # face +x first
            action[2] = -theta
            return np.clip(action, self.low, self.high)
        while len(self.waypoints) > 1 and np.hypot(self.waypoints[0][0] - x, self.waypoints[0][1] - y) < 1e-3:
            self.waypoints.pop(0)
        tx, ty = self.waypoints[0]
        action[0], action[1] = tx - x, ty - y
        return np.clip(action, self.low, self.high)
