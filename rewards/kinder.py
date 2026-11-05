"""
Reward for the KinDER robotics domains (domains/kinder).

Follows the rewards/ registry contract
    reward(result, execution_ok, entry, **kwargs) -> {"value": float, "message": str, ...}
with value 1.0 iff replaying the action sequence from the instance's reset reaches the
goal (the environment terminates) within the instance's step budget.  KinDER's
kinematic environments are deterministic, so the replay is an exact check that is
independent of whichever policy produced the actions.

``result`` is an action sequence: the compact encoding written into result rows
({"format": "float32le-zlib-base64", ...}), a list of action vectors, or a dict with an
``actions`` key holding either.  ``entry`` is a task record with ``family``, ``count``,
``seed`` and ``max_steps``.
"""
import zlib
from typing import Any, Dict


def reward(result: Any, execution_ok: bool, entry: Dict, **kwargs) -> Dict:
    if not execution_ok:
        return {"value": 0.0, "message": "Solution raised a runtime error during execution."}
    if result is None:
        return {"value": 0.0, "message": "No action sequence."}
    from domains.kinder.sim import replay
    try:
        out = replay(entry["family"], entry["count"], entry["seed"], result, max_steps=entry.get("max_steps"))
    except (ValueError, TypeError, KeyError, zlib.error) as exc:
        return {"value": 0.0, "message": f"Malformed action sequence: {exc}"}
    return {"value": 1.0 if out["solved"] else 0.0, "message": out["message"], "num_steps": out["num_steps"]}
