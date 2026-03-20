"""
Reward function registry.

Each reward lives in its own module: rewards/{name}.py
Every module must expose:

    def reward(result: Any, execution_ok: bool, entry: dict) -> dict:
        ...
        return {"value": float, "message": str}   # message is optional

    - result       : raw Python object returned by execute_with_library(), or None on failure
    - execution_ok : False if the solution code raised an exception
    - entry        : the full task record dict from the JSONL (includes question, answer, metadata)
    - value        : float in [0.0, 1.0] — 1.0 means fully correct
    - message      : optional natural-language feedback for the agent

Usage:
    from rewards import load_reward
    reward_fn = load_reward("reasoning_gym")
    result = reward_fn(raw_result, execution_ok, entry)
"""

import importlib
from typing import Any, Callable


def load_reward(name: str) -> Callable:
    """
    Load and return the reward function for *name*.

    Looks for rewards/{name}.py (with hyphens/spaces normalised to underscores).
    Raises NotImplementedError with a clear message if the module does not exist
    or does not define a top-level reward() function.
    """
    module_name = name.strip().replace("-", "_").replace(" ", "_")
    try:
        module = importlib.import_module(f"rewards.{module_name}")
    except ModuleNotFoundError:
        raise NotImplementedError(
            f"No reward function implemented for '{name}'. "
            f"Create rewards/{module_name}.py and define:\n\n"
            f"    def reward(result, execution_ok, entry) -> dict:\n"
            f"        ...  # return {{\"value\": float, \"message\": str}}"
        )

    if not hasattr(module, "reward"):
        raise NotImplementedError(
            f"rewards/{module_name}.py exists but does not define a reward() function."
        )

    return module.reward
