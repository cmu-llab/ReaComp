"""KinDER environment families used as ReaComp domains.

One family is one task distribution: a KinDER environment class whose instances
differ in their initial state (the reset seed) and in the number of count-defining
objects (obstructions, blocks, passages, buttons).  ReaComp induces one solver per
family, so each family is registered as its own domain (``kinder_<name>``).

The counts and step budgets follow the generalized-planning protocol of
AgenticGenTAMP (tomsilver/robocode, ``experiments/conf/environment/*_generalized.yaml``
and ``VariableObjectCountEnv``):

* ``design_counts``: the counts training instances are drawn from;
* ``eval_counts``: the counts the 100 test instances cycle through; counts outside
  ``design_counts`` are the extrapolation (hard) tier;
* step budget per instance: ``base_steps + steps_per_object * count``.

This module is import-light (no KinDER import) so registries and prompts can use it
without the simulator installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class FamilySpec:
    name: str                       # registry suffix, e.g. "obstruction2d"
    title: str                      # KinDER class name without "Env", e.g. "Obstruction2D"
    env_class: str                  # "module:Class" of the KinDER ConstantObjectKinDEREnv
    count_kwarg: str                # constructor keyword that sets the object count
    count_object_prefix: str        # name prefix of the count-defining objects
    count_noun: str                 # what the count counts, for prompts
    design_counts: Tuple[int, ...]
    eval_counts: Tuple[int, ...]
    base_steps: int = 300
    steps_per_object: int = 150

    def max_steps(self, count: int) -> int:
        return self.base_steps + self.steps_per_object * int(count)

    def regime(self, count: int) -> str:
        """``design`` for counts seen in training, ``extrapolation`` beyond them."""
        return "design" if int(count) in self.design_counts else "extrapolation"

    @property
    def all_counts(self) -> Tuple[int, ...]:
        return tuple(sorted(set(self.design_counts) | set(self.eval_counts)))


_K2D = "kinder.envs.kinematic2d"

FAMILIES: Dict[str, FamilySpec] = {f.name: f for f in (
    FamilySpec("obstruction2d", "Obstruction2D", f"{_K2D}.obstruction2d:Obstruction2DEnv",
               "num_obstructions", "obstruction", "obstructions on the target surface",
               design_counts=(0, 1, 2), eval_counts=(0, 1, 2, 3, 4)),
    FamilySpec("clutteredretrieval2d", "ClutteredRetrieval2D",
               f"{_K2D}.clutteredretrieval2d:ClutteredRetrieval2DEnv",
               "num_obstructions", "obstruction", "obstructions around the target",
               design_counts=(1, 3, 5, 10), eval_counts=(1, 3, 5, 10, 15, 20, 25)),
    FamilySpec("clutteredstorage2d", "ClutteredStorage2D",
               f"{_K2D}.clutteredstorage2d:ClutteredStorage2DEnv",
               "num_blocks", "block", "blocks to store on the shelf",
               design_counts=(1, 3, 7), eval_counts=(1, 3, 7, 11, 15)),
    FamilySpec("motion2d", "Motion2D", f"{_K2D}.motion2d:Motion2DEnv",
               "num_passages", "obstacle", "narrow passages (two obstacles each)",
               design_counts=(0, 1, 2, 3), eval_counts=(0, 1, 2, 3, 4, 5)),
    FamilySpec("stickbutton2d", "StickButton2D", f"{_K2D}.stickbutton2d:StickButton2DEnv",
               "num_buttons", "button", "buttons to press",
               design_counts=(1, 2, 3), eval_counts=(1, 2, 3, 5, 10)),
)}

FAMILY_NAMES = tuple(FAMILIES)


def get_family(name: str) -> FamilySpec:
    if name not in FAMILIES:
        raise KeyError(f"unknown KinDER family {name!r}; available: {sorted(FAMILIES)}")
    return FAMILIES[name]
