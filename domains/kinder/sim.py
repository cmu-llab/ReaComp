"""Simulator side of the KinDER domain: instances, policy rollouts, exact replay.

Everything that imports KinDER lives here (and in ``server.py``).  An instance is a
(family, count, seed) triple; it is built exactly as AgenticGenTAMP builds it (the
family's ``ConstantObjectKinDEREnv`` constructed with the count keyword and
``allow_state_access=True``, then the object-centric env underneath reset with the
seed), so a test split built with robocode's evaluation seed reproduces robocode's
evaluation suite.

Policies only ever see the pure-Python mirrors of ``client.py``.
"""
from __future__ import annotations

import base64
import contextlib
import importlib
import re
import signal
import threading
import time
import traceback
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import client as C
from .families import FamilySpec, get_family

ACTIONS_FORMAT = "float32le-zlib-base64"


# ------------------------------------------------------------------------- instances
def _env_class(family: FamilySpec):
    module, _, cls = family.env_class.partition(":")
    return getattr(importlib.import_module(module), cls)


def build_wrapper(family: FamilySpec, count: int):
    """The family's ConstantObjectKinDEREnv for one object count."""
    return _env_class(family)(**{family.count_kwarg: int(count)}, allow_state_access=True)


def _types_payload(type_features: Dict[Any, List[str]]) -> List[dict]:
    def depth(t) -> int:
        n, p = 0, t.parent
        while p is not None:
            n, p = n + 1, p.parent
        return n

    return [{"name": t.name, "parent": t.parent.name if t.parent is not None else None, "features": list(f)}
            for t, f in sorted(type_features.items(), key=lambda kv: (depth(kv[0]), kv[0].name))]


def _action_names(action_space) -> List[str]:
    try:
        md = action_space.create_markdown_description()
    except Exception:  # noqa: BLE001
        md = ""
    names = re.findall(r"^\|\s*\d+\s*\|\s*([A-Za-z_][\w]*)\s*\|", md, flags=re.M)
    return names if len(names) == action_space.shape[0] else [f"a{i}" for i in range(action_space.shape[0])]


class Instance:
    """One KinDER instance driven through its object-centric environment."""

    def __init__(self, family: FamilySpec | str, count: int, seed: int) -> None:
        self.family = get_family(family) if isinstance(family, str) else family
        self.count, self.seed = int(count), int(seed)
        self.max_steps = self.family.max_steps(self.count)
        self._wrapper = build_wrapper(self.family, self.count)
        self._env = self._wrapper._object_centric_env  # noqa: SLF001 - KinDER's documented object-centric layer
        self.types_payload = _types_payload(self._env.type_features)
        self._type_by_name, self._type_features = C.types_from_payload(self.types_payload)
        space = self._env.action_space
        self.action_space = C.ActionSpace(space.low, space.high, _action_names(space))
        self.observation_space = C.ObservationSpace(self._type_features)

    # spaces as sent over the wire
    def spaces_payload(self) -> dict:
        return {"types": self.types_payload,
                "action": {"low": self.action_space.low.tolist(), "high": self.action_space.high.tolist(),
                           "names": self.action_space.names}}

    def _mirror(self, ocs) -> C.ObjectCentricState:
        data = {C.Object(o.name, self._type_by_name[o.type.name]): np.array(ocs.data[o], dtype=np.float32)
                for o in ocs}
        return C.ObjectCentricState(data, self._type_features)

    def reset(self) -> Tuple[C.ObjectCentricState, dict]:
        ocs, info = self._env.reset(seed=self.seed)
        return self._mirror(ocs), {**info, "object_count": self.count}

    def step(self, action: Any) -> Tuple[C.ObjectCentricState, float, bool, bool, dict]:
        a = C.validate_action(action, self.action_space)
        ocs, reward, terminated, truncated, info = self._env.step(a)
        return self._mirror(ocs), float(reward), bool(terminated), bool(truncated), {**info, "object_count": self.count}

    def render_png(self) -> bytes:
        import io
        import imageio.v2 as imageio
        buf = io.BytesIO()
        imageio.imwrite(buf, self._env.render(), format="png")
        return buf.getvalue()

    def close(self) -> None:
        for fn in (self._wrapper.close, self._env.close):
            with contextlib.suppress(Exception):
                fn()


def initial_state_payload(family: FamilySpec | str, count: int, seed: int) -> dict:
    inst = Instance(family, count, seed)
    try:
        state, _ = inst.reset()
        return C.state_to_payload(state)
    finally:
        inst.close()


# --------------------------------------------------------------------------- actions
def encode_actions(actions: Sequence[np.ndarray], dim: int = 5) -> dict:
    """Exact, compact encoding of an action sequence for result rows."""
    arr = np.asarray(actions, dtype="<f4").reshape(-1, dim) if len(actions) else np.zeros((0, dim), "<f4")
    return {"format": ACTIONS_FORMAT, "shape": list(arr.shape),
            "data": base64.b64encode(zlib.compress(arr.tobytes(), 9)).decode("ascii")}


def decode_actions(obj: Any) -> np.ndarray:
    """Inverse of ``encode_actions``; also accepts a plain list of action vectors."""
    if isinstance(obj, dict) and obj.get("format") == ACTIONS_FORMAT:
        raw = zlib.decompress(base64.b64decode(obj["data"]))
        return np.frombuffer(raw, dtype="<f4").reshape(obj["shape"]).astype(np.float32)
    if isinstance(obj, dict) and "actions" in obj:
        return decode_actions(obj["actions"])
    arr = np.asarray(obj, dtype=np.float32)
    return arr.reshape(len(arr), -1) if arr.size else arr.reshape(0, 0)


# -------------------------------------------------------------------- policy rollouts
class PolicyTimeout(BaseException):
    """The policy exhausted its time budget for this instance.

    A BaseException (like robocode's EpisodeTimeout) so that ``except Exception`` inside a
    policy cannot swallow it."""


class PolicyClock:
    """Charges time spent inside policy calls against a per-instance budget.

    On the main thread a running call is interrupted with SIGALRM when the budget runs out,
    and the alarm repeats every second until the call returns, so a policy that catches the
    timeout cannot keep running (robocode's episode clock does the same).  Elsewhere the
    budget is checked after each call.  Use as::

        clock = PolicyClock(60.0)
        with clock.installed():
            with clock.charge():
                policy.get_action(state)
    """

    REPEAT_S = 1.0

    def __init__(self, budget_s: float) -> None:
        self.budget_s = float(budget_s)
        self.used_s = 0.0
        self._armed = False
        self._use_alarm = False
        self._previous = None

    @property
    def remaining_s(self) -> float:
        return self.budget_s - self.used_s

    def _timeout(self) -> PolicyTimeout:
        return PolicyTimeout(f"policy time budget of {self.budget_s:g}s exhausted")

    def _fire(self, _signum, _frame) -> None:
        if self._armed:
            raise self._timeout()

    @contextlib.contextmanager
    def installed(self):
        """Install the SIGALRM handler for one rollout; always disarms and restores on exit."""
        self._use_alarm = (np.isfinite(self.budget_s) and hasattr(signal, "setitimer")
                           and threading.current_thread() is threading.main_thread())
        if self._use_alarm:
            self._previous = signal.signal(signal.SIGALRM, self._fire)
        try:
            yield self
        finally:
            self._armed = False
            if self._use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, self._previous)

    @contextlib.contextmanager
    def charge(self):
        if self.remaining_s <= 0:
            raise self._timeout()
        if self._use_alarm:
            self._armed = True
            signal.setitimer(signal.ITIMER_REAL, self.remaining_s, self.REPEAT_S)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._armed = False
            if self._use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
            self.used_s += time.perf_counter() - t0
        if self.remaining_s < 0:
            raise self._timeout()


@dataclass
class Rollout:
    actions: List[np.ndarray] = field(default_factory=list)
    solved: bool = False
    num_steps: int = 0
    max_steps: int = 0
    total_reward: float = 0.0
    policy_time_s: float = 0.0
    env_time_s: float = 0.0
    error: Optional[str] = None       # "kind: message" for crashes, invalid actions, timeouts
    error_kind: Optional[str] = None  # init_error | policy_error | invalid_action | timeout

    def summary(self) -> dict:
        return {"solved": self.solved, "num_steps": self.num_steps, "max_steps": self.max_steps,
                "total_reward": self.total_reward, "policy_time_s": round(self.policy_time_s, 3),
                "env_time_s": round(self.env_time_s, 3), "error": self.error, "error_kind": self.error_kind}


def run_policy(approach_cls, family: FamilySpec | str, count: int, seed: int,
               policy_budget_s: float = 60.0, max_steps: Optional[int] = None) -> Rollout:
    """Roll out a ``GeneratedApproach`` class on one instance.

    Mirrors AgenticGenTAMP's evaluation episode: the approach is constructed with the
    spaces and an empty primitives dict, ``reset(state, info)`` once, then
    ``get_action(state)`` until the episode terminates (solved) or the step budget is
    spent.  Time inside ``reset``/``get_action`` is charged to ``policy_budget_s``.
    """
    inst = Instance(family, count, seed)
    out = Rollout(max_steps=inst.max_steps if max_steps is None else int(max_steps))
    clock = PolicyClock(policy_budget_s)
    try:
        try:
            approach = approach_cls(inst.action_space, inst.observation_space, {})
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            out.error_kind, out.error = "init_error", _describe(exc)
            return out
        state, info = inst.reset()
        with clock.installed():
            try:
                with clock.charge():
                    approach.reset(state, info)
                for _ in range(out.max_steps):
                    with clock.charge():
                        action = approach.get_action(state)
                    action = C.validate_action(action, inst.action_space)
                    t0 = time.perf_counter()
                    state, reward, terminated, truncated, info = inst.step(action)
                    out.env_time_s += time.perf_counter() - t0
                    out.actions.append(action)
                    out.num_steps += 1
                    out.total_reward += reward
                    if terminated or truncated:
                        out.solved = bool(terminated)
                        break
            except PolicyTimeout as exc:
                out.error_kind, out.error = "timeout", str(exc)
            except C.InvalidAction as exc:
                out.error_kind, out.error = "invalid_action", f"step {out.num_steps}: {exc}"
            except (Exception, SystemExit) as exc:  # noqa: BLE001 - a crashing policy is a failed episode
                out.error_kind, out.error = "policy_error", _describe(exc)
    finally:
        out.policy_time_s = clock.used_s
        inst.close()
    return out


def _describe(exc: BaseException) -> str:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__, limit=-6)
    return "".join(tb)[-3000:]


def replay(family: FamilySpec | str, count: int, seed: int, actions: Any,
           max_steps: Optional[int] = None) -> dict:
    """Exact verification: replay an action sequence from the instance's reset.

    Solved iff the episode terminates within the first ``max_steps`` actions.  Actions
    after termination are ignored; an invalid action ends the replay unsolved.
    """
    fam = get_family(family) if isinstance(family, str) else family
    budget = fam.max_steps(count) if max_steps is None else int(max_steps)
    acts = decode_actions(actions)
    inst = Instance(fam, count, seed)
    try:
        inst.reset()
        for i, a in enumerate(acts[:budget]):
            try:
                _, _, terminated, _, _ = inst.step(a)
            except C.InvalidAction as exc:
                return {"solved": False, "num_steps": i, "message": f"invalid action at step {i}: {exc}"}
            if terminated:
                return {"solved": True, "num_steps": i + 1, "message": f"goal reached after {i + 1} steps"}
        n = min(len(acts), budget)
        why = "step budget exhausted" if len(acts) >= budget else "actions ended"
        return {"solved": False, "num_steps": n, "message": f"goal not reached after {n} steps ({why})"}
    finally:
        inst.close()


# ------------------------------------------------------------------ environment card
def env_card(family: FamilySpec | str) -> str:
    """Markdown description of a family for agents (the strict black-box card of
    AgenticGenTAMP's variable-count environments: count-invariant prose from KinDER's
    metadata, the object-centric observation, the action space and the reward)."""
    fam = get_family(family) if isinstance(family, str) else family
    metas = {}
    ref_state = ref_types = None
    for count in fam.all_counts:
        wrapper = build_wrapper(fam, count)
        metas[count] = dict(wrapper.metadata)
        if count == max(fam.design_counts):
            inner = wrapper._object_centric_env  # noqa: SLF001
            ref_state, _ = inner.reset(seed=0)
            ref_types = inner.type_features
        with contextlib.suppress(Exception):
            wrapper.close()

    def invariant(key: str) -> str:
        per = [metas[c][key].split("\n\n") for c in fam.all_counts]
        first, *others = per
        return "\n\n".join(p for p in first if all(p in o for o in others)).strip("\n")

    prefix = fam.count_object_prefix
    by_type: Dict[str, List[str]] = {}
    for name in sorted(ref_state.get_object_names()):
        obj = ref_state.get_object_from_name(name)
        by_type.setdefault(obj.type.name, []).append(name)
    feats = {t.name: list(f) for t, f in ref_types.items()}
    rows = []
    for tname in sorted(by_type):
        names = [n for n in by_type[tname] if not n.startswith(prefix)]
        if len(names) < len(by_type[tname]):
            names += [f"{prefix}0", f"{prefix}1", "..."]
        rows.append(f"| {tname} | {', '.join(feats[tname])} | {', '.join(names)} |")
    parents = {t.name: t.parent.name for t in ref_types if t.parent is not None}
    hierarchy = ", ".join(f"`{c}` is a `{p}`" for c, p in sorted(parents.items()))
    budget = (f"{fam.base_steps} + {fam.steps_per_object} x count steps "
              f"(count = number of {fam.count_noun})")
    return "\n".join([
        f"# {fam.title} (variable object count)",
        "",
        invariant("description"),
        "",
        "## Generalization",
        "",
        "This environment contains a VARIABLE number of objects. A program must handle ANY number of "
        "them, in principle unbounded. Do not assume a fixed object count, a fixed number of objects of "
        "any type, or any fixed index layout; iterate the objects in the state and act on whatever is there.",
        "",
        f"The count of an instance is the number of {fam.count_noun}; it is given as `info['object_count']` "
        f"at reset. Training instances use counts {list(fam.design_counts)}; evaluation instances use "
        f"counts {list(fam.eval_counts)}.",
        "",
        "## Observation",
        "",
        "Each observation is an `ObjectCentricState`: a set of typed objects, each with named features. "
        "The number of objects VARIES between episodes, so there is no fixed-length vector and no fixed "
        "index layout. Read it with:",
        "",
        "- `state.get_objects(type)` / `state.get_object_names()` / `state.get_object_from_name(name)` "
        "to enumerate objects (`type = observation_space.get_type(\"rectangle\")`),",
        "- `state.get(obj, feature)` to read a feature; `obj.name`, `obj.type.name`, `obj.is_instance(type)`.",
        "",
        "| **Type** | **Features** | **Example objects** |",
        "| --- | --- | --- |",
        *rows,
        "",
        f"Type hierarchy: {hierarchy}." if hierarchy else "",
        f"The count-defining objects are named `{prefix}0`, `{prefix}1`, ... and their number changes per "
        "episode.",
        "",
        "## Action Space",
        "",
        invariant("action_space_description"),
        "",
        "## Reward",
        "",
        invariant("reward_description"),
        "",
        "## Episode",
        "",
        f"An episode succeeds when the environment terminates (the goal above holds) within the step "
        f"budget of {budget}. Coordinates are in world units; angles are in radians.",
        "",
    ])
