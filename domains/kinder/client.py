"""Pure-Python view of a KinDER environment (standard library + NumPy only).

This file is the single definition of what a KinDER solver or agent sees:

* ``ObjectCentricState`` / ``Object`` / ``Type`` and the observation/action spaces:
  local mirrors of ``relational_structs`` with the API subset of AgenticGenTAMP's
  strict black-box client (tomsilver/robocode ``utils/env_client.py``), so programs
  written against either run unchanged on the other.  The evaluation harness builds
  these mirrors from the simulator state, so a solver sees exactly the same objects
  during development (through the client below) and at evaluation.
* ``validate_action``: the action contract shared by the client, the environment
  server and the evaluator.
* ``KinderEnv`` / ``make_env``: a client for the environment server
  (``domains.kinder.server``) used from sandboxed code that cannot import KinDER.
* ``load_approach``, ``rollout``, ``evaluate``, ``format_state``: helpers for testing
  ``GeneratedApproach`` policies.

The file is copied into sandbox working directories as ``kinder_client.py`` and must
stay importable without the rest of the repository.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

SOCKET_ENV_VAR = "KINDER_ENV_SOCKET"
DEFAULT_SOCKET_NAME = "kinder_env.sock"


# --------------------------------------------------------------------------- mirrors
class Type:
    """Mirror of ``relational_structs.Type``: a name and an optional parent type."""

    def __init__(self, name: str, parent: Optional["Type"] = None) -> None:
        self.name = name
        self.parent = parent

    def get_ancestors(self) -> set:
        """This type and all of its ancestors."""
        out, cur = set(), self
        while cur is not None:
            out.add(cur)
            cur = cur.parent
        return out

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Type) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"Type(name={self.name!r})"


class Object:
    """Mirror of ``relational_structs.Object``: a name and a type."""

    def __init__(self, name: str, typ: Type) -> None:
        self.name = name
        self.type = typ

    def is_instance(self, typ: Type) -> bool:
        """True if this object's type is ``typ`` or a descendant of it."""
        return typ in self.type.get_ancestors()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Object) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"{self.name}:{self.type.name}"


class ObjectCentricState:
    """Mirror of ``relational_structs.ObjectCentricState``.

    ``data`` maps each ``Object`` to its float32 feature vector and ``type_features``
    maps each ``Type`` to its feature names.  Iteration yields objects sorted by name.
    """

    def __init__(self, data: Dict[Object, np.ndarray], type_features: Dict[Type, List[str]]) -> None:
        self.data = data
        self.type_features = type_features

    def __iter__(self):
        return iter(sorted(self.data, key=lambda o: o.name))

    def __len__(self) -> int:
        return len(self.data)

    def get_object_names(self) -> set:
        return {o.name for o in self.data}

    def get_object_from_name(self, name: str) -> Object:
        for obj in self.data:
            if obj.name == name:
                return obj
        raise ValueError(f"No object named {name!r}")

    def get_objects(self, typ: Type) -> List[Object]:
        """Objects whose type is ``typ`` or a subtype of it, sorted by name."""
        return sorted((o for o in self.data if o.is_instance(typ)), key=lambda o: o.name)

    def get(self, obj: Object, feature: str) -> float:
        return float(self.data[obj][self.type_features[obj.type].index(feature)])

    def set(self, obj: Object, feature: str, value: float) -> None:
        self.data[obj][self.type_features[obj.type].index(feature)] = value

    def vec(self, objects: Iterable[Object]) -> np.ndarray:
        objects = list(objects)
        return np.concatenate([self.data[o] for o in objects]) if objects else np.array([], dtype=np.float32)

    def copy(self) -> "ObjectCentricState":
        return ObjectCentricState({o: a.copy() for o, a in self.data.items()},
                                  {t: list(f) for t, f in self.type_features.items()})

    def allclose(self, other: "ObjectCentricState") -> bool:
        if self.get_object_names() != other.get_object_names():
            return False
        return all(np.allclose(self.data[o], other.data[other.get_object_from_name(o.name)]) for o in self.data)

    def __repr__(self) -> str:
        return f"ObjectCentricState({len(self.data)} objects: {', '.join(sorted(self.get_object_names()))})"


class ObservationSpace:
    """The object-centric observation space: the domain's types and their features."""

    def __init__(self, type_features: Dict[Type, List[str]]) -> None:
        self._type_features = type_features
        self._by_name = {t.name: t for t in type_features}

    @property
    def types(self) -> set:
        return set(self._type_features)

    @property
    def type_features(self) -> Dict[Type, List[str]]:
        return {t: list(f) for t, f in self._type_features.items()}

    def get_type(self, name: str) -> Type:
        return self._by_name[name]


class ActionSpace:
    """A Box action space: ``low``/``high`` bounds of a float32 vector."""

    def __init__(self, low, high, names: Optional[List[str]] = None) -> None:
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape
        self.dtype = np.dtype(np.float32)
        self.names = list(names) if names else [f"a{i}" for i in range(self.shape[0])]

    def sample(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(np.float32)

    def contains(self, action: Any) -> bool:
        try:
            validate_action(action, self)
            return True
        except InvalidAction:
            return False

    def __repr__(self) -> str:
        return f"Box({self.low.tolist()}, {self.high.tolist()}, {self.shape}, float32)"


class InvalidAction(ValueError):
    """An action outside the action-space contract."""


def validate_action(action: Any, space: ActionSpace) -> np.ndarray:
    """Return ``action`` as a float32 vector or raise ``InvalidAction``.

    The contract is the simulator's: a finite vector of the space's shape whose
    entries, cast to float32, lie within ``[low, high]``.  Nothing is clipped; use
    ``np.clip(action, space.low, space.high)`` in the policy.
    """
    try:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise InvalidAction(f"action is not a numeric vector: {action!r}") from exc
    if arr.shape != space.shape:
        raise InvalidAction(f"action has shape {arr.shape}, expected {space.shape}")
    if not np.all(np.isfinite(arr)):
        raise InvalidAction(f"action has non-finite entries: {arr.tolist()}")
    a32 = arr.astype(np.float32)
    bad = np.flatnonzero((a32 < space.low) | (a32 > space.high))
    if bad.size:
        detail = ", ".join(f"{space.names[i]}={a32[i]:.6g} not in [{space.low[i]:.6g}, {space.high[i]:.6g}]" for i in bad)
        raise InvalidAction(f"action out of bounds: {detail}")
    return a32


# ----------------------------------------------------------------------------- codec
def types_from_payload(types_payload: List[dict]) -> Tuple[Dict[str, Type], Dict[Type, List[str]]]:
    """Rebuild types from a parents-first list of {name, parent, features}."""
    by_name: Dict[str, Type] = {}
    features: Dict[Type, List[str]] = {}
    for entry in types_payload:
        parent = by_name[entry["parent"]] if entry.get("parent") is not None else None
        typ = Type(entry["name"], parent)
        by_name[entry["name"]] = typ
        features[typ] = list(entry["features"])
    return by_name, features


def state_from_payload(payload: dict) -> ObjectCentricState:
    by_name, features = types_from_payload(payload["types"])
    data = {Object(e["name"], by_name[e["type"]]): np.asarray(e["features"], dtype=np.float32)
            for e in payload["objects"]}
    return ObjectCentricState(data, features)


def state_to_payload(state: ObjectCentricState) -> dict:
    def depth(t: Type) -> int:
        n, p = 0, t.parent
        while p is not None:
            n, p = n + 1, p.parent
        return n

    types = sorted(state.type_features, key=lambda t: (depth(t), t.name))
    return {
        "types": [{"name": t.name, "parent": t.parent.name if t.parent is not None else None,
                   "features": list(state.type_features[t])} for t in types],
        "objects": [{"name": o.name, "type": o.type.name, "features": [float(v) for v in state.data[o]]}
                    for o in sorted(state.data, key=lambda o: o.name)],
    }


def spaces_from_payload(payload: dict) -> Tuple[ObservationSpace, ActionSpace]:
    _, features = types_from_payload(payload["types"])
    act = payload["action"]
    return ObservationSpace(features), ActionSpace(act["low"], act["high"], act.get("names"))


def format_state(state: ObjectCentricState, digits: int = 4) -> str:
    """A readable table of every object and feature (robot first, then by name)."""
    objs = sorted(state, key=lambda o: (o.type.name != "crv_robot", o.name))
    lines = []
    for o in objs:
        feats = ", ".join(f"{f}={state.get(o, f):.{digits}g}" for f in state.type_features[o.type])
        lines.append(f"{o.name} ({o.type.name}): {feats}")
    return "\n".join(lines)


# ------------------------------------------------------------------------- env client
class KinderEnv:
    """Client for ``domains.kinder.server``: a gym-like view of the served instances.

    ``reset()`` starts the instance this session is pinned to (instance 0);
    ``reset(instance=i)`` picks one of ``num_instances`` pinned instances;
    ``reset(seed=s, count=k)`` samples a fresh instance when the server allows it
    (``free_reset``; counts in ``counts``).  Seeds of pinned instances are never revealed.
    """

    def __init__(self, socket_path: str) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)
        self._file = self._sock.makefile("rwb")
        hello = self._request({"cmd": "hello"})
        self.observation_space, self.action_space = spaces_from_payload(hello["spaces"])
        self.family: str = hello["family"]
        self.num_instances: int = hello["num_instances"]
        self.free_reset: bool = hello["free_reset"]
        self.counts: List[int] = hello["counts"]
        self.max_steps: Optional[int] = None     # step budget of the current instance
        self.count: Optional[int] = None         # object count of the current instance

    def _request(self, payload: dict) -> dict:
        self._file.write((json.dumps(payload) + "\n").encode())
        self._file.flush()
        line = self._file.readline()
        if not line:
            raise ConnectionError("environment server closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            err = reply.get("error", "unknown error")
            if err.startswith("InvalidAction:"):
                raise InvalidAction(err.split(":", 1)[1].strip())
            raise RuntimeError(err)
        return reply

    def reset(self, instance: Optional[int] = None, seed: Optional[int] = None,
              count: Optional[int] = None) -> Tuple[ObjectCentricState, dict]:
        req: Dict[str, Any] = {"cmd": "reset"}
        if seed is not None or count is not None:
            req.update(seed=seed, count=count)
        else:
            req["instance"] = 0 if instance is None else int(instance)
        r = self._request(req)
        self.max_steps, self.count = r["max_steps"], r["count"]
        return state_from_payload(r["state"]), r["info"]

    def step(self, action: Any) -> Tuple[ObjectCentricState, float, bool, bool, dict]:
        a = validate_action(action, self.action_space)
        r = self._request({"cmd": "step", "action": [float(x) for x in a]})
        return state_from_payload(r["state"]), r["reward"], r["terminated"], r["truncated"], r["info"]

    def render(self, path: str = "render.png") -> str:
        """Save an image of the current state (PNG) and return its path."""
        r = self._request({"cmd": "render"})
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(r["png_b64"]))
        return path

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            self._sock.close()

    def __enter__(self) -> "KinderEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_env(socket_path: Optional[str] = None) -> KinderEnv:
    """Connect to the environment server of this sandbox."""
    path = socket_path or os.environ.get(SOCKET_ENV_VAR) or DEFAULT_SOCKET_NAME
    if not os.path.exists(path):
        raise FileNotFoundError(f"no KinDER environment server at {path!r} (set ${SOCKET_ENV_VAR})")
    return KinderEnv(path)


# ---------------------------------------------------------------------- policy helpers
def load_approach(path: str):
    """Load the ``GeneratedApproach`` class from a policy file.

    The file is executed as a real module registered in ``sys.modules`` (so dataclasses,
    pickling and ``from __future__ import annotations`` work), and its directory is put on
    ``sys.path`` so sibling helper modules import.
    """
    import hashlib
    import importlib.util
    path = os.path.abspath(path)
    folder = os.path.dirname(path)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    name = "generated_approach_" + hashlib.sha1(path.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if not hasattr(module, "GeneratedApproach"):
        raise AttributeError(f"{path} does not define GeneratedApproach")
    return module.GeneratedApproach


def rollout(approach, env: KinderEnv, instance: Optional[int] = None, seed: Optional[int] = None,
            count: Optional[int] = None, max_steps: Optional[int] = None, log_every: int = 0) -> dict:
    """Run one episode of a policy on the served environment.

    ``approach`` is a ``GeneratedApproach`` class (instantiated with the env's spaces)
    or an instance.  Returns ``solved``, ``num_steps``, ``max_steps``, ``count``,
    ``total_reward``, ``error`` (traceback of a policy or action failure),
    ``final_state`` and ``policy_time_s``.
    """
    if isinstance(approach, type):
        approach = approach(env.action_space, env.observation_space, {})
    state, info = env.reset(instance=instance, seed=seed, count=count)
    budget = env.max_steps if max_steps is None else int(max_steps)
    total, steps, solved, error, policy_s = 0.0, 0, False, None, 0.0
    try:
        t0 = time.perf_counter()
        approach.reset(state, info)
        policy_s += time.perf_counter() - t0
        for _ in range(budget):
            t0 = time.perf_counter()
            action = approach.get_action(state)
            policy_s += time.perf_counter() - t0
            state, reward, terminated, truncated, info = env.step(action)
            steps += 1
            total += float(reward)
            if log_every and steps % log_every == 0:
                print(f"[step {steps}]\n{format_state(state)}")
            if terminated or truncated:
                solved = bool(terminated)
                break
    except Exception:  # noqa: BLE001 - report policy failures to the caller
        error = traceback.format_exc(limit=8)
    return {"solved": solved, "num_steps": steps, "max_steps": budget, "count": env.count,
            "total_reward": total, "error": error, "final_state": state, "policy_time_s": round(policy_s, 3)}


def evaluate(approach_cls, env: KinderEnv, instances: Optional[Iterable[int]] = None,
             limit: Optional[int] = None, verbose: bool = True) -> List[dict]:
    """Roll out ``approach_cls`` on pinned instances (default: all) and summarise by count."""
    idx = list(range(env.num_instances)) if instances is None else list(instances)
    if limit is not None:
        idx = idx[:limit]
    results = []
    for i in idx:
        r = rollout(approach_cls, env, instance=i)
        r = {k: v for k, v in r.items() if k != "final_state"}
        r["instance"] = i
        results.append(r)
        if verbose:
            status = "solved" if r["solved"] else ("ERROR" if r["error"] else "unsolved")
            print(f"instance {i:3d} count={r['count']} {status} steps={r['num_steps']}/{r['max_steps']}"
                  + (f"\n{r['error'].strip().splitlines()[-1]}" if r["error"] else ""))
    if verbose and results:
        by: Dict[Any, List[bool]] = {}
        for r in results:
            by.setdefault(r["count"], []).append(r["solved"])
        summary = ", ".join(f"count {c}: {sum(v)}/{len(v)}" for c, v in sorted(by.items()))
        print(f"solved {sum(r['solved'] for r in results)}/{len(results)} ({summary})")
    return results
