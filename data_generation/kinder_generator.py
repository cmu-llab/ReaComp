"""Generate and verify the KinDER task files (data/kinder/).

A task is one instance of a KinDER environment family: ``(count, seed)`` plus its step
budget.  Two splits per family:

* ``train`` (100): instances for per-task agent traces (demos) and for the solver
  builder's own testing; counts cycle through the family's ``design_counts``.
* ``test`` (100): the evaluation suite; counts cycle through ``eval_counts``, so it holds
  both the design regime and the extrapolation (hard) tier.

The protocol is AgenticGenTAMP's (tomsilver/robocode): instance seeds are
``np.random.default_rng(master).integers(0, 2**63)`` (robocode ``generate_eval_seeds``),
counts are assigned round-robin, an instance is the family's ``ConstantObjectKinDEREnv``
built with the count and reset with the seed (robocode ``VariableObjectCountEnv``), and
the step budget is ``300 + 150 * count``.  Building ``test`` with robocode's private
evaluation seed (``--test-seed``) reproduces robocode's evaluation instances.

Files: ``data/kinder/<family>_{train,test}.jsonl``, ``data/kinder/<family>_env.md`` (the
environment description given to agents) and ``data/kinder/manifest.json``.  Each
record stores a SHA-256 of its initial state.

    python -m data_generation.kinder_generator build
    python -m data_generation.kinder_generator check      # stored instances still reproduce
    python -m data_generation.kinder_generator verify [--robocode-src <robocode checkout>]

``verify`` checks the generated instances against the original benchmark code:
  1. KinDER: for every instance whose count is a registered KinDER variant
     (e.g. ``kinder/Obstruction2D-o2-v0``), the registered environment and ours agree on
     the action space, the initial state and the states and terminations along a random
     action sequence.
  2. AgenticGenTAMP (with ``--robocode-src``): the family configs match robocode's
     ``experiments/conf/environment/*_generalized.yaml``, robocode's own
     ``generate_eval_seeds`` yields our seeds, and robocode's ``VariableObjectCountEnv``
     regenerates every instance (same initial state, same step budget, same transitions).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from domains.kinder.families import FAMILIES, FamilySpec, get_family  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data", "kinder")
SPLITS = ("train", "test")
TRAIN_MASTER_SEED = 20261105
TEST_MASTER_SEED = 20261106
DEFAULT_SIZES = {"train": 100, "test": 100}


def generate_seeds(master_seed: int, n: int) -> List[int]:
    """AgenticGenTAMP's evaluation-suite seeds (robocode ``generate_eval_seeds``)."""
    rng = np.random.default_rng(master_seed)
    return [int(rng.integers(0, 2 ** 63)) for _ in range(n)]


def state_digest(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def split_path(family: str, split: str, root: str = DATA_DIR) -> str:
    return os.path.join(root, f"{family}_{split}.jsonl")


def card_path(family: str, root: str = DATA_DIR) -> str:
    return os.path.join(root, f"{family}_env.md")


def load_split(family: str, split: str, root: str = DATA_DIR) -> List[dict]:
    with open(split_path(family, split, root)) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------------------ build
def build_split(family: FamilySpec, split: str, n: int, master_seed: int) -> List[dict]:
    from domains.kinder.sim import initial_state_payload
    pool = family.design_counts if split == "train" else family.eval_counts
    records = []
    for i, seed in enumerate(generate_seeds(master_seed, n)):
        count = pool[i % len(pool)]
        records.append({
            "task_id": f"{family.name}-{split}-{i:03d}",
            "family": family.name,
            "split": split,
            "count": count,
            "regime": family.regime(count),
            "seed": seed,
            "max_steps": family.max_steps(count),
            "initial_state_sha256": state_digest(initial_state_payload(family, count, seed)),
        })
    return records


def build(families: Sequence[str], out: str, sizes: Dict[str, int], seeds: Dict[str, int]) -> None:
    from importlib.metadata import version
    from domains.kinder.sim import env_card
    os.makedirs(out, exist_ok=True)
    manifest_path = os.path.join(out, "manifest.json")
    manifest = {"kindergarden": version("kindergarden"), "master_seeds": seeds, "sizes": sizes, "families": {}}
    if os.path.exists(manifest_path):  # keep families not rebuilt now if the protocol is unchanged
        with open(manifest_path) as fh:
            old = json.load(fh)
        if old.get("master_seeds") == seeds and old.get("sizes") == sizes:
            manifest["families"] = old.get("families", {})
    for name in families:
        fam = get_family(name)
        owner: Dict[int, str] = {}
        for split, n in sizes.items():
            rows = build_split(fam, split, n, seeds[split])
            for r in rows:
                if owner.setdefault(r["seed"], split) != split:
                    raise ValueError(f"{name}: seed {r['seed']} appears in two splits")
            with open(split_path(name, split, out), "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
        with open(card_path(name, out), "w") as fh:
            fh.write(env_card(fam))
        manifest["families"][name] = {
            "env_class": fam.env_class, "count_kwarg": fam.count_kwarg,
            "design_counts": list(fam.design_counts), "eval_counts": list(fam.eval_counts),
            "max_steps": {str(c): fam.max_steps(c) for c in fam.all_counts}}
        print(f"  {name}: " + ", ".join(f"{s}={n}" for s, n in sizes.items()))
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)


def check(families: Sequence[str], root: str = DATA_DIR, limit: Optional[int] = None) -> List[str]:
    """Task ids whose stored initial state no longer reproduces (KinDER version drift)."""
    from domains.kinder.sim import initial_state_payload
    bad = []
    for name in families:
        for split in SPLITS:
            for r in load_split(name, split, root)[:limit]:
                if state_digest(initial_state_payload(name, r["count"], r["seed"])) != r["initial_state_sha256"]:
                    bad.append(r["task_id"])
    return bad


# ----------------------------------------------------------------------------- verify
def _objects(state) -> Dict[str, tuple]:
    """{object name: (type name, float32 features)} of a relational_structs or mirror state."""
    return {o.name: (o.type.name, np.asarray(state.data[o], dtype=np.float32)) for o in state}


def _same(a, b) -> bool:
    """Same objects, same types, bit-identical features."""
    oa, ob = _objects(a), _objects(b)
    return oa.keys() == ob.keys() and all(oa[k][0] == ob[k][0] and np.array_equal(oa[k][1], ob[k][1]) for k in oa)


def _random_actions(space_low, space_high, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(space_low, space_high, size=(n, len(space_low))).astype(np.float32)


def _registered_id(fam: FamilySpec, count: int) -> Optional[str]:
    import kinder
    kinder.register_all_environments()
    variants = kinder.get_env_classes().get(fam.title, {}).get("variants", [])
    prefix = f"kinder/{fam.title}-"
    for vid in variants:
        tag = vid[len(prefix):].split("-v")[0]
        digits = "".join(ch for ch in tag if ch.isdigit())
        if digits and int(digits) == count:
            return vid
    return None


def verify_registered(families: Sequence[str], root: str, steps: int, log: Callable[[str], None]) -> List[str]:
    """Compare our instances with KinDER's registered environments (``kinder.make``)."""
    import kinder
    from domains.kinder.sim import Instance
    failures: List[str] = []
    for name in families:
        fam = get_family(name)
        checked, skipped = 0, set()
        for split in SPLITS:
            for r in load_split(name, split, root):
                vid = _registered_id(fam, r["count"])
                if vid is None:
                    skipped.add(r["count"])
                    continue
                reg = kinder.make(vid).unwrapped
                ours = Instance(fam, r["count"], r["seed"])
                try:
                    if not (np.array_equal(reg.action_space.low, ours.action_space.low)
                            and np.array_equal(reg.action_space.high, ours.action_space.high)):
                        failures.append(f"{r['task_id']}: action space differs from {vid}")
                        continue
                    vec, _ = reg.reset(seed=r["seed"])
                    mine, _ = ours.reset()
                    if not _same(reg.observation_space.devectorize(vec), mine):
                        failures.append(f"{r['task_id']}: initial state differs from {vid}")
                        continue
                    for t, a in enumerate(_random_actions(ours.action_space.low, ours.action_space.high, steps,
                                                          r["seed"] % (2 ** 32))):
                        vec, _, term_r, _, _ = reg.step(a)
                        mine, _, term_o, _, _ = ours.step(a)
                        if term_r != term_o or not _same(reg.observation_space.devectorize(vec), mine):
                            failures.append(f"{r['task_id']}: transition {t} differs from {vid}")
                            break
                    checked += 1
                finally:
                    ours.close()
                    reg.close()
        unregistered = sorted(skipped)
        log(f"  kinder  {name:22s} {checked} instances identical to the registered variants"
            + (f"; counts {unregistered} have no registered variant (built from the class, as robocode does)"
               if unregistered else ""))
    return failures


def _robocode_function(path: str, name: str):
    """Load one top-level function from a robocode source file without importing the file."""
    with open(path) as fh:
        tree = ast.parse(fh.read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)  # noqa: S102
    return namespace[name]


def verify_robocode(families: Sequence[str], root: str, robocode_src: str, steps: int,
                    log: Callable[[str], None]) -> List[str]:
    """Regenerate every instance with AgenticGenTAMP's own code and compare."""
    import yaml
    from domains.kinder.sim import Instance
    src = os.path.join(robocode_src, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from robocode.environments.variable_object_count_env import VariableObjectCountEnv
    gen_seeds = _robocode_function(os.path.join(robocode_src, "experiments", "run_experiment.py"),
                                   "generate_eval_seeds")
    with open(os.path.join(root, "manifest.json")) as fh:
        master = json.load(fh)["master_seeds"]
    with open(os.path.join(robocode_src, "experiments", "run_experiment.py")) as fh:
        if "count_pool[i % len(count_pool)]" not in fh.read():
            return ["robocode no longer assigns evaluation counts round-robin (count_pool[i % len(count_pool)])"]
    failures: List[str] = []
    for name in families:
        fam = get_family(name)
        with open(os.path.join(robocode_src, "experiments", "conf", "environment", f"{name}_generalized.yaml")) as fh:
            cfg = yaml.safe_load(fh)
        expect = {"constant_object_env_path": fam.env_class, "count_kwarg": fam.count_kwarg,
                  "count_object_prefix": fam.count_object_prefix,
                  "design_counts": list(fam.design_counts), "eval_counts": list(fam.eval_counts)}
        diff = {k: (cfg.get(k), v) for k, v in expect.items() if cfg.get(k) != v}
        if diff:
            failures.append(f"{name}: config differs from robocode (robocode, ours): {diff}")
            continue
        env = VariableObjectCountEnv(**{k: v for k, v in cfg.items() if not k.startswith("_")})
        n_ok = 0
        try:
            for split in SPLITS:
                rows = load_split(name, split, root)
                pool = fam.design_counts if split == "train" else fam.eval_counts
                if [r["seed"] for r in rows] != gen_seeds(master[split], len(rows)):
                    failures.append(f"{name}/{split}: seeds differ from robocode generate_eval_seeds")
                if [r["count"] for r in rows] != [pool[i % len(pool)] for i in range(len(rows))]:
                    failures.append(f"{name}/{split}: count schedule differs from robocode")
                for r in rows:
                    if env.max_steps_for_count(r["count"]) != r["max_steps"]:
                        failures.append(f"{r['task_id']}: step budget {r['max_steps']} != robocode "
                                        f"{env.max_steps_for_count(r['count'])}")
                        continue
                    state, info = env.reset(seed=r["seed"], options={"object_count": r["count"]})
                    ours = Instance(fam, r["count"], r["seed"])
                    try:
                        mine, _ = ours.reset()
                        if info.get("object_count") != r["count"] or not _same(state, mine):
                            failures.append(f"{r['task_id']}: robocode regenerates a different instance")
                            continue
                        for t, a in enumerate(_random_actions(ours.action_space.low, ours.action_space.high,
                                                              steps, r["seed"] % (2 ** 32))):
                            state, _, term_r, _, _ = env.step(a)
                            mine, _, term_o, _, _ = ours.step(a)
                            if term_r != term_o or not _same(state, mine):
                                failures.append(f"{r['task_id']}: transition {t} differs from robocode")
                                break
                        else:
                            n_ok += 1
                    finally:
                        ours.close()
        finally:
            env.close()
        log(f"  robocode {name:22s} {n_ok} instances regenerated identically by VariableObjectCountEnv")
    return failures


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write data/kinder/")
    b.add_argument("--families", nargs="+", default=list(FAMILIES))
    b.add_argument("--out", default=DATA_DIR)
    b.add_argument("--train-n", type=int, default=DEFAULT_SIZES["train"])
    b.add_argument("--test-n", type=int, default=DEFAULT_SIZES["test"])
    b.add_argument("--train-seed", type=int, default=TRAIN_MASTER_SEED)
    b.add_argument("--test-seed", type=int, default=TEST_MASTER_SEED,
                   help="master seed of the test suite (robocode's eval_seed reproduces its evaluation instances)")
    c = sub.add_parser("check", help="stored instances still reproduce with the installed KinDER")
    c.add_argument("--families", nargs="+", default=list(FAMILIES))
    c.add_argument("--root", default=DATA_DIR)
    c.add_argument("--limit", type=int, default=None, help="instances per split")
    v = sub.add_parser("verify", help="compare the instances with KinDER and AgenticGenTAMP")
    v.add_argument("--families", nargs="+", default=list(FAMILIES))
    v.add_argument("--root", default=DATA_DIR)
    v.add_argument("--steps", type=int, default=50, help="random transitions compared per instance")
    v.add_argument("--robocode-src", default=None, help="checkout of github.com/tomsilver/robocode")
    args = ap.parse_args()

    if args.cmd == "build":
        build(args.families, args.out, {"train": args.train_n, "test": args.test_n},
              {"train": args.train_seed, "test": args.test_seed})
    elif args.cmd == "check":
        bad = check(args.families, args.root, args.limit)
        print("all instances reproduce" if not bad else f"{len(bad)} instances changed: {bad[:10]}")
        sys.exit(1 if bad else 0)
    else:
        failures = [f"{t}: stored initial state does not reproduce" for t in check(args.families, args.root)]
        print(f"  stored initial-state digests: {'all reproduce' if not failures else 'MISMATCH'}")
        failures += verify_registered(args.families, args.root, args.steps, print)
        if args.robocode_src:
            failures += verify_robocode(args.families, args.root, args.robocode_src, args.steps, print)
        print("verified" if not failures else f"{len(failures)} mismatches:\n  " + "\n  ".join(failures[:20]))
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
