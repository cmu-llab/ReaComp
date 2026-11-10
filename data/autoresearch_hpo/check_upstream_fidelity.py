"""Check rewards/autoresearch_hpo.py against the pinned upstream benchmark sources.

Fetches three files at the commits pinned in ``UPSTREAM`` and compares them with the
constants this repository uses:

  autoresearch-automl  core/search_space.py  KNOWN_HP_METADATA (names, order, types,
                                             bounds, log flags, choices),
                                             POWER_OF_2_HPS, KARPATHY_STARTING_CONFIG
  autoresearch         train.py              sha256 pin; every hyperparameter is a
                                             module-level assignment whose default is
                                             the starting config; injection round-trips
  autoresearch         prepare.py            sha256 pin; TIME_BUDGET, MAX_SEQ_LEN

and checks that the task files under data/autoresearch_hpo/ are what the generator
produces today. Upstream files are parsed with ``ast`` only; nothing is executed.

Usage (from the project root):
    python data/autoresearch_hpo/check_upstream_fidelity.py
    python data/autoresearch_hpo/check_upstream_fidelity.py --offline-dir <dir>

``--offline-dir`` reads search_space.py / train.py / prepare.py from a directory
instead of the network (the sha256 comparison still applies to train.py / prepare.py).

Not covered: that training the default configuration on an H200 reproduces the
paper's val_bpb. That needs the GPU run described in COMMANDS.md.
"""
import argparse
import ast
import hashlib
import json
import operator
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from data_generation.autoresearch_hpo_generator import OUT_DIR, PRESETS, build_preset  # noqa: E402
from rewards.autoresearch_hpo import (  # noqa: E402
    HP_NAMES, MAX_SEQ_LEN, POWER_OF_2_HPS, SEARCH_SPACE, STARTING_CONFIG,
    TRIAL_TIME_BUDGET_S, UPSTREAM, inject_config, normalize_config, sample_config,
)

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
SOURCES = {
    "search_space.py": ("automl_repo", "automl_commit", "autoresearch_automl/core/search_space.py"),
    "train.py": ("train_repo", "train_commit", "train.py"),
    "prepare.py": ("train_repo", "train_commit", "prepare.py"),
}
_BINOPS = {ast.Pow: operator.pow, ast.Mult: operator.mul, ast.Add: operator.add,
           ast.Sub: operator.sub}


def fetch(name, offline_dir):
    if offline_dir:
        with open(os.path.join(offline_dir, name), "rb") as f:
            return f.read()
    repo_key, commit_key, path = SOURCES[name]
    url = RAW_URL.format(repo=UPSTREAM[repo_key], commit=UPSTREAM[commit_key], path=path)
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def constant(node):
    """Evaluate a literal expression, allowing arithmetic on numbers (``2**19``)."""
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](constant(node.left), constant(node.right))
    if isinstance(node, ast.Dict):
        return {constant(k): constant(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, (ast.List, ast.Tuple)):
        return [constant(element) for element in node.elts]
    if isinstance(node, ast.Set):
        return {constant(element) for element in node.elts}
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "frozenset" and len(node.args) == 1 and not node.keywords):
        return frozenset(constant(node.args[0]))
    return ast.literal_eval(node)


def module_constants(source, names):
    """Values of module-level ``NAME = <literal>`` / ``NAME: T = <literal>`` statements."""
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id in names:
            if target.id in values:
                raise ValueError(f"{target.id} is assigned more than once")
            values[target.id] = constant(value)
    return values


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, ok, what, detail=""):
        self.failures += 0 if ok else 1
        print(f"  [{'ok' if ok else 'MISMATCH'}] {what}" + (f": {detail}" if detail and not ok else ""))


def check_search_space(source, report):
    upstream = module_constants(source, {"KNOWN_HP_METADATA", "POWER_OF_2_HPS",
                                         "KARPATHY_STARTING_CONFIG"})
    metadata = upstream["KNOWN_HP_METADATA"]
    report.check(list(metadata) == list(HP_NAMES), "hyperparameter names and order",
                 f"upstream {list(metadata)}")
    for hp in SEARCH_SPACE:
        theirs = metadata.get(hp["name"], {})
        if hp["type"] == "cat":
            ours = {"type": "cat", "choices": hp["choices"]}
        else:
            ours = {"type": hp["type"], "low": hp["low"], "high": hp["high"]}
            if hp["log"]:
                ours["log"] = True
            theirs = {k: v for k, v in theirs.items() if not (k == "log" and v is False)}
        report.check(ours == theirs, f"{hp['name']} type / bounds / scale", f"upstream {theirs}")
    report.check(upstream["POWER_OF_2_HPS"] == POWER_OF_2_HPS, "power-of-two hyperparameters",
                 f"upstream {sorted(upstream['POWER_OF_2_HPS'])}")
    report.check(upstream["KARPATHY_STARTING_CONFIG"] == STARTING_CONFIG, "starting configuration",
                 f"upstream {upstream['KARPATHY_STARTING_CONFIG']}")
    report.check(normalize_config(STARTING_CONFIG) == STARTING_CONFIG,
                 "starting configuration is unchanged by normalisation")


def check_train_sources(train_bytes, prepare_bytes, report):
    report.check(hashlib.sha256(train_bytes).hexdigest() == UPSTREAM["train_py_sha256"],
                 "train.py sha256 pin", hashlib.sha256(train_bytes).hexdigest())
    report.check(hashlib.sha256(prepare_bytes).hexdigest() == UPSTREAM["prepare_py_sha256"],
                 "prepare.py sha256 pin", hashlib.sha256(prepare_bytes).hexdigest())

    train_source = train_bytes.decode("utf-8")
    defaults = module_constants(train_source, set(HP_NAMES))
    report.check(defaults == STARTING_CONFIG, "train.py defaults equal the starting configuration",
                 f"train.py {defaults}")

    prepare = module_constants(prepare_bytes.decode("utf-8"), {"TIME_BUDGET", "MAX_SEQ_LEN"})
    report.check(prepare.get("TIME_BUDGET") == TRIAL_TIME_BUDGET_S, "prepare.py TIME_BUDGET",
                 str(prepare.get("TIME_BUDGET")))
    report.check(prepare.get("MAX_SEQ_LEN") == MAX_SEQ_LEN, "prepare.py MAX_SEQ_LEN",
                 str(prepare.get("MAX_SEQ_LEN")))

    import random
    rng = random.Random(0)
    configs = [dict(STARTING_CONFIG)] + [sample_config(rng) for _ in range(200)]
    try:
        round_trips = all(
            module_constants(inject_config(train_source, config), set(HP_NAMES)) == config
            for config in configs)
        detail = ""
    except Exception as exc:  # InjectionError carries the reason
        round_trips, detail = False, str(exc)
    report.check(round_trips, f"injection round-trips {len(configs)} configurations exactly", detail)

    injected = inject_config(train_source, STARTING_CONFIG)
    untouched = [a for a, b in zip(train_source.splitlines(), injected.splitlines())
                 if a != b and a.split("=")[0].strip() not in HP_NAMES]
    report.check(not untouched and len(train_source.splitlines()) == len(injected.splitlines()),
                 "injection changes hyperparameter lines only", str(untouched[:3]))


def check_task_files(report):
    for name, preset in PRESETS.items():
        path = os.path.join(ROOT, OUT_DIR, preset["file"])
        try:
            with open(path, encoding="utf-8") as f:
                on_disk = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            report.check(False, f"{OUT_DIR}/{preset['file']}", "file is missing")
            continue
        report.check(on_disk == json.loads(json.dumps(build_preset(name))),
                     f"{OUT_DIR}/{preset['file']} matches the generator",
                     "regenerate with data_generation/autoresearch_hpo_generator.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--offline-dir", help="Directory holding search_space.py, train.py "
                                              "and prepare.py instead of fetching them")
    args = parser.parse_args()

    report = Report()
    print(f"search space vs {UPSTREAM['automl_repo']}@{UPSTREAM['automl_commit'][:12]}")
    check_search_space(fetch("search_space.py", args.offline_dir).decode("utf-8"), report)
    print(f"training sources vs {UPSTREAM['train_repo']}@{UPSTREAM['train_commit'][:12]}")
    check_train_sources(fetch("train.py", args.offline_dir), fetch("prepare.py", args.offline_dir), report)
    print("task files")
    check_task_files(report)
    print(f"\n{'PASS' if report.failures == 0 else f'FAIL ({report.failures} mismatch(es))'}")
    sys.exit(0 if report.failures == 0 else 1)


if __name__ == "__main__":
    main()
