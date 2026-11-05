"""
Task records, repository preparation and container provisioning for the
environment-setup domain.

Sources
-------
* SetupBench (microsoft/SetupBench, MIT): 93 scenarios vendored under
  ``data/envsetup/setupbench/{scenarios,fixtures}``.
* EnvBench-Python (JetBrains-Research/EnvBench, MIT): the published
  ``python_baseline_failure_{train,test}`` splits vendored under
  ``data/envsetup/envbench/splits``; repository tarballs are downloaded on demand
  from the Hugging Face dataset into the local cache.

Normalised record (one JSON object per line in ``data/envsetup/tasks/<split>.jsonl``)::

    task_id, source, category, language, ecosystem, base_image, image_substituted,
    workdir, problem_statement, initial_commands, repo, fixture_dir, prerunner,
    verify {mode, command, success_rule, start_new_session, timeout}, hidden {...}

Only the keys in ``EnvSetupDomain.agent_visible_keys`` are ever shown to an agent or
solver; ``verify`` and ``hidden`` never are.

CLI::

    python -m domains.envsetup.tasks build           # (re)write all task files
    python -m domains.envsetup.tasks prepare --split setupbench_dev   # warm the repo cache
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
from typing import Dict, List, Optional

from .container import DockerTaskContainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "envsetup")
TASKS_DIR = os.path.join(DATA_DIR, "tasks")
SETUPBENCH_DIR = os.path.join(DATA_DIR, "setupbench")
ENVBENCH_DIR = os.path.join(DATA_DIR, "envbench")
CACHE_DIR = os.environ.get("REACOMP_ENVSETUP_CACHE", os.path.join(DATA_DIR, "cache"))

ENVBENCH_HF_DATASET = "JetBrains-Research/EnvBench"
ENVBENCH_IMAGE = "ghcr.io/jetbrains-research/envbench-python:latest"
ENVBENCH_WORKDIR = "/data/project"

# The SetupBench npm tasks reference a private Azure registry image; node:16 is the
# closest public equivalent (Node 16 + npm on Debian).  Records carry image_substituted=True.
IMAGE_SUBSTITUTIONS = {"codeexecservice.azurecr.io/setupbench-node-16:latest": "node:16"}

SETUPBENCH_HIDDEN_KEYS = ("notes", "description", "build_commands", "human_actions_lower_bound",
                          "human_actions_upper_bound", "image_tag", "repo", "issue_number", "issue_url",
                          "comment_id", "snippet", "license_spdx", "commit_hash")

# Small, cheap subset used for smoke tests and solver development.
DEV_TASK_IDS = [
    "benjaminp-six-c1b416f",
    "bolsote-isoduration-ae0bd61",
    "python-hyper-rfc3986-75e77ba",
    "pytesseract-df9fce0",
    "dbsetup-sqlite-1",
    "dbsetup-redis-1",
]

ENVBENCH_PROBLEM_STATEMENT = """\
You are configuring a Python repository so that its code can be imported and run.

Environment Details
The repository is located at /data/project (your working directory) inside a Docker container
where you are root. Preinstalled: pyenv with Python 3.8-3.13 (run `eval "$(pyenv init -)"` first),
conda, poetry, uv, pip, pipenv, Node.js, git, build-essential and common system libraries.
Additional system packages can be installed with `apt-get install -y ...`.

Task:
Install the project's dependencies (system packages via apt-get, Python packages via the project's
dependency manager) and select or install the right Python version, so that the `python` command
resolves to an interpreter in which every import used by the repository's Python files can be resolved.
If you use a virtual environment (poetry, venv, conda), activate it with `source ...` so that
`python` points to it. Everything must be non-interactive.

Additional Instructions:
- Success is checked by static analysis: pyright must report zero unresolved imports
  (reportMissingImports) for the repository under the final `python`.
- Only the commands that exit successfully are kept and replayed in a fresh container to verify.
"""


# --------------------------------------------------------------------------- build
def _setupbench_record(raw: dict) -> dict:
    base_image = raw["base_image"]
    substituted = base_image in IMAGE_SUBSTITUTIONS
    task_type = raw["task_type"]
    commit = raw.get("base_commit") or raw.get("commit_hash")
    repo = {"url": raw["repo_url"], "commit": commit} if raw.get("repo_url") else None
    fixture = os.path.join("fixtures", raw["instance_id"])
    fixture_dir = fixture if os.path.isdir(os.path.join(SETUPBENCH_DIR, fixture)) else None
    prerunner = os.path.join("fixtures", f"prerunner-{raw['instance_id']}", "prerunner.sh")
    prerunner = prerunner if os.path.isfile(os.path.join(SETUPBENCH_DIR, prerunner)) else None
    return {
        "task_id": raw["instance_id"],
        "source": "setupbench",
        "category": task_type,
        "language": raw.get("language"),
        "ecosystem": raw.get("ecosystem"),
        "base_image": IMAGE_SUBSTITUTIONS.get(base_image, base_image),
        "image_substituted": substituted,
        "original_base_image": base_image if substituted else None,
        "workdir": "/testbed",
        "problem_statement": raw["problem_statement"].strip(),
        "initial_commands": [],
        "repo": repo,
        "fixture_dir": fixture_dir,
        "prerunner": prerunner,
        "verify": {
            "mode": "setupbench_live",
            "command": raw["success_command"],
            "success_rule": "exit0" if task_type == "dependency_resolution" else "substring",
            "start_new_session": bool(raw.get("start_new_session", False)),
            "timeout": 1200,
        },
        "hidden": {k: raw[k] for k in SETUPBENCH_HIDDEN_KEYS if k in raw},
    }


def build_setupbench() -> List[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(SETUPBENCH_DIR, "scenarios", "*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    records.append(_setupbench_record(json.loads(line)))
    return records


def _envbench_record(raw: dict) -> dict:
    repository, revision = raw["repository"], raw["revision"]
    safe = repository.replace("/", "__")
    return {
        "task_id": f"envbench-{safe}@{revision[:7]}",
        "source": "envbench",
        "category": "envbench_python",
        "language": "python",
        "ecosystem": raw.get("dm"),
        "base_image": ENVBENCH_IMAGE,
        "image_substituted": False,
        "original_base_image": None,
        "workdir": ENVBENCH_WORKDIR,
        "problem_statement": ENVBENCH_PROBLEM_STATEMENT.strip(),
        "initial_commands": ['eval "$(pyenv init -)"'],
        "repo": {"hf_dataset": ENVBENCH_HF_DATASET, "tarball": f"repos/final/python/{safe}.tar.gz",
                 "repository": repository, "revision": revision},
        "fixture_dir": None,
        "prerunner": None,
        "verify": {"mode": "envbench_python_replay", "timeout": 1800},
        "hidden": {},
    }


def _iter_json_objects(text: str):
    """Yield JSON objects from text that may be JSONL *or* concatenated objects on one line."""
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        yield obj
        i = end


def build_envbench(split: str) -> List[dict]:
    path = os.path.join(ENVBENCH_DIR, "splits", f"python_baseline_failure_{split}.jsonl")
    with open(path) as fh:
        return [_envbench_record(o) for o in _iter_json_objects(fh.read())]


def write_jsonl(records: List[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def build_all() -> Dict[str, int]:
    sb = build_setupbench()
    counts = {}
    write_jsonl(sb, os.path.join(TASKS_DIR, "setupbench.jsonl")); counts["setupbench"] = len(sb)
    dev = [r for r in sb if r["task_id"] in DEV_TASK_IDS]
    write_jsonl(dev, os.path.join(TASKS_DIR, "setupbench_dev.jsonl")); counts["setupbench_dev"] = len(dev)
    for split in ("train", "test"):
        recs = build_envbench(split)
        write_jsonl(recs, os.path.join(TASKS_DIR, f"envbench_python_{split}.jsonl"))
        counts[f"envbench_python_{split}"] = len(recs)
    return counts


# ------------------------------------------------------------------- repositories
def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 1800) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _clone_github(url: str, commit: Optional[str], dest: str) -> None:
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if commit:
        steps = [["git", "init", "-q"], ["git", "remote", "add", "origin", url],
                 ["git", "fetch", "-q", "--depth", "1", "origin", commit], ["git", "checkout", "-q", "FETCH_HEAD"]]
        ok = True
        for s in steps:
            cp = subprocess.run(s, cwd=tmp, capture_output=True, text=True, timeout=1800, env=env)
            if cp.returncode != 0:
                ok = False
                break
        if not ok:  # fall back to a full clone (some servers refuse fetch-by-sha)
            shutil.rmtree(tmp, ignore_errors=True)
            cp = subprocess.run(["git", "clone", "-q", url, tmp], capture_output=True, text=True, timeout=3600, env=env)
            if cp.returncode != 0:
                raise RuntimeError(f"git clone failed for {url}: {cp.stderr[-500:]}")
            cp = subprocess.run(["git", "checkout", "-q", commit], cwd=tmp, capture_output=True, text=True, env=env)
            if cp.returncode != 0:
                raise RuntimeError(f"git checkout {commit} failed for {url}: {cp.stderr[-500:]}")
    else:
        cp = subprocess.run(["git", "clone", "-q", "--depth", "1", url, tmp], capture_output=True, text=True,
                            timeout=3600, env=env)
        if cp.returncode != 0:
            raise RuntimeError(f"git clone failed for {url}: {cp.stderr[-500:]}")
    os.rename(tmp, dest)


def _download_hf_tarball(dataset: str, path_in_repo: str, dest: str) -> None:
    url = f"https://huggingface.co/datasets/{dataset}/resolve/main/{path_in_repo}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "reacomp-envsetup"})
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    os.rename(tmp, dest)


def prepare_repo(record: dict) -> Optional[str]:
    """Materialise the task's repository in the local cache and return its host path
    (None for fixture-only tasks).  Idempotent: a ``.reacomp_ready`` marker guards reuse."""
    repo = record.get("repo")
    if not repo:
        return None
    dest = os.path.join(CACHE_DIR, "repos", record["task_id"])
    marker = os.path.join(dest, ".reacomp_ready")
    if os.path.isfile(marker):
        return dest
    shutil.rmtree(dest, ignore_errors=True)
    if "url" in repo:
        _clone_github(repo["url"], repo.get("commit"), dest)
    else:
        tarball = os.path.join(CACHE_DIR, "tarballs", os.path.basename(repo["tarball"]))
        if not os.path.isfile(tarball):
            _download_hf_tarball(repo["hf_dataset"], repo["tarball"], tarball)
        tmp = dest + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        with tarfile.open(tarball) as tf:
            tf.extractall(tmp)
        entries = os.listdir(tmp)
        if len(entries) == 1 and os.path.isdir(os.path.join(tmp, entries[0])):
            os.rename(os.path.join(tmp, entries[0]), dest)
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            os.rename(tmp, dest)
    with open(marker, "w") as fh:
        fh.write(json.dumps(repo))
    return dest


_MANIFESTS = ("README.md", "README.rst", "README.txt", "README", "INSTALL.md", "CONTRIBUTING.md",
              "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "requirements-dev.txt",
              "environment.yml", "Pipfile", "package.json", "Gemfile", "go.mod", "Cargo.toml", "pom.xml",
              "build.gradle", "build.gradle.kts", "Makefile", "tox.ini", "Dockerfile", ".python-version",
              ".nvmrc", ".ruby-version")


def repo_snapshot(host_path: Optional[str], max_chars: int = 7000, max_entries: int = 120) -> str:
    """Compact, LLM-readable description of a repository: listing plus manifest heads.
    Used in demo traces so the coding agent sees what the trace's author saw."""
    if not host_path or not os.path.isdir(host_path):
        return "(no repository; fixture-only task)"
    entries = []
    for root, dirs, files in os.walk(host_path):
        dirs[:] = sorted(d for d in dirs if d != ".git")
        depth = os.path.relpath(root, host_path).count(os.sep) if root != host_path else -1
        if depth >= 1:
            dirs[:] = []
        for name in sorted(files):
            entries.append(os.path.relpath(os.path.join(root, name), host_path))
        for d in dirs:
            entries.append(os.path.relpath(os.path.join(root, d), host_path) + "/")
        if len(entries) > max_entries:
            break
    out = ["Repository listing (depth 2):", *("  " + e for e in entries[:max_entries])]
    if len(entries) > max_entries:
        out.append(f"  ... ({len(entries) - max_entries} more)")
    budget = max_chars - sum(len(x) + 1 for x in out)
    for name in _MANIFESTS:
        p = os.path.join(host_path, name)
        if not os.path.isfile(p) or budget <= 200:
            continue
        try:
            with open(p, errors="replace") as fh:
                head = "".join(fh.readlines()[:60])
        except OSError:
            continue
        head = head[: min(len(head), 1500, budget - 40)]
        block = f"\n--- {name} (head) ---\n{head}"
        out.append(block)
        budget -= len(block)
    return "\n".join(out)


# ---------------------------------------------------------------------- containers
def prepare_container(record: dict, run_label: str = "adhoc", memory: str = "6g", cpus: float = 2.0,
                      network: bool = True) -> DockerTaskContainer:
    """Start a container for ``record`` with repository / fixtures in place and any
    prerunner executed.  The caller owns the container and must ``stop()`` it."""
    host_repo = prepare_repo(record)
    labels = {"reacomp.task_id": record["task_id"], "reacomp.run": run_label}
    c = DockerTaskContainer(image=record["base_image"], workdir=record["workdir"], memory=memory, cpus=cpus,
                            network=network, labels=labels)
    c.start()
    try:
        if host_repo:
            c.copy_in(host_repo, record["workdir"])
        if record.get("fixture_dir"):
            c.copy_in(os.path.join(SETUPBENCH_DIR, record["fixture_dir"]), record["workdir"])
        if record.get("prerunner"):
            c.copy_in(os.path.join(SETUPBENCH_DIR, record["prerunner"]), "/tmp/reacomp_prerunner.sh")
            r = c.exec("bash /tmp/reacomp_prerunner.sh", timeout=900)
            if r.exit_code != 0:
                raise RuntimeError(f"prerunner failed for {record['task_id']}: {r.output[-500:]}")
    except Exception:
        c.stop()
        raise
    return c


# ---------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="write data/envsetup/tasks/*.jsonl from the vendored sources")
    p = sub.add_parser("prepare", help="clone/download the repositories of a split into the cache")
    p.add_argument("--split", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "build":
        print(json.dumps(build_all(), indent=2))
    else:
        path = os.path.join(TASKS_DIR, f"{args.split}.jsonl")
        with open(path) as fh:
            recs = [json.loads(l) for l in fh if l.strip()]
        for r in recs[: args.limit]:
            try:
                print(r["task_id"], "->", prepare_repo(r))
            except Exception as exc:  # noqa: BLE001
                print(r["task_id"], "FAILED:", exc)


if __name__ == "__main__":
    main()
