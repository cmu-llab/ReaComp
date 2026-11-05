"""
Hidden verifiers for the environment-setup domain.

* ``setupbench_live``: run the task's success command inside the *same* container the
  solver/agent worked in (services it started must still be up).  Success is the
  substring "Setup successful" in stdout+stderr, or exit code 0 for
  dependency-resolution tasks, exactly as in SetupBench's evaluation harness.
* ``envbench_python_replay``: EnvBench's protocol.  The replayable script (successful
  commands only) is executed by EnvBench's own build script in a *fresh* container of
  the EnvBench image; success is script exit code 0 and zero pyright
  ``reportMissingImports`` diagnostics.

Both return the ``rewards/`` contract dict: {"value": 1.0|0.0, "message": str, ...}.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from .container import DockerTaskContainer

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
SUCCESS_MARK = "Setup successful"


def verify_setupbench_live(container: DockerTaskContainer, record: dict) -> dict:
    v = record["verify"]
    t0 = time.time()
    r = container.exec(v["command"], timeout=v.get("timeout", 1200), workdir=record["workdir"],
                       new_session=v.get("start_new_session", False))
    if v.get("success_rule") == "exit0":
        ok = r.exit_code == 0 and not r.timed_out
    else:
        ok = SUCCESS_MARK in r.output and not r.timed_out
    return {"value": 1.0 if ok else 0.0, "message": r.output[-3000:], "exit_code": r.exit_code,
            "timed_out": r.timed_out, "mode": "setupbench_live", "elapsed": round(time.time() - t0, 1)}


def verify_envbench_replay(record: dict, script: str, memory: str = "6g", cpus: float = 2.0,
                           run_label: str = "verify") -> dict:
    from .tasks import prepare_container  # local import: tasks imports container, not verifier
    t0 = time.time()
    container = prepare_container(record, run_label=run_label, memory=memory, cpus=cpus)
    try:
        wd = record["workdir"]
        container.write_file(f"{wd}/bootstrap_script.sh", script or "", mode="0755")
        with open(os.path.join(VENDOR_DIR, "envbench_python_build.sh")) as fh:
            container.write_file(f"{wd}/build.sh", fh.read(), mode="0755")
        # EnvBench initialises pyenv before the agent acts; the build script sources the
        # bootstrap script in the same shell, so we replicate that prefix here.
        r = container.exec('eval "$(pyenv init -)"; bash build.sh', timeout=record["verify"].get("timeout", 1800),
                           workdir=wd)
        issues: Optional[int] = None
        res = container.exec(f"cat {wd}/build_output/results.json", timeout=60)
        if res.exit_code == 0:
            try:
                issues = int(json.loads(res.output).get("issues_count", -1))
            except (json.JSONDecodeError, ValueError, TypeError):
                issues = None
        ok = (r.exit_code == 0) and (issues == 0) and not r.timed_out
        return {"value": 1.0 if ok else 0.0, "message": r.output[-3000:], "exit_code": r.exit_code,
                "timed_out": r.timed_out, "issues_count": issues, "mode": "envbench_python_replay",
                "elapsed": round(time.time() - t0, 1)}
    finally:
        container.stop()


def verify(record: dict, container: Optional[DockerTaskContainer] = None, script: Optional[str] = None,
           memory: str = "6g", cpus: float = 2.0, run_label: str = "verify") -> dict:
    mode = record["verify"]["mode"]
    if mode == "setupbench_live":
        if container is None:
            # Replay path: rebuild the state from the script in a fresh container, then check.
            from .tasks import prepare_container
            container = prepare_container(record, run_label=run_label, memory=memory, cpus=cpus)
            try:
                sh = container.shell(default_timeout=900)
                for cmd in record.get("initial_commands", []):
                    sh.run(cmd)
                if script:
                    sh.run(script, timeout=record["verify"].get("timeout", 1200) * 2)
                return verify_setupbench_live(container, record)
            finally:
                container.stop()
        return verify_setupbench_live(container, record)
    if mode == "envbench_python_replay":
        return verify_envbench_replay(record, script or "", memory=memory, cpus=cpus, run_label=run_label)
    raise ValueError(f"unknown verify mode {mode!r}")
