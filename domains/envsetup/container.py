"""
Docker-backed task containers for the environment-setup domain.

A task container is the unit of isolation for one task attempt: it is started from
the task's base image, the repository / fixtures are copied in, and then either an
induced solver or an LLM agent drives it through a persistent bash session.  The
hidden verifier is executed in the same container afterwards (or, for replay-style
benchmarks, in a fresh one).

Only the Docker CLI is required (no daemon SDK), so the same code runs on a laptop
and on any Docker-capable host.  An Apptainer backend would implement the same
interface; root-inside-the-container tasks (apt, services) are the reason Docker is
the default.

    with DockerTaskContainer(image="ubuntu:22.04", workdir="/testbed", name="t1") as c:
        c.copy_in("/host/repo", "/testbed")
        sh = c.shell()
        r = sh.run("apt-get update && apt-get install -y python3", timeout=600)
        print(r.exit_code, r.output[-500:])
        print(c.exec("echo hi").output)
"""
from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_ENV = {
    "DEBIAN_FRONTEND": "noninteractive",
    "TZ": "UTC",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PIP_NO_INPUT": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "PYTHONUNBUFFERED": "1",
    "CI": "true",
}


class ContainerError(RuntimeError):
    pass


@dataclass
class ShellResult:
    exit_code: int
    output: str                 # combined stdout+stderr (tail, see max_output_chars)
    timed_out: bool = False
    elapsed: float = 0.0
    truncated: bool = False


@dataclass
class ShellStep:
    command: str
    exit_code: int
    timed_out: bool
    elapsed: float
    output_tail: str = ""


def _docker(*args: str, timeout: Optional[float] = None, check: bool = True, input_bytes: Optional[bytes] = None
            ) -> subprocess.CompletedProcess:
    cp = subprocess.run(["docker", *args], capture_output=True, timeout=timeout, input=input_bytes)
    if check and cp.returncode != 0:
        raise ContainerError(f"docker {' '.join(args[:3])}... failed ({cp.returncode}): "
                             f"{cp.stderr.decode(errors='replace')[-800:]}")
    return cp


def image_available(image: str) -> bool:
    return _docker("image", "inspect", image, check=False).returncode == 0


def pull_image(image: str, timeout: float = 3600) -> None:
    if not image_available(image):
        _docker("pull", "-q", image, timeout=timeout)


def remove_containers_by_label(label: str) -> int:
    """Force-remove every container carrying ``label`` (``key=value``). Returns the count removed."""
    cp = _docker("ps", "-aq", "--filter", f"label={label}", check=False)
    ids = cp.stdout.decode().split()
    for cid in ids:
        _docker("rm", "-f", cid, check=False)
    return len(ids)


class PersistentShell:
    """One long-lived ``bash`` inside the container.  cwd and exported variables persist
    across ``run`` calls, background processes survive, and a timed-out command has its
    whole process group killed before a fresh shell is opened (cwd/env are then reset)."""

    _DONE = "__REACOMP_DONE__"

    def __init__(self, container: "DockerTaskContainer", default_timeout: float = 600,
                 max_output_chars: int = 20000, log_path: Optional[str] = None):
        self.container = container
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self.log_path = log_path
        self.log: List[ShellStep] = []
        self.deadline: Optional[float] = None       # absolute time budget for the driver (solver/agent)
        self._proc: Optional[subprocess.Popen] = None
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._pid = None
        self._pgid = None
        self._lock = threading.Lock()
        self._open()

    # -- lifecycle ------------------------------------------------------------
    def _open(self) -> None:
        cmd = ["docker", "exec", "-i", "-w", self.container.workdir, self.container.cid,
               "bash", "--noprofile", "--norc"]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, bufsize=0)
        self._q = queue.Queue()
        t = threading.Thread(target=self._reader, args=(self._proc, self._q), daemon=True)
        t.start()
        # Learn pid / pgid so a hung command's whole group can be killed later.
        r = self._exchange('echo "__PIDINFO__ $$ $(ps -o pgid= -p $$ | tr -d " ")"', timeout=30)
        m = re.search(r"__PIDINFO__ (\d+) (\d+)", r.output)
        if m:
            self._pid, self._pgid = int(m.group(1)), int(m.group(2))
        for k, v in self.container.env.items():
            self._exchange(f"export {k}={shlex.quote(v)}", timeout=30)

    @staticmethod
    def _reader(proc: subprocess.Popen, q: "queue.Queue[Optional[str]]") -> None:
        try:
            for raw in iter(proc.stdout.readline, b""):
                q.put(raw.decode("utf-8", errors="replace"))
        finally:
            q.put(None)

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    def _kill_group_and_reopen(self) -> None:
        if self._pgid:
            self.container.exec(f"kill -9 -- -{self._pgid} 2>/dev/null; kill -9 {self._pid} 2>/dev/null; true",
                                timeout=30, check=False)
        try:
            self._proc.kill()
        except Exception:
            pass
        self._proc = None
        self._open()

    # -- execution ------------------------------------------------------------
    def _exchange(self, command: str, timeout: float) -> ShellResult:
        nonce = uuid.uuid4().hex[:12]
        wrapped = (
            "{\n" + command + "\n} </dev/null 2>&1\n"
            f"printf '\\n{self._DONE}{nonce} %s\\n' \"$?\"\n"
        )
        t0 = time.time()
        try:
            self._proc.stdin.write(wrapped.encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._kill_group_and_reopen()
            return ShellResult(-1, "shell was closed; reopened a fresh session (cwd/env reset)", False,
                               time.time() - t0)
        chunks: List[str] = []
        total = 0
        marker = f"{self._DONE}{nonce} "
        while True:
            remaining = timeout - (time.time() - t0)
            if remaining <= 0:
                self._kill_group_and_reopen()
                out = "".join(chunks)
                return ShellResult(124, self._tail(out) + f"\n[timeout after {timeout:.0f}s; process group killed, "
                                   "shell reset: cwd and exported variables are gone]", True, time.time() - t0)
            try:
                line = self._q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:  # shell died (e.g. the command ran `exit`)
                self._kill_group_and_reopen()
                return ShellResult(-1, self._tail("".join(chunks)) + "\n[shell exited; fresh session opened]",
                                   False, time.time() - t0)
            if marker in line:
                rc_str = line.split(marker, 1)[1].strip()
                try:
                    rc = int(rc_str)
                except ValueError:
                    rc = -1
                out = "".join(chunks)
                res = ShellResult(rc, self._tail(out), False, time.time() - t0, truncated=len(out) > self.max_output_chars)
                return res
            chunks.append(line)
            total += len(line)
            if total > 50 * self.max_output_chars:      # keep memory bounded on chatty commands
                chunks = [self._tail("".join(chunks))]
                total = len(chunks[0])

    def _tail(self, s: str) -> str:
        if len(s) <= self.max_output_chars:
            return s
        return "[... output truncated ...]\n" + s[-self.max_output_chars:]

    def run(self, command: str, timeout: Optional[float] = None) -> ShellResult:
        """Run ``command`` in the persistent session and record it in ``log``."""
        with self._lock:
            if self.deadline is not None and time.time() > self.deadline:
                r = ShellResult(124, "[time budget exhausted; no further commands are executed]", True, 0.0)
                self._record(command, r)
                return r
            syntax = subprocess.run(["bash", "-n"], input=command.encode(), capture_output=True)
            if syntax.returncode != 0:
                r = ShellResult(2, "bash syntax error:\n" + syntax.stderr.decode(errors="replace")[-2000:], False, 0.0)
                self._record(command, r)
                return r
            t = timeout or self.default_timeout
            if self.deadline is not None:
                t = max(1.0, min(t, self.deadline - time.time()))
            r = self._exchange(command, t)
            self._record(command, r)
            return r

    def _record(self, command: str, r: ShellResult) -> None:
        self.log.append(ShellStep(command, r.exit_code, r.timed_out, round(r.elapsed, 3), r.output[-2000:]))
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(f"\n$ {command}\n{r.output}\n[exit {r.exit_code}{' TIMEOUT' if r.timed_out else ''} "
                         f"{r.elapsed:.1f}s]\n")

    def script(self, include_failed: bool = False) -> str:
        """Replayable script: the successful commands in order (failed ones commented out)."""
        lines = []
        for s in self.log:
            if s.exit_code == 0 and not s.timed_out:
                lines.append(s.command)
            elif include_failed:
                lines.append("# " + s.command.replace("\n", "\n# "))
        return "\n".join(lines)


class DockerTaskContainer:
    def __init__(self, image: str, workdir: str, name: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None, memory: str = "6g", cpus: float = 2.0,
                 network: bool = True, labels: Optional[Dict[str, str]] = None, pids_limit: int = 4096):
        self.image = image
        self.workdir = workdir
        self.name = name or f"reacomp-{uuid.uuid4().hex[:10]}"
        self.env = {**DEFAULT_ENV, **(env or {})}
        self.memory = memory
        self.cpus = cpus
        self.network = network
        self.labels = {"reacomp": "1", **(labels or {})}
        self.pids_limit = pids_limit
        self.cid: Optional[str] = None
        self._shells: List[PersistentShell] = []

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> "DockerTaskContainer":
        pull_image(self.image)
        args = ["run", "-d", "--name", self.name, "--memory", self.memory, "--cpus", str(self.cpus),
                "--pids-limit", str(self.pids_limit), "--init"]
        if not self.network:
            args += ["--network", "none"]
        for k, v in self.labels.items():
            args += ["--label", f"{k}={v}"]
        for k, v in self.env.items():
            args += ["-e", f"{k}={v}"]
        args += [self.image, "sh", "-c", "while true; do sleep 3600; done"]
        self.cid = _docker(*args, timeout=600).stdout.decode().strip()
        self.exec(f"mkdir -p {shlex.quote(self.workdir)}", timeout=60, check=False)
        return self

    def stop(self) -> None:
        for sh in self._shells:
            sh.close()
        self._shells = []
        if self.cid:
            _docker("rm", "-f", self.cid, check=False, timeout=120)
            self.cid = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- operations -----------------------------------------------------------
    def copy_in(self, host_path: str, container_path: str) -> None:
        """Copy a host file, or the *contents* of a host directory, into the container."""
        if os.path.isdir(host_path):
            self.exec(f"mkdir -p {shlex.quote(container_path)}", timeout=60, check=False)
            _docker("cp", os.path.join(host_path, "."), f"{self.cid}:{container_path}", timeout=1800)
        else:
            _docker("cp", host_path, f"{self.cid}:{container_path}", timeout=600)

    def copy_out(self, container_path: str, host_path: str) -> bool:
        return _docker("cp", f"{self.cid}:{container_path}", host_path, check=False, timeout=600).returncode == 0

    def write_file(self, container_path: str, content: str, mode: str = "0755") -> None:
        self.exec(f"mkdir -p $(dirname {shlex.quote(container_path)}) && cat > {shlex.quote(container_path)} "
                  f"&& chmod {mode} {shlex.quote(container_path)}",
                  timeout=60, stdin=content.encode())

    def exec(self, command: str, timeout: Optional[float] = 600, workdir: Optional[str] = None,
             new_session: bool = False, check: bool = False, stdin: Optional[bytes] = None) -> ShellResult:
        """One-off ``bash -c`` execution (separate process, no state shared with the shell)."""
        args = ["exec", "-i" if stdin is not None else "-t"]
        args = ["exec"] + (["-i"] if stdin is not None else [])
        args += ["-w", workdir or self.workdir, self.cid]
        if new_session:
            args += ["setsid", "-w"]
        args += ["bash", "-c", command]
        t0 = time.time()
        try:
            cp = _docker(*args, timeout=timeout, check=False, input_bytes=stdin)
        except subprocess.TimeoutExpired as exc:
            partial = (exc.stdout or b"").decode(errors="replace") + (exc.stderr or b"").decode(errors="replace")
            return ShellResult(124, partial[-20000:] + f"\n[timeout after {timeout}s]", True, time.time() - t0)
        out = cp.stdout.decode(errors="replace") + cp.stderr.decode(errors="replace")
        if check and cp.returncode != 0:
            raise ContainerError(f"command failed ({cp.returncode}): {out[-800:]}")
        return ShellResult(cp.returncode, out[-20000:], False, time.time() - t0)

    def shell(self, default_timeout: float = 600, max_output_chars: int = 20000,
              log_path: Optional[str] = None) -> PersistentShell:
        sh = PersistentShell(self, default_timeout, max_output_chars, log_path)
        self._shells.append(sh)
        return sh
