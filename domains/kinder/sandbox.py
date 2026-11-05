"""Sandboxes for agent-written code in the KinDER domain.

A ``KinderSandbox`` is a working directory, an environment server
(``domains.kinder.server``) serving a fixed set of instances, and a runner that executes
Python in a fresh process under the import guard of ``sandbox_boot.py``.  The working
directory holds ``kinder_client.py`` (the client and state mirrors) plus any files the
caller provides (``ENV.md``, ``DEMOS.json``); it persists for the sandbox's lifetime,
so code can write a policy file in one call and test it in the next.

* The per-task agent (``openhands_agents/kinder_direct_solve``) uses one sandbox per task,
  pinned to that instance.
* ``BuilderSandbox`` adapts a sandbox to the ``run_code`` interface of
  ``openhands_agents.sandbox.ApptainerSandbox`` for the OpenHands solver builder: it
  serves the training instances, allows fresh instances of any configured count, and
  mirrors the builder's output directory (``SOLVER_KINDER.py``) into the working
  directory before every run.

Isolation: the code process gets a scrubbed environment (no API keys), its own
process group (killed on timeout) and the import guard; it reaches the simulator only
through the server's reset/step protocol.  This is a methodological boundary, not a
security sandbox for hostile code.
"""
from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import weakref
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .families import FamilySpec, get_family

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
BOOT = os.path.join(HERE, "sandbox_boot.py")
CLIENT = os.path.join(HERE, "client.py")
CLIENT_NAME = "kinder_client.py"
ALLOWED_PACKAGES = ("numpy", "scipy")


def short_tmpdir(prefix: str) -> str:
    """A private temporary directory whose path is short enough for a Unix socket."""
    base = tempfile.gettempdir()
    if len(base) > 40 and os.path.isdir("/tmp"):
        base = "/tmp"
    return tempfile.mkdtemp(prefix=prefix, dir=base)


def clip(text: str, limit: int) -> str:
    """Keep the head and the tail of long output."""
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head
    return f"{text[:head]}\n... [{len(text) - limit} characters omitted] ...\n{text[-tail:]}"


#: sandboxes not yet closed, for close_all_sandboxes()
_LIVE: "weakref.WeakSet" = weakref.WeakSet()


def close_all_sandboxes() -> None:
    """Close every open sandbox (servers and working directories), e.g. from a SIGTERM handler."""
    for sb in list(_LIVE):
        sb.close()


# ----------------------------------------------------------------------------- server
class EnvServer:
    """``domains.kinder.server`` in a child process, listening on a Unix socket."""

    def __init__(self, family: FamilySpec | str, instances: Sequence[Tuple[int, int]], free_reset: bool,
                 python: Optional[str] = None, startup_timeout: float = 120.0) -> None:
        self.family = get_family(family) if isinstance(family, str) else family
        self.instances = [(int(c), int(s)) for c, s in instances]
        self.free_reset = free_reset
        self.python = python or sys.executable
        self.startup_timeout = startup_timeout
        self.socket_path: Optional[str] = None
        self._dir: Optional[str] = None
        self._proc: Optional[subprocess.Popen] = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> "EnvServer":
        if self.running:
            return self
        self._dir = short_tmpdir("kenv_")
        self.socket_path = os.path.join(self._dir, "env.sock")
        cfg = os.path.join(self._dir, "session.json")
        with open(cfg, "w") as fh:
            json.dump({"family": self.family.name, "socket": self.socket_path, "free_reset": self.free_reset,
                       "instances": [{"count": c, "seed": s} for c, s in self.instances],
                       "exit_with_parent": True, "cleanup_dir": self._dir}, fh)
        log = open(os.path.join(self._dir, "server.log"), "w")
        self._proc = subprocess.Popen([self.python, "-m", "domains.kinder.server", "--config", cfg],
                                      cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=log, text=True,
                                      start_new_session=True)
        log.close()
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            ready, _, _ = select.select([self._proc.stdout], [], [], 0.5)
            if ready and self._proc.stdout.readline().strip() == "READY":
                return self
            if self._proc.poll() is not None:
                break
        err = self._log_tail()
        self.stop()
        raise RuntimeError(f"KinDER environment server failed to start:\n{err}")

    def _log_tail(self) -> str:
        try:
            with open(os.path.join(self._dir, "server.log")) as fh:
                return fh.read()[-3000:]
        except OSError:
            return ""

    def stop(self) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    os.killpg(self._proc.pid, signal.SIGTERM)
                    self._proc.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(self._proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self._proc.wait()
            if self._proc.stdout:
                self._proc.stdout.close()
            self._proc = None
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    def __enter__(self) -> "EnvServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


# ----------------------------------------------------------------------------- runner
@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: Optional[int]
    timed_out: bool
    elapsed_s: float

    def text(self) -> str:
        """Observation text for an LLM."""
        parts = []
        if self.timed_out:
            parts.append(f"[timed out after {self.elapsed_s:.0f}s; process killed]")
        elif not self.ok:
            parts.append(f"[exit code {self.exit_code}]")
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append("--- stderr ---\n" + self.stderr.rstrip())
        return "\n".join(parts) if parts else "(no output)"


class KinderSandbox:
    """Working directory + environment server + strict Python runner."""

    def __init__(self, family: FamilySpec | str, instances: Sequence[Tuple[int, int]], free_reset: bool = False,
                 files: Optional[Dict[str, str]] = None, workdir: Optional[str] = None, timeout: float = 120.0,
                 max_output_chars: int = 10000, python: Optional[str] = None) -> None:
        self.family = get_family(family) if isinstance(family, str) else family
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.python = python or sys.executable
        self._owns_workdir = workdir is None
        self.workdir = os.path.realpath(workdir or tempfile.mkdtemp(prefix=f"kinder_{self.family.name}_"))
        os.makedirs(os.path.join(self.workdir, ".cells"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "tmp"), exist_ok=True)
        shutil.copyfile(CLIENT, os.path.join(self.workdir, CLIENT_NAME))
        for name, src in (files or {}).items():
            shutil.copyfile(src, os.path.join(self.workdir, name))
        self.server = EnvServer(self.family, instances, free_reset, python=self.python)
        self._cell = 0
        _LIVE.add(self)
        self._finalizer = weakref.finalize(self, KinderSandbox._cleanup, self.server,
                                           self.workdir if self._owns_workdir else None)

    @staticmethod
    def _cleanup(server: EnvServer, workdir: Optional[str]) -> None:
        server.stop()
        if workdir and os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)

    def _child_env(self) -> Dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": self.workdir,
            "TMPDIR": os.path.join(self.workdir, "tmp"),
            "LANG": "C.UTF-8",
            "KINDER_ENV_SOCKET": self.server.socket_path or "",
            "KINDER_SANDBOX_ALLOWED": ",".join(ALLOWED_PACKAGES),
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }

    def run(self, code: str, timeout: Optional[float] = None) -> RunResult:
        """Run ``code`` as a script in the working directory."""
        self.server.start()
        self._cell += 1
        cell = os.path.join(self.workdir, ".cells", f"cell_{self._cell:03d}.py")
        with open(cell, "w") as fh:
            fh.write(code)
        limit = self.timeout if timeout is None else timeout
        t0 = time.time()
        proc = subprocess.Popen([self.python, "-I", "-B", BOOT, self.workdir, cell], cwd=self.workdir,
                                env=self._child_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.DEVNULL, text=True, errors="replace", start_new_session=True)
        timed_out = False
        try:
            out, err = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
        return RunResult(ok=(proc.returncode == 0 and not timed_out), stdout=clip(out, self.max_output_chars),
                         stderr=clip(err, self.max_output_chars // 2), exit_code=proc.returncode,
                         timed_out=timed_out, elapsed_s=round(time.time() - t0, 2))

    def close(self) -> None:
        self._finalizer()

    def __enter__(self) -> "KinderSandbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class BuilderSandbox(KinderSandbox):
    """``run_code`` adapter for ``openhands_agents.solver_builder`` (see module docstring)."""

    def __init__(self, family: FamilySpec | str, instances: Sequence[Tuple[int, int]], demos_path: str,
                 card_path: str, output_dir: str, timeout: float = 300.0, **kw) -> None:
        super().__init__(family, instances, free_reset=True,
                         files={"DEMOS.json": demos_path, "ENV.md": card_path}, timeout=timeout, **kw)
        self.output_dir = os.path.abspath(output_dir)

    def _sync_outputs(self) -> None:
        if not os.path.isdir(self.output_dir):
            return
        for name in os.listdir(self.output_dir):
            src = os.path.join(self.output_dir, name)
            if os.path.isfile(src) and name not in (CLIENT_NAME, "DEMOS.json", "ENV.md"):
                shutil.copyfile(src, os.path.join(self.workdir, name))

    def run_code(self, code: str, lib_dir: Optional[str] = None, timeout: Optional[int] = None,
                 extra_binds: Optional[List] = None) -> Tuple[bool, str, str]:
        del lib_dir, extra_binds  # files are provided through the working directory
        self._sync_outputs()
        r = self.run(code, timeout=timeout)
        if r.ok:
            return True, r.stdout, r.stderr
        # the builder shows only the first ~800 characters of stderr for a failed run: lead with the
        # end of the traceback (the error itself), then the end of the output that preceded it
        status = (f"timed out after {r.elapsed_s:.0f}s; process killed" if r.timed_out
                  else f"exit code {r.exit_code}")
        parts = [f"[{status}]"]
        if r.stderr.strip():
            parts.append(r.stderr.strip()[-600:])
        if r.stdout.strip():
            parts.append("--- end of stdout ---\n" + r.stdout.strip()[-1500:])
        return False, r.stdout, "\n".join(parts)
