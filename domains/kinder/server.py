"""Environment server: serves KinDER instances to sandboxed code over a Unix socket.

Sandboxed code (the per-task agent's ``python`` tool, the solver builder's
``execute_code`` tool, an external coding agent's test scripts) never imports KinDER;
it talks to this server through ``kinder_client.make_env()``.  The protocol is the
strict black-box protocol of AgenticGenTAMP: ``reset``, ``step``, ``render`` and
connection closure; no state access, no simulator internals, no seeds.

    python -m domains.kinder.server --config session.json

``session.json``: {"family": str, "socket": path, "instances": [{"count": int, "seed": int}, ...],
                   "free_reset": bool, "exit_with_parent": bool, "cleanup_dir": path | null}

Each connection gets its own environment.  Requests and replies are JSON lines:
    {"cmd": "hello"}                          -> spaces, family, num_instances, free_reset, counts
    {"cmd": "reset", "instance": i}           -> state, info, max_steps, count
    {"cmd": "reset", "seed": s, "count": k}   -> (free_reset only; either may be null; a null count
                                                  is drawn from the seed over the training counts)
    {"cmd": "step", "action": [...]}          -> state, reward, terminated, truncated, info
    {"cmd": "render"}                         -> png_b64
Errors reply {"ok": false, "error": "<Type>: <message>"} and leave the session usable.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import socketserver
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from domains.kinder import client as C  # noqa: E402
from domains.kinder import sim  # noqa: E402
from domains.kinder.families import get_family  # noqa: E402


class Session:
    """Server-side state of one connection."""

    def __init__(self, config: dict) -> None:
        self.family = get_family(config["family"])
        self.instances = [(int(i["count"]), int(i["seed"])) for i in config.get("instances", [])]
        self.free_reset = bool(config.get("free_reset", False))
        self.rng = random.Random()
        self.current: sim.Instance | None = None

    def hello(self, spaces: dict) -> dict:
        return {"spaces": spaces, "family": self.family.name, "num_instances": len(self.instances),
                "free_reset": self.free_reset, "counts": list(self.family.all_counts)}

    def reset(self, req: dict) -> dict:
        if req.get("seed") is not None or req.get("count") is not None:
            if not self.free_reset:
                raise PermissionError("this session only serves its pinned instance(s); use reset() or reset(instance=i)")
            seed = self.rng.randrange(2 ** 63) if req.get("seed") is None else int(req["seed"])
            count = req.get("count")
            # an unpinned count follows the seed, as in robocode's VariableObjectCountEnv
            count = (int(np.random.default_rng(seed).choice(self.family.design_counts)) if count is None
                     else int(count))
            if count not in self.family.all_counts:
                raise ValueError(f"count {count} not available; choose one of {list(self.family.all_counts)}")
        else:
            idx = int(req.get("instance", 0))
            if not 0 <= idx < len(self.instances):
                raise IndexError(f"instance {idx} out of range (num_instances={len(self.instances)})")
            count, seed = self.instances[idx]
        if self.current is not None:
            self.current.close()
        self.current = sim.Instance(self.family, count, seed)
        state, info = self.current.reset()
        return {"state": C.state_to_payload(state), "info": info, "max_steps": self.current.max_steps,
                "count": count}

    def step(self, req: dict) -> dict:
        if self.current is None:
            raise RuntimeError("call reset() before step()")
        state, reward, terminated, truncated, info = self.current.step(req["action"])
        return {"state": C.state_to_payload(state), "reward": reward, "terminated": terminated,
                "truncated": truncated, "info": info}

    def render(self) -> dict:
        if self.current is None:
            raise RuntimeError("call reset() before render()")
        return {"png_b64": base64.b64encode(self.current.render_png()).decode("ascii")}

    def close(self) -> None:
        if self.current is not None:
            self.current.close()


def make_handler(config: dict, spaces: dict):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            session = Session(config)
            try:
                for line in self.rfile:
                    try:
                        req = json.loads(line)
                        cmd = req.get("cmd")
                        if cmd == "hello":
                            body = session.hello(spaces)
                        elif cmd == "reset":
                            body = session.reset(req)
                        elif cmd == "step":
                            body = session.step(req)
                        elif cmd == "render":
                            body = session.render()
                        else:
                            raise ValueError(f"unknown command {cmd!r}")
                        reply = {"ok": True, **body}
                    except Exception as exc:  # noqa: BLE001 - report to the client, keep serving
                        reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    self.wfile.write((json.dumps(reply) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                session.close()

    return Handler


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def _exit_with_parent(config: dict) -> None:
    """Exit (removing the socket and ``cleanup_dir``) once the process that started us is gone,
    so a killed runner does not leave servers behind."""
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(1.0)
        if os.path.exists(config["socket"]):
            os.unlink(config["socket"])
        if config.get("cleanup_dir"):
            shutil.rmtree(config["cleanup_dir"], ignore_errors=True)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def serve(config: dict) -> None:
    if config.get("exit_with_parent"):
        _exit_with_parent(config)
    family = get_family(config["family"])
    probe = sim.Instance(family, max(family.design_counts), 0)
    spaces = probe.spaces_payload()
    probe.close()
    path = config["socket"]
    if os.path.exists(path):
        os.unlink(path)
    with _Server(path, make_handler(config, spaces)) as server:
        os.chmod(path, 0o600)
        print("READY", flush=True)
        try:
            server.serve_forever()
        finally:
            if os.path.exists(path):
                os.unlink(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as fh:
        serve(json.load(fh))


if __name__ == "__main__":
    main()
