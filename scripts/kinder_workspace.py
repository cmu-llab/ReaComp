#!/usr/bin/env python3
"""Working directory for inducing a KinDER solver with an external coding agent (Claude Code, Codex).

    # 1. create the directory: BUILDING_PROMPT.md, DEMOS.json, ENV.md, kinder_client.py
    python scripts/kinder_workspace.py create --domain kinder_motion2d --out workspaces/kinder_motion2d

    # 2. serve the simulator to it (keep running while the agent works; Ctrl-C to stop)
    python scripts/kinder_workspace.py serve workspaces/kinder_motion2d

    # 3. in another terminal, start the agent inside the directory and tell it:
    #    "Read BUILDING_PROMPT.md and implement the solver as described."
    cd workspaces/kinder_motion2d && claude

The simulator is reached only through ``kinder_env.sock`` (reset/step/render on the training
instances and on fresh instances of any configured count), as in the OpenHands builder's
sandbox.  For the strict black-box setting, run the agent where KinDER is not importable,
e.g. a container with Python, numpy and scipy and this directory mounted; the solver's
imports are checked again (standard library, numpy, scipy only) when it is evaluated.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from domains import get_domain, list_domains  # noqa: E402
from domains.kinder import prompts as P  # noqa: E402
from domains.kinder.sandbox import CLIENT, CLIENT_NAME  # noqa: E402

SOCKET_NAME = "kinder_env.sock"
UNIX_SOCKET_PATH_MAX = 104   # the smallest sun_path limit among Linux (108) and macOS (104)


def create(domain_name: str, out: str, demos: str | None) -> None:
    dom = get_domain(domain_name)
    demos = demos or dom.abspath(next(iter(dom.demos.values())))
    if not os.path.isfile(demos):
        sys.exit(f"demos not found: {demos} (build them with scripts/build_demos.py --domain {domain_name})")
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(demos, os.path.join(out, "DEMOS.json"))
    shutil.copyfile(dom.card_path(), os.path.join(out, "ENV.md"))
    shutil.copyfile(CLIENT, os.path.join(out, CLIENT_NAME))
    guide = P.WORKSPACE_GUIDE.format(solver_file=dom.solver_filename, algorithm_file=dom.algorithm_filename,
                                     client_usage=P.client_usage(pinned_only=False))
    with open(os.path.join(out, "BUILDING_PROMPT.md"), "w") as fh:
        fh.write(dom.render_building_prompt() + "\n\n---\n\n" + guide)
    with open(os.path.join(out, "workspace.json"), "w") as fh:
        json.dump({"domain": domain_name, "demos": os.path.abspath(demos)}, fh, indent=1)
    print(f"created {out} for {domain_name}; next: python scripts/kinder_workspace.py serve {out}")


def serve(workspace: str) -> None:
    from domains.kinder.server import serve as serve_env
    with open(os.path.join(workspace, "workspace.json")) as fh:
        meta = json.load(fh)
    dom = get_domain(meta["domain"])
    sock = os.path.join(os.path.abspath(workspace), SOCKET_NAME)
    if len(sock) >= UNIX_SOCKET_PATH_MAX:
        sys.exit(f"socket path too long for a Unix socket ({len(sock)} chars): use a shorter workspace path")
    train = [{"count": r["count"], "seed": r["seed"]} for r in dom.load_tasks("train")]
    print(f"serving {meta['domain']} ({len(train)} training instances, fresh instances of counts "
          f"{list(dom.family.all_counts)}) at {sock}", flush=True)
    try:
        serve_env({"family": dom.family.name, "socket": sock, "instances": train, "free_reset": True})
    except KeyboardInterrupt:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--domain", required=True, choices=[d for d in list_domains() if d.startswith("kinder_")])
    c.add_argument("--out", required=True)
    c.add_argument("--demos", default=None, help="default: the domain's demos file")
    s = sub.add_parser("serve")
    s.add_argument("workspace")
    args = ap.parse_args()
    if args.cmd == "create":
        create(args.domain, args.out, args.demos)
    else:
        serve(args.workspace)


if __name__ == "__main__":
    main()
