"""
End-to-end smoke test of the agent runner with a scripted mock LLM (no model needed).
Requires Docker and network access (installs python3 in an ubuntu:22.04 container).

    .venv/bin/python -m domains.envsetup.tests.smoke_agent
"""
from __future__ import annotations

import json
import logging
import os
import sys

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

from domains import get_domain  # noqa: E402
from domains.envsetup.agent import EnvSetupAgentRunner  # noqa: E402
from domains.envsetup.tests.mock_llm import MockLLMServer  # noqa: E402

SCRIPT = [
    ("bash", {"command": "ls -a && sed -n 1,5p setup.py", "reason": "inspect the repository"}),
    ("bash", {"command": "apt-get update -qq && apt-get install -y -qq python3 python3-pip >/dev/null && python3 --version",
              "reason": "install the python toolchain"}),
    ("bash", {"command": "pip3 install --no-input -q -e . pytest && python3 -c 'import six; print(six.__version__)'",
              "reason": "install the project and pytest, verify import"}),
    ("finish", {"report": "installed python3, pip, the project and pytest; import verified", "success": True}),
]


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    rec = next(r for r in get_domain("envsetup").load_tasks("setupbench_dev") if r["task_id"] == "benjaminp-six-c1b416f")
    with MockLLMServer(SCRIPT) as srv:
        runner = EnvSetupAgentRunner(base_url=srv.base_url, model="openai/mock", max_steps=8, max_tokens=512,
                                     memory="2g", cpus=2.0, command_timeout=600)
        row = runner.run_task(rec)
        n_calls = len(srv.calls)
        offered = [t["function"]["name"] for t in (srv.calls[0].get("tools") or [])]
    checks = {
        "tools offered are bash+finish": offered == ["bash", "finish"],
        "exactly 4 LLM calls": n_calls == 4,
        "3 bash steps recorded": row["steps_used"] == 3,
        "finish captured": row["claimed_success"] is True and "verified" in row["finish_summary"],
        "hidden verifier passed": row["solved"] is True,
        "token usage accumulated": row["token_usage"]["input"] > 0,
        "script has 3 lines": len(row["answer"].splitlines()) == 3,
    }
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        print(json.dumps({k: row[k] for k in ("steps_used", "claimed_success", "finish_summary", "solved", "token_usage")},
                         indent=1))
        return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
