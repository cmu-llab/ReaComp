"""
Build solver-induction demos (reasoning traces) for the environment-setup domain.

Two trace sources, same output schema:
  * our own agent runs: rows written by ``domains.envsetup.agent`` (``trajectory`` steps
    with reason/command/exit_code/output and the verifier verdict in ``solved``);
  * EnvBench's published GPT-4o bash-agent trajectories (Hugging Face dataset
    JetBrains-Research/EnvBench-trajectories, run ``python/bash_agent-4o``), whose
    tool calls carry a ``reason`` string per command; verdicts come from the run's
    ``results.jsonl`` (exit code 0 and zero missing imports).

Each demo::

    {"task_id", "source", "category", "language", "trace_source",
     "prompt": <problem statement + repository snapshot>,
     "cot": [{"reason", "command", "exit_code", "output"}, ...],
     "final_response": <replayable script of successful commands>,
     "success": <bool, hidden verifier verdict>}

Selection is balanced over (category, success) so the coding agent sees both what
works and what fails in every task family, as in the paper.

    python -m domains.envsetup.demos --traces outputs/envsetup/agent_setupbench.jsonl \
        --envbench-published train --n 100 --seed 42 --out demos/DEMOS_ENVSETUP.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional

from . import tasks as T

TRAJ_DATASET = "JetBrains-Research/EnvBench-trajectories"
TRAJ_RUN = "python/bash_agent-4o"


def _all_records() -> Dict[str, dict]:
    out = {}
    for name in ("setupbench", "envbench_python_train", "envbench_python_test"):
        path = os.path.join(T.TASKS_DIR, f"{name}.jsonl")
        if os.path.exists(path):
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        out[r["task_id"]] = r
    return out


def _domain():
    from domains import get_domain
    return get_domain("envsetup")


def _prompt_for(record: dict) -> str:
    return _domain().demo_prompt(record)


def _script_from_steps(steps: List[dict]) -> str:
    return "\n".join(s["command"] for s in steps if s.get("exit_code") == 0)


# ----------------------------------------------------------------- our traces
def demos_from_agent_rows(paths: List[str], records: Dict[str, dict], trace_source: str) -> List[dict]:
    """Rows of domains.envsetup.agent -> demos via the domain's converter (shared schema)."""
    dom = _domain()
    demos = []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                rec = records.get(row["task_id"])
                if rec is None:
                    continue
                row = dict(row, trace_source=trace_source)
                d = dom.demo_from_trace(rec, row)
                if d is not None:
                    demos.append(d)
    return demos


# ------------------------------------------------------- EnvBench published
def _hf_get(path_in_repo: str, dest: str) -> Optional[str]:
    if os.path.exists(dest):
        return dest
    url = f"https://huggingface.co/datasets/{TRAJ_DATASET}/resolve/main/{path_in_repo}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "reacomp-envsetup"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest + ".part", "wb") as out:
            out.write(resp.read())
        os.rename(dest + ".part", dest)
        return dest
    except Exception:  # noqa: BLE001
        return None


def _parse_envbench_trajectory(path: str) -> List[dict]:
    nodes = [json.loads(l) for l in open(path) if l.strip()]
    steps: List[dict] = []
    pending: List[dict] = []
    history = []
    for node in nodes:
        kind = node.get("node")
        if kind == "agent":
            for m in node.get("messages", []):
                for tc in (m.get("message_content") or {}).get("tool_calls", []) or []:
                    a = tc.get("args") or {}
                    if "command" in a:
                        pending.append({"reason": a.get("reason", ""), "command": a["command"], "exit_code": None, "output": ""})
        elif kind == "tools":
            for m in node.get("messages", []):
                content = (m.get("message_content") or {}).get("content", "")
                if pending:
                    st = pending.pop(0)
                    st["output"] = (content or "")[-1500:]
                    steps.append(st)
        elif kind == "commands_history":
            history = node.get("commands") or []
            if isinstance(history, str):
                history = json.loads(history)
    steps.extend(pending)
    # exit codes: align by command text in order (history also contains the initial pyenv command)
    hist = [h for h in history if isinstance(h, dict)]
    hi = 0
    for st in steps:
        for j in range(hi, len(hist)):
            if hist[j].get("command") == st["command"]:
                st["exit_code"] = hist[j].get("exit_code")
                hi = j + 1
                break
    return steps


def demos_from_envbench_published(split: str, records: Dict[str, dict], limit: Optional[int] = None) -> List[dict]:
    cache = os.path.join(T.CACHE_DIR, "envbench_trajectories", TRAJ_RUN.replace("/", "__"))
    results_path = _hf_get(f"{TRAJ_RUN}/results.jsonl", os.path.join(cache, "results.jsonl"))
    verdict = {}
    if results_path:
        for line in open(results_path):
            if line.strip():
                r = json.loads(line)
                verdict[(r["repo_name"], r["commit_sha"])] = (r.get("exit_code") == 0 and r.get("issues_count") == 0)
    wanted = [r for r in T.build_envbench(split)]
    demos = []
    for raw in wanted[:limit]:
        rec = records.get(raw["task_id"])
        if rec is None:
            continue
        repo, rev = rec["repo"]["repository"], rec["repo"]["revision"]
        fname = f"{repo.replace('/', '__')}@{rev}.jsonl"
        local = _hf_get(f"{TRAJ_RUN}/trajectories/{fname}", os.path.join(cache, "trajectories", fname))
        if not local or (repo, rev) not in verdict:
            continue
        steps = _parse_envbench_trajectory(local)
        if not steps:
            continue
        demos.append({
            "task_id": rec["task_id"], "source": rec["source"], "category": rec["category"],
            "language": "python", "trace_source": f"envbench:{TRAJ_RUN}",
            "prompt": _prompt_for(rec), "cot": steps, "final_response": _script_from_steps(steps),
            "success": bool(verdict[(repo, rev)]),
        })
    return demos


# ------------------------------------------------------------------ sampling
def balanced_sample(demos: List[dict], n: int, seed: int = 42) -> List[dict]:
    from domains.base import balanced_sample as _bs
    return _bs(demos, n, _domain().demo_group_key, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="*", default=[], help="agent output JSONL files (domains.envsetup.agent)")
    ap.add_argument("--trace-source", default="ours", help="label for --traces demos, e.g. ours:qwen3.8-27b")
    ap.add_argument("--envbench-published", default=None, choices=[None, "train", "test"],
                    help="also pull EnvBench's published GPT-4o trajectories for this split")
    ap.add_argument("--envbench-limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = _all_records()
    pool = demos_from_agent_rows(args.traces, records, args.trace_source)
    if args.envbench_published:
        pool += demos_from_envbench_published(args.envbench_published, records, args.envbench_limit)
    chosen = balanced_sample(pool, args.n, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(chosen, fh, indent=1)
    by = defaultdict(int)
    for d in chosen:
        by[(d["category"], d["success"])] += 1
    print(f"pool={len(pool)} chosen={len(chosen)} -> {args.out}")
    for k in sorted(by):
        print(f"  {k[0]:<24} success={k[1]!s:<5} {by[k]}")


if __name__ == "__main__":
    main()
