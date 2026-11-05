"""
Local tests for the KinDER domains (CPU only; no GPU, no real LLM, no network).

    .venv/bin/python -m domains.tests.test_kinder            # everything (~30 s)
    .venv/bin/python -m domains.tests.test_kinder --quick    # skip the end-to-end agent/builder runs (~15 s)

Covers:
  * task files: protocol (seeds, count schedule, budgets, disjoint splits), stored instances
    reproduce, environment descriptions present
  * simulator layer: state mirrors and codec, action contract, exact replay, verifier
  * solver path: a scripted Motion2D policy solves instances of every count through
    KinderDomain.solve; crashes, invalid actions, timeouts and forbidden imports fail cleanly
  * pipeline scripts: eval_domain_solver -> hybrid_gate -> kinder_report on real rows
  * sandbox: import guard, pinned instances, scrubbed environment, timeouts
  * per-task agent (openhands_agents.run_kinder_direct_solve) end to end against a scripted
    mock LLM: rows, checkpoint, reasoning capture, submit validation; demos from its rows
  * solver builder (openhands_agents.build_solver --domain kinder_motion2d) end to end against
    a scripted mock LLM: the KinDER sandbox serves DEMOS.json, ENV.md and the simulator
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np  # noqa: E402

from domains import get_domain, list_domains  # noqa: E402
from domains.kinder import client as C  # noqa: E402
from domains.kinder import sim  # noqa: E402
from domains.kinder.families import FAMILIES  # noqa: E402
from data_generation import kinder_generator as G  # noqa: E402

RESULTS = []
FIXTURE = os.path.join(REPO_ROOT, "domains", "tests", "fixtures", "kinder_motion2d_policy.py")
POLICY = open(FIXTURE).read()
PY = sys.executable


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + str(detail)) if (detail and not cond) else ''}")


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, **kw)


# ----------------------------------------------------------------------------- tasks
def test_task_files():
    names = [d for d in list_domains() if d.startswith("kinder_")]
    check("registry: one domain per family", sorted(names) == sorted(f"kinder_{f}" for f in FAMILIES), names)
    manifest = json.load(open(os.path.join(G.DATA_DIR, "manifest.json")))
    for fam in FAMILIES.values():
        dom = get_domain(f"kinder_{fam.name}")
        train, test = dom.load_tasks("train"), dom.load_tasks("test")
        ok = len(train) == 100 and len(test) == 100
        for split, rows, pool in (("train", train, fam.design_counts), ("test", test, fam.eval_counts)):
            ok &= [r["seed"] for r in rows] == G.generate_seeds(manifest["master_seeds"][split], len(rows))
            ok &= [r["count"] for r in rows] == [pool[i % len(pool)] for i in range(len(rows))]
            ok &= all(r["max_steps"] == 300 + 150 * r["count"] and r["regime"] == fam.regime(r["count"])
                      for r in rows)
        ok &= not ({r["seed"] for r in train} & {r["seed"] for r in test})
        ok &= all(r["regime"] == "design" for r in train) and \
            {r["count"] for r in test if r["regime"] == "extrapolation"} == set(fam.eval_counts) - set(fam.design_counts)
        check(f"{fam.name}: 100 train + 100 test, robocode protocol, disjoint", ok)
        card = dom.env_card()
        check(f"{fam.name}: environment description", fam.title in card and "## Observation" in card
              and "crv_robot" in card and "## Reward" in card)
    bad = G.check(list(FAMILIES), limit=5)
    check("stored instances reproduce with the installed KinDER (5 per split per family)", not bad, bad)


# -------------------------------------------------------------------------- simulator
def test_simulator_layer():
    inst = sim.Instance("obstruction2d", 2, 7)
    state, info = inst.reset()
    check("mirror: info carries object_count", info == {"object_count": 2})
    rect = inst.observation_space.get_type("rectangle")
    names = [o.name for o in state.get_objects(rect)]
    check("mirror: get_objects follows the type hierarchy", names == ["obstruction0", "obstruction1",
                                                                       "target_block", "target_surface"], names)
    robot = state.get_object_from_name("robot")
    check("mirror: get(obj, feature)", isinstance(state.get(robot, "x"), float))
    back = C.state_from_payload(C.state_to_payload(state))
    check("codec: payload round trip is exact", back.allclose(state) and all(
        np.array_equal(back.data[back.get_object_from_name(o.name)], state.data[o]) for o in state))
    for bad, why in (([0.06, 0, 0, 0, 0], "bounds"), ([0, 0, 0, 0], "shape"), ([np.nan, 0, 0, 0, 0], "nan")):
        try:
            C.validate_action(bad, inst.action_space)
            check(f"action contract rejects {why}", False)
        except C.InvalidAction:
            check(f"action contract rejects {why}", True)
    check("action contract accepts the bounds", C.validate_action(inst.action_space.high, inst.action_space)
          .dtype == np.float32)
    inst.close()
    acts = np.random.default_rng(0).uniform([-0.05] * 2 + [-0.19, -0.1, 0], [0.05] * 2 + [0.19, 0.1, 1],
                                            size=(30, 5)).astype(np.float32)
    enc = sim.encode_actions(list(acts))
    check("actions: encoding is exact", np.array_equal(sim.decode_actions(enc), acts))
    check("actions: plain lists decode", np.array_equal(sim.decode_actions(acts.tolist()), acts))
    rec = get_domain("kinder_obstruction2d").load_tasks("train")[1]
    from rewards.kinder import reward
    r = reward(enc, True, rec)
    check("verifier: random actions do not solve", r["value"] == 0.0 and r["num_steps"] == 30, r)
    check("verifier: malformed input scores 0", reward({"format": sim.ACTIONS_FORMAT, "shape": [1, 5],
                                                         "data": "!!"}, True, rec)["value"] == 0.0)
    check("verifier: execution failure scores 0", reward(enc, False, rec)["value"] == 0.0)


# ----------------------------------------------------------------------------- solver
def test_solver_path(tmp):
    dom = get_domain("kinder_motion2d")
    cls = dom.load_solver(FIXTURE)
    rows = dom.load_tasks("test")[:6]     # counts 0..5 (3 design + 2 extrapolation counts)
    outs = [dom.solve(cls, r, dom.split("test")) for r in rows]
    check("solve: scripted policy solves counts 0-5", all(o.success for o in outs),
          [(r["count"], o.extras.get("rollout_error")) for r, o in zip(rows, outs)])
    check("solve: replay agrees with the rollout", all(o.extras["replay_agrees"] for o in outs))
    from rewards.kinder import reward
    check("verifier: solver answers verify independently", all(reward(o.program, True, r)["value"] == 1.0
                                                               for r, o in zip(rows, outs)))
    truncated = dict(outs[5].program)
    acts = sim.decode_actions(truncated)[:-1]
    check("verifier: one action short does not solve", reward(acts.tolist(), True, rows[5])["value"] == 0.0)
    check("complexity: episode length", dom.complexity(outs[0].program) == outs[0].extras["rollout_num_steps"])

    rec = rows[1]
    cases = {
        "policy_error": "class GeneratedApproach:\n    def __init__(self, a, o, p): pass\n"
                        "    def reset(self, s, i): pass\n    def get_action(self, s): raise ValueError('boom')\n",
        "invalid_action": "class GeneratedApproach:\n    def __init__(self, a, o, p): pass\n"
                          "    def reset(self, s, i): pass\n    def get_action(self, s): return [1, 0, 0, 0, 0]\n",
        "timeout": "import time\nclass GeneratedApproach:\n    def __init__(self, a, o, p): pass\n"
                   "    def reset(self, s, i): time.sleep(5)\n    def get_action(self, s): return [0, 0, 0, 0, 0]\n",
        "init_error": "class GeneratedApproach:\n    def __init__(self, a, o, p): raise RuntimeError('x')\n",
    }
    from domains.base import SplitSpec
    sp = dom.split("test")
    short = SplitSpec(sp.name, sp.path, sp.label, {**sp.params, "policy_budget_s": 1.0})
    for kind, code in cases.items():
        out = dom.solve(dom.load_solver(write(os.path.join(tmp, f"{kind}.py"), code)), rec, short)
        check(f"solve: {kind} is a clean failure", not out.success and out.extras["rollout_error_kind"] == kind,
              out.extras)

    from domains.kinder.strict import StrictImportError, check_solver
    write(os.path.join(tmp, "helper_ok.py"), "import math\nX = 1\n")
    for code, allowed in (("import numpy, scipy.spatial, math, heapq\nimport helper_ok\n", True),
                          ("import kinder\n", False), ("from relational_structs import Type\n", False),
                          ("import importlib\n", False), ("x = __import__('os')\n", False),
                          ("import sys\nsys.modules['x'] = 1\n", False), ("import kinder_client\n", False),
                          ("import shapely\n", False)):
        v = check_solver(write(os.path.join(tmp, "probe.py"), code))
        check(f"strict imports: {'allows' if allowed else 'rejects'} {code.splitlines()[0][:40]!r}",
              (not v) == allowed, v)
    try:
        dom.load_solver(write(os.path.join(tmp, "bad_solver.py"), "import kinder\nclass GeneratedApproach: pass\n"))
        check("load_solver refuses a solver with forbidden imports", False)
    except StrictImportError:
        check("load_solver refuses a solver with forbidden imports", True)


# --------------------------------------------------------------------------- scripts
def test_pipeline_scripts(tmp):
    ev = os.path.join(tmp, "eval")
    r = run([PY, "scripts/eval_domain_solver.py", "--domain", "kinder_motion2d", "--split", "test", "--solver",
             FIXTURE, "--limit", "6", "--workers", "3", "--task-timeout", "300", "--output-dir", ev])
    rows = [json.loads(l) for l in open(os.path.join(ev, "test.jsonl"))] if r.returncode == 0 else []
    check("eval_domain_solver: rows for every task, all solved", len(rows) == 6 and all(x["solved"] for x in rows),
          r.stderr[-800:])
    check("eval_domain_solver: rows carry count/regime and zero tokens",
          rows and all({"count", "regime", "task_id", "answer"} <= set(x) and x["token_usage"]["input"] == 0
                       for x in rows))
    # a fake LLM baseline that fails everything, with token costs
    fb = os.path.join(tmp, "fallback.jsonl")
    with open(fb, "w") as fh:
        for i, rec in enumerate(get_domain("kinder_motion2d").load_tasks("test")[:8]):
            fh.write(json.dumps({"task_id": rec["task_id"], "task_index": rec["task_index"], "solved": False,
                                 "best_reward": 0.0, "answer": None, "count": rec["count"], "regime": rec["regime"],
                                 "token_usage": {"input": 1000, "output": 100, "reasoning": 0}}) + "\n")
    hy = os.path.join(tmp, "hybrid.jsonl")
    r = run([PY, "scripts/hybrid_gate.py", "--domain", "kinder_motion2d", "--solver", os.path.join(ev, "test.jsonl"),
             "--fallback", fb, "--out", hy])
    summary = json.loads(r.stdout[r.stdout.index("{"):]) if r.returncode == 0 else {}
    check("hybrid_gate: solver covers 6/8 at zero tokens",
          summary.get("solver_coverage") == 0.75 and summary.get("hybrid_tokens") == 2200, summary or r.stderr[-500:])
    r = run([PY, "scripts/kinder_report.py", os.path.join(ev, "test.jsonl"), hy, "--metrics-json",
             os.path.join(tmp, "m.json")])
    check("kinder_report: table and metrics", r.returncode == 0 and "extrapolation" in r.stdout
          and os.path.exists(os.path.join(tmp, "m.json")), r.stderr[-500:])


# ------------------------------------------------------------- review regressions
def test_review_regressions(tmp):
    """Defects found in review; each check fails without its fix."""
    import multiprocessing
    from domains.base import SplitSpec, wait_for_child_result
    dom = get_domain("kinder_motion2d")
    rec = dom.load_tasks("test")[1]
    sp = dom.split("test")

    # results larger than the pipe buffer must not deadlock the parent
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_put_big, args=(q,), daemon=True)
    t0 = time.time()
    proc.start()
    res, status = wait_for_child_result(proc, q, 60)
    proc.join()
    check("isolation: a 1 MB result is read without deadlock", status == "ok" and len(res) == 1_000_000
          and time.time() - t0 < 30)
    proc = ctx.Process(target=os._exit, args=(3,), daemon=True)
    proc.start()
    t0 = time.time()
    res, status = wait_for_child_result(proc, ctx.Queue(maxsize=1), 60)
    proc.join()
    check("isolation: a child that dies is reported at once", status == "died" and time.time() - t0 < 10)

    # policies using dataclasses with postponed annotations load and run
    code = ("from __future__ import annotations\nfrom dataclasses import dataclass\n\n@dataclass\nclass Waypoint:\n"
            "    x: float\n    y: float\n\n" + POLICY)
    out = dom.solve(dom.load_solver(write(os.path.join(tmp, "dc_policy.py"), code)), rec, sp)
    check("load: dataclass + `from __future__ import annotations`", out.success, out.extras)

    # the policy timeout cannot be swallowed by `except Exception`
    swallow = ("import time\nclass GeneratedApproach:\n    def __init__(self, a, o, p): pass\n"
               "    def reset(self, s, i):\n        while True:\n            try:\n                time.sleep(0.05)\n"
               "            except Exception:\n                continue\n    def get_action(self, s): return [0] * 5\n")
    short = SplitSpec(sp.name, sp.path, sp.label, {**sp.params, "policy_budget_s": 1.0})
    t0 = time.time()
    out = dom.solve(dom.load_solver(write(os.path.join(tmp, "swallow.py"), swallow)), rec, short)
    check("timeout: `except Exception` cannot swallow it", out.extras["rollout_error_kind"] == "timeout"
          and time.time() - t0 < 15, out.extras)
    unlimited = SplitSpec(sp.name, sp.path, sp.label, {**sp.params, "policy_budget_s": float("inf")})
    out = dom.solve(dom.load_solver(FIXTURE), rec, unlimited)
    check("timeout: an infinite budget disables the clock", out.success, out.extras)
    exiting = ("import sys\nclass GeneratedApproach:\n    def __init__(self, a, o, p): pass\n"
               "    def reset(self, s, i): sys.exit(3)\n    def get_action(self, s): return [0] * 5\n")
    out = dom.solve(dom.load_solver(write(os.path.join(tmp, "exiting.py"), exiting)), rec, sp)
    check("solve: sys.exit in a policy is a policy error", out.extras["rollout_error_kind"] == "policy_error")
    from domains.kinder.domain import evaluate_policy_source
    t0 = time.time()
    ev = evaluate_policy_source("kinder_motion2d", rec, "import os\nos._exit(3)\n", timeout=120)
    check("isolated evaluation: a crashing process is reported as a crash, not a timeout",
          ev["extras"].get("rollout_error_kind") == "crash" and time.time() - t0 < 30, ev["extras"])

    # static import check: sibling packages are scanned whole; kinder_client is never allowed
    from domains.kinder.strict import check_solver
    pkg = os.path.join(tmp, "pkgcase")
    write(os.path.join(pkg, "helpers", "__init__.py"), "")
    write(os.path.join(pkg, "helpers", "geom.py"), "import kinder\n")
    v = check_solver(write(os.path.join(pkg, "solver.py"), "from helpers import geom\n"))
    check("strict imports: submodules of sibling packages are checked", any("kinder" in x for x in v), v)
    ws = os.path.join(tmp, "wscase")
    write(os.path.join(ws, "kinder_client.py"), "")
    v = check_solver(write(os.path.join(ws, "solver.py"), "from kinder_client import make_env\n"))
    check("strict imports: kinder_client rejected even next to the solver", bool(v), v)
    v = check_solver(write(os.path.join(tmp, "sysmod.py"), "from sys import modules\nimport os\nos.__import__\n"))
    check("strict imports: `from sys import modules` and `x.__import__` rejected", len(v) == 2, v)

    # environment description: no false claim about static objects
    check("env description: no claim that static objects are unobserved",
          all("not of the observation" not in get_domain(f"kinder_{f}").env_card() for f in FAMILIES))


def _put_big(q):
    q.put("x" * 1_000_000)


# ---------------------------------------------------------------------------- sandbox
def test_sandbox():
    from domains.kinder.sandbox import KinderSandbox
    os.environ["VLLM_API_KEY"] = "secret-test-key"
    with KinderSandbox("motion2d", [(1, 11)], free_reset=False, timeout=60) as sb:
        r = sb.run("import kinder")
        check("sandbox: kinder is not importable", not r.ok and "not allowed" in r.text())
        r = sb.run("import importlib; importlib.import_module('gymnasium')")
        check("sandbox: importlib cannot bypass the guard", not r.ok and "not allowed" in r.text())
        r = sb.run("import os, numpy, scipy.optimize; print(os.environ.get('VLLM_API_KEY'))")
        check("sandbox: numpy/scipy import; API keys are not inherited", r.ok and r.stdout.strip() == "None",
              r.text())
        r = sb.run(f"open('approach.py', 'w').write({POLICY!r})\n"
                   "from kinder_client import make_env, load_approach, rollout\n"
                   "env = make_env(); res = rollout(load_approach('approach.py'), env)\n"
                   "print(res['solved'], res['count'], env.num_instances, env.free_reset)")
        check("sandbox: policy file persists and solves the pinned instance via the client",
              r.ok and r.stdout.split() == ["True", "1", "1", "False"], r.text())
        r = sb.run("from kinder_client import make_env\nmake_env().reset(seed=1, count=2)")
        check("sandbox: pinned sessions refuse fresh instances", not r.ok and "PermissionError" in r.text())
        r = sb.run("import time; time.sleep(30)", timeout=2)
        check("sandbox: timeout kills the run", r.timed_out and r.elapsed_s < 10)
        r = sb.run("import os; os.makedirs('kinder/renders', exist_ok=True)\nimport kinder")
        check("sandbox: a directory named like a package does not open the guard",
              not r.ok and "not allowed" in r.text(), r.text())
    os.environ.pop("VLLM_API_KEY", None)

    from domains.kinder.sandbox import BuilderSandbox
    demos = write(os.path.join(tempfile.mkdtemp(prefix="kinder_b_"), "d.json"), "[]")
    out_dir = tempfile.mkdtemp(prefix="kinder_bo_")
    with BuilderSandbox("motion2d", [(1, 11)], demos_path=demos, card_path=get_domain("kinder_motion2d").card_path(),
                        output_dir=out_dir, timeout=60) as sb:
        r = sb.run("from kinder_client import make_env\nenv = make_env()\n"
                   "a = env.reset(seed=5)[0]; b = env.reset(seed=5)[0]\nprint(env.count, a.allclose(b))")
        check("free reset: a seed fixes the count and the scene", r.ok and r.stdout.split()[-1] == "True", r.text())
        ok, out, err = sb.run_code("for i in range(60): print('progress line', i)\nraise RuntimeError('THE REAL BUG')")
        check("builder sandbox: the error leads the failure text", not ok and "THE REAL BUG" in err[:800], err[:800])

    code = ("import os, sys, time\nsys.path.insert(0, %r)\nfrom domains.kinder.sandbox import KinderSandbox\n"
            "sb = KinderSandbox('motion2d', [(1, 11)])\nsb.server.start()\nprint(sb.server._proc.pid, flush=True)\n"
            "os._exit(0)\n" % REPO_ROOT)
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120)
    pid = int(r.stdout.split()[-1]) if r.stdout.strip() else None
    gone = False
    for _ in range(20):
        try:
            os.kill(pid, 0)
            time.sleep(0.5)
        except (ProcessLookupError, TypeError):
            gone = True
            break
    check("sandbox: the simulator server exits when its parent is killed", gone, r.stderr[-300:])


# ------------------------------------------------------------------ agent end to end
def test_agent_end_to_end(tmp):
    from domains.tests.mock_llm import MockLLMServer
    look = ("from kinder_client import make_env, format_state\nenv = make_env()\nstate, info = env.reset()\n"
            "print(info, env.max_steps)\nprint(format_state(state))")
    test = (f"open('approach.py', 'w').write({POLICY!r})\nfrom kinder_client import make_env, load_approach, rollout\n"
            "res = rollout(load_approach('approach.py'), make_env())\nprint('solved', res['solved'])")
    script = [
        ("python", {"code": look, "reason": "inspect the scene"}, "The robot must cross the passages."),
        ("python", {"code": "import kinder", "reason": "check access"}),
        ("python", {"code": test, "reason": "test the waypoint policy"}),
        ("submit", {"code": "import kinder\nclass GeneratedApproach:\n    pass\n", "notes": "forbidden import"}),
        ("submit", {"code": POLICY, "notes": "waypoints through the gap centres"}),
    ]
    # second task: the model stops talking without submitting and is reminded once
    script2 = [("python", {"code": test, "reason": "test"}), (None, "The policy works, so I am done."),
               ("submit", {"code": POLICY, "notes": "after the reminder"})]
    out = os.path.join(tmp, "agent.jsonl")
    with MockLLMServer(script + script2, fallback=("submit", {"code": POLICY, "notes": "fallback"})) as srv:
        r = run([PY, "-m", "openhands_agents.run_kinder_direct_solve", "--domain", "kinder_motion2d", "--split",
                 "train", "--limit", "2", "--workers", "1", "--output-path", out, "--base-url", srv.base_url,
                 "--model", "openai/mock", "--max-steps", "10", "--debug-dir", os.path.join(tmp, "dbg")],
                timeout=900)
    rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    check("agent: one row per task", len(rows) == 2, r.stderr[-1500:])
    if not rows:
        return
    row = rows[0]
    check("agent: submitted policy solves and is verified", row["solved"] and row["best_reward"] == 1.0
          and row["rollout"].get("replay_agrees"), row.get("rollout"))
    check("agent: row schema", {"task_id", "task_index", "dataset", "answer", "token_usage", "steps_used",
                                "trajectory", "policy_code", "count", "regime"} <= set(row))
    steps = row["trajectory"]
    check("agent: five tool calls recorded", row["steps_used"] == 5 and [s["tool"] for s in steps]
          == ["python", "python", "python", "submit", "submit"], [s["tool"] for s in steps])
    check("agent: reasoning captured", steps[0]["reasoning"] == "The robot must cross the passages.")
    check("agent: observations captured", "crv_robot" in steps[0]["output"] and "not allowed" in steps[1]["output"]
          and "solved True" in steps[2]["output"], [s["output"][:200] for s in steps[:3]])
    check("agent: invalid submission rejected, agent continued", "rejected" in steps[3]["output"])
    check("agent: tokens counted", row["token_usage"]["input"] > 0)
    check("agent: checkpoint and debug files", os.path.exists(out.replace(".jsonl", ".ckpt.json"))
          and len(os.listdir(os.path.join(tmp, "dbg"))) == 2)
    check("agent: a text-only reply is followed by a reminder, then the submission counts",
          len(rows) == 2 and rows[1]["submitted"] and rows[1]["solved"] and rows[1]["nudges"] == 1
          and rows[1]["end_reason"] == "submitted" and row["end_reason"] == "submitted",
          {k: rows[-1].get(k) for k in ("submitted", "nudges", "end_reason")})
    shard_a, shard_b = os.path.join(tmp, "shard_a.jsonl"), os.path.join(tmp, "shard_b.jsonl")
    write(shard_a, json.dumps(rows[0]) + "\n")
    write(shard_b, json.dumps(rows[1]) + "\n")
    merged = os.path.join(tmp, "merged.jsonl")
    r = run([PY, "scripts/merge_direct_solve_shards.py", shard_a, shard_b, "--output", merged])
    check("shards: KinDER rows merge (string task ids)", r.returncode == 0 and
          [json.loads(l)["task_id"] for l in open(merged)] == [rows[0]["task_id"], rows[1]["task_id"]], r.stderr[-300:])
    demos = os.path.join(tmp, "demos.json")
    r = run([PY, "scripts/build_demos.py", "--domain", "kinder_motion2d", "--split", "train", "--traces", out,
             "--n", "10", "--out", demos])
    d = json.load(open(demos)) if os.path.exists(demos) else []
    first = next((x for x in d if x["task_id"] == rows[0]["task_id"]), None)
    check("demos: built from agent rows", len(d) == 2 and first is not None and first["success"]
          and first["final_response"] == POLICY and first["cot"][0]["reasoning"] and "Initial state" in first["prompt"],
          r.stderr[-800:])
    no_cot = os.path.join(tmp, "demos_no_cot.json")
    run([PY, "scripts/build_demos.py", "--domain", "kinder_motion2d", "--split", "train", "--traces", out,
         "--n", "10", "--no-cot", "--out", no_cot])
    nd = json.load(open(no_cot)) if os.path.exists(no_cot) else []
    check("demos: --no-cot drops the reasoning trace", len(nd) == 2 and all("cot" not in x for x in nd))
    return demos


# ---------------------------------------------------------------- builder end to end
def test_builder_end_to_end(tmp, demos):
    from domains.tests.mock_llm import MockLLMServer
    probe = ("import json, os\nfrom kinder_client import make_env, load_approach, evaluate, rollout\n"
             "print(sorted(f for f in os.listdir('.') if not f.startswith('.')))\n"
             "print('demos', len(json.load(open('DEMOS.json'))))\nenv = make_env()\n"
             "print('instances', env.num_instances, 'free', env.free_reset)\n"
             "res = evaluate(load_approach('SOLVER_KINDER.py'), env, limit=3, verbose=False)\n"
             "print('train solved', sum(r['solved'] for r in res))\n"
             "print('fresh solved', rollout(load_approach('SOLVER_KINDER.py'), env, seed=5, count=5)['solved'])")
    script = [
        ("sb_write_file", {"filename": "SOLVER_KINDER.py", "content": POLICY}),
        ("sb_execute_code", {"code": probe}),
        ("sb_write_file", {"filename": "SOLVER_KINDER_ALGORITHM.md", "content": "# Waypoints through gaps\n"}),
        ("sb_finish", {"summary": "done"}),
    ]
    out_dir, dbg = os.path.join(tmp, "built"), os.path.join(tmp, "sbdbg")
    with MockLLMServer(script, fallback=("sb_finish", {"summary": "done"})) as srv:
        r = run([PY, "-m", "openhands_agents.build_solver", "--domain", "kinder_motion2d", "--demos-path", demos,
                 "--rewards-dir", "rewards", "--output-dir", out_dir, "--base-url", srv.base_url,
                 "--model", "openai/mock", "--max-steps", "8", "--debug-dir", dbg], timeout=900)
        first_request = srv.calls[0] if srv.calls else {}
    check("builder: runs without an Apptainer image", r.returncode == 0, r.stderr[-1500:])
    check("builder: solver and algorithm written", os.path.exists(os.path.join(out_dir, "SOLVER_KINDER.py"))
          and os.path.exists(os.path.join(out_dir, "SOLVER_KINDER_ALGORITHM.md")))
    msgs = first_request.get("messages", [])
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    user = user if isinstance(user, str) else " ".join(c.get("text", "") for c in user)
    check("builder: prompt = file note + building prompt + environment description",
          "@DEMOS.json" in user and "GeneratedApproach" in user and "# Motion2D (variable object count)" in user)
    traj = json.load(open(os.path.join(dbg, "solver_builder_trajectory.json"))) if os.path.exists(dbg) else []
    obs = " ".join(json.dumps(e) for e in traj if e.get("kind") == "ObservationEvent")
    check("builder: sandbox serves DEMOS.json, ENV.md, the client and the written solver",
          all(s in obs for s in ("DEMOS.json", "ENV.md", "kinder_client.py", "SOLVER_KINDER.py", "demos 2")), obs[:600])
    check("builder: training instances and fresh instances reachable",
          "instances 100 free True" in obs and "train solved 3" in obs and "fresh solved True" in obs, obs[:600])


def main() -> int:
    quick = "--quick" in sys.argv
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="kinder_tests_") as tmp:
        for name, fn, args in (("task files", test_task_files, ()), ("simulator layer", test_simulator_layer, ()),
                               ("solver path", test_solver_path, (tmp,)),
                               ("review regressions", test_review_regressions, (tmp,)), ("sandbox", test_sandbox, ()),
                               ("pipeline scripts", test_pipeline_scripts, (tmp,))):
            print(name)
            fn(*args)
        if not quick:
            print("per-task agent (mock LLM)")
            demos = test_agent_end_to_end(tmp)
            print("solver builder (mock LLM)")
            if demos:
                test_builder_end_to_end(tmp, demos)
            else:
                check("builder: needs demos from the agent test", False)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed ({time.time() - t0:.0f}s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
