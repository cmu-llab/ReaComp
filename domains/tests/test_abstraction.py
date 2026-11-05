"""
Regression tests for the domain abstraction (no LLM, no Docker).

    .venv/bin/python -m domains.tests.test_abstraction

Checks that the generic layer reproduces the legacy behaviour exactly:
  * List Functions demo builder == demos/DEMOS_LISTFN_balanced.json (byte-for-byte content)
  * PBEBench / SLR-Bench demo schemas == the schemas of the paper's demo files
  * solver-builder and direct-solve prompts obtained through the domain == the legacy constants
  * the OpenHands controllers compose identical prompts with --domain and with --solver-type
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from domains import get_domain  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + detail) if (detail and not cond) else ''}")


def test_listfn_demos_reproduce_file():
    dom = get_domain("list_functions")
    demos = dom.build_demos(["outputs/listfn_full_bok_traces.jsonl"], "full", n=100, seed=42, per_task=4)
    ref = json.load(open("demos/DEMOS_LISTFN_balanced.json"))
    check("list_functions: 100 demos built", len(demos) == 100, str(len(demos)))
    check("list_functions: identical to DEMOS_LISTFN_balanced.json", demos == ref)


def test_pbebench_demo_schema():
    dom = get_domain("pbebench")
    trace = json.load(open("EXAMPLE_RECORD.json"))
    recs = {r["task_index"]: r for r in dom.load_tasks("lite")}
    demo = dom.demo_from_trace(recs[trace["task_index"]], trace, dom.split("lite"))
    ref_keys = list(json.load(open("demos/DEMOS_PBEBENCH_seed_42_12_examples_with_CoT.json"))[0].keys())
    check("pbebench: demo keys == paper demo keys", list(demo.keys()) == ref_keys, f"{list(demo.keys())} vs {ref_keys}")
    check("pbebench: cot from reasoning_content", demo["cot"].startswith("We need to find"))
    check("pbebench: final_response is the model's message", demo["final_response"].startswith("```python"))
    check("pbebench: lite -> easy", demo["difficulty"] == "easy")
    check("pbebench: success from best_reward", demo["success"] == (trace["best_reward"] >= 1.0))


def test_slr_demo_schema():
    dom = get_domain("slr_bench")
    rec = dom.load_tasks("slr", limit=1)[0]
    trace = {"task_index": 0, "best_reward": 1.0, "answer": "eastbound(T) :- has_car(T,C), car_len(C, short).",
             "agent_messages": [{"response": {"content": "```prolog\neastbound(T) :- has_car(T,C).\n```",
                                              "reasoning_content": "think"}}]}
    demo = dom.demo_from_trace(rec, trace, dom.split("slr"))
    ref_keys = list(json.load(open("demos/DEMOS_SLRBENCH_seed_42_92_examples_with_CoT.json"))[0].keys())
    check("slr_bench: demo keys == paper demo keys", list(demo.keys()) == ref_keys, f"{list(demo.keys())} vs {ref_keys}")
    check("slr_bench: examples parsed from prompt", len(demo["input_examples"]) == len(demo["output_examples"]) > 0)
    check("slr_bench: difficulty is the curriculum tier", demo["difficulty"] == rec["curriculum tier"])


def test_specs_match_legacy():
    from openhands_agents.solver_builder import controller as sb
    from openhands_agents.solver_builder import listfn_controller as lf
    from openhands_agents.direct_solve import controller as ds
    pbe, slr, lfn = get_domain("pbebench"), get_domain("slr_bench"), get_domain("list_functions")
    s = pbe.solver_builder_spec()
    check("builder spec pbebench == legacy prompt", s["system_prompt"] == sb._SYSTEM_PROMPT_PBE and s["file_note"] == sb._FILE_NOTE_PBE)
    s = slr.solver_builder_spec()
    check("builder spec slr_bench == legacy prompt", s["system_prompt"] == sb._SYSTEM_PROMPT_SLR and s["file_note"] == sb._FILE_NOTE_SLR)
    s = lfn.solver_builder_spec()
    check("builder spec list_functions == legacy prompt", s["system_prompt"] == lf._SYSTEM_PROMPT_LISTFN and s["file_note"] == lf._FILE_NOTE)
    d = pbe.direct_solve_spec(pbe.split("lite"))
    rec = pbe.load_tasks("lite", limit=1)[0]
    check("direct-solve spec pbebench == legacy", d["system_prompt"] == ds._make_system_prompt_pbe(5)
          and d["task_message"](rec) == ds._build_task_message_pbe(rec))
    d = slr.direct_solve_spec()
    rec = slr.load_tasks("slr", limit=1)[0]
    check("direct-solve spec slr_bench == legacy", d["system_prompt"] == ds._SYSTEM_PROMPT_SLR
          and d["task_message"](rec) == ds._build_task_message_slr(rec))
    g = get_domain("envsetup").solver_builder_spec()
    check("builder spec envsetup names its files", g["solver_filename"] == "SOLVER_ENVSETUP.py" and "SOLVER_ENVSETUP.py" in g["system_prompt"])


def test_controllers_compose_identically():
    from openhands_agents.solver_builder.controller import SolverBuilderController
    from openhands_agents.direct_solve.controller import DirectSolveController
    tmp = tempfile.mkdtemp()
    common = dict(base_url="http://localhost:1/v1", model="openai/x", sandbox=None, demos_path="demos/x.json",
                  rewards_dir="rewards", output_dir=tmp)
    for dname, stype in (("pbebench", "pbe"), ("slr_bench", "slr")):
        legacy = SolverBuilderController(solver_type=stype, **common)
        viadom = SolverBuilderController(domain=get_domain(dname), **common)
        same_sys = open(legacy._system_prompt_file).read() == open(viadom._system_prompt_file).read()
        same_msg = legacy.user_message("SPEC") == viadom.user_message("SPEC")
        same_files = legacy._output_filenames() == viadom._output_filenames()
        check(f"SolverBuilderController --domain {dname} == --solver-type {stype}", same_sys and same_msg and same_files)
    for dname, stype, split in (("pbebench", "pbe", "lite"), ("slr_bench", "slr", "slr")):
        dom = get_domain(dname)
        legacy = DirectSolveController("http://localhost:1/v1", "openai/x", None, "rewards", task_type=stype, max_programs=5)
        viadom = DirectSolveController("http://localhost:1/v1", "openai/x", None, "rewards", domain=dom,
                                       split=dom.split(split), max_programs=5)
        rec = dom.load_tasks(split, limit=1)[0]
        same_sys = open(legacy._system_prompt_files[stype]).read() == open(viadom._system_prompt_files["domain"]).read()
        same_msg = viadom._domain_spec["task_message"](rec) == (
            __import__("openhands_agents.direct_solve.controller", fromlist=["x"])._build_task_message_slr(rec)
            if stype == "slr" else
            __import__("openhands_agents.direct_solve.controller", fromlist=["x"])._build_task_message_pbe(rec))
        check(f"DirectSolveController --domain {dname} == task_type {stype}", same_sys and same_msg)


def test_envsetup_demo_from_agent_row():
    dom = get_domain("envsetup")
    rec = next(r for r in dom.load_tasks("setupbench_dev") if r["task_id"] == "benjaminp-six-c1b416f")
    row = {"task_id": rec["task_id"], "solved": True, "answer": "ls\npip3 install -e .",
           "trajectory": [{"reason": "look", "command": "ls", "exit_code": 0, "output": "setup.py"},
                          {"reason": "install", "command": "pip3 install -e .", "exit_code": 0, "output": "ok"}]}
    d = dom.demo_from_trace(rec, row)
    check("envsetup: demo has shared keys", all(k in d for k in ("prompt", "cot", "final_response", "success")))
    check("envsetup: prompt carries repository snapshot", "Repository listing" in d["prompt"] and "setup.py" in d["prompt"])
    check("envsetup: group key is (category, success)", dom.demo_group_key(d) == ("reposetup", True))


def main() -> int:
    for fn in (test_listfn_demos_reproduce_file, test_pbebench_demo_schema, test_slr_demo_schema,
               test_specs_match_legacy, test_controllers_compose_identically, test_envsetup_demo_from_agent_row):
        print(fn.__name__)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check(f"{fn.__name__} raised {type(exc).__name__}", False)
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
