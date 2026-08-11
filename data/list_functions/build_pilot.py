#!/usr/bin/env python3
"""Build a 10-task List Functions pilot set in ReaComp record format.

New-files-only: does not touch the PBEBench/SLR pipeline. Reads the raw BIG-bench
task.json files in raw/ and emits:
  - pilot_tasks.jsonl        : one record per task, keys {task_id, inputs, outputs,
                               held_in_inputs, held_in_outputs, held_out_inputs,
                               held_out_outputs, reward}. NO ground-truth text.
  - pilot_targets.json       : analysis-only map task_id -> natural-language target
                               (WITHHELD from the solver / model).

Each raw task has 32 example pairs of stringified integer lists. We split 24 shown /
8 held-out so "solved" means the induced program generalizes, not memorizes.
"""
import json
import os

RAW = os.path.join(os.path.dirname(__file__), "raw")
OUT_DIR = os.path.dirname(__file__)

PICKS = ["c006", "c020", "c025", "c037", "c058",
         "c099", "c110", "c150", "c199", "c245"]

N_SHOWN = 24  # of 32


def parse_list(s):
    return json.loads(s)


def main():
    tasks = []
    targets = {}
    for name in PICKS:
        d = json.load(open(os.path.join(RAW, name + ".json")))
        ex = d["examples"]
        assert len(ex) == 32, (name, len(ex))
        inputs = [parse_list(e["input"]) for e in ex]
        outputs = [parse_list(e["target"]) for e in ex]
        rec = {
            "task_id": name,
            # full set (for solvers that want all shown pairs as I/O examples)
            "inputs": inputs[:N_SHOWN],
            "outputs": outputs[:N_SHOWN],
            "held_in_inputs": inputs[:N_SHOWN],
            "held_in_outputs": outputs[:N_SHOWN],
            "held_out_inputs": inputs[N_SHOWN:],
            "held_out_outputs": outputs[N_SHOWN:],
            "reward": "list_functions",
        }
        tasks.append(rec)
        # analysis only, never shown to the model
        targets[name] = d["description"].split("target function is")[-1].strip().strip('."')

    with open(os.path.join(OUT_DIR, "pilot_tasks.jsonl"), "w") as f:
        for rec in tasks:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(OUT_DIR, "pilot_targets.json"), "w") as f:
        json.dump(targets, f, indent=2)

    print("wrote", len(tasks), "tasks ->", os.path.join(OUT_DIR, "pilot_tasks.jsonl"))
    print("shown pairs/task:", N_SHOWN, "held-out/task:", 32 - N_SHOWN)


if __name__ == "__main__":
    main()
