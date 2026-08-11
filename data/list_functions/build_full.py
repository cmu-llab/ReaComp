#!/usr/bin/env python3
"""Build the FULL 250-task List Functions set in ReaComp record format.

Additive: same format and 24-shown/8-held-out split as build_pilot.py, but over all
raw/*.json tasks instead of the 10 hardcoded picks. No ground-truth text is written
into the tasks file (targets are analysis-only, withheld from solver and model).
"""
import glob
import json
import os

RAW = os.path.join(os.path.dirname(__file__), "raw")
OUT_DIR = os.path.dirname(__file__)
N_SHOWN = 24  # of 32


def parse_list(s):
    return json.loads(s)


def main():
    tasks, targets = [], {}
    files = sorted(glob.glob(os.path.join(RAW, "c*.json")))
    skipped = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        d = json.load(open(path))
        ex = d.get("examples", [])
        if len(ex) < N_SHOWN + 1:
            skipped.append((name, len(ex)))
            continue
        inputs = [parse_list(e["input"]) for e in ex]
        outputs = [parse_list(e["target"]) for e in ex]
        tasks.append({
            "task_id": name,
            "inputs": inputs[:N_SHOWN],
            "outputs": outputs[:N_SHOWN],
            "held_in_inputs": inputs[:N_SHOWN],
            "held_in_outputs": outputs[:N_SHOWN],
            "held_out_inputs": inputs[N_SHOWN:],
            "held_out_outputs": outputs[N_SHOWN:],
            "reward": "list_functions",
        })
        desc = d.get("description", "")
        targets[name] = desc.split("target function is")[-1].strip().strip('."')

    with open(os.path.join(OUT_DIR, "full_tasks.jsonl"), "w") as f:
        for rec in tasks:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(OUT_DIR, "full_targets.json"), "w") as f:
        json.dump(targets, f, indent=2)
    print(f"wrote {len(tasks)} tasks -> full_tasks.jsonl (skipped {len(skipped)}: {skipped[:5]})")


if __name__ == "__main__":
    main()
