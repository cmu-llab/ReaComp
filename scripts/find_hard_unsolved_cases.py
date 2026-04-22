import os
import json

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]
    
def write_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for rec in data:
            f.write(json.dumps(rec) + "\n")

# main
if __name__ == "__main__":
    metrics = json.load(open("metrics/pbebench_lite_direct_feedback.json", "r"))
    outputs = read_jsonl("outputs/lite_tasks_full_og_direct_feedback.jsonl")
    unsolved_cases = []
    unsolved_indices = [rec['task_index'] for rec in metrics["unsolved"]]
    for rec in outputs:
        if rec['task_index'] in unsolved_indices:
            unsolved_cases.append(rec)
    write_jsonl(unsolved_cases, "failure_analysis/pbebench_lite_direct_feedback.jsonl")