import os
import sys
import pathlib
import datasets

sys.path.append(pathlib.Path(str(os.path.abspath(__file__))).parent)

from utils import write_jsonl

def load_pbebench_lite(path: str="changelinglab/PBEBench-Lite", first_k: int=-1):
    """Load the PBEBench-Lite dataset from Hugging Face. If `first_k` is set to a positive integer, only the first `k` records will be returned."""
    ds = datasets.load_dataset(path)['test']
    ds = [rec for rec in ds]
    if first_k > 0: ds = ds[:first_k]

    return ds

# main
if __name__ == "__main__":
    data = load_pbebench_lite(first_k=50)
    write_path = "data/pbebench/lite_pilot_tasks.jsonl"
    if not os.path.exists(write_path):
        write_jsonl(data, write_path)