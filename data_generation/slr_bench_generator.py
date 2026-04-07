import os
import sys
import pathlib
import datasets

sys.path.append(pathlib.Path(str(os.path.abspath(__file__))).parent)

from utils import write_jsonl

def load_slr_bench(path: str="AIML-TUDA/SLR-Bench", subset: str="v1-All", first_k: int=-1):
    """Load the PBEBench-Lite dataset from Hugging Face. If `first_k` is set to a positive integer, only the first `k` records will be returned."""
    ds = datasets.load_dataset(path, subset)['test']
    sel_data = []
    for rec in ds:
        rec = dict(rec)
        rec['reward'] = "slr_bench"
        sel_data.append(rec)

    if first_k > 0: sel_data = sel_data[:first_k]

    return sel_data

# main
if __name__ == "__main__":
    data = load_slr_bench(first_k=-1)
    write_path = "data/slr_bench/v1_All_full.jsonl"
    write_jsonl(data, write_path)