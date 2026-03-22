"""
Interleaved dataset generator.

Produces a single JSONL file that alternates records from multiple source
datasets in round-robin order (source 0, source 1, source 0, source 1, ...).
Each record already carries the correct "reward" field from its source
generator, so the interleaved file works directly as a --tasks-file input.

Usage (from repo root):
    python data_generation/interleaved_generator.py

Outputs:
    data/interleaved/pbebench_rg_string_pilot.jsonl
        50 pbebench-lite records interleaved with 50 reasoning_gym
        string_manipulation records (100 total, alternating).
"""

import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(os.path.abspath(__file__)).parent))

from utils import write_jsonl
from pbebench_generator import load_pbebench_lite
from reasoning_gym_generator import build_rg_split


def interleave(sources: list[list]) -> list:
    """
    Round-robin interleave any number of lists.

    Example: interleave([[A, B], [X, Y]]) → [A, X, B, Y]

    If lists have unequal lengths, the shorter ones are exhausted first
    and the remaining records from longer lists are appended at the end
    in their original round-robin order.
    """
    result = []
    iters = [iter(s) for s in sources]
    active = list(range(len(iters)))
    while active:
        next_active = []
        for idx in active:
            try:
                result.append(next(iters[idx]))
                next_active.append(idx)
            except StopIteration:
                pass
        active = next_active
    return result


def create_pbebench_rg_string_pilot(
    n: int = 50,
    seed: int = 42,
    out_path: str = "data/interleaved/pbebench_rg_string_pilot.jsonl",
    overwrite: bool = False,
) -> list:
    """
    Build a pilot interleaved dataset with:
      - n records from PBEBench-Lite      (reward="pbebench")
      - n records from reasoning_gym's    (reward="reasoning_gym")
        string_manipulation task

    Returns the interleaved list (and writes to out_path unless it already
    exists and overwrite=False).
    """
    pbebench_records = load_pbebench_lite(first_k=n)

    rg_records, _ = build_rg_split(
        N=n,
        f=0,
        seed=seed,
        datasets=["string_manipulation"],
        shuffle=False,
    )
    rg_records = rg_records[:n]

    interleaved = interleave([pbebench_records, rg_records])

    if not os.path.exists(out_path) or overwrite:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_jsonl(interleaved, out_path)
        print(f"Wrote {len(interleaved)} records → {out_path}")
    else:
        print(f"File already exists, skipping: {out_path}")

    return interleaved


if __name__ == "__main__":
    create_pbebench_rg_string_pilot()
