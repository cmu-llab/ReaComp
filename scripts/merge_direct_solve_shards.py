"""
Merge two or more DirectSolve shard outputs (.jsonl + .ckpt.json) into a
single combined file.

Usage:
    python scripts/merge_direct_solve_shards.py \
        outputs/lite_direct_solve_openhands_0_504.jsonl \
        outputs/lite_direct_solve_openhands_504_1008.jsonl \
        --output outputs/lite_direct_solve_openhands.jsonl

The script:
  - Concatenates all JSONL rows, deduplicating by task_id (last shard wins).
  - Sorts rows by task_index (falling back to task_id) for consistent ordering.
  - Merges the completed_ids sets from all shard checkpoints into a single
    combined .ckpt.json next to the output file.
"""
import argparse
import json
import os
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ckpt(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def ckpt_path_for(jsonl_path: str) -> str:
    return jsonl_path + ".ckpt.json"


def sort_key(row: dict) -> int:
    if "task_index" in row:
        return int(row["task_index"])
    return int(row.get("task_id", 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", help="Shard .jsonl files to merge")
    parser.add_argument("--output", required=True, help="Output .jsonl path")
    args = parser.parse_args()

    merged: dict[int, dict] = {}
    all_completed_ids: set = set()

    for shard_path in args.shards:
        rows = load_jsonl(shard_path)
        for row in rows:
            key = int(row.get("task_id", row.get("task_index", 0)))
            merged[key] = row

        ckpt = load_ckpt(ckpt_path_for(shard_path))
        all_completed_ids.update(ckpt.get("completed_ids", []))

    sorted_rows = sorted(merged.values(), key=sort_key)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for row in sorted_rows:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"Wrote {len(sorted_rows)} rows → {args.output}")

    out_ckpt = ckpt_path_for(args.output)
    with open(out_ckpt, "w") as f:
        json.dump({"completed_ids": sorted(all_completed_ids)}, f, indent=2)
    print(f"Wrote checkpoint ({len(all_completed_ids)} completed ids) → {out_ckpt}")


if __name__ == "__main__":
    main()
