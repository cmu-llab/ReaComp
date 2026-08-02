#!/usr/bin/env python3
"""Post-hoc analysis of a TroVE run JSONL output.

Reads the per-task JSONL file produced by main.py --output-file and reports:
  - Overall accuracy
  - Final toolbox size
  - Per-mode wins
  - IMPORT-mode tool-use breakdown
  - Top-10 most-called toolbox functions

Usage:
    python scripts/analyze_trove_run.py path/to/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: line {lineno} is not valid JSON: {exc}", file=sys.stderr)
    return rows


def _result_dict(row: dict) -> dict:
    """Tolerant accessor: results are nested under 'result' in main.py's output."""
    return row.get("result") or row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the TroVE results JSONL file")
    args = parser.parse_args()

    rows = _load_rows(args.path)
    if not rows:
        print("ERROR: no rows loaded", file=sys.stderr)
        sys.exit(1)

    n = len(rows)
    results = [_result_dict(r) for r in rows]

    solved = sum(1 for r in results if r.get("solved"))
    print(f"=== Run summary: {args.path.name} ===")
    print(f"Tasks: {n}")
    print(f"Solved: {solved}/{n} ({100 * solved / n:.1f}%)")

    last_snapshot = results[-1].get("library_snapshot") or []
    print(f"Final toolbox size: {len(last_snapshot)}")

    mode_counter = Counter(r.get("won_mode", "?") for r in results)
    print(f"Mode wins: {dict(mode_counter)}")

    import_eligible = [r for r in results if r.get("import_eligible")]
    if not import_eligible:
        print("No IMPORT-eligible tasks observed.")
    else:
        with_calls = [r for r in import_eligible if (r.get("tool_call_count") or 0) >= 1]
        n_eligible = len(import_eligible)
        n_with = len(with_calls)
        mean_calls = (
            sum((r.get("tool_call_count") or 0) for r in import_eligible) / n_eligible
        )
        all_calls = [tc for r in import_eligible for tc in (r.get("tool_calls") or [])]
        n_calls_total = len(all_calls)
        n_calls_ok = sum(1 for tc in all_calls if tc.get("ok"))
        success_rate = (100 * n_calls_ok / n_calls_total) if n_calls_total else 0.0
        print(
            f"IMPORT-eligible tasks: {n_eligible}\n"
            f"  Tasks with >=1 tool call: {n_with}/{n_eligible} ({100 * n_with / n_eligible:.1f}%)\n"
            f"  Mean tool calls / task:   {mean_calls:.2f}\n"
            f"  Tool-call success rate:   {n_calls_ok}/{n_calls_total} ({success_rate:.1f}%)"
        )

    name_counter: Counter = Counter()
    for r in results:
        for tc in r.get("tool_calls") or []:
            name = (tc.get("name") or "").split("<|", 1)[0].strip()
            if name:
                name_counter[name] += 1
    if name_counter:
        print("Top-10 most-called toolbox functions:")
        for name, cnt in name_counter.most_common(10):
            print(f"  {cnt:4d}  {name}")
    else:
        print("No tool calls recorded in this run.")


if __name__ == "__main__":
    main()
