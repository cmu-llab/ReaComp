"""
Extract agent_messages (reasoning traces) for unsolved tasks and optionally
upload the result to Google Drive via rclone.

Usage:
    python scripts/export_traces.py
    python scripts/export_traces.py --upload
    python scripts/export_traces.py --metrics metrics/pbebench_lite_best_of_k.json \
                                     --outputs outputs/lite_tasks_full_og_best_of_k.jsonl \
                                     --out failure_analysis/best_of_k_traces.jsonl \
                                     --gdrive-dir PBETrain/traces \
                                     --upload
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ── defaults (edit these for each run) ────────────────────────────────────────
DEFAULT_METRICS = "metrics/pbebench_lite_direct_feedback.json"
DEFAULT_OUTPUTS = "outputs/lite_tasks_full_og_direct_feedback.jsonl"
DEFAULT_OUT     = "failure_analysis/pbebench_lite_direct_feedback_traces.jsonl"
DEFAULT_GDRIVE  = "PBETrain/traces"   # path inside your Google Drive
RCLONE_REMOTE   = "gdrive"            # rclone remote name (from rclone config)
# ──────────────────────────────────────────────────────────────────────────────


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics",    default=DEFAULT_METRICS, metavar="FILE",
                        help=f"Metrics JSON with unsolved list (default: {DEFAULT_METRICS})")
    parser.add_argument("--outputs",    default=DEFAULT_OUTPUTS, metavar="FILE",
                        help=f"Output JSONL from the run (default: {DEFAULT_OUTPUTS})")
    parser.add_argument("--out",        default=DEFAULT_OUT,     metavar="FILE",
                        help=f"Where to write the traces JSONL (default: {DEFAULT_OUT})")
    parser.add_argument("--gdrive-dir", default=DEFAULT_GDRIVE,  metavar="DIR",
                        help=f"Google Drive destination folder (default: {DEFAULT_GDRIVE})")
    parser.add_argument("--upload",     action="store_true",
                        help="Upload the output file to Google Drive via rclone after writing")
    args = parser.parse_args()

    # Load unsolved task indices from metrics
    metrics = json.load(open(args.metrics))
    unsolved_ids = {rec["task_index"] for rec in metrics["unsolved"]}
    print(f"Unsolved tasks to extract: {len(unsolved_ids)}")

    # Stream through outputs and collect matching records, keeping only
    # task_index, best_reward, reward_history, and agent_messages
    records = []
    for rec in read_jsonl(args.outputs):
        if rec.get("task_index") not in unsolved_ids:
            continue
        records.append({
            "task_index":     rec.get("task_index"),
            "best_reward":    rec.get("best_reward"),
            "reward_history": rec.get("reward_history", []),
            "agent_messages": rec.get("agent_messages", []),
        })

    records.sort(key=lambda r: r["task_index"])
    write_jsonl(records, args.out)
    size_mb = Path(args.out).stat().st_size / 1024 / 1024
    print(f"Wrote {len(records)} traces → {args.out}  ({size_mb:.1f} MB)")

    if args.upload:
        dest = f"{RCLONE_REMOTE}:{args.gdrive_dir}/"
        cmd = ["rclone", "copy", args.out, dest, "--progress"]
        print(f"Uploading to {dest} ...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("rclone upload failed.", file=sys.stderr)
            sys.exit(1)
        print("Upload complete.")


if __name__ == "__main__":
    main()
