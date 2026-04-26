"""
Strip agent_messages (and optionally reasoning_content) from output JSONL files
to produce compact versions safe to copy off the cluster.

Usage:
    python scripts/strip_agent_messages.py outputs/foo.jsonl
    python scripts/strip_agent_messages.py outputs/foo.jsonl --out outputs/foo_stripped.jsonl
    python scripts/strip_agent_messages.py outputs/foo.jsonl --keep-reasoning
    python scripts/strip_agent_messages.py outputs/  # strip all *.jsonl in a directory
"""

import argparse
import json
from pathlib import Path


def strip_record(record: dict, keep_reasoning: bool = False) -> dict:
    """Remove agent_messages; optionally strip reasoning_content inside token_usage too."""
    out = {k: v for k, v in record.items() if k != "agent_messages"}
    if not keep_reasoning:
        # token_usage may carry a top-level reasoning field — keep it (it's just a count)
        pass
    return out


def strip_file(src: Path, dst: Path, keep_reasoning: bool) -> int:
    lines_written = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            fout.write(json.dumps(strip_record(record, keep_reasoning)) + "\n")
            lines_written += 1
    return lines_written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="JSONL file or directory of JSONL files")
    parser.add_argument("--out", default="",
                        help="Output path (file or dir). Defaults to <input>_stripped.jsonl "
                             "or <dir>_stripped/")
    parser.add_argument("--keep-reasoning", action="store_true",
                        help="Keep reasoning_content inside agent_messages responses")
    args = parser.parse_args()

    src = Path(args.input)

    if src.is_dir():
        files = sorted(src.glob("*.jsonl"))
        out_dir = Path(args.out) if args.out else src.parent / (src.name + "_stripped")
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            dst = out_dir / f.name
            n = strip_file(f, dst, args.keep_reasoning)
            print(f"  {f.name} -> {dst.name}  ({n} records)")
        print(f"\nWrote {len(files)} files to {out_dir}/")
    else:
        if args.out:
            dst = Path(args.out)
        else:
            dst = src.with_name(src.stem + "_stripped" + src.suffix)
        dst.parent.mkdir(parents=True, exist_ok=True)
        n = strip_file(src, dst, args.keep_reasoning)
        orig_mb = src.stat().st_size / 1e6
        new_mb = dst.stat().st_size / 1e6
        print(f"{src.name} -> {dst.name}")
        print(f"  {n} records  |  {orig_mb:.1f} MB -> {new_mb:.1f} MB  "
              f"({100*(1-new_mb/orig_mb):.0f}% reduction)")


if __name__ == "__main__":
    main()
