"""
Convert data/real_forward_reconstruction/*.csv into a single JSONL file
compatible with quick_eval.py / eval_solver.py.

Each CSV has two columns: (input, output) or (proto_ipa, attested_ipa).
Each row is one word-pair example for that language pair.
One JSONL record = one language pair task.

Output fields (quick_eval-compatible):
  task_index  int
  language    str   e.g. "Proto-Malayo-Polynesian-Papuma"
  source_lang str   e.g. "Proto-Malayo-Polynesian"
  target_lang str   e.g. "Papuma"
  inputs      list[str]
  outputs     list[str]
  n_examples  int

Usage:
    python scripts/convert_real_forward_reconstruction.py
    python scripts/convert_real_forward_reconstruction.py --out data/real_forward_reconstruction.jsonl
    python scripts/convert_real_forward_reconstruction.py --min-examples 3
"""
import argparse
import csv
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IN_DIR = os.path.join(REPO_ROOT, "data", "real_forward_reconstruction")
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "real_forward_reconstruction.jsonl")

INPUT_COLS = {"input", "proto_ipa"}
OUTPUT_COLS = {"output", "attested_ipa"}


def parse_language_name(stem: str):
    """Split 'Proto-X-Y-Z-Language' into source='Proto-X-Y-Z', target='Language'."""
    parts = stem.split("-")
    # Find the last token that starts uppercase and is alphabetically a language name.
    # Convention: filename = <proto-group>-<daughter-language>, where proto-group may
    # itself contain hyphens (e.g. "Proto-Central-Eastern Malayo-Polynesian").
    # The daughter language is the last hyphen-separated token (may contain spaces).
    # We split on the LAST hyphen that separates group from daughter.
    # Heuristic: the group always starts with "Proto-", the daughter is everything after
    # the last "-" that follows a capital-letter token.
    idx = stem.rfind("-")
    if idx == -1:
        return stem, ""
    return stem[:idx], stem[idx + 1 :]


def load_csv(path: str):
    inputs, outputs = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        in_col = out_col = None
        for col in (reader.fieldnames or []):
            if col in INPUT_COLS:
                in_col = col
            if col in OUTPUT_COLS:
                out_col = col
        if in_col is None or out_col is None:
            return [], []
        for row in reader:
            inp = (row.get(in_col) or "").strip()
            out = (row.get(out_col) or "").strip()
            if inp and out:
                inputs.append(inp)
                outputs.append(out)
    return inputs, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default=DEFAULT_IN_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--min-examples", type=int, default=1,
        help="Skip tasks with fewer than this many word pairs (default: 1)",
    )
    args = parser.parse_args()

    csv_files = sorted(
        f for f in os.listdir(args.in_dir) if f.endswith(".csv")
    )

    written = skipped = 0
    with open(args.out, "w", encoding="utf-8") as out_fh:
        for task_index, fn in enumerate(csv_files):
            stem = fn[:-4]  # strip .csv
            source_lang, target_lang = parse_language_name(stem)
            path = os.path.join(args.in_dir, fn)
            inputs, outputs = load_csv(path)
            if len(inputs) < args.min_examples:
                skipped += 1
                continue
            record = {
                "task_index": task_index,
                "language": stem,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "inputs": inputs,
                "outputs": outputs,
                "n_examples": len(inputs),
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written {written} tasks to {args.out}  (skipped {skipped})")
    # Summary stats
    import statistics
    with open(args.out) as f:
        records = [json.loads(l) for l in f]
    sizes = [r["n_examples"] for r in records]
    print(f"Examples per task: min={min(sizes)}, max={max(sizes)}, "
          f"median={statistics.median(sizes):.0f}, mean={statistics.mean(sizes):.1f}")


if __name__ == "__main__":
    main()
