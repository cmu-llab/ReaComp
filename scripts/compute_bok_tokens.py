"""
Estimate token usage for BoK-32 outputs from the cluster file format.

The cluster file has the same schema as outputs/gpt_oss_120b_pbebench_outputs.jsonl
but includes a `chains_of_thought` key (list of K CoT strings, one per sample).

Per task, per sample:
  - input tokens    = tokens(prompt)          [same for all K samples]
  - output tokens   = tokens(answer_string)
  - reasoning tokens = tokens(cot_string)     [only if chains_of_thought present]

Token counting priority:
  1. --tokenizer HF_MODEL_NAME  — loads via transformers (exact for that model)
  2. tiktoken cl100k_base        — if tiktoken is installed (~GPT-4 tokenizer)
  3. char/4 approximation        — fallback

Join key for --solver: I/O example pairs (frozenset of (input, output) tuples),
which gives a perfect 1216/1216 match between the BoK file and eval_solver.py output.

Usage:
    # Cluster file with CoT:
    python scripts/compute_bok_tokens.py PATH/TO/cluster_bok.jsonl

    # Local file (no CoT — estimates input+output only):
    python scripts/compute_bok_tokens.py outputs/gpt_oss_120b_pbebench_outputs.jsonl

    # Use exact HuggingFace tokenizer (recommended on cluster):
    python scripts/compute_bok_tokens.py cluster_bok.jsonl --tokenizer Qwen/Qwen2.5-72B-Instruct

    # Effi mode: zero LLM tokens for tasks solved by the symbolic solver
    python scripts/compute_bok_tokens.py cluster_bok.jsonl \\
        --solver evals/solver_results/claude_code/Thu_Apr_23_807_PM/hard.jsonl

    # Write per-task JSONL + aggregate summary:
    python scripts/compute_bok_tokens.py cluster_bok.jsonl \\
        --out outputs/bok_tokens.jsonl --metrics-json metrics/bok_hard_tokens.json
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


def _make_counter(tokenizer_name: str):
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            method = f"transformers({tokenizer_name})"
            def count(text: str) -> int:
                return len(tok.encode(text, add_special_tokens=False))
            return count, method
        except Exception as e:
            print(f"Warning: could not load tokenizer '{tokenizer_name}': {e}", file=sys.stderr)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        method = "tiktoken(cl100k_base)"
        def count(text: str) -> int:
            return len(enc.encode(text))
        return count, method
    except ImportError:
        pass

    method = "char/4 approximation"
    def count(text: str) -> int:
        return max(1, len(text) // 4)
    return count, method


def _io_key(inputs, outputs) -> frozenset:
    return frozenset(zip(inputs, outputs))


def load_solver(path: str) -> dict[frozenset, float]:
    """Return {io_key: best_reward} joined on I/O example pairs."""
    rewards: dict[frozenset, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            r = rec.get("best_reward")
            if r is None:
                r = 1.0 if (rec.get("solved") or rec.get("success")) else 0.0
            key = _io_key(rec["inputs"], rec["outputs"])
            rewards[key] = float(r)
    return rewards


def compute(path: str, count, solver_rewards: dict | None = None) -> tuple[list[dict], dict]:
    records = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for i, line in enumerate(tqdm(lines, desc="computing tokens", unit="task")):
        rec = json.loads(line)
        inp = rec.get("input") or {}
        prompt = inp.get("prompt") or ""
        outputs = rec.get("outputs") or []
        cots = rec.get("chains_of_thought") or []
        task_index = inp.get("index", i)

        prompt_toks = count(prompt)
        K = len(outputs)

        input_toks = K * prompt_toks
        output_toks = sum(count(str(o)) for o in outputs)
        reasoning_toks = sum(count(str(c)) for c in cots) if cots else 0

        # Effi: zero cost if symbolic solver solved this task perfectly
        effi_input = effi_output = effi_reasoning = 0
        sym_solved = False
        if solver_rewards is not None:
            key = _io_key(inp.get("inputs", []), inp.get("outputs", []))
            sym_reward = solver_rewards.get(key, 0.0)
            sym_solved = sym_reward >= 1.0
            if not sym_solved:
                effi_input, effi_output, effi_reasoning = input_toks, output_toks, reasoning_toks
        else:
            effi_input, effi_output, effi_reasoning = input_toks, output_toks, reasoning_toks

        records.append({
            "task_index": task_index,
            "K": K,
            "prompt_tokens": prompt_toks,
            "input_tokens": input_toks,
            "output_tokens": output_toks,
            "reasoning_tokens": reasoning_toks,
            "total_tokens": input_toks + output_toks + reasoning_toks,
            "sym_solved": sym_solved,
            "effi_input_tokens": effi_input,
            "effi_output_tokens": effi_output,
            "effi_reasoning_tokens": effi_reasoning,
            "effi_total_tokens": effi_input + effi_output + effi_reasoning,
        })

    n = len(records)
    has_cot = any(r["reasoning_tokens"] > 0 for r in records)

    def _agg(key):
        total = sum(r[key] for r in records)
        return {"total": total, "avg": round(total / n, 1) if n else 0}

    summary = {
        "n_tasks": n,
        "K": records[0]["K"] if records else 0,
        "has_cot": has_cot,
        "input":     _agg("input_tokens"),
        "output":    _agg("output_tokens"),
        "reasoning": _agg("reasoning_tokens"),
        "total":     _agg("total_tokens"),
    }
    if solver_rewards is not None:
        n_sym = sum(1 for r in records if r["sym_solved"])
        summary["effi"] = {
            "solver_solved": n_sym,
            "solver_pct": round(100 * n_sym / n, 1) if n else 0,
            "input":     _agg("effi_input_tokens"),
            "output":    _agg("effi_output_tokens"),
            "reasoning": _agg("effi_reasoning_tokens"),
            "total":     _agg("effi_total_tokens"),
        }
        total_full = summary["total"]["total"]
        total_effi = summary["effi"]["total"]["total"]
        if total_full > 0:
            summary["effi"]["token_savings_pct"] = round(100 * (1 - total_effi / total_full), 1)

    return records, summary


def print_summary(summary: dict, token_method: str) -> None:
    n = summary["n_tasks"]
    K = summary["K"]
    print(f"\n{'='*60}")
    print(f"  BoK token usage  (n={n} tasks, K={K})")
    print(f"  Token method: {token_method}")
    print(f"  Has CoT: {summary['has_cot']}")
    print(f"{'='*60}")

    def _row(label, d):
        print(f"  {label:<18}: {d['total']:>14,}  (avg {d['avg']:>10,.1f}/task)")

    print()
    _row("Input",     summary["input"])
    _row("Output",    summary["output"])
    if summary["has_cot"]:
        _row("Reasoning", summary["reasoning"])
    _row("Total",     summary["total"])

    if "effi" in summary:
        e = summary["effi"]
        print(f"\n  Effi mode (solver-first, zero LLM cost when solver correct)")
        print(f"    Solver solved    : {e['solver_solved']}/{n} = {e['solver_pct']}%")
        _row("  Effi input",     e["input"])
        _row("  Effi output",    e["output"])
        if summary["has_cot"]:
            _row("  Effi reasoning", e["reasoning"])
        _row("  Effi total",     e["total"])
        print(f"    Token savings    : {e.get('token_savings_pct', 0):.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", metavar="FILE", help="BoK JSONL (local or cluster format)")
    parser.add_argument("--tokenizer", metavar="HF_MODEL", default="",
                        help="HuggingFace model name for exact tokenization (e.g. Qwen/Qwen2.5-72B-Instruct)")
    parser.add_argument("--solver", metavar="FILE", default="",
                        help="Solver JSONL for effi token savings estimate (joined on I/O examples)")
    parser.add_argument("--out", metavar="FILE", default="",
                        help="Write per-task token JSONL to this path")
    parser.add_argument("--metrics-json", metavar="FILE", default="",
                        help="Write aggregate summary JSON to this path")
    args = parser.parse_args()

    count, token_method = _make_counter(args.tokenizer)
    solver_rewards = load_solver(args.solver) if args.solver else None
    records, summary = compute(args.input, count, solver_rewards)

    print_summary(summary, token_method)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Per-task tokens written to {args.out}")

    if args.metrics_json:
        summary["token_method"] = token_method
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.metrics_json}")


if __name__ == "__main__":
    main()
