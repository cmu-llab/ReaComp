"""
Estimate token usage from an OpenHands solver-builder trajectory JSON.

Counts only meaningful turns (execute_code, write_file, and one finish call).
Ignores repeated sb_finish retries that are an OpenHands implementation artefact.

Usage:
    python scripts/compute_trajectory_tokens.py debug_oh_solver_builder/Fri_Apr_24_200_AM/solver_builder_trajectory.json

    python scripts/compute_trajectory_tokens.py PATH/TO/trajectory.json \\
        --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --metrics-json metrics/solver_build_tokens.json
"""

import argparse
import json
from pathlib import Path

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
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
                print(f"Warning: could not load tokenizer '{tokenizer_name}': {e}")
        method = "tiktoken(cl100k_base)"
        def count(text: str) -> int:
            return len(_enc.encode(text))
        return count, method
except ImportError:
    def _make_counter(tokenizer_name: str):
        method = "char/4 approximation"
        def count(text: str) -> int:
            return max(1, len(text) // 4)
        return count, method


_REAL_TOOLS = {"sb_execute_code", "sb_write_file"}
_FINISH_TOOL = "sb_finish"
# OpenHands truncates observation content to this many chars before sending to LLM
# (LLM_MAX_MESSAGE_CHARS, default 30000)
_MAX_OBS_CHARS = 30_000


def _thought_text(rec: dict) -> str:
    t = rec.get("thought") or ""
    if isinstance(t, list):
        return " ".join(x.get("text", "") for x in t if isinstance(x, dict))
    return str(t)


def _action_text(rec: dict) -> str:
    a = rec.get("action") or {}
    return json.dumps(a) if isinstance(a, dict) else str(a)


def compute(path: str, count) -> tuple[dict, dict]:
    with open(path) as f:
        traj = json.load(f)

    sp_event = next((r for r in traj if r["kind"] == "SystemPromptEvent"), None)
    sp_toks = count(json.dumps(sp_event.get("system_prompt", ""))) if sp_event else 0

    action_events = [r for r in traj if r["kind"] == "ActionEvent"]
    obs_events    = [r for r in traj if r["kind"] == "ObservationEvent"]

    # Separate real turns from finish retries
    real_turns   = [r for r in action_events if r.get("tool_name") in _REAL_TOOLS]
    finish_turns = [r for r in action_events if r.get("tool_name") == _FINISH_TOOL]
    # Keep only the first (successful) finish call
    first_finish = finish_turns[-1:] if finish_turns else []
    counted_turns = real_turns + first_finish

    # Map each real turn to its observation (they are interleaved in order)
    obs_iter = iter(obs_events)
    obs_map: dict[int, int] = {}
    for i, r in enumerate(action_events):
        if r.get("tool_name") in _REAL_TOOLS:
            try:
                obs = next(obs_iter)
                # OpenHands truncates observation content to _MAX_OBS_CHARS before
                # sending to the LLM; cap here to reflect actual tokens seen by model
                obs_content = str(obs.get("observation", ""))[:_MAX_OBS_CHARS]
                obs_map[i] = count(obs_content)
            except StopIteration:
                pass

    counted_indices = {id(r): i for i, r in enumerate(action_events)}

    # KV-cache mode: turn 0 pays for system prompt; each subsequent turn pays
    # only for the new tokens added since the previous turn (output + observation).
    kv_input = kv_output = 0
    new_tokens_this_turn = sp_toks
    for r in counted_turns:
        i = counted_indices[id(r)]
        out_toks = count(_thought_text(r) + _action_text(r))
        kv_input  += new_tokens_this_turn
        kv_output += out_toks
        new_tokens_this_turn = out_toks + obs_map.get(i, 0)

    # No-cache mode: each turn pays for the full accumulated context so far.
    nc_input = nc_output = 0
    context = sp_toks
    for r in counted_turns:
        i = counted_indices[id(r)]
        out_toks = count(_thought_text(r) + _action_text(r))
        nc_input  += context
        nc_output += out_toks
        context   += out_toks + obs_map.get(i, 0)

    # final context window = context after all turns (same value as nc loop end)
    total_input  = kv_input
    total_output = kv_output

    # Session metadata
    timestamps = sorted(r["timestamp"] for r in traj if r.get("timestamp"))
    start, end = (timestamps[0], timestamps[-1]) if timestamps else ("", "")

    per_event = {
        "system_prompt_tokens": sp_toks,
        "real_work_turns": len(real_turns),
        "finish_turns_total": len(finish_turns),
        "finish_turns_counted": len(first_finish),
        "observations": len(obs_events),
        "tool_counts": {},
    }
    from collections import Counter
    per_event["tool_counts"] = dict(Counter(r.get("tool_name") for r in real_turns))

    n = max(len(counted_turns), 1)
    summary = {
        "trajectory": str(path),
        "session_start": start,
        "session_end": end,
        "turns_counted": len(counted_turns),
        "turns_ignored_finish_retries": len(finish_turns) - len(first_finish),
        "kv_cache": {
            "input_tokens":  {"total": kv_input,  "avg_per_turn": round(kv_input  / n, 1)},
            "output_tokens": {"total": kv_output, "avg_per_turn": round(kv_output / n, 1)},
            "total_tokens":  kv_input + kv_output,
        },
        "no_cache": {
            "input_tokens":  {"total": nc_input,  "avg_per_turn": round(nc_input  / n, 1)},
            "output_tokens": {"total": nc_output, "avg_per_turn": round(nc_output / n, 1)},
            "total_tokens":  nc_input + nc_output,
        },
        # legacy keys kept for backwards compatibility
        "input_tokens":  {"total": kv_input,  "avg_per_turn": round(kv_input  / n, 1)},
        "output_tokens": {"total": kv_output, "avg_per_turn": round(kv_output / n, 1)},
        "total_tokens":  kv_input + kv_output,
        "final_context_window": context,
        "detail": per_event,
    }
    return summary


def print_summary(summary: dict, token_method: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Solver-builder trajectory token usage")
    print(f"  File   : {Path(summary['trajectory']).name}")
    print(f"  Start  : {summary['session_start']}")
    print(f"  End    : {summary['session_end']}")
    print(f"  Method : {token_method}")
    print(f"{'='*60}")
    d = summary["detail"]
    print(f"\n  Turns counted   : {summary['turns_counted']}")
    print(f"    Tool breakdown: {d['tool_counts']}")
    print(f"  Finish retries ignored: {summary['turns_ignored_finish_retries']}")
    print(f"  System prompt   : {d['system_prompt_tokens']:,} tokens")
    print(f"  Final context   : {summary['final_context_window']:,} tokens")
    kv = summary["kv_cache"]
    nc = summary["no_cache"]
    print(f"\n  {'':30s}  {'KV-cache':>12}  {'No-cache':>12}")
    print(f"  {'Input tokens':30s}  {kv['input_tokens']['total']:>12,}  {nc['input_tokens']['total']:>12,}")
    print(f"  {'Output tokens':30s}  {kv['output_tokens']['total']:>12,}  {nc['output_tokens']['total']:>12,}")
    print(f"  {'Total tokens':30s}  {kv['total_tokens']:>12,}  {nc['total_tokens']:>12,}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trajectory", metavar="FILE")
    parser.add_argument("--tokenizer", metavar="HF_MODEL", default="",
                        help="HuggingFace model name for exact tokenization")
    parser.add_argument("--metrics-json", metavar="FILE", default="",
                        help="Write summary JSON to this path")
    args = parser.parse_args()

    count, token_method = _make_counter(args.tokenizer)
    summary = compute(args.trajectory, count)
    summary["token_method"] = token_method
    print_summary(summary, token_method)

    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to {args.metrics_json}")


if __name__ == "__main__":
    main()
