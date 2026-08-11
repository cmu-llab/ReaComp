#!/usr/bin/env python3
"""Generate gpt-oss-120b CoT reasoning traces on the List Functions pilot tasks.

New-files-only: reuses ``BestOfKController.generate_all`` (domain-agnostic async
generation) but scores candidates with the List Functions verifier on HELD-OUT inputs,
so a "solved" trace is one whose induced program generalizes, not one that fits the
shown pairs. Emits a demos file in the same shape SolverBuilder consumes for PBEBench
(keys: prompt, input_examples, output_examples, final_response, cot, success), so the
List Functions building prompt can point at it directly.

Usage (once gpt-oss-120b is served, e.g. on port 8004):
    python -m openhands_agents.run_listfn_traces \
        --tasks-file data/list_functions/pilot_tasks.jsonl \
        --base-url http://localhost:8004/v1 --model openai/gpt-oss-120b \
        --k 8 --max-tokens 32768 \
        --out-demos demos/DEMOS_LISTFN_with_CoT.json \
        --out-raw outputs/listfn_pilot_traces.jsonl
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openhands_agents.best_of_k.controller import BestOfKController, _extract_code
from rewards.list_functions import _extract_callable, score_program

_PROMPT_TEMPLATE = """\
You are solving a List Functions task: infer the single function that maps each input \
list of natural numbers to its output list, then express it as a Python program.

Here are input/output examples of the same unknown function:

{examples}

Write a Python function

    def program(xs):
        # xs: a list of ints; return the transformed list of ints
        ...

that reproduces every example above and generalizes to unseen inputs of the same \
function. Prefer a simple, general list operation (slicing, filtering, sorting, \
counting, arithmetic over elements, insertion/deletion at a position, concatenation) \
over anything that special-cases particular inputs. Do not hardcode outputs.

Respond with a single JSON object:
{{"reasoning": "<step-by-step thinking>", "code": "<a complete program defining program(xs)>"}}"""


def _fmt_examples(inputs, outputs):
    return "\n".join(f"  {inp} -> {out}" for inp, out in zip(inputs, outputs))


def build_prompt(rec):
    return _PROMPT_TEMPLATE.format(
        examples=_fmt_examples(rec["held_in_inputs"], rec["held_in_outputs"]))


def score_candidate(code, rec):
    """Compile the candidate program and score it on shown + held-out pairs."""
    if not code:
        return {"shown": 0.0, "heldout": 0.0, "ok": False}
    fn = _extract_callable(code)
    if fn is None:
        return {"shown": 0.0, "heldout": 0.0, "ok": False}
    shown, _ = score_program(fn, rec["held_in_inputs"], rec["held_in_outputs"])
    held, _ = score_program(fn, rec["held_out_inputs"], rec["held_out_outputs"])
    return {"shown": shown, "heldout": held, "ok": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", default="data/list_functions/pilot_tasks.jsonl")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--request-timeout", type=float, default=600,
                    help="per-request aiohttp timeout (s); raise for long reasoning generations")
    ap.add_argument("--chunk-size", type=int, default=10,
                    help="tasks per concurrent batch; raw rows stream to disk after each chunk")
    ap.add_argument("--out-demos", default="demos/DEMOS_LISTFN_with_CoT.json")
    ap.add_argument("--out-raw", default="outputs/listfn_pilot_traces.jsonl")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.tasks_file) if l.strip()]
    task_inputs = [{"prompt": build_prompt(r)} for r in recs]

    controller = BestOfKController(
        base_url=args.base_url, model=args.model, api_key=args.api_key,
        k=args.k, max_tokens=args.max_tokens, temperature=args.temperature,
        request_timeout=args.request_timeout)

    os.makedirs(os.path.dirname(args.out_demos), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_raw), exist_ok=True)

    demos = []
    raw_f = open(args.out_raw, "w")  # stream raw candidates as each chunk finishes
    total = len(recs)
    done = 0

    # Process in chunks so output grows live and progress is trackable, instead of
    # one giant batch that only writes at the very end.
    chunk = max(1, args.chunk_size)
    for start in range(0, total, chunk):
        recs_chunk = recs[start:start + chunk]
        inputs_chunk = [{"prompt": build_prompt(r)} for r in recs_chunk]
        grouped = asyncio.run(controller.generate_all(inputs_chunk))

        for rec, samples in zip(recs_chunk, grouped):
            best = None
            for s in samples:
                raw = s.get("raw", "")
                code = s.get("code") or _extract_code(raw)
                sc = score_candidate(code, rec)
                reasoning = ""
                try:
                    obj = json.loads(raw.strip())
                    if isinstance(obj, dict):
                        reasoning = obj.get("reasoning", "")
                except Exception:
                    reasoning = raw
                row = {
                    "task_id": rec["task_id"], "code": code, "reasoning": reasoning,
                    "shown_score": sc["shown"], "heldout_score": sc["heldout"],
                    "solved": sc["heldout"] >= 1.0 and sc["shown"] >= 1.0,
                    "token_usage": s.get("usage", {"input": 0, "output": 0, "reasoning": 0}),
                }
                raw_f.write(json.dumps(row) + "\n")
                if best is None or sc["heldout"] > best["heldout_score"] or (
                        sc["heldout"] == best["heldout_score"] and sc["shown"] > best["shown_score"]):
                    best = row
            if best is not None:
                demos.append({
                    "prompt": build_prompt(rec),
                    "input_examples": rec["held_in_inputs"],
                    "output_examples": rec["held_in_outputs"],
                    "final_response": best["code"] or "",
                    "cot": best["reasoning"] or "",
                    "success": bool(best["solved"]),
                    "task_id": rec["task_id"],
                    "shown_score": best["shown_score"],
                    "heldout_score": best["heldout_score"],
                })
        raw_f.flush()
        done += len(recs_chunk)
        n_solved = sum(1 for d in demos if d["success"])
        print(f"[{done}/{total}] tasks done | {n_solved} solved on held-out so far", flush=True)

    raw_f.close()
    with open(args.out_demos, "w") as f:
        json.dump(demos, f, indent=2)

    n_solved = sum(1 for d in demos if d["success"])
    print(f"wrote {len(demos)} demos ({n_solved} solved on held-out) -> {args.out_demos}", flush=True)
    print(f"wrote {len(raw_rows)} raw candidates -> {args.out_raw}")


if __name__ == "__main__":
    main()
