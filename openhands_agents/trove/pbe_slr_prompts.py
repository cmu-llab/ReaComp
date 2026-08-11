"""
DSL-aware TroVE prompt builders for PBEBench and SLR-Bench.

This is an ADDITIVE module (new file) — it does not modify the generic
`openhands_agents/trove/prompts.py`. It supplies task-type-aware IMPORT / CREATE
/ SKIP prompts so the TroVE baseline is a *fair* comparison: the model is told
the exact DSL and the exact stdout format the reward function parses.

Faithful to TroVE (Wang et al., 2024b): the model writes a standalone Python
program whose stdout IS the task answer. Here the "answer" is:
  - PBEBench: a Python list of replace(A, B) call strings, printed to stdout,
    e.g.  ["replace('a', 'b')", "replace('cd', 'ef')"]
  - SLR-Bench: a single Prolog rule string, printed to stdout,
    e.g.  eastbound(T) :- has_car(T, C), car_color(C, red).

The three modes only differ in whether/how the toolbox is used; the DSL, the
task examples, and the required stdout format are identical across modes.
"""

from typing import Any

_BASE_SYSTEM = """\
You are an expert program-synthesis assistant. You solve each task by writing a
single, self-contained Python program that PRINTS the final answer to stdout.
Respond with a single JSON object — no markdown fences, no prose outside the JSON."""

_TOOLBOX_HEADER = "Available toolbox functions (import from toolbox):"


# ──────────────────────────────────────────────────────────────────────────────
# Task-text formatting (per benchmark)
# ──────────────────────────────────────────────────────────────────────────────

def _pbe_task_text(record: dict, max_programs: int) -> str:
    inputs = record.get("inputs", [])
    outputs = record.get("outputs", [])
    lines = [
        "TASK: Programming by Example (string transformation).",
        "",
        "Find an ordered sequence of replace(A, B) operations that transforms every",
        "input string into its paired output string.",
        "",
        "DSL constraints:",
        "  - Each program has the form replace('A', 'B') using str.replace semantics.",
        "  - 1 <= len(A) <= 3 ;  0 <= len(B) <= 3 (B may be empty to delete).",
        f"  - At most {max_programs} programs in the sequence.",
        "  - Programs are applied LEFT TO RIGHT:",
        "        s = input; for p in programs: s = apply(p, s); assert s == output",
        "  - No regex, no imports needed for the transformation itself.",
        "",
        "Examples:",
    ]
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        lines.append(f"  [{i + 1}] input:  {inp!r}")
        lines.append(f"       output: {out!r}")
    lines += [
        "",
        "REQUIRED OUTPUT FORMAT: your program must print a Python list of",
        "replace() call strings (and nothing else), e.g.:",
        "  print([\"replace('a', 'b')\", \"replace('cd', 'ef')\"])",
    ]
    return "\n".join(lines)


def _slr_task_text(record: dict) -> str:
    prompt = record.get("prompt", "")
    validation_program = record.get("validation program", "")
    lines = [
        "TASK: Inductive logic programming (SLR-Bench).",
        "",
        "Find a Prolog rule that correctly classifies every train as eastbound or",
        "westbound, consistent with the labelled examples in the task prompt.",
        "",
        "DSL: the rule must be a Prolog clause of the form",
        "  eastbound(T) :- Body.",
        "where Body is a conjunction of literals over predicates such as:",
        "  has_car(T, C), car_num(C, N), car_color(C, Color), car_len(C, Length),",
        "  has_wall(C, WallType)   (plus any predicates that appear in the prompt).",
        "Prefer rules with the fewest body literals.",
        "",
        "Task prompt:",
        "---",
        prompt,
        "---",
    ]
    if validation_program:
        lines += [
            "",
            "(For reference only — do NOT print this. It is the validation program:)",
            "---",
            validation_program,
            "---",
        ]
    lines += [
        "",
        "REQUIRED OUTPUT FORMAT: your program must print exactly one Prolog rule",
        "string (and nothing else), e.g.:",
        "  print(\"eastbound(T) :- has_car(T, C), car_color(C, red).\")",
    ]
    return "\n".join(lines)


def task_text(record: dict, task_type: str, max_programs: int = 5) -> str:
    if task_type == "slr":
        return _slr_task_text(record)
    return _pbe_task_text(record, max_programs)


# ──────────────────────────────────────────────────────────────────────────────
# Mode prompts (skip / import / create) — signature matches trove.prompts
# ──────────────────────────────────────────────────────────────────────────────

def build_skip_prompt(record: dict, task_type: str, max_programs: int = 5) -> tuple[str, str]:
    system = _BASE_SYSTEM + """

MODE: solve the problem using only Python primitives and the standard library.

Response format:
{"function": null, "code": "<complete Python program that prints the answer>"}"""
    return system, task_text(record, task_type, max_programs)


def build_import_prompt(record: dict, task_type: str, toolbox_listing: str,
                        max_programs: int = 5) -> tuple[str, str]:
    system = _BASE_SYSTEM + f"""

MODE: solve the problem by importing and using functions from the toolbox below.

{_TOOLBOX_HEADER}
{toolbox_listing}

Import exactly what you use: `from toolbox import fn_name`

Response format:
{{"function": null, "code": "<complete program that imports from toolbox and prints the answer>"}}"""
    return system, task_text(record, task_type, max_programs)


def build_create_prompt(record: dict, task_type: str, toolbox_listing: str,
                        max_programs: int = 5) -> tuple[str, str]:
    system = _BASE_SYSTEM + f"""

MODE: (1) write a NEW reusable Python helper function, then (2) use it to solve
the problem.

Rules for the new function:
- Must be completely standalone: import only from stdlib or installed packages.
- Must NOT import from toolbox or call other toolbox functions.
- Should be generic enough to reuse on similar {('SLR rule-induction' if task_type == 'slr' else 'string-transformation')} tasks.
- Include a concise docstring.

Existing toolbox (for reference — do not duplicate these):
{toolbox_listing}

Response format:
{{
  "function": {{
    "name": "<snake_case_name>",
    "description": "<one-line description>",
    "code": "<complete function definition including any needed imports>"
  }},
  "code": "<complete solution program — defines the function above, then calls it and prints the answer>"
}}"""
    return system, task_text(record, task_type, max_programs)
