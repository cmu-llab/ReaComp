# Method: Solver Induction

This document describes the solver induction procedure used to produce the symbolic solvers evaluated in the paper. It covers the demonstration collection strategy, building prompts, agent configuration, and the resulting solver interface for both PBEBench and SLR-Bench.

---

## Overview

Solver induction is a one-time offline procedure. A coding agent reads a set of LLM reasoning traces (*demos*) and a task specification (*building prompt*), then writes a standalone Python solver (`SOLVER.py`) that can be run at zero per-task LLM cost. The agent has access to the verifier so it can test candidates during construction.

Two agent backends are used:
- **Claude Code (CC)**: claude-sonnet-4-6 in an interactive Claude Code CLI session. The agent reads the demos file and building prompt directly and writes the solver to the project directory.
- **OpenHands + Qwen (OH)**: Qwen3.6-35B-A3B inside an openhands-sdk 1.16.1 agent loop. The building prompt is passed as the task description; the demos file and verifier are bind-mounted into an Apptainer sandbox (`python:3.11-slim` + numpy/scipy/sympy/evaluate). The agent uses `execute_code` and `write_file` tools.

---

## Step 1: Demonstration Collection

### PBEBench

**File:** `demos/DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json` (100 traces)

Traces are sampled from gpt-oss-120b solving PBEBench tasks, balanced across a 2×2 grid:

| | Easy (cascade length 2–3) | Hard (cascade length 4–5) |
|---|---|---|
| **Success** | 25 | 25 |
| **Failure** | 25 | 25 |

Each trace contains:
- `prompt` — the task prompt shown to the LLM
- `input_examples` / `output_examples` — the I/O pairs
- `cascade_length`, `bfcc_string`, `difficulty` — task metadata
- `cot` — the model's chain-of-thought reasoning (full scratchpad)
- `final_response` — the model's final answer (the `replace(A,B)` sequence)
- `success` — whether the final response achieved reward = 1.0

Failure traces are included deliberately: the coding agent is instructed to study *what went wrong* in failures and design the solver to avoid those failure modes.

**Ablation variants** (Qwen solver only):
| File | Examples | CoT |
|------|---:|---|
| `DEMOS_PBEBENCH_seed_42_100_examples_with_CoT.json` | 100 | Yes (default) |
| `DEMOS_PBEBENCH_seed_42_48_examples_with_CoT.json` | 48 | Yes |
| `DEMOS_PBEBENCH_seed_42_12_examples_with_CoT.json` | 12 | Yes |
| `DEMOS_PBEBENCH_seed_42_100_examples.json` | 100 | No |

### SLR-Bench

**File:** `demos/DEMOS_SLRBENCH_seed_42_92_examples_with_CoT.json` (92 traces)

Traces are sampled from a 2×3×4 grid (success/failure × effort level × curriculum tier). One cell was empty, yielding 92 instead of 96. Tier distribution: basic 20, easy 24, medium 24, hard 24. Success/failure split: 48/44.

Each trace contains:
- `prompt` — task prompt with background facts and direction labels
- `input_examples` / `output_examples` — (facts_string, direction_label) pairs
- `curriculum_level`, `difficulty`, `rule_complexity` — task metadata
- `rule_sampling` (`random` or `llm guided`), `background_sampling` (`mirror` or `uniform`) — sampling strategy metadata
- `validation_program` — the ground-truth Prolog rule
- `cot` — chain-of-thought reasoning
- `final_response` — the model's final Prolog rule
- `success` — whether the rule scored 1.0

---

## Step 2: Building Prompts

The building prompt is the full task specification given to the coding agent. It instructs the agent to analyse the demos, implement a solver, and document its algorithm.

### PBEBench Building Prompt

**File:** `building_prompts/SOLVER_BUILDING_PROMPT_PBE.md`

```
You are an expert in symbolic program induction.

Write a single Python file implementing a solver for a given Programming by Example (PBE) task.

## Task

Write a Python-based symbolic program synthesizer that infers a transformation program from a set of (input_string, output_string) pairs.

You will be shown examples of an LLM solving similar tasks in @DEMOS_PBEBENCH.json, including reasoning traces from both successful and unsuccessful attempts across easy and hard cases. Use these to understand the task structure and take inspiration from the problem-solving strategies, especially in cases where the LLM struggles.

Output:
- a Python solver file @SOLVER.py
- a markdown file @SOLVER_ALGORITHM.md explaining the algorithm

The solver should use the verifier defined in @rewards/pbebench.py to evaluate candidate programs. If no correct program is found, it should return the top-K highest scoring programs, where K is a parameter taken by the solver.

## Requirements

* Output exactly one Python file and one markdown file.
* Use only the Python standard library.
* No external data, APIs, or dataset-specific assumptions.
* The solver must generalize across tasks in @DEMOS.json

## Interface

Implement:

def solve_pbe(examples):
    """
    examples: list of (input_string, output_string)
    returns: dict with at least:
        - "success": bool
        - "program": structured representation of the inferred transformation
    """

## Behavior

The solver should:
* infer a program consistent with the examples and compatible with the verifier
* use (not reimplement) the verifier to score candidate programs
* prefer simple, compositional rules with low description complexity
* follow the domain-specific language (DSL) defined in @DEMOS.json
* search over candidate transformations and select ones that match all examples
* if no fully consistent program is found, return the top-K highest scoring programs
* return structured programs or hypotheses that could be useful for downstream refinement if partially incorrect
```

### SLR-Bench Building Prompt

**File:** `building_prompts/SOLVER_BUILDING_PROMPT_SLR.md`

```
You are an expert in symbolic program induction.

Write a single Python file implementing a solver for a given SLR-Bench task.

## Task

Write a Python-based symbolic program synthesizer that infers a Prolog rule from a set of (background_facts, direction_label) pairs, where direction_label is either "eastbound" or "westbound".

You will be shown examples of an LLM solving similar tasks in @DEMOS_SLRBENCH.json, including reasoning traces from both successful and unsuccessful attempts across easy and hard cases. Use these to understand the task structure and take inspiration from the problem-solving strategies, especially in cases where the LLM struggles.

Output:
- a Python solver file @SOLVER_SLR.py
- a markdown file @SOLVER_SLR_ALGORITHM.md explaining the algorithm

The solver should use the verifier defined in @rewards/slr_bench.py to evaluate candidate rules. If no correct rule is found, it should return the top-K highest scoring rules, where K is a parameter taken by the solver.

## Requirements

* Output exactly one Python file and one markdown file.
* Use only the Python standard library.
* No external data, APIs, or dataset-specific assumptions.
* The solver must generalize across tasks in @DEMOS_SLRBENCH.json

## Interface

Implement:

def solve_slr(examples):
    """
    examples: list of (facts_string, direction_label)
              facts_string  — space-separated Prolog ground facts for one train
              direction_label — "eastbound" or "westbound"
    returns: dict with at least:
        - "success": bool
        - "program": Prolog rule string of the form "eastbound(T) :- Body."
    """

## Domain-Specific Language

The rule must be a Prolog clause of the form `eastbound(T) :- Body.` where Body is a conjunction of
literals drawn from these predicates:

- `has_car(Train, Car)` — Car is part of Train
- `car_num(Car, CarNumber)` — position of Car (positive integer)
- `car_color(Car, Color)` — Color ∈ {red, blue, green, yellow, white}
- `car_len(Car, Length)` — Length ∈ {short, long}
- `has_wall(Car, WallType)` — WallType ∈ {full, railing}

Prefer rules with the fewest body literals (use `rule_complexity()` from `rewards/slr_bench.py`
to measure this).

## Performance

Each call to the verifier (`judge.compute`) invokes SWI-Prolog as a subprocess and costs ~300ms.
Hard tasks can have hundreds of thousands of candidate rules. Minimize the number of verifier
calls — use the examples to prune the candidate space in Python before invoking the verifier.

## Behavior

The solver should:
* infer a rule consistent with the examples and compatible with the verifier
* use (not reimplement) the verifier to score candidate rules
* prefer simple rules with the fewest body literals
* follow the domain-specific language defined above and in @DEMOS_SLRBENCH.json
* search over candidate rules and select ones that correctly classify all examples
* if no fully consistent rule is found, return the top-K highest scoring rules
* return structured rules or hypotheses that could be useful for downstream refinement if partially incorrect
```

---

## Step 3: Agent Configuration

### Claude Code (CC)

- **Model:** claude-sonnet-4-6
- **Session type:** interactive Claude Code CLI session from project root
- **Procedure:** the agent is shown the building prompt and demos file via `@` references in the session; it writes `SOLVER.py` and `SOLVER_ALGORITHM.md` directly to the project directory
- **No fixed seed** — session is interactive and non-deterministic
- **Build cost:** estimated ~$10 (PBE) / ~$24 (SLR) — no trajectory log from interactive session

### OpenHands + Qwen

- **Model:** Qwen3.6-35B-A3B
- **Framework:** openhands-sdk 1.16.1
- **Sandbox:** Apptainer image `python:3.11-slim` + `pip install numpy scipy sympy evaluate datasets` (built via `scripts/build_sandbox_openhands.sh`)
- **Max steps:** 100 per solver-builder session
- **Max tokens:** model default (no explicit cap)
- **Temperature:** 1.0 (vLLM server default, recommended by serving docs)
- **Script:** `scripts/run_solver_builder_openhands.sh`
- **No fixed seed** — run-to-run variance is large by design (each run invents a different algorithm)
- **Build cost:** $0.26–$1.34 per PBE run, $0.51–$1.28 per SLR run (exact, native Qwen3.6-35B-A3B tokenizer via `transformers.AutoTokenizer`; AtlasCloud pricing $0.1612/M input, $0.9653/M output). Full per-run breakdown in `findings.md` §Solver Construction Ablations and `metrics/solver_build_tokens_qwen_slr.json`

To run a solver induction:

```bash
# PBEBench solver (default: 100 examples + CoT)
SOLVER_TYPE=pbe bash scripts/run_solver_builder_openhands.sh <PORT>

# SLR-Bench solver
SOLVER_TYPE=slr bash scripts/run_solver_builder_openhands.sh <PORT>

# Ablation: override demos file
SOLVER_TYPE=pbe DEMOS_PATH=demos/DEMOS_PBEBENCH_seed_42_48_examples_with_CoT.json \
    bash scripts/run_solver_builder_openhands.sh <PORT>
```

Output lands in `built_solvers/qwen3.6_35b_a3b/<TIMESTAMP>[_<demos_tag>]/`.

---

## Step 4: Solver Interface

### PBEBench (`SOLVER.py`)

```python
def solve_pbe(examples, k=10, max_programs=5):
    """
    examples    : list of (input_string, output_string) pairs
    k           : number of top-scoring candidates to return if no perfect solution
    max_programs: maximum cascade length (5 for Lite, 20 for Hard)
    returns     : dict with:
        - "success"  : bool — True if a reward-1.0 program was found
        - "program"  : list of replace(A,B) strings (best solution or top-1 candidate)
        - "programs" : list of up to k candidates sorted by (reward desc, complexity asc)
    """
```

Evaluated via `scripts/eval_solver.py --solver SOLVER.py --dataset lite|hard|both`.

### SLR-Bench (`SOLVER_SLR.py`)

```python
def solve_slr(examples, k=10):
    """
    examples : list of (facts_string, direction_label)
               facts_string     — space-separated Prolog ground atoms for one train
               direction_label  — "eastbound" or "westbound"
    k        : number of top-scoring candidates to return if no perfect solution
    returns  : dict with:
        - "success"  : bool — True if a reward-1.0 rule was found
        - "program"  : Prolog rule string "eastbound(T) :- Body."
        - "programs" : list of up to k candidates sorted by (reward desc, complexity asc)
    """
```

Evaluated via `scripts/eval_solver.py --solver SOLVER_SLR.py --dataset slr`.

---

## Induced Solver Paths

| Solver | Agent | Path |
|--------|-------|------|
| CC Solver (PBE) | claude-sonnet-4-6 | `built_solvers/claude_code/Thu_Apr_23_807_PM/SOLVER.py` |
| CC Solver (SLR) | claude-sonnet-4-6 | `built_solvers/claude_code/Sat_Apr_25_251_AM/SOLVER_SLR.py` |
| Qwen Solver run 2 (PBE) | Qwen3.6-35B-A3B | `built_solvers/qwen3.6_35b_a3b/Sun_Apr_26_402_PM_DEMOS_PBEBENCH_seed_42_100_examples_with_CoT/SOLVER.py` |
| Qwen Solver run 2 (SLR) | Qwen3.6-35B-A3B | `built_solvers/qwen3.6_35b_a3b/Sun_Apr_26_131_PM/SOLVER_SLR.py` |

Algorithm descriptions for each solver are in `findings.md` §Solver Construction Ablations and in the corresponding `SOLVER_ALGORITHM.md` files alongside each solver.
