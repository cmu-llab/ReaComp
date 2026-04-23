You are an expert in symbolic program induction.

Write a single Python file implementing a solver for a given Programming by Example (PBE) task.

## Task

Write a Python-based symbolic program synthesizer that infers a transformation program from a set of (input_string, output_string) pairs.

You will be shown examples of an LLM solving similar tasks in @DEMOS.json, including reasoning traces from both successful and unsuccessful attempts across easy and hard cases. Use these to understand the task structure and take inspiration from the problem-solving strategies, especially in cases where the LLM struggles.

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

```python
def solve_pbe(examples):
    """
    examples: list of (input_string, output_string)
    returns: dict with at least:
        - "success": bool
        - "program": structured representation of the inferred transformation
    """
```

## Behavior

The solver should:
* infer a program consistent with the examples and compatible with the verifier
* use (not reimplement) the verifier to score candidate programs
* prefer simple, compositional rules with low description complexity
* follow the domain-specific language (DSL) defined in @DEMOS.json
* search over candidate transformations and select ones that match all examples
* if no fully consistent program is found, return the top-K highest scoring programs
* return structured programs or hypotheses that could be useful for downstream refinement if partially incorrect