You are building a reusable Python helper library for a weaker reasoning model.

You will be given a file @DEMOS.json which contains:
1. A set of EASY and HARD problem instances.
2. 25 random examples for each of the following categories:
   - EASY, SUCCESS
   - HARD, SUCCESS
   - EASY, FAILURE
   - HARD, FAILURE
3. The task specification and input-output examples for each instance.

Your goal is NOT to solve the specific instances directly.
Your goal is to identify recurring procedural patterns that appear across successful attempts, contrast them with failure modes in unsuccessful attempts, and compress those recurring procedures into a small library of reusable Python helper functions.

Important constraints:
- Do NOT write one giant solver.
- Do NOT write functions specialized to a single instance ID or exact string pattern.
- Do NOT hardcode answers for any provided example.
- Do NOT build helper functions that only wrap trivial Python built-ins unless they correspond to a meaningful reasoning primitive.
- Prefer small, compositional, general-purpose symbolic routines that could transfer from easy to hard cases.
- Each helper function should correspond to a recurring latent procedure, not surface wording in the chain-of-thought.

What to look for:
- Common subproblems repeatedly solved in successful traces
- Steps that successful traces perform reliably but unsuccessful traces miss, skip, or do incorrectly
- Procedures that help decompose hard problems into easier local operations
- Intermediate symbolic operations that can be abstracted and reused
- Validation or search procedures that prevent common errors

Examples of acceptable helper-function types:
- finding changed regions between input and output strings
- aligning example pairs to infer local edits
- proposing candidate rewrite rules from examples
- applying a candidate rule across a dataset
- checking consistency of a rule across all examples
- composing multiple rewrite rules
- scoring, ranking, or filtering candidate hypotheses
- constructing intermediate representations useful across multiple tasks

Examples of bad helper-function types:
- solve_task_7()
- infer_exact_rule_for_this_dataset()
- giant end-to-end search functions with task-specific logic
- wrappers that merely rename basic Python string methods without adding reasoning structure

Your deliverable must have the following sections in order:

SECTION 1: PATTERN SUMMARY
Summarize the most important recurring successful procedures and the most important recurring failure modes.
For each pattern:
- give it a short name
- describe it in 1-3 sentences
- state whether it appears mostly in easy cases, hard cases, or both
- explain why it is worth turning into a reusable function

SECTION 2: LIBRARY DESIGN
Propose a compact helper library of at most N functions.
Favor fewer, higher-utility functions over many narrow ones.
For each proposed function include:
- function name
- signature
- short docstring
- what recurring pattern it captures
- why it should transfer to unseen harder cases
- what common failure mode it helps avoid

SECTION 3: PYTHON IMPLEMENTATION
Implement the library in pure Python.
Requirements:
- clean, readable code
- type hints
- docstrings
- no external dependencies unless absolutely necessary
- functions should be modular and individually testable
- avoid task-specific assumptions

SECTION 4: LIGHTWEIGHT TESTS
Write simple tests or usage examples showing the intended behavior of each helper function on small synthetic toy examples.
Do NOT reuse the original task instances verbatim unless necessary for illustration.
Tests should demonstrate generality, not memorization.

SECTION 5: USAGE GUIDE FOR A WEAKER MODEL
Write concise tool descriptions for each function as if they will be shown to a weaker coding/reasoning model.
For each function include:
- when to call it
- what inputs to provide
- what output to expect
- one short example of use

SECTION 6: RISK CHECK
Briefly explain:
- which functions might be too specific
- which functions might be too broad
- what parts of the library are most likely to fail on unseen tasks
- how you would simplify the library further if forced to reduce it by half

Selection criteria:
Choose functions that maximize:
1. reuse across both easy and hard cases
2. support for easy-to-hard generalization
3. interpretability
4. compactness
5. executable utility for a weaker model

When deciding whether to include a function, ask:
- Does this correspond to a real recurring procedure across traces?
- Would a weaker model plausibly benefit from calling this instead of re-deriving it?
- Is it general enough to apply to unseen tasks?
- Is it meaningfully distinct from other proposed functions?

Output format:
Return:
1. the analysis sections (ANALYSIS.md)
2. one complete Python file containing the final library (LIBRARY.py)
3. one short recommendation for how a weaker model should be prompted to use this library (PROMPTING_GUIDE.md)

Do not optimize for elegance alone.
Optimize for reusable symbolic procedures that improve easy-to-hard transfer.