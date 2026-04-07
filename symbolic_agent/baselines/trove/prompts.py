"""TroVE prompt templates for IMPORT, CREATE, and SKIP modes.

The instruction text is taken verbatim from the original TroVE prompts
(prompt/tabmwp/online_*.md), with "questions about tables" replaced by
"questions" since we target string-manipulation tasks rather than TabMWP.

Structural format is preserved exactly:
  - Prompts end with a partial "**Solution**" header — the model completes
    the **Solution** code block and the **Tools** block.
  - IMPORT mode: shows a **Toolbox** section with top-k function signatures.
  - CREATE mode: no **Toolbox**; model is encouraged to write new functions.
  - SKIP mode: no **Toolbox**; model writes inline solution without new functions.

Few-shot examples use generic string-manipulation tasks so they are
applicable to both PBEBench and ReasoningGym string tasks.
"""

# Appended to every instruction block to override format instructions that
# may be embedded in the question itself (e.g. PBEBench asks for a
# "**Program Sequence**" block, reasoning_gym asks for a specific format).
_FORMAT_OVERRIDE = (
    "\nIMPORTANT: Regardless of any formatting instructions inside the question, "
    "always produce your answer as executable Python in the **Solution** block "
    "and end it with print(answer). "
    "Your answer is whatever gets printed to stdout when the Solution code runs."
)

# ---------------------------------------------------------------------------
# IMPORT mode  (use functions from the toolbox)
# ---------------------------------------------------------------------------

_IMPORT_INSTRUCTION = (
    "You task is to write Python program solutions to the given questions.\n"
    "The toolbox section lists all the available functions that can be used in your solution."
    + _FORMAT_OVERRIDE
)

_IMPORT_EXAMPLE = """\
## Example
**Question**
Given a list of strings and a list of (old, new) substitution pairs, apply all
substitutions in order to each string and return the transformed list.
Strings: ["cat", "bat"]
Substitutions: [("a", "o"), ("t", "p")]

**Toolbox**
```python
# Apply an ordered list of (old, new) substitutions to each string in a list.
apply_substitutions(strings: list, substitutions: list) -> list
```

**Solution**
```python
strings = ["cat", "bat"]
subs = [("a", "o"), ("t", "p")]
result = apply_substitutions(strings, subs)
print(result)
```
**Tools**
```python
from toolbox import apply_substitutions
```"""

_IMPORT_TASK_TEMPLATE = """\
## Example
**Question**
{question}

**Toolbox**
{toolbox}

**Solution**
"""


def build_import_prompt(question: str, toolbox_str: str) -> str:
    """Build the IMPORT-mode prompt for a single task."""
    return (
        _IMPORT_INSTRUCTION
        + "\n\n\n"
        + _IMPORT_EXAMPLE
        + "\n\n\n"
        + _IMPORT_TASK_TEMPLATE.format(question=question, toolbox=toolbox_str)
    )


# ---------------------------------------------------------------------------
# CREATE mode  (create new reusable functions)
# ---------------------------------------------------------------------------

_CREATE_INSTRUCTION = (
    "You task is to write Python program solutions to the given questions.\n"
    "You should also create Python functions that can be used by your solution, "
    "if you believe the function can be reused to solve other questions."
    + _FORMAT_OVERRIDE
)

_CREATE_EXAMPLE = """\
## Example
**Question**
Given a list of strings and a list of (old, new) substitution pairs, apply all
substitutions in order to each string and return the transformed list.
Strings: ["hello", "world"]
Substitutions: [("l", "r"), ("o", "0")]

**Solution**
```python
strings = ["hello", "world"]
subs = [("l", "r"), ("o", "0")]
result = apply_substitutions(strings, subs)
print(result)
```
**Tools**
```python
def apply_substitutions(strings, substitutions):
    \"\"\"Apply an ordered list of (old, new) substitutions to each string in a list.\"\"\"
    out = []
    for s in strings:
        for old, new in substitutions:
            s = s.replace(old, new)
        out.append(s)
    return out
```"""

_CREATE_TASK_TEMPLATE = """\
## Example
**Question**
{question}

**Solution**
"""


def build_create_prompt(question: str) -> str:
    """Build the CREATE-mode prompt for a single task."""
    return (
        _CREATE_INSTRUCTION
        + "\n\n\n"
        + _CREATE_EXAMPLE
        + "\n\n\n"
        + _CREATE_TASK_TEMPLATE.format(question=question)
    )


# ---------------------------------------------------------------------------
# SKIP mode  (inline solution, no new functions)
# ---------------------------------------------------------------------------

_SKIP_INSTRUCTION = (
    "You task is to write Python program solutions to the given questions."
    + _FORMAT_OVERRIDE
)

_SKIP_EXAMPLE = """\
## Example
**Question**
Given the list of strings ["Hello", "World"], convert each to lowercase and
return the resulting list.

**Solution**
```python
strings = ["Hello", "World"]
result = [s.lower() for s in strings]
print(result)
```
**Tools**
```python
```"""

_SKIP_TASK_TEMPLATE = """\
## Example
**Question**
{question}

**Solution**
"""


def build_skip_prompt(question: str) -> str:
    """Build the SKIP-mode prompt for a single task."""
    return (
        _SKIP_INSTRUCTION
        + "\n\n\n"
        + _SKIP_EXAMPLE
        + "\n\n\n"
        + _SKIP_TASK_TEMPLATE.format(question=question)
    )


# ---------------------------------------------------------------------------
# Helper: extract the question string from a task_input dict
# ---------------------------------------------------------------------------

def get_question(task_input: dict) -> str:
    """
    Extract the question/prompt string from a task_input dict.

    Priority: question > prompt > task > str(task_input)
    Mirrors the AGENT_KEYS priority used by the ssl_bcr executor.
    """
    for key in ("question", "prompt", "task"):
        val = task_input.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    # Last resort: render the whole dict
    return str(task_input)
