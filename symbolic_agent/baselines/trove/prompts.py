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


# ---------------------------------------------------------------------------
# IMPORT mode  (use functions from the toolbox)
# ---------------------------------------------------------------------------

_IMPORT_INSTRUCTION = (
    "You task is to write Python program solutions to the given questions.\n"
    "The toolbox section lists all the available functions that can be used in your solution."
)

_IMPORT_EXAMPLE = """\
## Example
**Question**
Given the string "foo-bar-baz", replace all hyphens with underscores and return the result.

**Toolbox**
```python
# Replace all occurrences of a target character in a string.
replace_char(s: str, old: str, new: str) -> str
```

**Solution**
```python
s = "foo-bar-baz"
result = replace_char(s, old="-", new="_")
print(result)
```
**Tools**
```python
from toolbox import replace_char
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
)

_CREATE_EXAMPLE = """\
## Example
**Question**
Given a string, extract all digit characters and return them as a single string.
Input: "abc123def456"

**Solution**
```python
s = "abc123def456"
result = extract_digits(s)
print(result)
```
**Tools**
```python
def extract_digits(s: str) -> str:
    \"\"\"Extract all digit characters from a string and return them concatenated.\"\"\"
    return "".join(c for c in s if c.isdigit())
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
)

_SKIP_EXAMPLE = """\
## Example
**Question**
Given the string "Hello World", convert it to lowercase and print it.

**Solution**
```python
s = "Hello World"
result = s.lower()
print(result)
```
**Tools**
```python
import re
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
