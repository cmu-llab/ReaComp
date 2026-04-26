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
# Format override (default-family only)
# ---------------------------------------------------------------------------

_FORMAT_OVERRIDE_DEFAULT = (
    "\nIMPORTANT: Regardless of any formatting instructions inside the question, "
    "always produce your answer as executable Python in the **Solution** block "
    "and end it with print(answer). "
    "Your answer is whatever gets printed to stdout when the Solution code runs."
)

_FORMAT_OVERRIDE_PBEBENCH = (
    "\nIMPORTANT: For PBEBench, the answer printed by the **Solution** block "
    "must be a Python list of replace() call strings, such as "
    "[\"replace('a', 'b')\", \"replace('cd', 'ef')\"]. Do not print the "
    "transformed output strings."
)


def _format_override(task_family: str) -> str:
    return _FORMAT_OVERRIDE_PBEBENCH if task_family == "pbebench" else _FORMAT_OVERRIDE_DEFAULT


# ---------------------------------------------------------------------------
# IMPORT mode (text-based, default and Anthropic fallback)
# ---------------------------------------------------------------------------

_IMPORT_INSTRUCTION_DEFAULT = (
    "You task is to write Python program solutions to the given questions.\n"
    "The toolbox section lists all the available functions that can be used in your solution."
)

_IMPORT_EXAMPLE_DEFAULT = """\
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

_IMPORT_EXAMPLE_PBEBENCH = """\
## Example
**Question**
You are given example input/output pairs. Produce a list of replace() calls
that transforms each input into its expected output.

Input:  "hello world"
Output: "HELLO_WORLD"

**Toolbox**
```python
# Apply a chain of (old, new) replacements to a string.
find_replace_chain(s: str, pairs: list) -> str
```

**Solution**
```python
result = find_replace_chain("hello world", [(" ", "_"), ("h", "H"), ("e", "E"), ("l", "L"), ("o", "O"), ("w", "W"), ("r", "R"), ("d", "D")])
print(result)
```
**Tools**
```python
from toolbox import find_replace_chain
```"""

_IMPORT_TASK_TEMPLATE = """\
## Task
**Question**
{question}

**Toolbox**
{toolbox}

**Solution**
"""


def build_import_prompt(question: str, toolbox_str: str, task_family: str = "default") -> str:
    """Build the text-based IMPORT-mode prompt (used for Anthropic and as fallback)."""
    instruction = _IMPORT_INSTRUCTION_DEFAULT + _format_override(task_family)
    example = _IMPORT_EXAMPLE_PBEBENCH if task_family == "pbebench" else _IMPORT_EXAMPLE_DEFAULT
    return (
        instruction
        + "\n\n\n"
        + example
        + "\n\n\n"
        + _IMPORT_TASK_TEMPLATE.format(question=question, toolbox=toolbox_str)
    )


# ---------------------------------------------------------------------------
# IMPORT-with-tools mode (native OpenAI tool calling; no **Toolbox** block)
# ---------------------------------------------------------------------------

_IMPORT_WITH_TOOLS_INSTRUCTION_DEFAULT = (
    "You task is to write Python program solutions to the given questions.\n"
    "You have a set of helper functions available as tools. Call any of them "
    "when they help you solve the question; otherwise solve directly. After "
    "you have computed the answer, output it as executable Python in a "
    "**Solution** block and end with print(answer)."
)

_IMPORT_WITH_TOOLS_INSTRUCTION_PBEBENCH = (
    "You task is to produce a list of replace() calls that transforms each "
    "input into its expected output for a Programming-by-Example task.\n"
    "You have a set of helper functions available as tools. Call any of them "
    "to test ideas or compute intermediate results; the final **Solution** "
    "block must print the program sequence as a Python list of replace() call "
    "strings, not the transformed outputs."
)

_IMPORT_WITH_TOOLS_EXAMPLE_DEFAULT = """\
## Example
**Question**
Apply substitutions [("a","o"),("t","p")] to ["cat","bat"] and return the list.

(After optionally calling `apply_substitutions` as a tool to confirm,
the assistant produces:)

**Solution**
```python
strings = ["cat", "bat"]
subs = [("a", "o"), ("t", "p")]
result = apply_substitutions(strings, subs)
print(result)
```"""

_IMPORT_WITH_TOOLS_EXAMPLE_PBEBENCH = """\
## Example
**Question**
Produce a sequence of replace() calls that transforms "hello world" into
"HELLO_WORLD".

(After optionally calling `find_replace_chain` as a tool to verify a
candidate sequence, the assistant produces:)

**Solution**
```python
programs = ["replace(' ', '_')", "replace('h', 'H')", "replace('e', 'E')", "replace('l', 'L')", "replace('o', 'O')", "replace('w', 'W')", "replace('r', 'R')", "replace('d', 'D')"]
print(programs)
```"""

_IMPORT_WITH_TOOLS_TASK_TEMPLATE = """\
## Task
**Question**
{question}

**Solution**
"""


def build_import_with_tools_prompt(question: str, task_family: str = "default") -> str:
    """
    Build the IMPORT-with-tools prompt. The toolbox is NOT shown as text — it
    is conveyed via the OpenAI tools=[...] parameter on the chat completion call.
    """
    if task_family == "pbebench":
        instruction = _IMPORT_WITH_TOOLS_INSTRUCTION_PBEBENCH
        example = _IMPORT_WITH_TOOLS_EXAMPLE_PBEBENCH
    else:
        instruction = _IMPORT_WITH_TOOLS_INSTRUCTION_DEFAULT
        example = _IMPORT_WITH_TOOLS_EXAMPLE_DEFAULT
    return (
        instruction
        + "\n\n\n"
        + example
        + "\n\n\n"
        + _IMPORT_WITH_TOOLS_TASK_TEMPLATE.format(question=question)
    )


# ---------------------------------------------------------------------------
# CREATE mode
# ---------------------------------------------------------------------------

_CREATE_INSTRUCTION_DEFAULT = (
    "You task is to write Python program solutions to the given questions.\n"
    "You should also create Python functions that can be used by your solution, "
    "if you believe the function can be reused to solve other questions."
)

_CREATE_EXAMPLE_DEFAULT = """\
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

_CREATE_EXAMPLE_PBEBENCH = """\
## Example
**Question**
Produce a sequence of replace() calls that transforms "hello world" into
"HELLO_WORLD".

**Solution**
```python
programs = ["replace(' ', '_')", "replace('h', 'H')", "replace('e', 'E')", "replace('l', 'L')", "replace('o', 'O')", "replace('w', 'W')", "replace('r', 'R')", "replace('d', 'D')"]
print(programs)
```
**Tools**
```python
def find_replace_chain(s, pairs):
    \"\"\"Apply a chain of (old, new) replacements to a string.\"\"\"
    for old, new in pairs:
        s = s.replace(old, new)
    return s
```"""

_CREATE_TASK_TEMPLATE = """\
## Task
**Question**
{question}

**Solution**
"""


def build_create_prompt(question: str, task_family: str = "default") -> str:
    """Build the CREATE-mode prompt for a single task."""
    instruction = _CREATE_INSTRUCTION_DEFAULT + _format_override(task_family)
    example = _CREATE_EXAMPLE_PBEBENCH if task_family == "pbebench" else _CREATE_EXAMPLE_DEFAULT
    return (
        instruction
        + "\n\n\n"
        + example
        + "\n\n\n"
        + _CREATE_TASK_TEMPLATE.format(question=question)
    )


# ---------------------------------------------------------------------------
# SKIP mode
# ---------------------------------------------------------------------------

_SKIP_INSTRUCTION_DEFAULT = (
    "You task is to write Python program solutions to the given questions."
)

_SKIP_EXAMPLE_DEFAULT = """\
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

_SKIP_EXAMPLE_PBEBENCH = """\
## Example
**Question**
Produce a sequence of replace() calls that transforms "hello world" into
"HELLO_WORLD".

**Solution**
```python
programs = ["replace(' ', '_')", "replace('h', 'H')", "replace('e', 'E')", "replace('l', 'L')", "replace('o', 'O')", "replace('w', 'W')", "replace('r', 'R')", "replace('d', 'D')"]
print(programs)
```
**Tools**
```python
```"""

_SKIP_TASK_TEMPLATE = """\
## Task
**Question**
{question}

**Solution**
"""


def build_skip_prompt(question: str, task_family: str = "default") -> str:
    """Build the SKIP-mode prompt for a single task."""
    instruction = _SKIP_INSTRUCTION_DEFAULT + _format_override(task_family)
    example = _SKIP_EXAMPLE_PBEBENCH if task_family == "pbebench" else _SKIP_EXAMPLE_DEFAULT
    return (
        instruction
        + "\n\n\n"
        + example
        + "\n\n\n"
        + _SKIP_TASK_TEMPLATE.format(question=question)
    )


def get_question(task_input: dict) -> str:
    """
    Extract the question/prompt string from a task_input dict.

    Priority: question > prompt > task > str(task_input).
    """
    for key in ("question", "prompt", "task"):
        val = task_input.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    return str(task_input)
