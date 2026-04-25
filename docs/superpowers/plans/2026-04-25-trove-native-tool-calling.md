# TroVE Native Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the existing TroVE port so that the IMPORT mode uses native OpenAI tool calling (vLLM-served `gpt-oss`) while CREATE / SKIP / selection / trimming remain faithful to the paper, then run a 50-task PBEBench smoke and report numbers.

**Architecture:** Keep `_multi_way_generation` unchanged for CREATE/SKIP. Replace the IMPORT branch (when toolbox non-empty AND backend is OpenAI) with a multi-turn loop that (a) translates top-k toolbox functions into OpenAI tool schemas, (b) lets the model emit `tool_calls` that are executed in a sandboxed subprocess, and (c) returns the final assistant text + recorded tool-call trajectory. Frequency credit comes from unique `tool_call.function.name` entries, not parsed `from toolbox import`. All other invariants (K-sampling, reward-based selection, AST tie-break, `C·log_{20}(n)` trimming) are unchanged.

**Tech Stack:** Python 3.11, OpenAI Python SDK against a vLLM ≥ v0.16.0 endpoint serving `openai/gpt-oss-20b` (or `120b`), `subprocess`-based executor, `inspect` + `ast` from stdlib.

**Spec:** [docs/superpowers/specs/2026-04-25-trove-native-tool-calling-design.md](../specs/2026-04-25-trove-native-tool-calling-design.md)

---

## File Structure

| File | Status | Purpose |
|---|---|---|
| `symbolic_agent/baselines/trove/toolbox.py` | Modify | Trim coefficient `C=1.0` |
| `symbolic_agent/baselines/trove/executor.py` | Modify | `DEFAULT_TIMEOUT=60` |
| `symbolic_agent/baselines/trove/llm.py` | Modify | `reasoning_content` fallback in `_call_openai`; new `chat_with_tools` method |
| `symbolic_agent/baselines/trove/parse.py` | Modify | `imported_callsites` helper; `task_family` parameter on `parse_response` |
| `symbolic_agent/baselines/trove/prompts.py` | Modify | PBEBench-shaped few-shots; `build_import_with_tools_prompt`; `task_family` dispatch |
| `symbolic_agent/baselines/trove/controller.py` | Modify | IMPORT-with-tools branch; telemetry fields; `task_family` + `selection` params |
| `symbolic_agent/baselines/trove/tools_api.py` | Create | `toolbox_to_openai_tools`; `dispatch_tool_call` |
| `symbolic_agent/baselines/trove/docs/deviations.md` | Create | Algorithmic deviations / faithful elements / infra patches |
| `symbolic_agent/baselines/trove/tests/__init__.py` | Create | Marker file for the new tests package |
| `symbolic_agent/baselines/trove/tests/test_tools_api.py` | Create | Unit tests for schema generation + dispatcher |
| `symbolic_agent/baselines/trove/tests/test_parse_callsites.py` | Create | Unit tests for `imported_callsites` |
| `main.py` | Modify | `--trove-selection` and `--trove-task-family` flags |
| `scripts/launch_vllm_gpt_oss_120b.sh` | Modify | Add three vLLM tool-calling flags |
| `scripts/analyze_trove_run.py` | Create | Post-hoc analysis of TroVE JSONL output |

---

## Task 1: Quick infrastructure patches (trim C, executor timeout, reasoning_content fallback)

**Files:**
- Modify: `symbolic_agent/baselines/trove/toolbox.py:117`
- Modify: `symbolic_agent/baselines/trove/executor.py:19`
- Modify: `symbolic_agent/baselines/trove/llm.py:192`

These are three independent one-line changes. Bundling them since each is too small to warrant its own commit and they're all on the "infrastructure" axis.

- [ ] **Step 1.1: Update trim coefficient default**

In `symbolic_agent/baselines/trove/toolbox.py`, change the default of `trim`:

```python
def trim(self, n_processed: int, C: float = 1.0) -> set:
    """
    Remove functions whose frequency is below the threshold
        C * log_{20}(n_processed)
    and return the set of example indices that had used those functions.

    Faithful to trim_library() in run_trove.py:
        threshold = math.log(n, 20)   # log base 20
    C defaults to 1.0, matching the original implementation (C·log_{20}(n)).
    Note: the original uses log base-20 not base-10; we keep base-20.
    """
```

- [ ] **Step 1.2: Update executor timeout default**

In `symbolic_agent/baselines/trove/executor.py`, change the constant:

```python
DEFAULT_TIMEOUT = 60  # seconds — generous for PBEBench replace() chains and multi-turn dispatch
```

- [ ] **Step 1.3: Add reasoning_content fallback in `_call_openai`**

In `symbolic_agent/baselines/trove/llm.py`, replace the line that reads `raw = response.choices[0].message.content or ""` with:

```python
                msg = response.choices[0].message
                raw = msg.content or getattr(msg, "reasoning_content", "") or ""
```

Context (the surrounding `try` block stays unchanged):

```python
                response = self._client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                msg = response.choices[0].message
                raw = msg.content or getattr(msg, "reasoning_content", "") or ""
                u = getattr(response, "usage", None)
```

- [ ] **Step 1.4: Sanity-check the changes**

Run: `python -c "from symbolic_agent.baselines.trove.toolbox import TroVEToolbox; from symbolic_agent.baselines.trove.executor import DEFAULT_TIMEOUT; import inspect; print(inspect.signature(TroVEToolbox.trim).parameters['C'].default, DEFAULT_TIMEOUT)"`

Expected: `1.0 60`

- [ ] **Step 1.5: Commit**

```bash
git add symbolic_agent/baselines/trove/toolbox.py symbolic_agent/baselines/trove/executor.py symbolic_agent/baselines/trove/llm.py
git commit -m "$(cat <<'EOF'
fix(trove): infra patches for native tool calling

- toolbox.trim default C=1.0 (matches original TroVE)
- executor DEFAULT_TIMEOUT=60s (PBEBench + multi-turn headroom)
- llm._call_openai falls back to message.reasoning_content when
  message.content is empty (gpt-oss Harmony channel split)
EOF
)"
```

---

## Task 2: `parse.imported_callsites` helper + `task_family` parameter

**Files:**
- Modify: `symbolic_agent/baselines/trove/parse.py:86,106-114`
- Create: `symbolic_agent/baselines/trove/tests/__init__.py`
- Create: `symbolic_agent/baselines/trove/tests/test_parse_callsites.py`

- [ ] **Step 2.1: Create the tests package marker**

Create `symbolic_agent/baselines/trove/tests/__init__.py` as an empty file.

- [ ] **Step 2.2: Write the failing test for `imported_callsites`**

Create `symbolic_agent/baselines/trove/tests/test_parse_callsites.py`:

```python
"""Unit tests for parse.imported_callsites and parse_response(task_family=)."""

from symbolic_agent.baselines.trove.parse import imported_callsites, parse_response


# ---------------------------------------------------------------------------
# imported_callsites
# ---------------------------------------------------------------------------

def test_callsites_bare_name():
    code = "result = find_replace_chain(s, [('a', 'b')])\nprint(result)"
    assert imported_callsites(code, tools_code="", candidate_names={"find_replace_chain", "other"}) == {"find_replace_chain"}


def test_callsites_attribute_access():
    code = "result = toolbox.find_replace_chain(s, pairs)\nprint(result)"
    assert imported_callsites(code, tools_code="", candidate_names={"find_replace_chain"}) == {"find_replace_chain"}


def test_callsites_no_match():
    code = "print(s.replace('a', 'b'))"
    assert imported_callsites(code, tools_code="", candidate_names={"find_replace_chain"}) == set()


def test_callsites_multiple_calls_same_name_dedup():
    code = "x = f(1)\ny = f(2)\nprint(x, y)"
    assert imported_callsites(code, tools_code="", candidate_names={"f", "g"}) == {"f"}


def test_callsites_syntax_error_returns_empty():
    code = "this is not valid python ::"
    assert imported_callsites(code, tools_code="", candidate_names={"f"}) == set()


def test_callsites_empty_inputs():
    assert imported_callsites("", "", set()) == set()
    assert imported_callsites("print(1)", "", set()) == set()


# ---------------------------------------------------------------------------
# parse_response(task_family=)
# ---------------------------------------------------------------------------

def test_parse_response_pbebench_strict_no_solution_block():
    text = "Here is some reasoning.\n```python\nprint('answer')\n```\n"
    out = parse_response(text, task_family="pbebench")
    assert out["solution_code"] == ""


def test_parse_response_pbebench_with_solution_block():
    text = "**Solution**\n```python\nprint('answer')\n```\n"
    out = parse_response(text, task_family="pbebench")
    assert out["solution_code"] == "print('answer')"


def test_parse_response_default_falls_back_to_any_python_block():
    text = "Here is some reasoning.\n```python\nprint('answer')\n```\n"
    out = parse_response(text, task_family="default")
    assert "print('answer')" in out["solution_code"]


def test_parse_response_default_call_signature_unchanged():
    text = "**Solution**\n```python\nprint('answer')\n```\n"
    out = parse_response(text)
    assert out["solution_code"] == "print('answer')"
```

- [ ] **Step 2.3: Run the tests to confirm they fail**

Run: `python -m pytest symbolic_agent/baselines/trove/tests/test_parse_callsites.py -v`

Expected: ImportError on `imported_callsites` (function does not exist) and one or more failures on `parse_response(text, task_family=...)` (unknown kwarg).

- [ ] **Step 2.4: Implement `imported_callsites` and add `task_family` to `parse_response`**

In `symbolic_agent/baselines/trove/parse.py`, add the helper at the end of the AST section (after `count_ast_nodes`):

```python
def imported_callsites(
    solution_code: str,
    tools_code: str,
    candidate_names: set,
) -> set:
    """
    Return the subset of `candidate_names` that appear as call-sites in
    `solution_code`. Used for the `actually_called` telemetry field.

    Detects two callee shapes:
      - bare Name:        find_replace_chain(...)
      - Attribute(name):  toolbox.find_replace_chain(...)

    `tools_code` is currently unused (kept in the signature so callers can
    pass through the **Tools** block context if we later want to filter by
    what was actually imported).

    Returns an empty set on empty input or SyntaxError.
    """
    if not solution_code or not candidate_names:
        return set()
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        return set()
    found: set = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in candidate_names:
            found.add(func.id)
        elif isinstance(func, ast.Attribute) and func.attr in candidate_names:
            found.add(func.attr)
    return found
```

Then modify `parse_response` (around line 86) to accept `task_family`:

```python
def parse_response(text: str, task_family: str = "default") -> dict:
    """
    Parse a TroVE-format LLM response.

    Returns
    -------
    {
        "solution_code": str,         # code inside **Solution** block
        "tools_code":    str,         # code inside **Tools** block
        "functions":     list[dict],  # parsed tool dicts from the Tools block
    }

    task_family
    -----------
    "default": if no **Solution** block is found, falls back to the first
    ```python``` block anywhere (legacy behaviour).
    "pbebench": no fallback. Strict **Solution**-block-only parsing avoids
    accidentally promoting CoT scratchpad to the answer.
    """
    solution_code = _extract_code_block(text, "Solution") or ""
    tools_code = _extract_code_block(text, "Tools") or ""

    if not solution_code and task_family != "pbebench":
        raw = _extract_any_python_block(text)
        if raw:
            solution_code = _make_executable(raw)

    functions = parse_tools_in_chunk(tools_code) if tools_code else []
    return {
        "solution_code": solution_code,
        "tools_code": tools_code,
        "functions": functions,
    }
```

- [ ] **Step 2.5: Run the tests to confirm they pass**

Run: `python -m pytest symbolic_agent/baselines/trove/tests/test_parse_callsites.py -v`

Expected: 10 passed.

- [ ] **Step 2.6: Commit**

```bash
git add symbolic_agent/baselines/trove/parse.py symbolic_agent/baselines/trove/tests/__init__.py symbolic_agent/baselines/trove/tests/test_parse_callsites.py
git commit -m "$(cat <<'EOF'
feat(trove): add imported_callsites helper and task_family to parse_response

- imported_callsites(solution, tools, names) -> set: AST-walks Solution
  code and returns names from the candidate set that are actually called.
  Handles bare Name and Attribute (toolbox.foo) callees.
- parse_response(text, task_family="default"): when task_family="pbebench"
  the parser does not fall back to the first python block when **Solution**
  is missing. Prevents CoT scratchpad from being promoted to the answer.
EOF
)"
```

---

## Task 3: PBEBench-shaped few-shots + IMPORT-with-tools prompt

**Files:**
- Modify: `symbolic_agent/baselines/trove/prompts.py` (full rewrite of constants and `build_*` functions)

This task has no automated test — prompts are validated by inspection and by the smoke run.

- [ ] **Step 3.1: Replace the prompts module with task-family-aware variants**

Open `symbolic_agent/baselines/trove/prompts.py` and replace the entire body below the module docstring with the following. Keep the docstring at the top of the file.

```python
# ---------------------------------------------------------------------------
# Format override (default-family only)
# ---------------------------------------------------------------------------

_FORMAT_OVERRIDE_DEFAULT = (
    "\nIMPORTANT: Regardless of any formatting instructions inside the question, "
    "always produce your answer as executable Python in the **Solution** block "
    "and end it with print(answer). "
    "Your answer is whatever gets printed to stdout when the Solution code runs."
)

# PBEBench prompts model the desired format directly via the few-shot example,
# so no override string is needed.
_FORMAT_OVERRIDE_PBEBENCH = ""


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
    "to test ideas or compute intermediate results; the final answer must be "
    "produced as a Python program in the **Solution** block."
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
result = find_replace_chain("hello world", [(" ", "_"), ("h", "H"), ("e", "E"), ("l", "L"), ("o", "O"), ("w", "W"), ("r", "R"), ("d", "D")])
print(result)
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
result = find_replace_chain("hello world", [(" ", "_"), ("h", "H"), ("e", "E"), ("l", "L"), ("o", "O"), ("w", "W"), ("r", "R"), ("d", "D")])
print(result)
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
s = "hello world"
s = s.replace(" ", "_")
s = s.replace("h", "H")
s = s.replace("e", "E")
s = s.replace("l", "L")
s = s.replace("o", "O")
s = s.replace("w", "W")
s = s.replace("r", "R")
s = s.replace("d", "D")
print(s)
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
```

- [ ] **Step 3.2: Smoke-test the new prompts compile and dispatch correctly**

Run: `python -c "from symbolic_agent.baselines.trove.prompts import build_import_prompt, build_create_prompt, build_skip_prompt, build_import_with_tools_prompt; print('--IMPORT default--'); print(build_import_prompt('Q?', 'TB')[:200]); print('--IMPORT pbebench--'); print(build_import_prompt('Q?', 'TB', task_family='pbebench')[:200]); print('--IMPORT_WITH_TOOLS pbebench--'); print(build_import_with_tools_prompt('Q?', task_family='pbebench')[:200])"`

Expected: three short prompt previews, no exceptions, no `IMPORTANT:` line in the pbebench variant.

- [ ] **Step 3.3: Commit**

```bash
git add symbolic_agent/baselines/trove/prompts.py
git commit -m "$(cat <<'EOF'
feat(trove): PBEBench-shaped few-shots and IMPORT-with-tools prompt

- Add task_family parameter to all build_* prompt builders.
- Add _CREATE_EXAMPLE_PBEBENCH and _SKIP_EXAMPLE_PBEBENCH demonstrating
  replace()-chain solutions and a find_replace_chain helper.
- Add build_import_with_tools_prompt for native tool calling: no
  **Toolbox** markdown block (toolbox is conveyed via tools=[...]).
- _FORMAT_OVERRIDE is empty for task_family="pbebench" (the example
  models the desired format directly).
EOF
)"
```

---

## Task 4: New `tools_api.py` (toolbox -> OpenAI schemas, dispatcher)

**Files:**
- Create: `symbolic_agent/baselines/trove/tools_api.py`
- Create: `symbolic_agent/baselines/trove/tests/test_tools_api.py`

- [ ] **Step 4.1: Write the failing tests**

Create `symbolic_agent/baselines/trove/tests/test_tools_api.py`:

```python
"""Unit tests for tools_api.toolbox_to_openai_tools and dispatch_tool_call."""

import json
from types import SimpleNamespace

from symbolic_agent.baselines.trove.toolbox import TroVEToolbox
from symbolic_agent.baselines.trove.tools_api import (
    dispatch_tool_call,
    toolbox_to_openai_tools,
)


def _make_toolbox_with(func_src: str, name: str, docstr: str = "") -> TroVEToolbox:
    tb = TroVEToolbox()
    tb.add(
        {
            "name": name,
            "docstr": docstr,
            "signature": f"def {name}(...)",
            "function": func_src,
            "type": "function",
        },
        example_idx=0,
    )
    return tb


def _tool_call(name: str, args: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


# ---------------------------------------------------------------------------
# toolbox_to_openai_tools
# ---------------------------------------------------------------------------

def test_schema_basic_function():
    src = (
        "def find_replace_chain(s: str, pairs: list) -> str:\n"
        "    \"\"\"Apply a chain of (old, new) replacements to a string.\"\"\"\n"
        "    for old, new in pairs:\n"
        "        s = s.replace(old, new)\n"
        "    return s\n"
    )
    tb = _make_toolbox_with(src, "find_replace_chain", docstr="Apply a chain of (old, new) replacements to a string.")
    tools = toolbox_to_openai_tools(tb, topk=10)
    assert len(tools) == 1
    fn = tools[0]
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "find_replace_chain"
    assert fn["function"]["description"] == "Apply a chain of (old, new) replacements to a string."
    params = fn["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"].keys()) == {"s", "pairs"}
    assert params["properties"]["s"]["type"] == "string"
    assert params["properties"]["pairs"]["type"] == "array"
    assert set(params["required"]) == {"s", "pairs"}


def test_schema_unannotated_falls_back_to_string():
    src = (
        "def f(x):\n"
        "    return x\n"
    )
    tb = _make_toolbox_with(src, "f")
    tools = toolbox_to_openai_tools(tb, topk=10)
    assert tools[0]["function"]["parameters"]["properties"]["x"]["type"] == "string"


def test_schema_skips_varargs_kwargs():
    src = (
        "def f(*args, **kwargs):\n"
        "    return args\n"
    )
    tb = _make_toolbox_with(src, "f")
    tools = toolbox_to_openai_tools(tb, topk=10)
    assert tools == []


def test_schema_required_excludes_defaults():
    src = (
        "def f(x: int, y: int = 5):\n"
        "    return x + y\n"
    )
    tb = _make_toolbox_with(src, "f")
    tools = toolbox_to_openai_tools(tb, topk=10)
    params = tools[0]["function"]["parameters"]
    assert params["required"] == ["x"]
    assert params["properties"]["y"]["type"] == "integer"


def test_schema_topk_respects_frequency():
    tb = TroVEToolbox()
    for n, freq in [("a", 3), ("b", 2), ("c", 1)]:
        tb.add(
            {
                "name": n,
                "docstr": "",
                "signature": f"def {n}()",
                "function": f"def {n}():\n    return 0\n",
                "type": "function",
            },
            example_idx=0,
        )
        for _ in range(freq - 1):
            tb.update_frequency(n, example_idx=0)
    tools = toolbox_to_openai_tools(tb, topk=2)
    assert [t["function"]["name"] for t in tools] == ["a", "b"]


def test_schema_empty_toolbox():
    assert toolbox_to_openai_tools(TroVEToolbox(), topk=10) == []


# ---------------------------------------------------------------------------
# dispatch_tool_call
# ---------------------------------------------------------------------------

def test_dispatch_runs_function_and_returns_stdout():
    src = (
        "def reverse_str(s):\n"
        "    return s[::-1]\n"
    )
    tb = _make_toolbox_with(src, "reverse_str")
    result = dispatch_tool_call(tb, _tool_call("reverse_str", {"s": "hello"}))
    assert "olleh" in result


def test_dispatch_unknown_tool_returns_error():
    tb = TroVEToolbox()
    result = dispatch_tool_call(tb, _tool_call("nonexistent", {}))
    assert "not in toolbox" in result


def test_dispatch_bad_json_returns_error():
    src = "def f(x):\n    return x\n"
    tb = _make_toolbox_with(src, "f")
    bad = SimpleNamespace(
        id="x",
        function=SimpleNamespace(name="f", arguments="{not json"),
    )
    result = dispatch_tool_call(tb, bad)
    assert "argument JSON parse failed" in result


def test_dispatch_sanitizes_harmony_contamination():
    src = "def reverse_str(s):\n    return s[::-1]\n"
    tb = _make_toolbox_with(src, "reverse_str")
    tc = _tool_call("reverse_str<|channel|>commentary", {"s": "abc"})
    result = dispatch_tool_call(tb, tc)
    assert "cba" in result


def test_dispatch_truncates_long_output():
    src = (
        "def long_output(n):\n"
        "    return 'x' * n\n"
    )
    tb = _make_toolbox_with(src, "long_output")
    result = dispatch_tool_call(tb, _tool_call("long_output", {"n": 10000}))
    assert len(result) <= 4096 + 100  # +slack for repr quotes and truncation marker
```

- [ ] **Step 4.2: Run the tests to confirm they fail**

Run: `python -m pytest symbolic_agent/baselines/trove/tests/test_tools_api.py -v`

Expected: ImportError on `tools_api` module.

- [ ] **Step 4.3: Create the `tools_api.py` module**

Create `symbolic_agent/baselines/trove/tools_api.py`:

```python
"""Translate the TroVE toolbox into OpenAI Chat Completions tool schemas
and dispatch tool calls back through the executor.

This module is the bridge between TroVE's in-memory toolbox and vLLM's
native tool-calling protocol. It is invoked only from the IMPORT-with-tools
controller branch.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from .executor import run_solution
from .toolbox import TroVEToolbox

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 4096

# Type inference: Python annotation -> JSON Schema type.
_TYPE_MAP = {
    int: "integer",
    float: "number",
    bool: "boolean",
    str: "string",
    list: "array",
    tuple: "array",
    dict: "object",
}


def _infer_type(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "string"
    # Plain types (int, str, etc.)
    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    # typing.List, typing.Dict, etc. — fall through to string if unrecognised.
    origin = getattr(annotation, "__origin__", None)
    if origin in _TYPE_MAP:
        return _TYPE_MAP[origin]
    return "string"


def _function_to_schema(name: str, fn: Any, docstr: str) -> dict | None:
    """
    Build one OpenAI tool dict from a callable. Returns None if the function
    has *args or **kwargs (we cannot generate a meaningful schema).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        logger.debug("Could not introspect %s: %s", name, exc)
        return None

    properties: dict = {}
    required: list = []

    for pname, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            logger.debug("Skipping %s — has *args/**kwargs", name)
            return None
        prop: dict = {"type": _infer_type(param.annotation)}
        if param.default is not inspect.Parameter.empty:
            if isinstance(param.default, (int, float, bool, str)):
                prop["default"] = param.default
        else:
            required.append(pname)
        properties[pname] = prop

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": docstr or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def toolbox_to_openai_tools(toolbox: TroVEToolbox, topk: int = 10) -> list:
    """
    Convert the top-k toolbox functions (by frequency) into OpenAI Chat
    Completions tool dicts.

    Functions with *args / **kwargs are silently excluded.
    Returns [] when the toolbox is empty.
    """
    entries = toolbox.snapshot()
    if not entries:
        return []
    entries.sort(key=lambda e: -int(e.get("frequency", 0)))
    selected = entries[:topk]

    namespace: dict = {}
    try:
        exec(toolbox.get_full_code(), namespace)
    except Exception as exc:
        logger.warning("Could not exec toolbox source for schema generation: %s", exc)
        return []

    tools: list = []
    for entry in selected:
        name = entry.get("name", "")
        if not name or name not in namespace:
            continue
        fn = namespace[name]
        schema = _function_to_schema(name, fn, entry.get("docstr", ""))
        if schema is not None:
            tools.append(schema)
    return tools


def _sanitize_name(name: str) -> str:
    """Defensive workaround for vLLM PR #35906 (Harmony control tokens
    leaking into tool names like `reverse_str<|channel|>commentary`)."""
    return name.split("<|", 1)[0].strip()


def _truncate(s: str, limit: int = _MAX_RESULT_CHARS) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated {len(s) - limit} chars]"


def dispatch_tool_call(toolbox: TroVEToolbox, tool_call) -> str:
    """
    Resolve `tool_call` against the toolbox, run it via the sandbox executor,
    and return the captured stdout (truncated to 4096 chars) or an error
    message string. Always returns a string — never raises.
    """
    name = _sanitize_name(getattr(tool_call.function, "name", "") or "")
    if not name:
        return json.dumps({"error": "tool_call has no function name"})
    if name not in {e["name"] for e in toolbox.snapshot()}:
        return json.dumps({"error": f"tool '{name}' not in toolbox"})

    raw_args = getattr(tool_call.function, "arguments", "") or "{}"
    try:
        args = json.loads(raw_args)
        if not isinstance(args, dict):
            return json.dumps({"error": f"argument JSON parse failed: expected object, got {type(args).__name__}"})
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"argument JSON parse failed: {exc}"})

    call_expr = f"print(repr({name}(**{args!r})))"
    is_ok, output = run_solution(
        solution_code=call_expr,
        tools_code="",
        toolbox_code=toolbox.get_full_code(),
    )
    if not is_ok:
        return json.dumps({"error": "execution failed", "stderr": _truncate(output)})
    return _truncate(output)
```

- [ ] **Step 4.4: Run the tests to confirm they pass**

Run: `python -m pytest symbolic_agent/baselines/trove/tests/test_tools_api.py -v`

Expected: 10 passed.

- [ ] **Step 4.5: Commit**

```bash
git add symbolic_agent/baselines/trove/tools_api.py symbolic_agent/baselines/trove/tests/test_tools_api.py
git commit -m "$(cat <<'EOF'
feat(trove): add tools_api for native OpenAI tool calling

- toolbox_to_openai_tools(toolbox, topk=10): converts top-k toolbox
  functions into OpenAI Chat Completions tool schemas. Infers parameter
  types from inspect.signature; functions with *args/**kwargs are
  silently excluded.
- dispatch_tool_call(toolbox, tool_call): runs the requested function
  in the sandbox executor, returns stdout truncated to 4096 chars or
  a JSON error string. Sanitizes Harmony control-token contamination
  in tool names (defensive vs. open vLLM PR #35906).
EOF
)"
```

---

## Task 5: `chat_with_tools` method on `TroVELLMClient`

**Files:**
- Modify: `symbolic_agent/baselines/trove/llm.py` (add new method, no signature changes to existing methods)

This task has no automated test — the multi-turn loop is validated by the controller-level integration plus the smoke run.

- [ ] **Step 5.1: Add `chat_with_tools` to `TroVELLMClient`**

In `symbolic_agent/baselines/trove/llm.py`, add the following imports near the top (`Callable` may already be implicit via `typing`):

```python
from typing import Any, Callable, Dict, List, Optional
```

Then add the new method to the `TroVELLMClient` class (insert after `_call_openai`, before `_record`):

```python
    # ------------------------------------------------------------------
    # Native tool calling (OpenAI/vLLM only)
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_iters: int = 8,
        on_tool_call: Optional[Callable[[Any], str]] = None,
        tag: str = "",
    ) -> Dict[str, Any]:
        """
        Multi-turn chat completion that supports native OpenAI tool calls.

        Returns
        -------
        {
            "final_text":     str,         # message.content (or reasoning_content fallback)
            "tool_calls":     list[dict],  # ordered, each {name, args_preview, result_preview, ok}
            "iterations":     int,         # number of round-trips actually used
            "stopped_reason": str,         # "no_tool_calls" | "max_iters" | "error"
        }

        The caller is responsible for providing `on_tool_call(tc) -> str`,
        which is invoked for every tool_call returned by the model. The
        return value (already a string) is sent back as the tool message.

        Anthropic backend is not supported — this method exists for the
        OpenAI/vLLM tool-calling flow only. It raises NotImplementedError
        on Anthropic as a defensive guard; controllers must check
        `self.backend == "openai"` before calling.
        """
        if self.backend != "openai":
            raise NotImplementedError("chat_with_tools requires the openai backend")

        if on_tool_call is None:
            raise ValueError("chat_with_tools requires an on_tool_call callback")

        recorded_calls: List[Dict[str, Any]] = []
        convo: List[Dict[str, Any]] = list(messages)
        iterations = 0
        final_text = ""
        stopped_reason = "no_tool_calls"

        for it in range(max_tool_iters + 1):
            iterations = it + 1
            iter_tag = f"{tag}_iter{it}" if tag else f"iter{it}"
            response = None
            last_exc = None

            for attempt in range(3):
                try:
                    response = self._client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=convo,
                        tools=tools,
                        tool_choice="auto",
                    )
                    break
                except Exception as exc:
                    last_exc = exc
                    if getattr(exc, "status_code", None) == 400:
                        logger.warning(
                            "OpenAI chat_with_tools 400 (tag=%s): %s", iter_tag, exc
                        )
                        self._record(iter_tag, model, json.dumps(convo)[:2000], "", max_tokens, {})
                        return {
                            "final_text": "",
                            "tool_calls": recorded_calls,
                            "iterations": iterations,
                            "stopped_reason": "error",
                        }
                    if attempt < 2:
                        wait = 5 * (2 ** attempt)
                        logger.warning(
                            "chat_with_tools failed (attempt %d/3, tag=%s): %s. Retrying in %ds.",
                            attempt + 1, iter_tag, exc, wait,
                        )
                        time.sleep(wait)

            if response is None:
                logger.warning("All chat_with_tools retries exhausted (tag=%s): %s", iter_tag, last_exc)
                stopped_reason = "error"
                break

            msg = response.choices[0].message
            content = msg.content or getattr(msg, "reasoning_content", "") or ""
            tool_calls = getattr(msg, "tool_calls", None) or []

            u = getattr(response, "usage", None)
            details = getattr(u, "completion_tokens_details", None)
            usage = {
                "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0 if details else 0,
            }
            self._record(
                iter_tag,
                model,
                json.dumps(convo)[:2000],
                json.dumps({"content": content, "tool_calls_count": len(tool_calls)}),
                max_tokens,
                usage,
            )

            if not tool_calls:
                final_text = content
                stopped_reason = "no_tool_calls"
                break

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
            convo.append(assistant_msg)

            for tc in tool_calls:
                try:
                    result = on_tool_call(tc)
                    ok = True
                except Exception as exc:
                    result = json.dumps({"error": f"on_tool_call raised: {exc}"})
                    ok = False
                args_preview = (tc.function.arguments or "")[:200]
                result_preview = (result or "")[:200]
                recorded_calls.append(
                    {
                        "name": tc.function.name,
                        "args_preview": args_preview,
                        "result_preview": result_preview,
                        "ok": ok,
                    }
                )
                convo.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

            if it >= max_tool_iters - 1:
                stopped_reason = "max_iters"
                final_text = content
                break

        return {
            "final_text": final_text,
            "tool_calls": recorded_calls,
            "iterations": iterations,
            "stopped_reason": stopped_reason,
        }
```

- [ ] **Step 5.2: Smoke-test the method does not break import**

Run: `python -c "from symbolic_agent.baselines.trove.llm import TroVELLMClient; print(hasattr(TroVELLMClient, 'chat_with_tools'))"`

Expected: `True`.

- [ ] **Step 5.3: Smoke-test the Anthropic guard fires**

Run: `python -c "from symbolic_agent.baselines.trove.llm import TroVELLMClient; c = TroVELLMClient(backend='anthropic', api_key='unused'); 
try:
    c.chat_with_tools([], [], model='x', on_tool_call=lambda x: '')
    print('no exception (BUG)')
except NotImplementedError as e:
    print('guard fires:', e)"`

Expected: `guard fires: chat_with_tools requires the openai backend`.

- [ ] **Step 5.4: Commit**

```bash
git add symbolic_agent/baselines/trove/llm.py
git commit -m "$(cat <<'EOF'
feat(trove): add TroVELLMClient.chat_with_tools for native tool calls

Multi-turn loop that handles tool_calls returned by gpt-oss/vLLM:
appends assistant message + tool result messages until the model returns
no tool_calls or max_tool_iters is reached. Records each call as
{name, args_preview, result_preview, ok} for downstream telemetry.
Reuses the existing 3-attempt retry, debug logging, and token accounting.

Anthropic backend raises NotImplementedError as a defensive guard;
controllers branch on self.backend == "openai" before calling.
EOF
)"
```

---

## Task 6: Controller IMPORT-with-tools branch + telemetry fields

**Files:**
- Modify: `symbolic_agent/baselines/trove/controller.py`

- [ ] **Step 6.1: Update imports and `__init__` signature**

In `symbolic_agent/baselines/trove/controller.py`, replace the imports block at the top (currently lines 36-44) with:

```python
import logging
from collections import Counter
from typing import Callable, Dict, List, Optional

from . import tools_api
from .executor import run_solution
from .llm import TroVELLMClient
from .parse import count_ast_nodes, imported_callsites, parse_response
from .prompts import (
    build_create_prompt,
    build_import_prompt,
    build_import_with_tools_prompt,
    build_skip_prompt,
    get_question,
)
from .toolbox import TroVEToolbox
```

Then update `TroVEController.__init__` (currently around lines 78-105) to accept the two new parameters:

```python
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        base_url: Optional[str] = None,
        debug_dir: Optional[str] = None,
        k: int = DEFAULT_K,
        trim_every: int = DEFAULT_TRIM_EVERY,
        trim_C: float = 1.0,
        temperature: float = 0.3,
        top_p: float = 0.95,
        task_family: str = "default",
        selection: str = "reward",
        max_tool_iters: int = 8,
        tool_schema_topk: int = 10,
    ):
        self.model = model
        self.k = k
        self.trim_every = trim_every
        self.trim_C = trim_C
        self.task_family = task_family
        self.selection = selection
        self.max_tool_iters = max_tool_iters
        self.tool_schema_topk = tool_schema_topk

        self.backend = "openai" if base_url else "anthropic"
        self.llm = TroVELLMClient(
            backend=self.backend,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            debug_dir=debug_dir,
        )
        self.toolbox = TroVEToolbox()
        self._n_processed: int = 0
```

(Note `trim_C` default is now 1.0 to match the toolbox change in Task 1; controllers passing the default get the new behavior.)

- [ ] **Step 6.2: Update existing build_* call-sites to pass `task_family`**

In `_multi_way_generation`, find each call to `build_create_prompt(question)` and `build_skip_prompt(question)` and the legacy `build_import_prompt(question, toolbox_str)`, replacing them with:

```python
                prompt = build_import_prompt(question, toolbox_str, task_family=self.task_family)
```

```python
            prompt = build_create_prompt(question, task_family=self.task_family)
```

```python
            prompt = build_skip_prompt(question, task_family=self.task_family)
```

Also update `parse_response(raw)` calls to `parse_response(raw, task_family=self.task_family)`.

- [ ] **Step 6.3: Insert the IMPORT-with-tools branch in `_multi_way_generation`**

Locate the `# --- IMPORT mode ---` section (currently around lines 254-274). Replace it with:

```python
        # --- IMPORT mode ---
        toolbox_nonempty = bool(toolbox_str)
        use_tools_branch = toolbox_nonempty and self.backend == "openai"

        if use_tools_branch:
            import_candidates = self._generate_import_with_tools(
                question, example_idx, reward_fn=reward_fn, entry=entry
            )
            best_import_idx, best_import_score = self._select_best(
                import_candidates, reward_fn=reward_fn, entry=entry
            )
            best_import = import_candidates[best_import_idx]
            best_import["_reward_score"] = best_import_score
        elif toolbox_nonempty:
            # Legacy text-based IMPORT (Anthropic or unforeseen non-OpenAI path).
            import_candidates = []
            for _ in range(self.k):
                prompt = build_import_prompt(question, toolbox_str, task_family=self.task_family)
                raw = self.llm.call(prompt, self.model, max_tokens=DEFAULT_MAX_TOKENS, tag="trove_import")
                parsed = parse_response(raw, task_family=self.task_family)
                is_ok, out = run_solution(
                    parsed["solution_code"],
                    parsed["tools_code"],
                    self.toolbox.get_full_code(),
                )
                import_candidates.append(
                    {**parsed, "is_success": is_ok, "exec_output": out, "tool_calls": [], "stopped_reason": "legacy"}
                )
            best_import_idx, best_import_score = self._select_best(
                import_candidates, reward_fn=reward_fn, entry=entry
            )
            best_import = import_candidates[best_import_idx]
            best_import["_reward_score"] = best_import_score
        else:
            best_import = {
                "solution_code": "", "tools_code": "", "functions": [],
                "is_success": False, "exec_output": "",
                "tool_calls": [], "stopped_reason": "empty_toolbox",
                "_reward_score": None,
            }
```

- [ ] **Step 6.4: Add the `_generate_import_with_tools` method**

Insert this new method into the `TroVEController` class, after `_multi_way_generation`:

```python
    def _generate_import_with_tools(
        self,
        question: str,
        example_idx: int,
        reward_fn: Optional[Callable] = None,
        entry: Optional[dict] = None,
    ) -> List[dict]:
        """
        IMPORT-mode generation using native OpenAI tool calling.
        Builds K trajectories; each trajectory may invoke toolbox functions
        via tool_calls during the multi-turn loop. Returns K candidate dicts
        compatible with _select_best.
        """
        prompt = build_import_with_tools_prompt(question, task_family=self.task_family)
        tools_schema = tools_api.toolbox_to_openai_tools(self.toolbox, topk=self.tool_schema_topk)

        candidates: List[dict] = []
        for i in range(self.k):
            tag = f"trove_import_t{example_idx}_{i}"
            messages = [{"role": "user", "content": prompt}]
            on_tc = lambda tc: tools_api.dispatch_tool_call(self.toolbox, tc)
            traj = self.llm.chat_with_tools(
                messages=messages,
                tools=tools_schema,
                model=self.model,
                max_tokens=DEFAULT_MAX_TOKENS,
                max_tool_iters=self.max_tool_iters,
                on_tool_call=on_tc,
                tag=tag,
            )
            parsed = parse_response(traj["final_text"], task_family=self.task_family)
            is_ok, out = run_solution(
                parsed["solution_code"],
                parsed["tools_code"],
                self.toolbox.get_full_code(),
            )
            candidates.append(
                {
                    **parsed,
                    "is_success": is_ok,
                    "exec_output": out,
                    "tool_calls": traj["tool_calls"],
                    "stopped_reason": traj["stopped_reason"],
                    "iterations": traj["iterations"],
                }
            )
        return candidates
```

- [ ] **Step 6.5: Wire `selection="consistency"` to the existing consistency selector**

Replace `_select_best` (currently around lines 337-361) with:

```python
    def _select_best(
        self,
        candidates: List[dict],
        reward_fn: Optional[Callable] = None,
        entry: Optional[dict] = None,
    ):
        """
        Select the best candidate from a list of response dicts.

        Returns (best_index, score_or_None) where score is (reward, message)
        when reward-based selection is used, or None otherwise.

        Selection strategy is governed by self.selection:
          - "reward" (default): reward-based when reward_fn+entry provided,
            falls back to consistency when not.
          - "consistency": original TroVE majority-vote algorithm.
        """
        if self.selection == "consistency":
            return self._select_best_by_consistency(candidates), None
        if reward_fn is not None and entry is not None:
            return self._select_best_by_reward(candidates, reward_fn, entry)
        return self._select_best_by_consistency(candidates), None
```

- [ ] **Step 6.6: Update `_update_library` to credit frequency from tool_calls**

Replace `_update_library` (currently around lines 419-432) with:

```python
    def _update_library(self, mode: str, resp: dict, example_idx: int) -> None:
        """Update toolbox based on winning mode (faithful to run_trove.py)."""
        if mode == "import":
            tool_calls = resp.get("tool_calls") or []
            if tool_calls:
                # Native tool-calling path: credit by unique tool_call.function.name
                # (defensive: sanitize and let toolbox.update_frequency filter unknowns).
                unique_names = {
                    tc["name"].split("<|", 1)[0].strip()
                    for tc in tool_calls
                    if tc.get("name")
                }
                for name in unique_names:
                    if name:
                        self.toolbox.update_frequency(name, example_idx)
            else:
                # Legacy text-based IMPORT: credit functions parsed from **Tools**.
                for func_dict in resp.get("functions", []):
                    name = func_dict.get("name", "")
                    if name:
                        self.toolbox.update_frequency(name, example_idx)
        elif mode == "create" and resp.get("is_success"):
            for func_dict in resp.get("functions", []):
                self.toolbox.add(func_dict, example_idx)

        # SKIP: no library changes
```

- [ ] **Step 6.7: Add telemetry fields to `_make_result`**

Replace `_make_result` (currently around lines 438-480) with:

```python
    def _make_result(
        self,
        task_input: dict,
        task_type: str,
        best_mode: str,
        best_resp: dict,
        is_success: bool,
        output: str,
        best_reward_score=None,
    ) -> dict:
        """
        Build a result dict compatible with main.py's _print_result() and
        _append_task_output(). Adds passive TroVE telemetry fields.
        """
        tool_calls = best_resp.get("tool_calls") or []
        tools_called = sorted({
            tc["name"].split("<|", 1)[0].strip()
            for tc in tool_calls
            if tc.get("name")
        })
        candidate_names = {e["name"] for e in self.toolbox.snapshot()}
        actually_called = sorted(
            imported_callsites(
                solution_code=best_resp.get("solution_code", ""),
                tools_code=best_resp.get("tools_code", ""),
                candidate_names=candidate_names,
            )
        )
        import_eligible = len(self.toolbox) > 0  # state AFTER this task's update
        # Note: import_eligible reflects the current toolbox state after
        # _update_library has already run for this task. The analyzer should
        # interpret this as "a non-empty toolbox existed at some point during
        # this task's processing". For pre-task eligibility, infer from
        # toolbox snapshots in adjacent tasks.

        return {
            "task_type": task_type,
            "original_prompt": str(task_input),
            "solved": is_success,
            "steps": 1,
            "trace": [
                {
                    "step": 0,
                    "agent": "trove",
                    "action": best_mode,
                    "is_success": is_success,
                }
            ],
            "solution": best_resp.get("solution_code", ""),
            "library_snapshot": self.toolbox.snapshot(),
            "cost_summary": {},
            "final_output": {
                "answer": output,
                "explanation": f"TroVE mode={best_mode}",
                "confidence": "high" if is_success else "low",
                "execution_result": output,
            },
            "agent_messages": self.llm.get_task_log(),
            "reward_history": [],
            "best_reward": None,
            "final_reward": None,
            "_best_reward_score": best_reward_score,
            # TroVE native-tool-calling telemetry
            "won_mode": best_mode,
            "import_eligible": import_eligible,
            "import_was_winner": best_mode == "import",
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "tools_called": tools_called,
            "actually_called": actually_called,
            "trove_stopped_reason": best_resp.get("stopped_reason", ""),
        }
```

- [ ] **Step 6.8: Sanity-check the controller imports and constructs**

Run: `python -c "from symbolic_agent.baselines.trove.controller import TroVEController; c = TroVEController(api_key='unused', model='x', task_family='pbebench', selection='reward'); print(c.task_family, c.selection, c.backend, c.max_tool_iters, c.tool_schema_topk)"`

Expected: `pbebench reward anthropic 8 10`.

- [ ] **Step 6.9: Run all tests to confirm no regressions**

Run: `python -m pytest symbolic_agent/baselines/trove/tests/ -v`

Expected: 16 passed (10 from tools_api + 6 from parse_callsites + 4 more = 20 actually; verify count matches what was added).

Actual expected: 6 (parse_callsites) + 10 (tools_api) = 16 passed.

- [ ] **Step 6.10: Commit**

```bash
git add symbolic_agent/baselines/trove/controller.py
git commit -m "$(cat <<'EOF'
feat(trove): controller branch for native IMPORT tool calling

- Add task_family and selection params to TroVEController.__init__.
- IMPORT branch dispatches to _generate_import_with_tools when toolbox
  is non-empty and backend is openai; otherwise falls back to legacy
  text-based IMPORT.
- _generate_import_with_tools builds K multi-turn trajectories via
  TroVELLMClient.chat_with_tools, parses **Solution** strictly for
  pbebench, and runs the result through the executor.
- _update_library credits frequency by unique tool_call.function.name
  for the native path; legacy path still credits parsed functions.
- _make_result emits won_mode, import_eligible, import_was_winner,
  tool_calls, tool_call_count, tools_called, actually_called,
  trove_stopped_reason as passive telemetry.
- _select_best honors selection="consistency" or "reward" (default).
EOF
)"
```

---

## Task 7: `main.py` CLI flags (`--trove-selection`, `--trove-task-family`)

**Files:**
- Modify: `main.py:794-810` (add new flags) and `main.py:1002-1011` (pass through to controller)

- [ ] **Step 7.1: Add the two new argparse flags**

In `main.py`, after the existing `--trove-trim-every` argument (around line 810), insert:

```python
    parser.add_argument(
        "--trove-selection",
        choices=["reward", "consistency"],
        default="reward",
        help="[TroVE] Candidate selection strategy. 'reward' (default) uses "
             "the per-task reward function with AST tie-breaking. "
             "'consistency' uses the original TroVE majority-vote algorithm. "
             "(default: reward)",
    )
    parser.add_argument(
        "--trove-task-family",
        choices=["default", "pbebench"],
        default="default",
        help="[TroVE] Task family for prompt selection and parser strictness. "
             "'pbebench' uses PBEBench-shaped few-shots and strict **Solution** "
             "parsing (no fallback to any python block). (default: default)",
    )
```

- [ ] **Step 7.2: Plumb the flags into the `TroVEController` constructor**

Find the `elif args.framework == "trove":` block (around line 1002) and replace the `controller = TroVEController(...)` call with:

```python
    elif args.framework == "trove":
        controller = TroVEController(
            api_key=api_key,
            model=model,
            base_url=base_url,
            debug_dir=args.debug_dir,
            k=args.trove_k,
            trim_every=args.trove_trim_every,
            task_family=args.trove_task_family,
            selection=args.trove_selection,
        )
        logger.info(
            "Framework: TroVE (k=%d, trim_every=%d, task_family=%s, selection=%s)",
            args.trove_k, args.trove_trim_every, args.trove_task_family, args.trove_selection,
        )
```

- [ ] **Step 7.3: Sanity-check the CLI parses both flags**

Run: `python main.py --help 2>&1 | grep -E "trove-selection|trove-task-family"`

Expected: two lines, one for each new flag, both showing the choices and defaults.

- [ ] **Step 7.4: Sanity-check controller wires through**

Construct an empty tasks file so the run finishes immediately after parsing args:

```bash
echo '[]' > /tmp/_pbebench_empty.json
VLLM_API_KEY=EMPTY python main.py \
  --framework trove \
  --trove-task-family pbebench \
  --trove-selection reward \
  --tasks-file /tmp/_pbebench_empty.json \
  --model openai/gpt-oss-20b \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  2>&1 | grep -E "Framework: TroVE|ERROR" | head -5
```

Expected: `Framework: TroVE (k=5, trim_every=500, task_family=pbebench, selection=reward)` then an `ERROR: no records found` from the loader. Both confirm the flags parsed and the controller was constructed.

- [ ] **Step 7.5: Commit**

```bash
git add main.py
git commit -m "$(cat <<'EOF'
feat(trove): CLI flags --trove-selection and --trove-task-family

- --trove-selection {reward,consistency} (default: reward).
- --trove-task-family {default,pbebench} (default: default). Plumbed
  through to TroVEController; PBEBench runs should pass --trove-task-family
  pbebench to enable PBEBench-shaped few-shots and strict **Solution**
  parsing.
EOF
)"
```

---

## Task 8: Update vLLM launcher script with tool-calling flags

**Files:**
- Modify: `scripts/launch_vllm_gpt_oss_120b.sh`

- [ ] **Step 8.1: Add the three vLLM flags**

Replace the body of `scripts/launch_vllm_gpt_oss_120b.sh` with:

```bash
#!/bin/bash

mkdir -p /tmp/$USER-tiktoken-cache /tmp/$USER-tmp
chmod 700 /tmp/$USER-tiktoken-cache /tmp/$USER-tmp
export TIKTOKEN_CACHE_DIR=/tmp/$USER-tiktoken-cache
export TMPDIR=/tmp/$USER-tmp

ts=$(date +%Y%m%d_%H%M%S)

# Required vLLM tool-calling flags (vLLM >= v0.16.0 for PR #28729):
#   --enable-auto-tool-choice  enables tool_choice="auto"
#   --tool-call-parser openai  parses gpt-oss Harmony commentary channel
#   --reasoning-parser openai_gptoss  routes analysis-channel content into
#                                     message.reasoning_content
nohup python -m vllm.entrypoints.openai.api_server \
  --model "openai/gpt-oss-120b" \
  --tokenizer "openai/gpt-oss-120b" \
  --dtype auto \
  --port ${1} \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --reasoning-parser openai_gptoss \
  > vllm_logs/vllm_${1}_${ts}.log 2>&1 & echo $! > vllm_logs/vllm_${1}_${ts}.pid
```

- [ ] **Step 8.2: Lint the script**

Run: `bash -n scripts/launch_vllm_gpt_oss_120b.sh && echo OK`

Expected: `OK`.

- [ ] **Step 8.3: Commit**

```bash
git add scripts/launch_vllm_gpt_oss_120b.sh
git commit -m "$(cat <<'EOF'
chore(launcher): enable native tool calling for gpt-oss-120b vLLM server

Add three flags required for OpenAI-compatible tool calling on gpt-oss
served by vLLM >= v0.16.0:
  --enable-auto-tool-choice
  --tool-call-parser openai
  --reasoning-parser openai_gptoss

Without these the controller's chat_with_tools loop sees no tool_calls
in the response and degrades to no-tool behavior.
EOF
)"
```

---

## Task 9: `scripts/analyze_trove_run.py`

**Files:**
- Create: `scripts/analyze_trove_run.py`

- [ ] **Step 9.1: Create the analysis script**

Create `scripts/analyze_trove_run.py`:

```python
#!/usr/bin/env python3
"""Post-hoc analysis of a TroVE run JSONL output.

Reads the per-task JSONL file produced by main.py --output-file and reports:
  - Overall accuracy
  - Final toolbox size
  - Per-mode wins
  - IMPORT-mode tool-use breakdown
  - Top-10 most-called toolbox functions

Usage:
    python scripts/analyze_trove_run.py path/to/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: line {lineno} is not valid JSON: {exc}", file=sys.stderr)
    return rows


def _result_dict(row: dict) -> dict:
    """Tolerant accessor: results are nested under 'result' in main.py's output."""
    return row.get("result") or row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to the TroVE results JSONL file")
    args = parser.parse_args()

    rows = _load_rows(args.path)
    if not rows:
        print("ERROR: no rows loaded", file=sys.stderr)
        sys.exit(1)

    n = len(rows)
    results = [_result_dict(r) for r in rows]

    # Overall accuracy
    solved = sum(1 for r in results if r.get("solved"))
    print(f"=== Run summary: {args.path.name} ===")
    print(f"Tasks: {n}")
    print(f"Solved: {solved}/{n} ({100 * solved / n:.1f}%)")

    # Final toolbox size — take the snapshot from the last row.
    last_snapshot = results[-1].get("library_snapshot") or []
    print(f"Final toolbox size: {len(last_snapshot)}")

    # Per-mode wins
    mode_counter = Counter(r.get("won_mode", "?") for r in results)
    print(f"Mode wins: {dict(mode_counter)}")

    # IMPORT-mode tool-use breakdown
    import_eligible = [r for r in results if r.get("import_eligible")]
    if not import_eligible:
        print("No IMPORT-eligible tasks observed.")
    else:
        with_calls = [r for r in import_eligible if (r.get("tool_call_count") or 0) >= 1]
        n_eligible = len(import_eligible)
        n_with = len(with_calls)
        mean_calls = (
            sum((r.get("tool_call_count") or 0) for r in import_eligible) / n_eligible
        )
        all_calls = [tc for r in import_eligible for tc in (r.get("tool_calls") or [])]
        n_calls_total = len(all_calls)
        n_calls_ok = sum(1 for tc in all_calls if tc.get("ok"))
        success_rate = (100 * n_calls_ok / n_calls_total) if n_calls_total else 0.0
        print(
            f"IMPORT-eligible tasks: {n_eligible}\n"
            f"  Tasks with >=1 tool call: {n_with}/{n_eligible} ({100 * n_with / n_eligible:.1f}%)\n"
            f"  Mean tool calls / task:   {mean_calls:.2f}\n"
            f"  Tool-call success rate:   {n_calls_ok}/{n_calls_total} ({success_rate:.1f}%)"
        )

    # Top-10 most-called functions
    name_counter: Counter = Counter()
    for r in results:
        for tc in r.get("tool_calls") or []:
            name = (tc.get("name") or "").split("<|", 1)[0].strip()
            if name:
                name_counter[name] += 1
    if name_counter:
        print("Top-10 most-called toolbox functions:")
        for name, cnt in name_counter.most_common(10):
            print(f"  {cnt:4d}  {name}")
    else:
        print("No tool calls recorded in this run.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Make the script executable and lint-check**

Run: `chmod +x scripts/analyze_trove_run.py && python -c "import ast; ast.parse(open('scripts/analyze_trove_run.py').read())" && echo OK`

Expected: `OK`.

- [ ] **Step 9.3: Smoke-test on synthetic data**

Run:

```bash
python -c "
import json, tempfile, subprocess
rows = [
    {'result': {'solved': True,  'won_mode': 'import', 'import_eligible': True,  'tool_call_count': 2, 'tool_calls': [{'name':'find_replace_chain','ok':True},{'name':'find_replace_chain','ok':True}], 'library_snapshot':[{'name':'find_replace_chain'}]}},
    {'result': {'solved': False, 'won_mode': 'create', 'import_eligible': False, 'tool_call_count': 0, 'tool_calls': [], 'library_snapshot':[{'name':'find_replace_chain'}]}},
]
with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
    for r in rows: f.write(json.dumps(r) + '\n')
    p = f.name
print(subprocess.check_output(['python','scripts/analyze_trove_run.py', p]).decode())
"
```

Expected output contains `Solved: 1/2 (50.0%)`, `Final toolbox size: 1`, `Mode wins: {'import': 1, 'create': 1}`, `IMPORT-eligible tasks: 1`, `Tool-call success rate: 2/2 (100.0%)`, and a row `2  find_replace_chain` in the top-10.

- [ ] **Step 9.4: Commit**

```bash
git add scripts/analyze_trove_run.py
git commit -m "$(cat <<'EOF'
feat(trove): add analyze_trove_run.py for post-hoc telemetry reports

Reads a TroVE JSONL output and reports overall accuracy, final toolbox
size, per-mode wins, IMPORT-mode tool-use breakdown (>=1 call rate,
mean calls/task, success rate), and the top-10 most-called toolbox
functions. Sanitizes Harmony control-token contamination in tool names
when aggregating.
EOF
)"
```

---

## Task 10: Rewrite `docs/deviations.md`

**Files:**
- Create: `symbolic_agent/baselines/trove/docs/deviations.md`

- [ ] **Step 10.1: Create the directory and the deviations doc**

Create `symbolic_agent/baselines/trove/docs/deviations.md`:

```markdown
# TroVE Implementation: Deviations and Faithful Elements

This document tracks how this port differs from — and where it stays
faithful to — the original TroVE algorithm
([Wang et al., 2024](https://arxiv.org/abs/2401.12869),
[zorazrw/trove](https://github.com/zorazrw/trove)).

## 1. Algorithmic deviations

### 1.1 Native OpenAI tool calling for IMPORT mode
The original TroVE shows the model a `**Toolbox**` markdown block
listing top-k function signatures and asks it to write a `**Solution**`
plus `**Tools**` block referencing those functions by name. We replace
this for the IMPORT mode (when `backend == "openai"` and the toolbox is
non-empty) with **native OpenAI tool calling**: the toolbox is exposed
via the `tools=[...]` parameter of `chat.completions.create`, the model
emits structured `tool_calls` during its reasoning, and `dispatch_tool_call`
runs each one in the sandboxed executor and returns the stdout. This
makes function usage observable and credit-able from the trajectory
itself.

### 1.2 Reward-based candidate selection (default)
The paper uses self-consistency (majority vote on stdout, AST tie-break)
to pick the best of K samples per mode. We default to **reward-based
selection**: every candidate is scored by the per-task reward function,
ties broken by minimum AST node count. This is more reliable on
PBEBench (program-list outputs rarely tie as strings). The original
self-consistency selector remains available via `--trove-selection consistency`.

### 1.3 PBEBench-shaped few-shot examples
For `task_family="pbebench"` we replace the generic CREATE / SKIP / IMPORT
example pairs with PBEBench-shaped pairs that demonstrate `replace()`
chains and a small reusable helper (`find_replace_chain`). The legacy
default examples remain for `task_family="default"`.

### 1.4 Strict **Solution** parsing for PBEBench
The legacy parser falls back to "first ```python``` block anywhere" when
no `**Solution**` block is present. For `task_family="pbebench"` this
fallback is disabled, preventing CoT scratchpad from being accidentally
promoted to the answer.

## 2. Faithful elements

- 3-mode generation (IMPORT, CREATE, SKIP).
- K samples per mode (default K=5, paper).
- AST-tie-breaking by node count (simplest solution wins).
- Periodic toolbox trimming with threshold `C·log_{20}(n)`, default
  `C=1.0`, matching the original implementation.
- Frequency-based top-k retrieval for the toolbox view.
- Dict-keyed toolbox structure mirroring `utils/code.py`.
- Library updates: IMPORT credits frequency, CREATE adds new functions
  on success, SKIP makes no library changes.

## 3. Infrastructural patches

- **JSONL-per-task checkpointing** via `--output-file`, with crash
  resumption.
- **`reasoning_content` fallback** in `_call_openai` for `gpt-oss` Harmony
  channel splits where the answer text lives in `message.reasoning_content`.
- **Executor timeout 60s** (vs. 10s in earlier versions of this port),
  closer to the original's ~100s.
- **`<|`-truncation sanitizer** in `dispatch_tool_call` and
  `_update_library`. Defensive workaround for the open vLLM
  [PR #35906](https://github.com/vllm-project/vllm/pull/35906) covering
  Harmony control-token leakage into tool names. When that PR lands
  upstream the sanitizer becomes a no-op and is left in place.

## 4. Backend coverage caveat

Anthropic backend code paths exist and are exercised by CREATE / SKIP and
the legacy text-based IMPORT fallback, but **the smoke run and reported
numbers are vLLM-served `gpt-oss` only**. IMPORT-with-tools requires
the OpenAI/vLLM backend and is the only path we test end-to-end.

## 5. vLLM version requirement

- Minimum vLLM: **v0.16.0** (branch-cut 2026-02-08).
- Required upstream change: [PR #28729](https://github.com/vllm-project/vllm/pull/28729)
  ("Multiple fixes for gpt-oss Chat Completion prompting"), merged
  2025-12-12. v0.16.0 is the first stable release branch-cut after the merge.
- Known open caveat: [PR #35906](https://github.com/vllm-project/vllm/pull/35906)
  ("Sanitize leaked Harmony control tokens"), still open as of late
  March 2026 — see §3 for the sanitizer mitigation.
```

- [ ] **Step 10.2: Verify the file renders**

Run: `head -20 symbolic_agent/baselines/trove/docs/deviations.md`

Expected: the document renders with the title on the first line.

- [ ] **Step 10.3: Commit**

```bash
git add symbolic_agent/baselines/trove/docs/deviations.md
git commit -m "$(cat <<'EOF'
docs(trove): rewrite deviations.md for native tool calling

Document algorithmic deviations (native OpenAI tool calling for IMPORT,
reward-based selection by default, PBEBench-shaped few-shots, strict
**Solution** parsing for pbebench), faithful elements (3-mode generation,
K-sampling, AST tie-break, C*log_20(n) trimming with C=1.0), and
infrastructural patches (JSONL checkpointing, reasoning_content
fallback, 60s executor timeout, defensive <|-truncation sanitizer).

Includes vLLM version requirement (>= v0.16.0 for PR #28729) and the
backend coverage caveat (smoke run is vLLM-served gpt-oss only).
EOF
)"
```

---

## Task 11: Pre-flight sanity check + 50-task smoke run + report

**Files:** none modified. This is the validation task.

- [ ] **Step 11.1: Re-launch vLLM with the new flags**

The existing launcher is named `launch_vllm_gpt_oss_120b.sh` but the spec calls for `gpt-oss-20b`. Two options — pick one:

(a) **Smoke on 120b directly** (no script change beyond Task 8). Run:

```bash
bash scripts/launch_vllm_gpt_oss_120b.sh 8000
```

Then in Tasks 11.2 and 11.4, replace `--model openai/gpt-oss-20b` with `--model openai/gpt-oss-120b`.

(b) **Smoke on 20b** (one-line edit). In `scripts/launch_vllm_gpt_oss_120b.sh`, change `openai/gpt-oss-120b` → `openai/gpt-oss-20b` for both `--model` and `--tokenizer`, and lower `--tensor-parallel-size 2` → `--tensor-parallel-size 1` (20b fits on one GPU). Then:

```bash
bash scripts/launch_vllm_gpt_oss_120b.sh 8000
```

(Do not commit the edit — restore the file before the final commit, or rename the script if you want the 20b variant kept.)

Then wait 60–120 seconds and confirm the server is up:

Run: `curl -sS http://localhost:8000/v1/models | head -5`

Expected: a JSON response listing the model you launched.

- [ ] **Step 11.2: Pre-flight: one-task smoke**

Run a single task to verify the tool-calling round-trip works end-to-end. The codebase has no `--num-tasks` flag, so we slice the first row out of the 50-task PBEBench-Lite file:

```bash
mkdir -p outputs/trove_pbebench_preflight
head -n 1 data/pbebench/lite_pilot_tasks.jsonl > /tmp/_pbebench_one.jsonl
VLLM_API_KEY=EMPTY python main.py \
  --framework trove \
  --tasks-file /tmp/_pbebench_one.jsonl \
  --output-file outputs/trove_pbebench_preflight/results.jsonl \
  --model openai/gpt-oss-20b \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  --trove-task-family pbebench \
  --trove-selection reward \
  --trove-k 3 \
  --trove-trim-every 9999 \
  --max-tokens 4096 \
  --debug-dir outputs/trove_pbebench_preflight/debug
```

Expected: the run completes without crashing. The output file should contain one row.

- [ ] **Step 11.3: Verify the tool-calling pre-flight check**

This task starts with an empty toolbox so the IMPORT-with-tools branch will not run. Inspect the most recent debug-dir log file with `trove_create` or `trove_skip` in the name and confirm it contains a non-empty response:

Run: `ls -t outputs/trove_pbebench_preflight/debug/trove_run_*/0001_*.json | head -1 | xargs python -c "import json,sys; d=json.load(open(sys.argv[1])); print('content length:', len(d['response']['content']))"`

Expected: non-zero content length. If zero, the `reasoning_content` fallback (Task 1.3) is not engaging — debug before proceeding.

- [ ] **Step 11.4: Run the 50-task smoke**

`data/pbebench/lite_pilot_tasks.jsonl` is exactly 50 PBEBench-Lite tasks with per-task `reward: pbebench`, so no slicing or `--default-reward` flag is required.

```bash
mkdir -p outputs/trove_pbebench_smoke
VLLM_API_KEY=EMPTY python main.py \
  --framework trove \
  --tasks-file data/pbebench/lite_pilot_tasks.jsonl \
  --output-file outputs/trove_pbebench_smoke/results.jsonl \
  --model openai/gpt-oss-20b \
  --backend vllm \
  --base-url http://localhost:8000/v1 \
  --trove-task-family pbebench \
  --trove-selection reward \
  --trove-k 3 \
  --trove-trim-every 9999 \
  --max-tokens 4096 \
  --debug-dir outputs/trove_pbebench_smoke/debug
```

Expected: ~30–60 minutes wall-clock on local vLLM. Run completes without crashes. Auto-resume from checkpoint is supported by `--output-file` if the run is interrupted.

- [ ] **Step 11.5: Run the analysis script and capture the report**

Run: `python scripts/analyze_trove_run.py outputs/trove_pbebench_smoke/results.jsonl | tee outputs/trove_pbebench_smoke/report.txt`

Expected: the report shows accuracy, toolbox size, mode wins, IMPORT-mode tool-use breakdown, and top-10 functions.

- [ ] **Step 11.6: Report numbers to the user (no prompt iteration)**

Per the spec's "done criteria", report the contents of `outputs/trove_pbebench_smoke/report.txt` plus a short narrative paragraph noting any anomalies (e.g. `<|channel|>` contamination from PR #35906, `max_iters` stops, JSON-arg parse failures).

**No prompt iteration. No threshold tuning. The numbers are what they are.**

---

## Self-Review

### 1. Spec coverage

| Spec section | Implementing task |
|---|---|
| §3 Architecture overview | Tasks 1–8 collectively |
| §4 Data flow for IMPORT-with-tools | Tasks 4–6 |
| §5.1 New `tools_api.py` | Task 4 |
| §5.2 `_call_openai` reasoning fallback | Task 1 |
| §5.2 `chat_with_tools` method | Task 5 |
| §5.3 Controller `__init__` params, IMPORT branch, `_update_library`, `_make_result` | Task 6 |
| §5.4 `imported_callsites`, `task_family` in parse_response | Task 2 |
| §5.5 PBEBench prompts and IMPORT-with-tools prompt | Task 3 |
| §5.6 Trim `C=1.0` | Task 1 |
| §5.7 Executor timeout 60s | Task 1 |
| §5.8 main.py CLI flags | Task 7 |
| §5.9 vLLM launcher flags | Task 8 |
| §5.10 `analyze_trove_run.py` | Task 9 |
| §5.11 deviations.md rewrite | Task 10 |
| §6 Telemetry fields | Task 6.7 |
| §7 Implementation defaults | Tasks 4–6 |
| §8 Smoke run + done criteria | Task 11 |

All sections accounted for.

### 2. Placeholder scan

No `TBD`, `TODO`, `implement later`, "appropriate", "various", or "fill in details" in any task. All test code is fully written (not "write tests for the above"). All file paths are exact. All commit messages are pre-written.

### 3. Type and signature consistency

- `imported_callsites(solution_code, tools_code, candidate_names)` — defined in Task 2, called in Task 6.7 with matching kwargs.
- `toolbox_to_openai_tools(toolbox, topk=10)` — defined in Task 4, called in Task 6.4.
- `dispatch_tool_call(toolbox, tool_call) -> str` — defined in Task 4, called via the `on_tc` closure in Task 6.4.
- `chat_with_tools(messages, tools, model, max_tokens, max_tool_iters, on_tool_call, tag)` — defined in Task 5, called in Task 6.4 with matching kwargs.
- `build_import_with_tools_prompt(question, task_family)` — defined in Task 3, called in Task 6.4.
- `build_import_prompt(question, toolbox_str, task_family)` — extended in Task 3, called in Task 6.3.
- `parse_response(text, task_family)` — extended in Task 2, called in Tasks 6.3 and 6.4.
- `TroVEController(__init__)` new params (`task_family`, `selection`, `max_tool_iters`, `tool_schema_topk`) — defined in Task 6.1, passed in Task 7.2 (only `task_family` and `selection` from CLI; the other two use defaults, which matches the spec's defaults table).

All consistent.

### 4. Plan quirks worth noting to the executor

- Task 11.4 relies on the user's `task_index_25_direct_feedback.json` having at least 50 tasks. If it has fewer, swap to whichever PBEBench-Lite tasks file is available (the spec calls for "50 PBEBench-Lite tasks"; the exact filename is not load-bearing).
- Task 11.5 `tee` output captures the report for the user-facing message in 11.6.
- The `import_eligible` field in `_make_result` is computed *after* `_update_library` runs for the current task. The doc-comment in Task 6.7 explains the consequence; the analyzer in Task 9 doesn't depend on the pre-task value.
- Task 6.5's `_select_best` change wraps the existing reward/consistency selectors. When `selection="consistency"` is set, the `reward_fn` and `entry` arguments are ignored — that is intentional and matches the user's choice to keep both flags as opt-ins.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-25-trove-native-tool-calling.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
