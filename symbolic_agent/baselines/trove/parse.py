"""TroVE response parser.

LLM responses are expected in the following format (from the TroVE .md prompts):

    **Solution**
    ```python
    <solution code>
    ```
    **Tools**
    ```python
    <imports and/or function definitions>
    ```

parse_response() extracts both sections and returns a structured dict.

parse_tools_in_chunk() converts the Tools code block into a list of
tool-dicts, faithful to the logic in the original TroVE utils/code.py.

count_ast_nodes() approximates solution complexity for tie-breaking best
candidate selection (TroVE §3.2: prefer fewest operations).
"""

import ast
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def _extract_code_block(text: str, header: str) -> Optional[str]:
    """
    Extract the first ```python...``` block after a **Header** marker.
    Returns None if not found.
    """
    pattern = (
        r"\*\*" + re.escape(header) + r"\*\*"
        r"[^\n]*\n+"
        r"```python\s*\n(.*?)```"
    )
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: look for any ```python block that follows the header on the same
    # or next non-empty line (handles models that skip the newline)
    fallback = (
        r"\*\*" + re.escape(header) + r"\*\*"
        r".*?```python\s*\n(.*?)```"
    )
    m2 = re.search(fallback, text, re.DOTALL | re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


def _extract_any_python_block(text: str) -> Optional[str]:
    """
    Fallback: return the content of the first ```python...``` block found anywhere.
    Returns None if no such block exists.
    """
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _make_executable(code: str) -> str:
    """
    Ensure code ends with a print() so execution captures the answer.

    If the code is a bare list literal (e.g. ["replace('a','b')"]), wrap it
    in print().  If it already contains a print() or is multi-line executable
    code, return as-is.
    """
    if not code:
        return code
    stripped = code.strip()
    # Already has a print call
    if "print(" in stripped:
        return stripped
    # Bare list or string literal — wrap in print()
    if stripped.startswith("[") or stripped.startswith('"') or stripped.startswith("'"):
        return f"print({stripped})"
    return stripped


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


# ---------------------------------------------------------------------------
# Tool parsing (faithful reimplementation of utils/code.py)
# ---------------------------------------------------------------------------

def _is_def_line(line: str) -> bool:
    return all(sym in line for sym in ["def", "(", ")", ":"])


def _is_import_line(line: str) -> bool:
    return "import" in line


def _get_function_docstr(function: str) -> str:
    """Extract docstring from a function string."""
    if '"""' in function:
        try:
            ds = function.index('"""') + 3
            de = function.index('"""', ds)
            return function[ds:de].strip()
        except ValueError:
            pass
    # Fall back to first # comment line after the def line
    lines = function.split("\n")
    for line in lines[1:]:
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
        if stripped:
            break
    return ""


def _get_function_name(function: str) -> str:
    m = re.search(r"def\s+(\w+)\s*\(", function)
    return m.group(1) if m else ""


def _get_function_signature(function: str) -> str:
    """Return 'def name(args) -> ret:' — the first def line."""
    m = re.search(r"(def\s+\w+\s*\(.*?\).*?):", function)
    return (m.group(1).strip() + ":") if m else ""


def _parse_function_tools(code_lines: list, def_indices: list) -> list:
    """Parse function definitions into tool dicts.
    Faithful to parse_function_tools() in utils/code.py."""
    tools = []
    prefix = "\n".join(code_lines[: def_indices[0]]).strip()
    for i, d in enumerate(def_indices):
        end = def_indices[i + 1] if i + 1 < len(def_indices) else len(code_lines)
        func_str = "\n".join(code_lines[d:end]).strip()
        full_func = (prefix + "\n" + func_str).strip() if prefix else func_str
        try:
            name = _get_function_name(func_str)
            if not name:
                continue
            tools.append(
                {
                    "name": name,
                    "docstr": _get_function_docstr(func_str),
                    "signature": _get_function_signature(func_str),
                    "function": full_func,
                    "type": "function",
                }
            )
        except Exception:
            continue
    return tools


def _parse_import_line(line: str) -> list:
    """Parse one import statement into a list of tool dicts.
    Faithful to parse_library_functions() in utils/code.py."""
    line = line.strip()
    if not line:
        return []
    if line.startswith("from") and "import" in line:
        m = re.match(r"from\s+(\S+)\s+import\s+(.*)", line)
        if not m:
            return []
        lib = m.group(1)
        names = [n.strip().split(" as")[0].strip() for n in m.group(2).split(",") if n.strip()]
        return [
            {
                "name": n if lib == "toolbox" else f"{lib}.{n}",
                "docstr": line,
                "signature": line,
                "function": line,
                "type": "import",
            }
            for n in names
        ]
    elif line.startswith("import"):
        m = re.match(r"import\s+(\S+)", line)
        if not m:
            return []
        lib = m.group(1).split(" as")[0].strip()
        return [
            {"name": lib, "docstr": line, "signature": line, "function": line, "type": "import"}
        ]
    return []


def parse_tools_in_chunk(code_chunk: str) -> list:
    """
    Parse all tools from a code chunk (the **Tools** section).

    Returns a list of dicts with keys: name, docstr, signature, function, type.
    Faithful to parse_tools_in_chunk() in the original TroVE utils/code.py:
      - if def lines are found → parse as function tools
      - else → parse as import tools
    """
    if not code_chunk:
        return []
    code_lines = code_chunk.split("\n")
    def_indices = [i for i, line in enumerate(code_lines) if _is_def_line(line)]
    if def_indices:
        return _parse_function_tools(code_lines, def_indices)
    # Only import lines
    tools = []
    for line in code_lines:
        if _is_import_line(line):
            tools.extend(_parse_import_line(line))
    return tools


# ---------------------------------------------------------------------------
# AST complexity (for best-candidate tie-breaking)
# ---------------------------------------------------------------------------

def count_ast_nodes(code: str) -> int:
    """
    Count total AST nodes in a program.

    TroVE §3.2 and Appendix B use AST depth per expression summed over all
    expressions.  We count total nodes across the whole tree, which is
    monotonically related to program complexity and achieves the same
    tie-breaking effect: simpler programs have fewer nodes.
    """
    if not code:
        return 0
    try:
        tree = ast.parse(code)
        return sum(1 for _ in ast.walk(tree))
    except SyntaxError:
        return 99_999


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
