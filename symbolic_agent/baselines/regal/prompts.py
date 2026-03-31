"""ReGAL prompts — faithful to paper Appendix D (Tables 10, 13, 14).

parse_result() mirrors PythonTuple.parse_result() in the original codebase:
parses "NEW PROGRAM {i}:" and "NEW HELPERS:" blocks from LLM output.
"""

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Thought instructions (Table 14 — ReAct-style)
# ---------------------------------------------------------------------------

_THOUGHT_STR = (
    "Begin your program with a comment that explains your reasoning. "
    "For example, you might write:\n"
    "# Thought: the query asks for X, so I will use function Y to accomplish Z."
)
_THOUGHT_AND = "Thought and "


# ---------------------------------------------------------------------------
# Stage 1: refactorBatch prompt (Table 10)
# ---------------------------------------------------------------------------

def build_refactor_batch_prompt(
    batch: List[Tuple[str, str]],
    codebank_str: str = "",
) -> str:
    """
    Build the batch refactoring prompt.
    Faithful to Table 10 in the paper.

    Parameters
    ----------
    batch : [(query, program), ...]
    codebank_str : str
        Current codebank function definitions (empty string if codebank is empty).
    """
    n = len(batch)
    noun = "two programs" if n == 2 else f"{n} programs"

    lines = [
        f"Please rewrite the following {noun} to be more efficient.",
        "The resulting programs MUST execute to the same result as the original programs.",
        "Start by writing helper functions that can reduce the size of the code.",
    ]

    if codebank_str and codebank_str.strip():
        lines.append("You can also choose from the following helper functions:")
        lines.append(codebank_str.strip())

    lines.append("")
    for i, (query, program) in enumerate(batch, 1):
        lines.append(f"QUERY {i}: {query}")
        lines.append(f"PROGRAM {i}:")
        lines.append(program.strip())
        lines.append("")

    lines.append("Please format your answer as:")
    for i in range(1, n + 1):
        lines.append(f"NEW PROGRAM {i}:")
    lines.append("NEW HELPERS:")
    lines.append("")

    lines.append("Do not include any text that is not valid Python code.")
    lines.append(
        "Recall that no matter what, your program MUST be formatted in the following fashion:"
    )
    for i in range(1, n + 1):
        lines.append(f"NEW PROGRAM {i}:")
        lines.append("# Thoughts:")
        lines.append(f"# 1. The query asks for: <query {i} intention>")
        lines.append(f"# 2. <query {i}> can be solved by <components>.")
        lines.append(f"# 3. I will use helper function <function> to <goal>.")
        lines.append(f"<code for program {i}>")
        lines.append("")
    lines.append("NEW HELPERS:")
    lines.append("<helper function definitions>")
    lines.append("")

    lines.append(
        "Try to make your new programs as short as possible by introducing shared helper functions."
    )
    lines.append(
        "Helper function parameters should be as general as possible and helper functions "
        "should be informatively named."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 2b: retry prompt (§3.2)
# ---------------------------------------------------------------------------

def build_retry_prompt(
    failed_items: List[Tuple[str, str, str, str, str]],
    codebank_str: str = "",
) -> str:
    """
    Build retry prompt for programs that failed verification.

    Parameters
    ----------
    failed_items : [(query, original_program, refactored_program, helpers_used, error_msg), ...]
    codebank_str : str
    """
    lines = [
        "The following programs failed to execute to the same result as the original. "
        "Please fix them.",
        "",
    ]

    if codebank_str and codebank_str.strip():
        lines.append("Available helper functions:")
        lines.append(codebank_str.strip())
        lines.append("")

    for i, (query, orig, refactored, helpers, error) in enumerate(failed_items, 1):
        lines.append(f"QUERY {i}: {query}")
        lines.append(f"ORIGINAL PROGRAM {i}:")
        lines.append(orig.strip())
        lines.append(f"REFACTORED PROGRAM {i} (failed):")
        lines.append(refactored.strip())
        if helpers and helpers.strip():
            lines.append(f"HELPERS USED {i}:")
            lines.append(helpers.strip())
        lines.append(f"ERROR {i}: {error}")
        lines.append("")

    n = len(failed_items)
    lines.append("Please format your answer as:")
    for i in range(1, n + 1):
        lines.append(f"NEW PROGRAM {i}:")
    lines.append("NEW HELPERS:")
    lines.append("")
    lines.append(
        "Fix the programs so they execute to the same result as the original programs. "
        "Do not include any text that is not valid Python code."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 3a: editCodeBank prompt (Table 13)
# ---------------------------------------------------------------------------

def build_edit_codebank_prompt(
    func_str: str,
    func_name: str,
    pass_perc: float,
    fail_perc: float,
    passing_demos: List[Tuple[str, str]],
    failing_demos: List[Tuple[str, str]],
    codebank_str: str = "",
) -> str:
    """
    Build the editCodeBank prompt for a single function.
    Faithful to Table 13 in the paper.

    Parameters
    ----------
    func_str : str
        Full source of the function to edit.
    func_name : str
    pass_perc, fail_perc : float
        Fraction of unit tests passing/failing (0.0–1.0).
    passing_demos : [(query, program), ...]
        Examples where the function helped produce a passing program.
    failing_demos : [(query, program), ...]
        Examples where the function was used but the program failed.
    codebank_str : str
    """
    lines = [
        "Refactor the following function to improve performance.",
        "FUNCTION:",
        "'''",
        func_str.strip(),
        "'''",
    ]

    if codebank_str and codebank_str.strip():
        lines.append("")
        lines.append("You may also use the following helper functions:")
        lines.append(codebank_str.strip())

    lines.append("")
    lines.append(
        "Try to increase the number of passing programs. Try to make programs general. "
        "For example, you can add parameters instead of hardcoded values or call other helper functions."
    )
    lines.append(
        "First, for each failing query, explain why the programs do not accomplish the query's goal. "
        "Output this reasoning as:"
    )
    lines.append("Thoughts:")
    lines.append("1. The function passes some tests and fails others because <reason>.")
    lines.append("2. The failing queries <repeat queries here> asked for <intent>.")
    lines.append("3. The program failed because <reason>.")
    lines.append("4. This can be addressed by <change>.")
    lines.append(
        "Then output your program so that all test cases pass, using the following format: "
        "NEW PROGRAM: <program>"
    )
    lines.append("")
    lines.append(
        f"Currently, {func_name} passes in {pass_perc * 100:.1f}% of cases "
        f"and fails in {fail_perc * 100:.1f}%."
    )

    if passing_demos:
        lines.append("SUCCEEDED:")
        q, p = passing_demos[0]
        lines.append(f"Query: {q}")
        lines.append(f"Program: {p.strip()[:300]}")

    if failing_demos:
        lines.append("FAILED:")
        q, p = failing_demos[0]
        lines.append(f"Query: {q}")
        lines.append(f"Program: {p.strip()[:300]}")

    lines.append("Thoughts:")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test-time agent prompt (Table 14)
# ---------------------------------------------------------------------------

def build_agent_prompt(
    query: str,
    codebank_str: str = "",
    icl_examples: Optional[List[Tuple[str, str]]] = None,
    include_thoughts: bool = True,
) -> str:
    """
    Build the test-time agent prompt.
    Faithful to Table 14 (Python / general tasks).

    Parameters
    ----------
    query : str
    codebank_str : str
        Helper function definitions retrieved from the CodeBank.
    icl_examples : [(query, program), ...]
        In-context learning examples (mix of primitive + demo bank).
    include_thoughts : bool
        Whether to include the ReAct-style thought instruction (Table 14).
    """
    lines = [
        "Your task is to solve problems by creating Python programs.",
    ]

    if codebank_str and codebank_str.strip():
        lines.append("")
        lines.append("You have access to the following helper functions:")
        lines.append(codebank_str.strip())

    lines.append("")
    thought_str = _THOUGHT_STR if include_thoughts else ""
    lines.append(f"You will be given a query and have to produce a program. {thought_str}")

    if icl_examples:
        lines.append("Examples:")
        for q, p in icl_examples:
            lines.append(f"Query: {q}")
            thought_and = _THOUGHT_AND if include_thoughts else ""
            lines.append(f"{thought_and}Program:")
            lines.append(p.strip())
            lines.append("")

    lines.append("Please generate ONLY the code to produce the answer and nothing else.")
    lines.append(f"Query: {query}")
    thought_and = _THOUGHT_AND if include_thoughts else ""
    lines.append(f"{thought_and}Program:")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# parse_result: faithful to PythonTuple.parse_result() in original codebase
# ---------------------------------------------------------------------------

def parse_result(
    response: str,
    n_programs: int,
) -> Tuple[List[str], str]:
    """
    Parse LLM output from refactorBatch or retry into programs + helpers.

    Expected format:
        NEW PROGRAM 1:
        # Thoughts: ...
        <code>
        NEW PROGRAM 2:
        <code>
        NEW HELPERS:
        <helper function definitions>

    Returns
    -------
    (programs, helpers_code) : (List[str], str)
        programs[i] is the refactored code for input program i+1.
        helpers_code is the combined helper definitions (may be empty string).
    """
    pattern = re.compile(
        r"NEW\s+PROGRAM\s*(\d+)\s*:|NEW\s+HELPERS\s*:",
        re.IGNORECASE,
    )

    segments: Dict[str, str] = {}
    last_key: Optional[str] = None
    last_end: int = 0

    for m in pattern.finditer(response):
        if last_key is not None:
            segments[last_key] = response[last_end : m.start()].strip()
        raw_key = m.group(0).upper()
        if "HELPERS" in raw_key:
            key = "NEW HELPERS"
        else:
            num = m.group(1)
            key = f"NEW PROGRAM {num}"
        last_key = key
        last_end = m.end()

    if last_key is not None:
        segments[last_key] = response[last_end:].strip()

    programs = [segments.get(f"NEW PROGRAM {i}", "") for i in range(1, n_programs + 1)]
    helpers_code = segments.get("NEW HELPERS", "")

    return programs, helpers_code


def parse_edit_result(response: str) -> str:
    """
    Parse 'NEW PROGRAM: <code>' from editCodeBank response.
    """
    m = re.search(r"NEW\s+PROGRAM\s*:\s*\n(.*)", response, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Utility: extract query from task_input dict
# ---------------------------------------------------------------------------

def get_question(task_input: dict) -> str:
    """
    Extract a natural-language query string from a task_input dict.
    Priority: question > prompt > task > str(dict).
    """
    for key in ("question", "prompt", "task"):
        if key in task_input and isinstance(task_input[key], str):
            return task_input[key]
    return str(task_input)
