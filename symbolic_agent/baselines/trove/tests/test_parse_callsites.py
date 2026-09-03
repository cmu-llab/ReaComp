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
