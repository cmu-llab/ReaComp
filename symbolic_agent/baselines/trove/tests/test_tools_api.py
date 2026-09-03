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
        '    """Apply a chain of (old, new) replacements to a string."""\n'
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
