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
        # compile(..., dont_inherit=True) so this module's `from __future__ import
        # annotations` is not applied to the toolbox source; we need real types in
        # `__annotations__` for inspect.signature() / _infer_type.
        _code = compile(
            toolbox.get_full_code(), "<trove-toolbox>", "exec", dont_inherit=True
        )
        exec(_code, namespace)
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
        return json.dumps({"error": "execution failed", "stdout": _truncate(output)})
    return _truncate(output)
