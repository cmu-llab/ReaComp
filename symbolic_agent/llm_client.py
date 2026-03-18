"""
Unified LLM client adapter.

Wraps either the Anthropic API or an OpenAI-compatible endpoint (e.g. vLLM)
and exposes a single .create() method.  All agents talk to this adapter, so
they never need to know which backend is in use.

Usage:
    # Anthropic
    client = LLMClient(backend="anthropic", api_key="sk-ant-...")

    # vLLM (OpenAI-compatible)
    client = LLMClient(backend="openai", base_url="http://localhost:8000/v1", api_key="EMPTY")

Reasoning-model note
--------------------
Some models (e.g. gpt-oss-120b with reasoning_backend='GptOss') return their
tool call as JSON embedded in the message text rather than in the tool_calls
API field.  _create_openai() falls back to _extract_tool_calls_from_text()
when tool_calls is empty, so these models work without any agent changes.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Unified response types (mirrors Anthropic's block objects)
# --------------------------------------------------------------------------

@dataclass
class ToolUseBlock:
    type: str = "tool_use"
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


class UnifiedResponse:
    """Minimal wrapper so agents can do: for block in response.content."""
    def __init__(self, blocks: List[ToolUseBlock]):
        self.content = blocks


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------

class LLMClient:
    """
    Backend-agnostic LLM client.

    Parameters
    ----------
    backend : "anthropic" | "openai"
        "openai" covers any OpenAI-compatible server, including vLLM.
    base_url : str, optional
        Required for the "openai" backend (e.g. "http://localhost:8000/v1").
    api_key : str, optional
        API key.  For local vLLM this can be any non-empty string.
    """

    def __init__(
        self,
        backend: str = "anthropic",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        debug_dir: Optional[str] = None,
    ):
        self.backend = backend
        self._call_counter = 0

        # Create a timestamped sub-directory for this run so multiple runs
        # never overwrite each other and logs survive mid-run crashes.
        if debug_dir:
            run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            self._debug_dir: Optional[str] = os.path.join(debug_dir, f"run_{run_ts}")
            os.makedirs(self._debug_dir, exist_ok=True)
            logger.info("Debug logs → %s", self._debug_dir)
        else:
            self._debug_dir = None

        if backend == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif backend == "openai":
            import openai
            if not base_url:
                raise ValueError("base_url is required for the 'openai' backend")
            self._client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key or "EMPTY",
            )
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'anthropic' or 'openai'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: List[Dict],
        tools: List[Dict],
        tool_choice: Optional[Dict] = None,
        tag: str = "",
    ) -> UnifiedResponse:
        if self.backend == "anthropic":
            return self._create_anthropic(model, max_tokens, system, messages, tools, tool_choice, tag)
        else:
            return self._create_openai(model, max_tokens, system, messages, tools, tag)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _create_anthropic(self, model, max_tokens, system, messages, tools, tool_choice, tag=""):
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice or {"type": "any"},
        )
        blocks = [
            ToolUseBlock(name=b.name, input=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        self._write_debug_log(
            tag=tag,
            model=model,
            request={
                "system": system,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice or {"type": "any"},
                "max_tokens": max_tokens,
            },
            response={
                "stop_reason": response.stop_reason,
                "content": [
                    {"type": b.type, **({"text": b.text} if b.type == "text" else {"name": b.name, "input": b.input})}
                    for b in response.content
                ],
                "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            },
            blocks=blocks,
        )
        return UnifiedResponse(blocks)

    def _create_openai(self, model, max_tokens, system, messages, tools, tag=""):
        oai_messages = [{"role": "system", "content": system}] + messages
        oai_tools = self._to_openai_tools(tools)

        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=oai_messages,
            tools=oai_tools,
            tool_choice="required",
        )

        msg = response.choices[0].message
        blocks: List[ToolUseBlock] = []

        # Primary path: structured tool_calls field
        for tc in (msg.tool_calls or []):
            try:
                inp = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                inp = {}
                logger.warning("Could not parse tool arguments for %s", tc.function.name)
            blocks.append(ToolUseBlock(name=tc.function.name, input=inp))

        # Fallback: reasoning models (e.g. gpt-oss-120b) sometimes embed the
        # tool call as JSON in the message content instead of tool_calls.
        if not blocks:
            content = getattr(msg, "content", "") or ""
            reasoning = getattr(msg, "reasoning_content", "") or ""

            # Try content first — it holds the final committed answer.
            # Only fall back to reasoning_content if content yields nothing,
            # because reasoning_content contains chain-of-thought drafts that
            # produce false-positive JSON matches.
            blocks = self._extract_tool_calls_from_text(content, oai_tools) if content.strip() else []

            if not blocks and reasoning.strip():
                blocks = self._extract_tool_calls_from_text(reasoning, oai_tools)
                if blocks:
                    logger.info(
                        "Extracted %d tool call(s) from reasoning_content (reasoning model fallback)",
                        len(blocks),
                    )
            elif blocks:
                logger.info(
                    "Extracted %d tool call(s) from content (reasoning model fallback)",
                    len(blocks),
                )

            if not blocks:
                logger.warning(
                    "Response returned no tool calls and text extraction found nothing.\n"
                    "finish_reason=%s  content=%s",
                    response.choices[0].finish_reason,
                    content[:300],
                )

        self._write_debug_log(
            tag=tag,
            model=model,
            request={
                "system": system,
                "messages": messages,
                "tools": oai_tools,
                "tool_choice": "required",
                "max_tokens": max_tokens,
            },
            response={
                "finish_reason": response.choices[0].finish_reason,
                "content": content,
                "reasoning_content": reasoning,
                "tool_calls_raw": [
                    {"name": tc.function.name, "arguments": tc.function.arguments}
                    for tc in (msg.tool_calls or [])
                ],
            },
            blocks=blocks,
        )
        return UnifiedResponse(blocks)

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _write_debug_log(
        self,
        tag: str,
        model: str,
        request: Dict,
        response: Dict,
        blocks: List[ToolUseBlock],
    ) -> None:
        """Write one JSON file per LLM call to the debug directory."""
        if not self._debug_dir:
            return
        self._call_counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        label = f"{tag}_" if tag else ""
        fname = f"{self._call_counter:04d}_{label}{ts}.json"
        path = os.path.join(self._debug_dir, fname)
        data = {
            "call_index": self._call_counter,
            "timestamp": ts,
            "tag": tag,
            "model": model,
            "request": request,
            "response": response,
            "extracted_blocks": [{"name": b.name, "input": b.input} for b in blocks],
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not write debug log %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Schema conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai_tools(anthropic_tools: List[Dict]) -> List[Dict]:
        """Convert Anthropic tool schemas → OpenAI function-calling schemas."""
        result = []
        for t in anthropic_tools:
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return result

    # ------------------------------------------------------------------
    # Text-based tool call extraction (reasoning model fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_calls_from_text(
        text: str, oai_tools: List[Dict]
    ) -> List[ToolUseBlock]:
        """
        Parse tool call JSON from plain text for models that don't use the
        tool_calls API field.

        Handles these common patterns emitted by reasoning models:

          1. ```json\n{"name": "fn", "arguments": {...}}\n```
          2. {"name": "fn", "arguments": {...}}
          3. {"name": "fn", "input": {...}}          (Anthropic-style)
          4. {"name": "fn", "parameters": {...}}
          5. A top-level object whose keys match a known tool's required fields
             (the model skipped the wrapper and wrote the arguments directly)
        """
        valid_names = {t["function"]["name"]: t for t in oai_tools}
        blocks: List[ToolUseBlock] = []

        # Collect JSON candidates: prefer fenced blocks, then bare objects
        candidates: List[str] = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if not candidates:
            # Extract top-level {...} objects (handles nested braces up to depth 5)
            candidates = re.findall(r"\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}", text)

        for raw in candidates:
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict):
                continue

            tool_name = data.get("name") or data.get("tool") or data.get("function")

            if tool_name and tool_name in valid_names:
                # Wrapper format: {"name": "...", "arguments"|"input"|"parameters": {...}}
                inp = (
                    data.get("arguments")
                    or data.get("input")
                    or data.get("parameters")
                    or {}
                )
                if isinstance(inp, str):
                    try:
                        inp = json.loads(inp)
                    except json.JSONDecodeError:
                        inp = {}
                blocks.append(ToolUseBlock(name=tool_name, input=inp))

            else:
                # Direct format: the object IS the arguments for a tool whose
                # required fields are all present in data.
                for name, tool_def in valid_names.items():
                    required = tool_def["function"].get("parameters", {}).get("required", [])
                    if required and all(k in data for k in required):
                        blocks.append(ToolUseBlock(name=name, input=data))
                        break

        return blocks
