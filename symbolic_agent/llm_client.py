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
"""

import json
import logging
from dataclasses import dataclass, field
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
    ):
        self.backend = backend

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
    ) -> UnifiedResponse:
        if self.backend == "anthropic":
            return self._create_anthropic(model, max_tokens, system, messages, tools, tool_choice)
        else:
            return self._create_openai(model, max_tokens, system, messages, tools)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _create_anthropic(self, model, max_tokens, system, messages, tools, tool_choice):
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
        return UnifiedResponse(blocks)

    def _create_openai(self, model, max_tokens, system, messages, tools):
        oai_messages = [{"role": "system", "content": system}] + messages
        oai_tools = self._to_openai_tools(tools)

        response = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=oai_messages,
            tools=oai_tools,
            tool_choice="required",   # equivalent to Anthropic's tool_choice={"type":"any"}
        )

        blocks: List[ToolUseBlock] = []
        msg = response.choices[0].message
        for tc in (msg.tool_calls or []):
            try:
                inp = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                inp = {}
                logger.warning("Could not parse tool arguments for %s", tc.function.name)
            blocks.append(ToolUseBlock(name=tc.function.name, input=inp))

        return UnifiedResponse(blocks)

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
                    # Anthropic uses "input_schema"; OpenAI uses "parameters"
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return result
