"""
Unified LLM client adapter.

Wraps either the Anthropic API or an OpenAI-compatible endpoint (e.g. vLLM)
and exposes a single .create() method that returns a parsed dict.

All agents instruct the model to respond with a JSON object via the system
prompt.  No tool-calling machinery is used, which avoids backend schema
differences and the GptOss single-output-message constraint.

Usage:
    client = LLMClient(backend="anthropic", api_key="sk-ant-...")
    client = LLMClient(backend="openai", base_url="http://localhost:8000/v1")

    result = client.create(model=..., max_tokens=..., system=..., messages=...)
    # result is a plain dict, e.g. {"action": "create", "name": "...", "code": "..."}
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_JSON_INSTRUCTION = (
    "\n\nRespond with a single valid JSON object and nothing else. "
    "No markdown fences, no prose before or after the JSON."
)


class LLMClient:
    """
    Backend-agnostic LLM client. Returns parsed JSON dicts.

    Parameters
    ----------
    backend : "anthropic" | "openai"
        "openai" covers any OpenAI-compatible server, including vLLM.
    base_url : str, optional
        Required for the "openai" backend.
    api_key : str, optional
        API key. For local vLLM any non-empty string works.
    debug_dir : str, optional
        Directory for per-call debug logs. Each run creates a timestamped
        subdirectory so logs from multiple runs never overwrite each other.
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
        self._task_log: List[Dict] = []
        # Token usage accumulators — reset per task and accumulated session-wide
        self._task_tokens: Dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}
        self._session_tokens: Dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}

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
            self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Use 'anthropic' or 'openai'.")

    # ------------------------------------------------------------------
    # Task-level message log
    # ------------------------------------------------------------------

    def reset_task_log(self) -> None:
        """Clear the in-memory message log and reset per-task token counters."""
        self._task_log = []
        self._task_tokens = {"input": 0, "output": 0, "reasoning": 0}

    def get_task_log(self) -> List[Dict]:
        """Return all messages recorded since the last reset_task_log() call."""
        return list(self._task_log)

    def get_task_token_usage(self) -> Dict[str, int]:
        """Return token counts accumulated since the last reset_task_log() call."""
        return dict(self._task_tokens)

    def get_session_token_usage(self) -> Dict[str, int]:
        """Return token counts accumulated across the entire session."""
        return dict(self._session_tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(
        self,
        model: str,
        max_tokens: int,
        system: str,
        messages: List[Dict],
        tag: str = "",
    ) -> Dict:
        """
        Call the LLM and return the response as a parsed dict.
        Returns {} on parse failure (always logged as a warning).
        """
        system_with_json = system + _JSON_INSTRUCTION
        if self.backend == "anthropic":
            return self._create_anthropic(model, max_tokens, system_with_json, messages, tag)
        else:
            return self._create_openai(model, max_tokens, system_with_json, messages, tag)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _create_anthropic(self, model, max_tokens, system, messages, tag):
        last_exc = None
        for attempt in range(3):
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
                raw = next(
                    (b.text for b in response.content if getattr(b, "type", None) == "text"),
                    "",
                )
                result = self._parse_json(raw, tag)
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "reasoning_tokens": 0,  # Anthropic does not separate reasoning tokens
                }
                self._write_debug_log(
                    tag=tag,
                    model=model,
                    request={"system": system, "messages": messages, "max_tokens": max_tokens},
                    response={
                        "stop_reason": response.stop_reason,
                        "content": raw,
                        "usage": usage,
                    },
                    parsed=result,
                    usage=usage,
                )
                return result
            except Exception as exc:
                last_exc = exc
                if getattr(exc, "status_code", None) == 400:
                    raise
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        "Anthropic call failed (attempt %d/3, tag=%s): %s. Retrying in %ds.",
                        attempt + 1, tag, exc, wait,
                    )
                    time.sleep(wait)
        raise last_exc

    def _create_openai(self, model, max_tokens, system, messages, tag):
        oai_messages = [{"role": "system", "content": system}] + messages
        last_exc = None
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=oai_messages,
                    response_format={"type": "json_object"},
                )
                msg = response.choices[0].message
                content = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", "") or ""
                # Extract token usage; reasoning_tokens is available for o1/o3/gpt-oss models
                u = getattr(response, "usage", None)
                input_toks = getattr(u, "prompt_tokens", 0) or 0
                output_toks = getattr(u, "completion_tokens", 0) or 0
                reasoning_toks = 0
                if u:
                    details = getattr(u, "completion_tokens_details", None)
                    if details:
                        reasoning_toks = getattr(details, "reasoning_tokens", 0) or 0
                usage = {
                    "input_tokens": input_toks,
                    "output_tokens": output_toks,
                    "reasoning_tokens": reasoning_toks,
                }
                result = self._parse_json(content, tag)
                self._write_debug_log(
                    tag=tag,
                    model=model,
                    request={"system": system, "messages": messages, "max_tokens": max_tokens},
                    response={
                        "finish_reason": response.choices[0].finish_reason,
                        "content": content,
                        "reasoning_content": reasoning,
                        "usage": usage,
                    },
                    parsed=result,
                    usage=usage,
                )
                return result
            except Exception as exc:
                last_exc = exc
                if getattr(exc, "status_code", None) == 400:
                    raise
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        "OpenAI call failed (attempt %d/3, tag=%s): %s. Retrying in %ds.",
                        attempt + 1, tag, exc, wait,
                    )
                    time.sleep(wait)
        raise last_exc

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str, tag: str = "") -> Dict:
        text = text.strip()
        # Strip markdown fences if the model wrapped its output anyway
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
            # Unwrap single-element list  e.g. [{"action": ...}]
            if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
                return result[0]
            logger.warning("JSON response is not a dict (tag=%s, type=%s)", tag, type(result).__name__)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON (tag=%s): %s | text: %s", tag, exc, text[:300])
        return {}

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------

    def _write_debug_log(
        self,
        tag: str,
        model: str,
        request: Dict,
        response: Dict,
        parsed: Dict,
        usage: Optional[Dict] = None,
    ) -> None:
        # Accumulate token counts
        if usage:
            for key, sess_key in [("input_tokens", "input"), ("output_tokens", "output"), ("reasoning_tokens", "reasoning")]:
                v = usage.get(key, 0) or 0
                self._task_tokens[sess_key] += v
                self._session_tokens[sess_key] += v
        # Always record in the per-task in-memory log for training data export
        self._task_log.append({
            "tag": tag,
            "model": model,
            "request": request,
            "response": response,
            "parsed_result": parsed,
            "token_usage": usage or {},
        })

        if not self._debug_dir:
            return
        self._call_counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        label = f"{tag}_" if tag else ""
        path = os.path.join(self._debug_dir, f"{self._call_counter:04d}_{label}{ts}.json")
        data = {
            "call_index": self._call_counter,
            "timestamp": ts,
            "tag": tag,
            "model": model,
            "request": request,
            "response": response,
            "parsed_result": parsed,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not write debug log %s: %s", path, exc)
