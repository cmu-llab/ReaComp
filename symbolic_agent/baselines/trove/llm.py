"""Plain-text LLM client for TroVE (and other baseline) prompts.

Unlike the main LLMClient, this client:
  - Does NOT append a JSON instruction to the system prompt
  - Returns the raw text response (str) rather than a parsed dict
  - Uses a completion-style interface (single prompt string → str)
  - Supports both Anthropic and OpenAI-compatible (vLLM) backends

The TroVE prompts end with a partial header ("**Solution**") and the
model is expected to complete the response freely.  Forcing JSON mode
would corrupt this format, so we need a separate client.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default generation parameters — match TroVE's original settings
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 512


class TroVELLMClient:
    """
    Backend-agnostic plain-text LLM client for TroVE generation.

    Parameters
    ----------
    backend : "anthropic" | "openai"
    base_url : str, optional
        Required for the "openai" backend.
    api_key : str, optional
    temperature : float
        Sampling temperature. Default: 0.3 (TroVE paper setting).
    top_p : float
        Nucleus sampling p. Default: 0.95 (TroVE paper setting).
    debug_dir : str, optional
        Directory for per-call debug JSON files.
    """

    def __init__(
        self,
        backend: str = "anthropic",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        debug_dir: Optional[str] = None,
    ):
        self.backend = backend
        self.temperature = temperature
        self.top_p = top_p
        self._call_counter = 0
        self._task_log: List[Dict] = []
        self._task_tokens: Dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}
        self._session_tokens: Dict[str, int] = {"input": 0, "output": 0, "reasoning": 0}

        if debug_dir:
            run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            self._debug_dir: Optional[str] = os.path.join(debug_dir, f"trove_run_{run_ts}")
            os.makedirs(self._debug_dir, exist_ok=True)
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
    # Task-level message log (mirrors LLMClient interface)
    # ------------------------------------------------------------------

    def reset_task_log(self) -> None:
        self._task_log = []
        self._task_tokens = {"input": 0, "output": 0, "reasoning": 0}

    def get_task_log(self) -> List[Dict]:
        return list(self._task_log)

    def get_task_token_usage(self) -> Dict[str, int]:
        return dict(self._task_tokens)

    def get_session_token_usage(self) -> Dict[str, int]:
        return dict(self._session_tokens)

    def restore_session_tokens(self, d: Dict[str, int]) -> None:
        """Restore accumulated session token counts from a checkpoint."""
        for k in ("input", "output", "reasoning"):
            self._session_tokens[k] = int(d.get(k, 0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        prompt: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tag: str = "",
    ) -> str:
        """
        Send a prompt to the LLM and return the raw text response.

        The prompt is sent as a user message.  No system prompt is added
        (TroVE prompts are self-contained).

        Returns "" on any error (consistent with run_solution returning
        is_success=False for empty output).
        """
        if self.backend == "anthropic":
            return self._call_anthropic(prompt, model, max_tokens, tag)
        else:
            return self._call_openai(prompt, model, max_tokens, tag)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _call_anthropic(self, prompt: str, model: str, max_tokens: int, tag: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        last_exc = None
        for attempt in range(3):
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                raw = next(
                    (b.text for b in response.content if getattr(b, "type", None) == "text"),
                    "",
                )
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "reasoning_tokens": 0,
                }
                self._record(tag, model, prompt, raw, max_tokens, usage)
                return raw
            except Exception as exc:
                last_exc = exc
                if getattr(exc, "status_code", None) == 400:
                    logger.warning(
                        "Anthropic call got 400 (tag=%s, likely multi-segment reasoning model output): %s",
                        tag, exc,
                    )
                    self._record(tag, model, prompt, "", max_tokens, {})
                    return ""
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        "Anthropic call failed (attempt %d/3, tag=%s): %s. Retrying in %ds.",
                        attempt + 1, tag, exc, wait,
                    )
                    time.sleep(wait)
        logger.warning("All Anthropic retries exhausted (tag=%s): %s", tag, last_exc)
        return ""

    def _call_openai(self, prompt: str, model: str, max_tokens: int, tag: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        last_exc = None
        for attempt in range(3):
            try:
                # Do NOT pass temperature/top_p for the openai/vLLM backend.
                # Reasoning models (gpt-oss-120b, o1, DeepSeek-R1, etc.) reject
                # non-default temperature values with a 400 BadRequest, which
                # caused TroVE to silently produce empty responses and an empty
                # toolbox.  temperature/top_p are passed for Anthropic only
                # (where they control sampling diversity for K-shot generation).
                response = self._client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    # No response_format — TroVE uses free-form text
                )
                msg = response.choices[0].message
                raw = msg.content or getattr(msg, "reasoning_content", "") or ""
                u = getattr(response, "usage", None)
                details = getattr(u, "completion_tokens_details", None)
                usage = {
                    "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0 if details else 0,
                }
                self._record(tag, model, prompt, raw, max_tokens, usage)
                return raw
            except Exception as exc:
                last_exc = exc
                if getattr(exc, "status_code", None) == 400:
                    logger.warning(
                        "OpenAI call got 400 (tag=%s, likely multi-segment reasoning model output): %s",
                        tag, exc,
                    )
                    self._record(tag, model, prompt, "", max_tokens, {})
                    return ""
                if attempt < 2:
                    wait = 5 * (2 ** attempt)
                    logger.warning(
                        "OpenAI call failed (attempt %d/3, tag=%s): %s. Retrying in %ds.",
                        attempt + 1, tag, exc, wait,
                    )
                    time.sleep(wait)
        logger.warning("All OpenAI retries exhausted (tag=%s): %s", tag, last_exc)
        return ""

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

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _record(self, tag: str, model: str, prompt: str, raw: str, max_tokens: int, usage: Dict) -> None:
        # Accumulate token counts
        for key, sess_key in [("input_tokens", "input"), ("output_tokens", "output"), ("reasoning_tokens", "reasoning")]:
            v = usage.get(key, 0) or 0
            self._task_tokens[sess_key] += v
            self._session_tokens[sess_key] += v
        entry = {
            "tag": tag,
            "model": model,
            "request": {"prompt": prompt, "max_tokens": max_tokens},
            "response": {"content": raw, "usage": usage},
            "token_usage": usage,
        }
        self._task_log.append(entry)

        if not self._debug_dir:
            return
        self._call_counter += 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        label = f"{tag}_" if tag else ""
        path = os.path.join(self._debug_dir, f"{self._call_counter:04d}_{label}{ts}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not write debug log %s: %s", path, exc)
