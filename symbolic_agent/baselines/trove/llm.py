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
from typing import Dict, List, Optional

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
                response = self._client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    # No response_format — TroVE uses free-form text
                )
                raw = response.choices[0].message.content or ""
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
