"""ReGAL LLM client — plain-text API calls for refactoring and synthesis.

Uses chat API (Anthropic or OpenAI-compatible vLLM). No JSON mode needed —
refactoring responses are parsed by regex in prompts.parse_result().
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY = 5.0


class RegalLLMClient:
    """
    Thin LLM client for ReGAL.

    Supports Anthropic (default) and OpenAI-compatible (vLLM) backends.
    Returns raw response text — callers do their own parsing.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        debug_dir: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug_dir = debug_dir

        self._task_log: list = []
        self._task_tokens: dict = {"input": 0, "output": 0, "reasoning": 0}
        self._session_tokens: dict = {"input": 0, "output": 0, "reasoning": 0}

        if base_url:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._backend = "openai"
        else:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._backend = "anthropic"

    def reset_task_log(self) -> None:
        self._task_log = []
        self._task_tokens = {"input": 0, "output": 0, "reasoning": 0}

    def get_task_log(self) -> list:
        return list(self._task_log)

    def get_task_token_usage(self) -> dict:
        return dict(self._task_tokens)

    def get_session_token_usage(self) -> dict:
        return dict(self._session_tokens)

    def restore_session_tokens(self, d: dict) -> None:
        for k in ("input", "output", "reasoning"):
            self._session_tokens[k] = int(d.get(k, 0))

    def call(
        self,
        prompt: str,
        tag: str = "regal",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call the LLM with a plain-text prompt.

        Returns the response text (stripped).
        """
        temp = temperature if temperature is not None else self.temperature
        mtok = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if self._backend == "anthropic":
                    response = self._client.messages.create(
                        model=self.model,
                        max_tokens=mtok,
                        temperature=temp,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.content[0].text.strip()
                    usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "reasoning_tokens": 0,
                    }
                else:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=mtok,
                        temperature=temp,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.choices[0].message.content.strip()
                    u = getattr(response, "usage", None)
                    details = getattr(u, "completion_tokens_details", None)
                    usage = {
                        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0 if details else 0,
                    }

                # Accumulate token counts
                for key, sess_key in [("input_tokens", "input"), ("output_tokens", "output"), ("reasoning_tokens", "reasoning")]:
                    v = usage.get(key, 0) or 0
                    self._task_tokens[sess_key] += v
                    self._session_tokens[sess_key] += v

                entry = {
                    "tag": tag,
                    "prompt": prompt,
                    "response": text,
                    "token_usage": usage,
                }
                self._task_log.append(entry)

                if self.debug_dir:
                    self._write_debug(entry, tag)

                return text

            except Exception as exc:
                logger.warning(
                    "LLM call attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY)
                else:
                    logger.error("All %d LLM attempts failed.", _MAX_RETRIES)
                    return ""

        return ""

    def _write_debug(self, entry: dict, tag: str) -> None:
        try:
            debug_path = Path(self.debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            count = len(list(debug_path.glob(f"{tag}_*.json")))
            fname = debug_path / f"{tag}_{count:04d}.json"
            fname.write_text(json.dumps(entry, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Could not write debug log: %s", exc)
