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

    def get_task_log(self) -> list:
        return list(self._task_log)

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
                else:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=mtok,
                        temperature=temp,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.choices[0].message.content.strip()

                entry = {
                    "tag": tag,
                    "prompt": prompt,
                    "response": text,
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
