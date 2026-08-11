"""
PBEBench/SLR-Bench TroVE controller.

ADDITIVE module (new file). Subclasses the paper-faithful `TroVEController`
without modifying it. The ONLY behavioural change is candidate generation: we
build DSL-aware IMPORT / CREATE / SKIP prompts (via pbe_slr_prompts) from the
full task record, instead of the generic `_task_text(str(dict))` path that
silently fed TroVE an empty task for PBEBench records (whose keys are
`inputs`/`outputs`, not `question`/`prompt`/`task`).

Everything else — 3K concurrent candidate generation, sandboxed execution,
reward-based selection with AST-node tiebreak, toolbox add/increment/trim,
checkpointing — is inherited unchanged from TroVEController.
"""

import asyncio
import json
import logging
import re

import aiohttp

from .controller import (
    TroVEController,
    _used_toolbox_names,
)
from .pbe_slr_prompts import build_create_prompt, build_import_prompt, build_skip_prompt

logger = logging.getLogger(__name__)


async def _call_llm_no_think(session, system, user, base_url, model, api_key, max_tokens,
                             enable_thinking: bool = False, temperature=0.7):
    """
    Local variant of controller._call_llm with configurable Qwen3.6 thinking mode.

    Two operating points:

    * ``enable_thinking=False`` (default, original TroVE port): Qwen emits JSON
      directly. Needed because with thinking ON and a small budget the model
      spends the whole allocation on chain-of-thought and never emits the JSON.

    * ``enable_thinking=True`` + large ``max_tokens`` + ``temperature=None``:
      the **DirectSolve-matched** setting. DirectSolve runs Qwen via OpenHands
      with reasoning_effort=high (thinking ON), max_tokens=16384, and the vLLM
      default temperature. Matching all three makes the TroVE-vs-DirectSolve
      comparison compute-fair (same per-call generation budget), so the accuracy
      gap can't be dismissed as "TroVE was under-resourced".

    When ``temperature`` is None the field is omitted from the payload so vLLM
    applies its own default (exactly what the OpenHands DirectSolve path does).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            # Qwen thinking mode may return CoT in a separate reasoning_content
            # field; keep it out of the parsed answer but count its tokens.
            usage = data.get("usage", {})
            return content, usage
    except Exception as exc:
        logger.warning("TroVE LLM call failed: %s", exc)
        return "", {}


def _extract_json(raw: str) -> dict:
    """
    Robust JSON extraction: handle pure JSON, ```json fences, and a JSON object
    embedded in surrounding prose (e.g. a stray thinking preamble). Falls back to
    the first balanced {...} block. Returns {} on failure.
    """
    if not raw:
        return {}
    s = raw.strip()
    # Strip a leading <think>...</think> block if the template emitted one.
    s = re.sub(r"^<think>.*?</think>\s*", "", s, flags=re.DOTALL)
    # Strip markdown fences.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Find the first balanced top-level {...} object.
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    block = s[start:i + 1]
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    return {}


class PbeSlrTroVEController(TroVEController):
    """
    TroVE for PBEBench / SLR-Bench.

    Parameters
    ----------
    task_type : "pbe" | "slr"
        Selects the DSL-aware prompt family and stdout-format instructions.
    max_programs : int
        PBE only: cascade budget surfaced in the prompt (5 for Lite, 20 for Hard).
    (all other parameters are identical to TroVEController)
    """

    def __init__(self, *args, task_type: str = "pbe", max_programs: int = 5,
                 enable_thinking: bool = False, temperature=0.7,
                 request_timeout: float = 120, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = task_type
        self.max_programs = max_programs
        # DirectSolve-matched generation knobs (see _call_llm_no_think docstring).
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        # Per-request HTTP timeout; large K + thinking mode needs a bigger value.
        self.request_timeout = request_timeout

    def solve(self, task_input, reward_fn, entry):
        """
        Override: build the aiohttp connector INSIDE the running event loop.

        The base ``TroVEController.solve`` constructs ``aiohttp.TCPConnector``
        before the loop is running; aiohttp >=3.10 requires a running loop at
        connector construction and raises ``RuntimeError: no running event
        loop``. We provide a running loop first, patch the candidate-generation
        step to use an in-loop session, then delegate the (synchronous)
        selection / toolbox-update logic to the base implementation unchanged.
        """
        # Pre-build the candidates once, inside a running loop, and stash them so
        # the base solve()'s own gather sees an already-built session.
        loop = asyncio.new_event_loop()
        try:
            async def _run():
                connector = aiohttp.TCPConnector(limit=3 * self.k)
                timeout = aiohttp.ClientTimeout(total=getattr(self, "request_timeout", 120))
                async with aiohttp.ClientSession(
                    connector=connector, timeout=timeout
                ) as session:
                    return await self._generate_candidates(task_input, session)

            candidates = loop.run_until_complete(_run())
        finally:
            loop.close()

        return self._finalize(candidates, reward_fn, entry)

    async def _generate_candidates(self, task_input, session: aiohttp.ClientSession):
        """
        Override: build DSL-aware prompts from the full record (task_input here is
        the full task record dict passed through from solve()).
        Returns list of dicts: {mode, raw, obj, code, fn_def, usage}.
        """
        record = task_input  # full record — see solve() below
        listing = self.toolbox.as_listing_with_signatures()
        has_toolbox = len(self.toolbox) > 0

        prompts = {
            "skip": build_skip_prompt(record, self.task_type, self.max_programs),
            "import": (
                build_import_prompt(record, self.task_type, listing, self.max_programs)
                if has_toolbox else None
            ),
            "create": build_create_prompt(record, self.task_type, listing, self.max_programs),
        }

        async def one(mode: str, system: str, user: str) -> dict:
            raw, usage = await _call_llm_no_think(
                session, system, user,
                self.base_url, self.model, self.api_key, self.max_tokens,
                enable_thinking=self.enable_thinking, temperature=self.temperature,
            )
            obj = _extract_json(raw)
            code = obj.get("code", "")
            fn_def = obj.get("function")
            return {"mode": mode, "raw": raw, "obj": obj, "code": code, "fn_def": fn_def, "usage": usage}

        coros = []
        for mode, prompt_pair in prompts.items():
            if prompt_pair is None:
                continue
            system, user = prompt_pair
            for _ in range(self.k):
                coros.append(one(mode, system, user))

        return await asyncio.gather(*coros)

    def _finalize(self, candidates, reward_fn, entry):
        """
        Synchronous selection + toolbox update, replicating the tail of the base
        ``TroVEController.solve`` verbatim (lines after candidate generation).
        Kept here only because we had to override ``solve`` to fix the aiohttp
        loop-construction bug; behaviour is identical to the base.
        """
        best = self._select_best(candidates, reward_fn, entry)

        total_prompt_tokens = sum(c.get("usage", {}).get("prompt_tokens", 0) for c in candidates)
        total_completion_tokens = sum(c.get("usage", {}).get("completion_tokens", 0) for c in candidates)

        if best is None:
            self._n_processed += 1
            return {
                "solved": False,
                "answer": None,
                "best_reward": 0.0,
                "mode": None,
                "toolbox_size": len(self.toolbox),
                "candidates": [],
                "token_usage": {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens},
            }

        if best["mode"] == "create" and best["fn_def"]:
            fn = best["fn_def"]
            name = fn.get("name", "").strip()
            desc = fn.get("description", "").strip()
            code = fn.get("code", "").strip()
            if name and code:
                self.toolbox.add(name, desc, code)
                logger.info("TroVE: added '%s' to toolbox (%d total)", name, len(self.toolbox))

        used = _used_toolbox_names(best.get("code", ""), self.toolbox)
        if used:
            self.toolbox.increment_usage(used)

        self._n_processed += 1

        if self._n_processed % self.trim_every == 0:
            removed = self.toolbox.trim(self._n_processed)
            if removed:
                logger.info("TroVE: trimmed %d functions: %s", len(removed), removed)

        return {
            "solved": best["reward"] >= 1.0,
            "answer": best.get("answer"),
            "best_reward": best["reward"],
            "mode": best["mode"],
            "toolbox_size": len(self.toolbox),
            "n_processed": self._n_processed,
            "token_usage": {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens},
            "candidates": [
                {k: v for k, v in c.items() if k not in ("raw", "usage")}
                for c in candidates
            ],
        }
