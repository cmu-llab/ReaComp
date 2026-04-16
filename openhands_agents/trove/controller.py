"""
TroVE controller — paper-faithful clean rewrite.

Algorithm (Section 3 of Wang et al. 2024):
  For each task (streaming, online):
    1. Generate K candidates in each of 3 modes (IMPORT, CREATE, SKIP) = 3K total.
       All 3K LLM calls are made concurrently within the task.
    2. Execute all candidates in the sandbox.
    3. Select the best by reward (our deviation from the paper's self-consistency,
       since we have external reward functions). Tiebreak: fewest AST nodes.
    4. If best candidate used CREATE mode, add its new function to the toolbox.
       Increment usage counts for any toolbox functions used in the solution.
    5. Every --trim-every tasks: trim functions below usage threshold.

Toolbox = PkgLibrary at pkg_dir/toolbox/ (per-function .py files).
"""

import ast
import asyncio
import json
import logging
import re
from typing import Any, Callable, Optional

import aiohttp

from ..pkg_library import PkgLibrary
from .prompts import build_create_prompt, build_import_prompt, build_skip_prompt

logger = logging.getLogger(__name__)

MODES = ("skip", "import", "create")


def _extract_response(raw: str) -> dict:
    """Parse JSON response, stripping markdown fences if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _ast_node_count(code: str) -> int:
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except SyntaxError:
        return 999


def _used_toolbox_names(code: str, toolbox: PkgLibrary) -> list[str]:
    """Find toolbox function names referenced in the solution code."""
    known = set(toolbox.names())
    found = []
    for name in known:
        if re.search(rf"\b{re.escape(name)}\b", code):
            found.append(name)
    return found


async def _call_llm(
    session: aiohttp.ClientSession,
    system: str,
    user: str,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            content = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            return content, usage
    except Exception as exc:
        logger.warning("TroVE LLM call failed: %s", exc)
        return "", {}


class TroVEController:
    """
    TroVE baseline with sandboxed execution and reward-based selection.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible vLLM endpoint.
    model : str
        Model name as served.
    api_key : str
    toolbox : PkgLibrary
        The shared toolbox (pkg_dir/toolbox/).
    sandbox : ApptainerSandbox
    k : int
        Samples per mode (total = 3k per task).
    max_tokens : int
    trim_every : int
        Trim toolbox every N tasks processed.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        toolbox: PkgLibrary,
        sandbox,
        api_key: str = "EMPTY",
        k: int = 5,
        max_tokens: int = 4096,
        trim_every: int = 200,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.toolbox = toolbox
        self.sandbox = sandbox
        self.k = k
        self.max_tokens = max_tokens
        self.trim_every = trim_every
        self._n_processed = 0

    async def _generate_candidates(
        self, task_input: Any, session: aiohttp.ClientSession
    ) -> list[dict]:
        """
        Generate 3K candidates concurrently (K per mode).
        Returns list of dicts: {mode, raw, obj, code, fn_def}.
        """
        listing = self.toolbox.as_listing_with_signatures()
        has_toolbox = len(self.toolbox) > 0

        # Build prompts for each mode
        prompts = {
            "skip": build_skip_prompt(task_input),
            "import": build_import_prompt(task_input, listing) if has_toolbox else None,
            "create": build_create_prompt(task_input, listing),
        }

        async def one(mode: str, system: str, user: str) -> dict:
            raw, usage = await _call_llm(
                session, system, user,
                self.base_url, self.model, self.api_key, self.max_tokens,
            )
            obj = _extract_response(raw)
            code = obj.get("code", "")
            fn_def = obj.get("function")  # None for skip/import
            return {"mode": mode, "raw": raw, "obj": obj, "code": code, "fn_def": fn_def, "usage": usage}

        coros = []
        for mode, prompt_pair in prompts.items():
            if prompt_pair is None:
                continue
            system, user = prompt_pair
            for _ in range(self.k):
                coros.append(one(mode, system, user))

        return await asyncio.gather(*coros)

    def _select_best(
        self,
        candidates: list[dict],
        reward_fn: Callable,
        entry: dict,
    ) -> Optional[dict]:
        """
        Execute all candidates, score with reward_fn, return best.
        Tiebreak: fewest AST nodes (simpler solution).
        Returns None if all candidates fail execution.
        """
        scored = []
        for c in candidates:
            code = c.get("code", "").strip()
            if not code:
                continue
            ok, stdout, stderr = self.sandbox.run_code(
                code, lib_dir=self.toolbox.pkg_dir if len(self.toolbox) > 0 else None
            )
            answer = stdout.strip() if ok else None
            reward_result = reward_fn(answer, ok, entry)
            reward_value = float(reward_result.get("value", 0.0))
            scored.append({
                **c,
                "ok": ok,
                "stdout": stdout,
                "stderr": stderr,
                "answer": answer,
                "reward": reward_value,
                "n_nodes": _ast_node_count(code),
            })

        if not scored:
            return None

        scored.sort(key=lambda x: (-x["reward"], x["n_nodes"]))
        return scored[0]

    def solve(
        self,
        task_input: Any,
        reward_fn: Callable,
        entry: dict,
    ) -> dict:
        """
        Solve one task. Generates 3K candidates, selects best, updates toolbox.
        Returns a result dict for JSONL output.
        """
        loop = asyncio.new_event_loop()
        try:
            connector = aiohttp.TCPConnector(limit=3 * self.k)
            timeout = aiohttp.ClientTimeout(total=120)

            async def _run():
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    return await self._generate_candidates(task_input, session)

            candidates = loop.run_until_complete(_run())
        finally:
            loop.close()

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

        # Update toolbox
        if best["mode"] == "create" and best["fn_def"]:
            fn = best["fn_def"]
            name = fn.get("name", "").strip()
            desc = fn.get("description", "").strip()
            code = fn.get("code", "").strip()
            if name and code:
                self.toolbox.add(name, desc, code)
                logger.info("TroVE: added '%s' to toolbox (%d total)", name, len(self.toolbox))

        # Increment usage for toolbox functions referenced in solution
        used = _used_toolbox_names(best.get("code", ""), self.toolbox)
        if used:
            self.toolbox.increment_usage(used)

        self._n_processed += 1

        # Periodic trim
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

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "n_processed": self._n_processed,
            "toolbox": self.toolbox.to_dict(),
        }

    def from_dict(self, data: dict) -> None:
        self._n_processed = data.get("n_processed", 0)
        # Toolbox state is already on disk (pkg_dir); _meta.json is the source of truth.
        # Just sync the in-memory meta.
        self.toolbox._load_meta()
