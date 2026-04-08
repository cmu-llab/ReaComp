"""
Best-of-K with CoT — fully async controller.

Two-stage design:
  Stage 1: Generate all K samples for ALL N tasks concurrently via async HTTP.
           No waiting for one task to finish before starting the next.
  Stage 2: For each task, execute all K codes in the sandbox and score with
           reward_fn. Pick best. This stage can be run after all generations
           are done, or interleaved — run.py decides.

Each LLM call asks for {"reasoning": "...", "code": "..."} in one shot.
The model reasons first (CoT), then writes code. We only execute the code.
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, Optional

import aiohttp

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert programmer. For each task you will:
1. Reason step-by-step about how to solve it (chain of thought).
2. Write a complete Python solution that prints the final answer.

Respond with a single JSON object:
{"reasoning": "<your step-by-step thinking>", "code": "<complete Python program>"}

The code must print the answer to stdout. No markdown fences."""


def _extract_code(raw: str) -> Optional[str]:
    """Parse code from {"reasoning":..., "code":...} JSON response."""
    raw = raw.strip()
    # Try JSON parse first
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "code" in obj:
            return obj["code"]
    except json.JSONDecodeError:
        pass
    # Fallback: find first ```python ... ``` block
    m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Last resort: find def/import lines
    lines = [l for l in raw.splitlines() if l.strip().startswith(("def ", "import ", "from ", "print("))]
    return "\n".join(lines) if lines else None


async def _generate_one(
    session: aiohttp.ClientSession,
    prompt: str,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    task_idx: int,
    attempt_idx: int,
) -> dict:
    """Single async LLM call. Returns {task_idx, attempt_idx, raw, code}."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            raw = data["choices"][0]["message"]["content"] or ""
            code = _extract_code(raw)
            return {
                "task_idx": task_idx,
                "attempt_idx": attempt_idx,
                "raw": raw,
                "code": code,
            }
    except Exception as exc:
        logger.warning("best_of_k generate failed (task=%d, attempt=%d): %s", task_idx, attempt_idx, exc)
        return {"task_idx": task_idx, "attempt_idx": attempt_idx, "raw": "", "code": None}


def _build_prompt(task_input: Any) -> str:
    if isinstance(task_input, dict):
        return (
            task_input.get("question")
            or task_input.get("prompt")
            or task_input.get("task")
            or str(task_input)
        )
    return str(task_input)


class BestOfKController:
    """
    Best-of-K with CoT.

    Usage
    -----
    controller = BestOfKController(base_url=..., model=..., k=8)

    # Stage 1: generate all samples for all tasks
    all_samples = await controller.generate_all(task_inputs)  # list[list[dict]]

    # Stage 2: score and pick best per task
    for i, (samples, entry) in enumerate(zip(all_samples, entries)):
        result = controller.score_and_pick(samples, reward_fn, entry, sandbox)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        k: int = 8,
        max_tokens: int = 4096,
        temperature: float = 0.8,
        max_concurrent: int = 64,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.k = k
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent

    async def generate_all(self, task_inputs: list[Any]) -> list[list[dict]]:
        """
        Generate K samples for every task concurrently.

        Returns a list of length len(task_inputs), each element is a list of
        K dicts with keys {task_idx, attempt_idx, raw, code}.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def bounded(coro):
            async with semaphore:
                return await coro

        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for i, task_input in enumerate(task_inputs):
                prompt = _build_prompt(task_input)
                for j in range(self.k):
                    tasks.append(bounded(_generate_one(
                        session, prompt, self.base_url, self.model,
                        self.api_key, self.max_tokens, self.temperature, i, j,
                    )))
            all_results = await asyncio.gather(*tasks)

        # Group by task_idx
        grouped: list[list[dict]] = [[] for _ in task_inputs]
        for r in all_results:
            grouped[r["task_idx"]].append(r)
        return grouped

    def score_and_pick(
        self,
        samples: list[dict],
        reward_fn: Callable,
        entry: dict,
        sandbox,
    ) -> dict:
        """
        Execute each sample's code in sandbox, score with reward_fn, return best.

        Returns a result dict compatible with run.py's JSONL output format.
        """
        best_reward = -1.0
        best_code = None
        best_answer = None
        scored = []

        for s in samples:
            code = s.get("code")
            if not code:
                scored.append({**s, "reward": 0.0, "stdout": "", "ok": False})
                continue

            ok, stdout, stderr = sandbox.run_code(code)
            answer = stdout.strip() if ok else None
            reward_result = reward_fn(answer, ok, entry)
            reward_value = float(reward_result.get("value", 0.0))

            scored.append({**s, "reward": reward_value, "stdout": stdout, "stderr": stderr, "ok": ok})

            if reward_value > best_reward:
                best_reward = reward_value
                best_code = code
                best_answer = answer

            if best_reward >= 1.0:
                break  # early exit

        logger.info("best_of_k: best_reward=%.3f over %d samples", best_reward, len(samples))
        return {
            "solved": best_reward >= 1.0,
            "answer": best_answer,
            "best_reward": best_reward,
            "best_code": best_code,
            "samples": scored,
        }
