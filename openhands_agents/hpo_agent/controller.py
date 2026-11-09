"""LLM hyperparameter-optimisation agent for the autoresearch HPO domain.

One LLM call per trial: the model sees the search space, the recent trial history and
the incumbent, and answers with the next configuration. This is the paper's
"Karpathy Agent (14 HPs)" (arXiv:2603.24647; upstream
``backends/karpathy_agent_hps_backend.py``) written as a policy for
``rewards.autoresearch_hpo.run_episode``:

    agent = HPOAgentController(base_url=..., model=...)
    run_episode(agent, entry, log_path=...)

Every call returns the proposed config plus the prompt, the model's answer and the
content of its reasoning channel (when the server exposes one), which ``run_episode``
stores in the episode log. Those per-trial records are the reasoning traces the
solver-induction step consumes.

Prompt styles:
  cot     the same context, but the model is asked to reason step by step and then
          give the configuration in a ```json block (default; gives a visible trace
          for models without a separate reasoning channel). The answer is the last
          JSON object that names a hyperparameter; an answer cut off by the token
          limit is not used (``parse_cot_answer``).
  paper   the upstream prompt: answer with a JSON object only, read by upstream's
          rule (``parse_paper_answer``). With ``max_tokens=2048`` and
          ``parse_retries=0`` this is the upstream agent.

Differences from upstream that apply to both styles:
  * search-space lines use ``describe_search_space()`` instead of ConfigSpace's repr,
    and the hardware sentence comes from the task record;
  * the answer goes through ``normalize_config``, so a WINDOW_PATTERN outside the
    listed choices or a non-numeric value is rejected (upstream passes the former to
    train.py unchecked and casts strings such as ``"12"``);
  * the request carries a seed derived from the task seed, trial and attempt.
As upstream, an answer that cannot be used falls back to a random configuration
(marked ``source="random_fallback"``), and a transport error stops the episode
(``EpisodeInterrupted``) instead of being recorded as a trial.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

import openai

from rewards.autoresearch_hpo import (
    HP_NAMES, STATUS_OK, EpisodeInterrupted, InvalidConfig, describe_search_space,
    normalize_config, sample_config,
)

PROMPT_STYLES = ("cot", "paper")
HISTORY_WINDOW = 20      # upstream shows the last 20 evaluations

_CONTEXT = """\
You are optimizing hyperparameters for a GPT-2 scale transformer trained on climbmix-400b-shuffle.
The goal is to minimize val_bpb (validation bits-per-byte).
Training runs on a single {gpu} GPU ({total_vram:g}GB VRAM, ~{training_vram:g}GB available for training) — configs with very high DEPTH, large HEAD_DIM,
or large DEVICE_BATCH_SIZE may cause OOM (out of memory) crashes.

Current search space with bounds:
{space}

Previous evaluations (config → result):
{history}

{incumbent}
"""

_INSTRUCTIONS = {
    "paper": """\
Suggest the next configuration to try. Learn from both successful runs AND failures (OOM crashes
mean the model was too large for the GPU). Respond with a JSON object mapping HP names to values.
Only include HPs from the search space. Be creative but principled — consider what you know
about transformer training dynamics.

Respond ONLY with a valid JSON object, no explanation.""",
    "cot": """\
{progress}

Suggest the next configuration to try. Learn from both successful runs AND failures (OOM crashes
mean the model was too large for the GPU). Only include HPs from the search space. Be creative
but principled — consider what you know about transformer training dynamics.

First reason step by step: what the evaluations so far show, which hyperparameters look
responsible for the gains and the failures, how much memory headroom is left, and what you
expect from the configuration you propose. Then give the configuration as a JSON object mapping
HP names to values, in a ```json code block at the very end of your answer.""",
}


def format_history(history: List[Dict[str, Any]], window: int = HISTORY_WINDOW) -> str:
    """Upstream's history block: the last ``window`` evaluations, one per line."""
    if not history:
        return "  (no evaluations yet)"
    lines = []
    for trial in history[-window:]:
        config = json.dumps(trial["config"]) if trial["config"] is not None else "(no valid configuration)"
        if trial["status"] == STATUS_OK:
            lines.append(f"  {config} → val_bpb={trial['val_bpb']}")
        elif trial["error"]:
            lines.append(f"  {config} → CRASHED ({trial['error']})")
        else:
            lines.append(f"  {config} → FAILED")
    return "\n".join(lines)


def incumbent(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The best successful trial so far (earliest on ties), or None."""
    best = None
    for trial in history:
        if trial["status"] == STATUS_OK and (best is None or trial["val_bpb"] < best["val_bpb"]):
            best = trial
    return best


def build_prompt(task: Dict[str, Any], history: List[Dict[str, Any]], style: str = "cot") -> str:
    best = incumbent(history)
    incumbent_info = "" if best is None else (
        f"Current best config (val_bpb={best['val_bpb']}):\n{json.dumps(best['config'], indent=2)}")
    hardware = task["hardware"]
    context = _CONTEXT.format(
        gpu=hardware["gpu"], total_vram=hardware["total_vram_gb"],
        training_vram=hardware["training_vram_gb"], space=describe_search_space(),
        history=format_history(history), incumbent=incumbent_info)
    used_h = sum(trial["wall_time_s"] for trial in history) / 3600.0
    progress = (f"Trials so far: {len(history)}. Training time used: {used_h:.2f} h of "
                f"{task['budget']['time_budget_s'] / 3600.0:g} h "
                f"(a full trial takes {task['budget']['trial_time_budget_s'] / 60.0:g} min).")
    return context + "\n" + _INSTRUCTIONS[style].format(progress=progress)


def parse_paper_answer(text: str) -> Dict[str, Any]:
    """Upstream's reading of an answer (``KarpathyAgentHPsBackend.ask``).

    The text between the first pair of ``` fences (minus a leading ``json``), or the
    whole text when there is no fence, has to be one JSON object. As upstream, an
    object that names no hyperparameter is the default configuration.
    """
    text = (text or "").strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        config = json.loads(text)
    except ValueError as exc:
        raise InvalidConfig(f"the answer is not a JSON object: {exc}") from None
    normalize_config(config)
    return config


def parse_cot_answer(text: str) -> Dict[str, Any]:
    """The configuration a step-by-step answer ends with.

    The answer is the last JSON object in the text that names a hyperparameter (or
    wraps one that does in ``{"config": ...}``); objects quoted earlier in the
    reasoning are never used in its place. Raises ``InvalidConfig`` when there is no
    such object, when it is not a valid configuration, or when the text after it
    opens another object that does not parse (a final answer that is malformed or
    was cut off).
    """
    text = text or ""
    decoder = json.JSONDecoder()
    answer, answer_end, unparsed = None, 0, []
    position = text.find("{")
    while position != -1:
        try:
            value, end = decoder.raw_decode(text, position)
        except ValueError:
            unparsed.append(position)
            position = text.find("{", position + 1)
            continue
        if isinstance(value.get("config"), dict):
            value = value["config"]
        if any(name in value for name in HP_NAMES):
            answer, answer_end = value, end
        position = text.find("{", end)
    if answer is None:
        raise InvalidConfig("incomplete JSON object in the answer" if unparsed
                            else "no configuration in the answer")
    if any(position >= answer_end for position in unparsed):
        raise InvalidConfig("the answer ends with an incomplete JSON object")
    normalize_config(answer)
    return answer


ANSWER_PARSERS = {"cot": parse_cot_answer, "paper": parse_paper_answer}


def agent_policy_name(model: str, prompt_style: str, temperature: float, max_tokens: int,
                      parse_retries: int) -> str:
    """Identity of an agent in episode logs and result tables.

    Every setting that changes what the agent proposes is part of it, so a log
    started with one set of settings is not resumed with another and results
    obtained with different settings are not averaged together.
    """
    return (f"hpo_agent[{served_model_name(model)}|{prompt_style}|T={temperature:g}|"
            f"max_tokens={max_tokens}|retries={parse_retries}]")


def served_model_name(model: str) -> str:
    """Launch scripts pass litellm-style names ("openai/<served name>"); the OpenAI
    client talks to the server directly and wants the served name."""
    return model[len("openai/"):] if model.startswith("openai/") else model


# Failures of the LLM service, as opposed to unusable answers. Upstream re-raises the
# same set so that an outage is not recorded as a run of random trials.
TRANSPORT_ERRORS = (openai.APIConnectionError, openai.APITimeoutError, openai.APIStatusError,
                    ConnectionError, TimeoutError)


class HPOAgentController:
    """Policy callable: ``agent(task, history) -> {"config": ..., <trace fields>}``."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 prompt_style: str = "cot", temperature: float = 0.7, max_tokens: int = 8192,
                 parse_retries: int = 2, request_timeout: float = 1800.0, client=None):
        if prompt_style not in PROMPT_STYLES:
            raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}")
        self.model = served_model_name(model)
        self.prompt_style = prompt_style
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parse_retries = parse_retries
        self.name = agent_policy_name(model, prompt_style, temperature, max_tokens, parse_retries)
        self._client = client or openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=request_timeout)

    def _complete(self, prompt: str, seed: int) -> Dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature, max_tokens=self.max_tokens, seed=seed)
        except TRANSPORT_ERRORS as exc:
            raise EpisodeInterrupted(f"LLM request failed: {type(exc).__name__}: {exc}") from exc
        message = response.choices[0].message
        extra = getattr(message, "model_extra", None) or {}
        reasoning = (extra.get("reasoning") or extra.get("reasoning_content")
                     or getattr(message, "reasoning_content", None) or "")
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        return {
            "content": message.content or "",
            "reasoning": reasoning if isinstance(reasoning, str) else "",
            "finish_reason": getattr(response.choices[0], "finish_reason", None),
            "usage": {
                "input": getattr(usage, "prompt_tokens", 0) or 0,
                "output": getattr(usage, "completion_tokens", 0) or 0,
                "reasoning": getattr(details, "reasoning_tokens", 0) or 0,
            },
        }

    def __call__(self, task: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = build_prompt(task, history, self.prompt_style)
        usage = {"input": 0, "output": 0, "reasoning": 0}
        trace = {"prompt": prompt, "prompt_style": self.prompt_style, "model": self.model,
                 "response": "", "reasoning": "", "finish_reason": None, "parse_error": None}
        for attempt in range(self.parse_retries + 1):
            seed = (task["seed"] * 1_000_003 + len(history)) * 101 + attempt
            answer = self._complete(prompt, seed)
            for key in usage:
                usage[key] += answer["usage"][key]
            # Upstream: when the visible answer is empty (some servers return everything
            # in the reasoning field), look for the configuration in the reasoning text.
            text = answer["content"] if answer["content"].strip() else answer["reasoning"]
            trace.update(response=answer["content"], reasoning=answer["reasoning"],
                         finish_reason=answer["finish_reason"], attempts=attempt + 1)
            try:
                if self.prompt_style == "cot" and answer["finish_reason"] == "length":
                    # The configuration comes last in this style, so whatever a
                    # truncated answer contains is not the model's answer.
                    raise InvalidConfig("the answer was cut off by the token limit")
                config = ANSWER_PARSERS[self.prompt_style](text)
            except InvalidConfig as exc:
                trace["parse_error"] = str(exc)
                continue
            trace["parse_error"] = None
            return dict(trace, config=config, source="llm", token_usage=usage)
        fallback = sample_config(random.Random(f"hpo_agent_fallback/{task['seed']}/{len(history)}"))
        return dict(trace, config=fallback, source="random_fallback", token_usage=usage)
