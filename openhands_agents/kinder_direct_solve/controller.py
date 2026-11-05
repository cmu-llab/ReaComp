"""
KinDER DirectSolve controller — one OpenHands conversation per KinDER instance.

The agent sees the environment description and the instance's initial state, runs
Python against a simulator of that instance (``domains.kinder.sandbox``: an environment
server plus a strict runner where only the standard library, numpy, scipy and
``kinder_client`` import), and submits a ``GeneratedApproach`` policy.  The harness
then rolls the policy out on the instance in a separate process and verifies the
executed actions by exact replay (``KinderDomain.solve`` / ``rewards/kinder.py``).

Rows serve three roles in the ReaComp pipeline:
  * traces: ``trajectory`` (reasoning, code, output per step) and the submitted
    ``policy_code`` become solver-induction demos (``scripts/build_demos.py``);
  * LLM baseline: the per-instance agent with simulator access, run on the test split;
  * fallback of the solver-first hybrid (``scripts/hybrid_gate.py``).
"""

import json
import logging
import os
import shutil
import tempfile
import time

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation
from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent

from domains.kinder import prompts as P
from domains.kinder.domain import evaluate_policy_source
from domains.kinder.sandbox import KinderSandbox

from .tools import KDSPythonTool, KDSSubmitTool

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert robotics programmer. You solve ONE instance of a 2D robot manipulation task
by writing a policy program, testing it against a simulator of that instance, and submitting it.

Tools
  python(code, reason)
      Runs `code` as a Python script in a fresh process inside your working directory; files you
      write there persist between calls. Importable: the standard library, numpy, scipy and
      `kinder_client` (the simulator client). Each call has a {code_timeout}s time limit and output
      is truncated to {max_output_chars} characters, so print summaries rather than every step.
  submit(code, notes)
      Submits your final policy: the complete source of a module defining `GeneratedApproach`.
      The harness runs it once on this instance and the session ends. Submit as soon as your own
      rollout reports solved=True, or your best policy before you run out of turns
      ({max_steps} in total). Never stop without calling submit.

Simulator (inside python)
{client_usage}

Policy interface
{policy_interface}
Approach
  - Study the initial state: positions and sizes of the robot, the objects and the goal region.
  - Plan in phases (for example: move, align, extend arm, vacuum on, carry, place, vacuum off) and
    implement them as a small state machine or as a plan computed in reset().
  - Test with rollout(); when it fails, step through the episode and print the state at the phase
    boundaries to see what went wrong (a move that would collide is not executed: the robot stays put).
  - Write the policy to a file (e.g. approach.py) and test it with load_approach("approach.py"), so
    the code you submit is exactly the code you tested.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory
# ──────────────────────────────────────────────────────────────────────────────

NUDGE = ("You have not called submit. Call submit now with the complete source of your best GeneratedApproach "
         "policy (for example the contents of approach.py), even if it does not solve the instance yet.")


def llm_turns(events) -> int:
    """LLM responses so far: agent messages plus distinct responses behind tool calls."""
    responses = {e.llm_response_id for e in events if isinstance(e, ActionEvent)}
    messages = sum(1 for e in events if isinstance(e, MessageEvent) and e.source == "agent")
    return len(responses) + messages


def _reasoning(event: ActionEvent) -> str:
    if event.reasoning_content:
        return event.reasoning_content
    blocks = [getattr(b, "thinking", "") or "" for b in (event.thinking_blocks or [])]
    return "\n".join(b for b in blocks if b)


def trajectory_steps(events) -> list:
    """One step per tool call: the model's reasoning and message, the call, and the observation."""
    steps, by_call = [], {}
    for e in events:
        if isinstance(e, ActionEvent) and e.action is not None:
            step = {
                "tool": e.tool_name,
                "reasoning": _reasoning(e),
                "thought": "".join(getattr(t, "text", "") for t in (e.thought or [])),
                "reason": getattr(e.action, "reason", None) or getattr(e.action, "notes", "") or "",
                "code": getattr(e.action, "code", ""),
                "output": "",
            }
            by_call[e.tool_call_id] = step
            steps.append(step)
        elif isinstance(e, ObservationEvent) and e.tool_call_id in by_call:
            content = e.observation.to_llm_content
            by_call[e.tool_call_id]["output"] = "\n".join(getattr(c, "text", "") for c in content)
    return steps


# ──────────────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────────────

class KinderDirectSolveController:
    """
    Parameters
    ----------
    domain : KinderDomain
    base_url, model, api_key : the OpenAI-compatible LLM endpoint (litellm model name)
    max_steps : LLM turns per task (OpenHands iterations; a turn may issue several tool calls)
    max_nudges : times the agent is reminded to submit after answering without a tool call
    max_tokens : max tokens per LLM response
    code_timeout : seconds per python call
    eval_timeout : seconds for evaluating the submitted policy (rollout + replay)
    """

    def __init__(self, domain, base_url: str, model: str, api_key: str = "EMPTY", max_steps: int = 40,
                 max_tokens: int = 16384, temperature: float | None = None, code_timeout: float = 120,
                 max_output_chars: int = 10000, eval_timeout: float = 600, max_nudges: int = 2):
        self.domain = domain
        self.max_steps = max_steps
        self.max_nudges = max_nudges
        self.code_timeout = code_timeout
        self.max_output_chars = max_output_chars
        self.eval_timeout = eval_timeout
        self._llm_kwargs = dict(model=model, base_url=base_url, api_key=SecretStr(api_key), max_tokens=max_tokens)
        if temperature is not None:
            self._llm_kwargs["temperature"] = temperature
        fd, self._system_prompt_file = tempfile.mkstemp(suffix=".j2", prefix="kds_system_")
        with os.fdopen(fd, "w") as f:
            f.write(SYSTEM_PROMPT.format(code_timeout=int(code_timeout), max_output_chars=max_output_chars,
                                         max_steps=max_steps, client_usage=P.client_usage(pinned_only=True),
                                         policy_interface=P.POLICY_INTERFACE))

    def solve(self, record: dict) -> dict:
        """Run one agent conversation on one instance."""
        t0 = time.time()
        task_dir = tempfile.mkdtemp(prefix="oh_kds_task_")
        llm = LLM(**self._llm_kwargs)
        done: dict = {}
        events, agent_error, end_reason, nudges = [], None, None, 0
        sandbox = KinderSandbox(self.domain.family, [(record["count"], record["seed"])], free_reset=False,
                                files={"ENV.md": self.domain.card_path()}, timeout=self.code_timeout,
                                max_output_chars=self.max_output_chars)
        try:
            tools = [*KDSPythonTool.create(sandbox), *KDSSubmitTool.create(done)]
            agent = Agent(llm=llm, tools=[], include_default_tools=[],
                          system_prompt_filename=self._system_prompt_file)
            agent.__pydantic_private__["_tools"] = {t.name: t for t in tools}
            agent.__pydantic_private__["_initialized"] = True
            conversation = Conversation(agent=agent, workspace=task_dir, max_iteration_per_run=self.max_steps,
                                        visualizer=None)
            conversation.send_message(self.domain.agent_task_text(record))
            nudges = 0
            while True:
                try:
                    conversation.run()
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    agent_error, end_reason = f"{type(exc).__name__}: {exc}", "error"
                    logger.warning("kinder direct_solve conversation error on %s: %s\n%s", record["task_id"], exc,
                                   traceback.format_exc())
                    break
                if done.get("code"):
                    end_reason = "submitted"
                    break
                remaining = self.max_steps - llm_turns(conversation.state.events)
                status = str(getattr(conversation.state.execution_status, "value", ""))
                if remaining <= 0:
                    end_reason = "max_steps"
                    break
                if status != "finished" or nudges >= self.max_nudges:
                    end_reason = "no_submission" if status == "finished" else status or "stopped"
                    break
                # the model answered without a tool call: remind it to submit, within the remaining turns
                nudges += 1
                conversation.max_iteration_per_run = remaining
                conversation.send_message(NUDGE)
            events = list(conversation.state.events)
        finally:
            sandbox.close()
            shutil.rmtree(task_dir, ignore_errors=True)

        steps = trajectory_steps(events)
        if done.get("code"):
            ev = evaluate_policy_source(self.domain.name, record, done["code"], self.eval_timeout)
        else:
            ev = {"success": False, "score": 0.0, "program": None, "extras": {}}

        usage = llm.metrics.accumulated_token_usage
        token_usage = {"input": usage.prompt_tokens, "output": usage.completion_tokens,
                       "reasoning": getattr(usage, "reasoning_tokens", 0) or 0}
        score = float(ev["score"])
        return {
            "solved": score >= 1.0,
            "answer": ev["program"],
            "best_reward": score,
            "reward_history": [{"iteration": 0, "reward": score}],
            "steps_used": len(steps),
            "token_usage": token_usage,
            "cost_summary": {"elapsed_s": round(time.time() - t0, 1)},
            "submitted": bool(done.get("code")),
            "policy_code": done.get("code", ""),
            "submit_notes": done.get("notes", ""),
            "rollout": ev["extras"],
            "agent_error": agent_error,
            "end_reason": end_reason,
            "nudges": nudges,
            "trajectory": steps,
            "_trajectory": [json.loads(e.model_dump_json()) for e in events],
        }
