"""
DirectSolve controller — one OpenHands conversation per task.

Supports both PBEBench (replace-program synthesis) and SLR-Bench (Prolog rule induction).
Task type is inferred from the record fields or passed explicitly via task_type=.
"""

import json
import logging
import os
import tempfile

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation

from .tools import DSExecuteCodeTool, DSSubmitAnswerTool

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_PBE = """\
You are an expert programmer solving Programming-by-Example (PBE) tasks.

Each task gives you a set of (input, output) string pairs. Your goal is to find
an ordered sequence of replace(A, B) operations that transforms every input into
its paired output.

DSL constraints:
  - Each program has the form: replace('A', 'B')
  - 1 <= len(A) <= 3 characters,  0 <= len(B) <= 3 characters
  - At most 5 programs in sequence (PBEBench-Lite)
  - Programs are applied left-to-right: output = prog1(prog2(...(progN(input))))
  - Only str.replace() semantics — no regex, no imports

You have two tools:

  execute_code(code)
      Run any Python code in a sandbox. Use this to:
        - Test candidate programs against the examples
        - Write systematic search / enumeration code
        - Import and call the reward function directly:
              import sys; sys.path.insert(0, '/workspace')
              from rewards.pbebench import reward
              result = reward(["replace('x','y')"], True, task_record)
              print(result['value'], result['feedback'])
          where task_record = {"inputs": [...], "outputs": [...]}

      IMPORTANT: The sandbox only has rewards/pbebench.py available.
      Do NOT try to read any files — no datasets, no DEMOS.json, nothing else exists.
      All task information is in this message — use it directly in your code.

  submit_answer(programs)
      Submit your final answer as a list of replace(A,B) strings.
      You will receive the reward score immediately.
      Submit as soon as you reach reward=1.0, or submit your best attempt
      before you run out of steps.

Strategy tips:
  - Examine what characters change between input and output.
  - Enumerate single-replace candidates that fix the most examples.
  - Greedily extend with additional replaces for remaining errors.
  - The reward function gives partial credit and feedback — use it iteratively.
  - Always submit something before your steps run out.

Constraints:
  - Do NOT attempt to read files or access the filesystem.
  - Only rewards/pbebench.py is available to import.
"""

_SYSTEM_PROMPT_SLR = """\
You are an expert at symbolic rule induction for SLR-Bench (Symbolic Logic Rules).

Each task gives you a set of labelled train descriptions. Your goal is to find a
Prolog rule that correctly classifies all trains as eastbound or westbound.

Rule format:
  eastbound(T) :- has_car(T, C), lit1, lit2, ...
where each literal is a ground atom over predicates like car_len/2, car_color/2,
car_shape/2, load_shape/2, load_num/2, has_load/2, closed/1.

You have two tools:

  execute_code(code)
      Run any Python code in a sandbox. Use this to:
        - Parse and analyse the train descriptions
        - Test candidate rules using the reward function:
              import sys; sys.path.insert(0, '/workspace')
              from rewards.slr_bench import reward, parse_prompt_examples
              examples = parse_prompt_examples(prompt_text)
              result = reward("eastbound(T) :- has_car(T,C), car_len(C,short).", True, task_record)
              print(result['value'], result.get('feedback',''))
          where task_record = {"prompt": "...the full prompt text..."}
        - Enumerate candidate predicates and conjunctions
        - Implement systematic search over rule bodies

      IMPORTANT: The sandbox only has rewards/slr_bench.py available.
      Do NOT try to read any files — no datasets, no DEMOS.json, nothing else exists.
      All task information is in this message — use it directly in your code.

  submit_answer(programs)
      Submit your final answer as a single-element list containing the Prolog rule string, e.g.:
          ["eastbound(T) :- has_car(T, C), car_len(C, short)."]
      You will receive the reward score immediately (1.0 = correct, 0.0 = wrong).
      Submit as soon as you reach reward=1.0, or submit your best attempt
      before you run out of steps.

Strategy tips:
  - Parse the prompt to extract eastbound/westbound train descriptions.
  - Look for predicates that appear in all eastbound trains but no westbound trains.
  - Start with single-literal rules, then try conjunctions of 2-3 literals.
  - Use execute_code to systematically enumerate and score candidates.
  - Always submit something before your steps run out.

Constraints:
  - Do NOT attempt to read files or access the filesystem.
  - Only rewards/slr_bench.py is available to import.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Task message formatters
# ──────────────────────────────────────────────────────────────────────────────

def _build_task_message_pbe(record: dict) -> str:
    inputs = record.get("inputs", [])
    outputs = record.get("outputs", [])
    lines = ["Solve this PBE task. Find a sequence of replace(A,B) programs that transforms each input to its output.\n"]
    lines.append("Examples:")
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        lines.append(f"  [{i+1}] input:  {repr(inp)}")
        lines.append(f"       output: {repr(out)}")
    lines.append("\nUse execute_code to test candidates, then submit_answer with your best solution.")
    return "\n".join(lines)


def _build_task_message_slr(record: dict) -> str:
    prompt = record.get("prompt", "")
    lines = [
        "Solve this SLR-Bench task. Find a Prolog rule that classifies all trains correctly.\n",
        "Task prompt:",
        "---",
        prompt,
        "---",
        "\nUse execute_code to test candidate rules, then submit_answer with your best rule as a single-element list.",
    ]
    return "\n".join(lines)


def _infer_task_type(record: dict, reward_name: str) -> str:
    """Infer 'pbe' or 'slr' from the record or reward name."""
    if reward_name and "slr" in reward_name.lower():
        return "slr"
    if "prompt" in record and "inputs" not in record:
        return "slr"
    return "pbe"


# ──────────────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────────────

class DirectSolveController:
    """
    Runs one OpenHands conversation per task.

    Parameters
    ----------
    base_url, model, sandbox, rewards_dir, api_key, max_steps, max_tokens : as before
    task_type : "pbe" | "slr" | "auto"
        Controls system prompt and task message format. "auto" infers from the record.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        sandbox,
        rewards_dir: str,
        api_key: str = "EMPTY",
        max_steps: int = 100,
        max_tokens: int = 16384,
        task_type: str = "auto",
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.sandbox = sandbox
        self.rewards_dir = os.path.abspath(rewards_dir)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.task_type = task_type

        self._llm_kwargs = dict(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            max_tokens=max_tokens,
        )

        # Pre-build both system prompt temp files
        self._system_prompt_files = {}
        for ttype, prompt in [("pbe", _SYSTEM_PROMPT_PBE), ("slr", _SYSTEM_PROMPT_SLR)]:
            fd, path = tempfile.mkstemp(suffix=".j2", prefix=f"ds_system_{ttype}_")
            with os.fdopen(fd, "w") as f:
                f.write(prompt)
            self._system_prompt_files[ttype] = path

    def _extra_binds(self) -> list:
        return [(self.rewards_dir, "/workspace/rewards", "ro")]

    def solve(self, record: dict, reward_fn, reward_name: str = "") -> dict:
        """Run one agent conversation to solve a single task."""
        task_type = self.task_type
        if task_type == "auto":
            task_type = _infer_task_type(record, reward_name)

        task_dir = tempfile.mkdtemp(prefix="oh_ds_task_")
        done_path = os.path.join(task_dir, "done.json")

        llm = LLM(**self._llm_kwargs)

        tool_instances = [
            *DSExecuteCodeTool.create(self.sandbox, self._extra_binds()),
            *DSSubmitAnswerTool.create(record, done_path, reward_fn),
        ]

        agent = Agent(
            llm=llm,
            tools=[],
            include_default_tools=[],
            system_prompt_filename=self._system_prompt_files[task_type],
        )
        agent.__pydantic_private__["_tools"] = {t.name: t for t in tool_instances}
        agent.__pydantic_private__["_initialized"] = True

        conversation = Conversation(
            agent=agent,
            workspace=task_dir,
            max_iteration_per_run=self.max_steps,
        )

        if task_type == "slr":
            user_message = _build_task_message_slr(record)
        else:
            user_message = _build_task_message_pbe(record)

        conversation.send_message(user_message)

        trajectory = []
        try:
            conversation.run()
            try:
                trajectory = [e.model_dump() for e in conversation.state.events]
            except Exception:
                pass
        except Exception as exc:
            import traceback
            logger.warning("direct_solve conversation error: %s\n%s", exc, traceback.format_exc())

        answer = None
        reward_value = 0.0
        if os.path.exists(done_path):
            try:
                data = json.load(open(done_path))
                answer = data.get("programs")
                reward_value = float(data.get("reward", 0.0))
            except Exception:
                pass

        steps_used = len([e for e in trajectory if e.get("kind") == "ActionEvent"])

        usage = llm.metrics.accumulated_token_usage
        token_usage = {
            "input":     usage.prompt_tokens,
            "output":    usage.completion_tokens,
            "reasoning": usage.reasoning_tokens,
        }

        return {
            "solved": reward_value >= 1.0,
            "answer": answer,
            "best_reward": reward_value,
            "steps_used": steps_used,
            "token_usage": token_usage,
            "_trajectory": trajectory,
        }
