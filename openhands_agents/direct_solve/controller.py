"""
DirectSolve controller — one OpenHands conversation per PBEBench task.

The agent receives:
  - The task (inputs/outputs) as the user message
  - execute_code: Python sandbox with rewards/pbebench.py importable
  - submit_answer: submits the final list of replace(A,B) programs

It can do anything it wants in up to max_steps steps: enumerate candidates,
write search code, test partial programs, etc. The only constraint is the
PBEBench DSL (replace sequences, max 5 programs for Lite).
"""

import json
import logging
import os
import tempfile

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation

from .tools import DSExecuteCodeTool, DSSubmitAnswerTool

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
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

  submit_answer(programs)
      Submit your final answer as a list of replace(A,B) strings.
      You will receive the reward score immediately.
      Submit as soon as you reach reward=1.0, or submit your best attempt
      before you run out of steps.

Strategy tips:
  - Start by examining what characters change between input and output.
  - Enumerate single-replace candidates that fix the most examples.
  - Greedily extend with additional replaces for remaining errors.
  - The reward function gives partial credit and feedback — use it iteratively.
  - Always submit something before your steps run out.
"""


def _build_task_message(record: dict) -> str:
    """Format a task record as a user message for the agent."""
    inputs = record.get("inputs", [])
    outputs = record.get("outputs", [])
    lines = ["Solve this PBE task. Find a sequence of replace(A,B) programs that transforms each input to its output.\n"]
    lines.append("Examples:")
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        lines.append(f"  [{i+1}] input:  {repr(inp)}")
        lines.append(f"       output: {repr(out)}")
    lines.append("\nUse execute_code to test candidates, then submit_answer with your best solution.")
    return "\n".join(lines)


class DirectSolveController:
    """
    Runs one OpenHands conversation per task to directly solve a PBEBench instance.

    Parameters
    ----------
    base_url : str
    model : str
    sandbox : ApptainerSandbox
    rewards_dir : str
        Host path to the rewards/ package directory.
    api_key : str
    max_steps : int
        Max agent steps per task conversation (default 100).
    max_tokens : int
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
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.sandbox = sandbox
        self.rewards_dir = os.path.abspath(rewards_dir)
        self.max_steps = max_steps
        self.max_tokens = max_tokens

        self._llm = LLM(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            max_tokens=max_tokens,
        )

        fd, self._system_prompt_file = tempfile.mkstemp(suffix=".j2", prefix="ds_system_prompt_")
        with os.fdopen(fd, "w") as f:
            f.write(_SYSTEM_PROMPT)

    def _extra_binds(self) -> list:
        return [(self.rewards_dir, "/workspace/rewards", "ro")]

    def solve(self, record: dict, reward_fn) -> dict:
        """
        Run one agent conversation to solve a single task.

        Returns
        -------
        dict with keys: solved, answer, best_reward, steps_used, _trajectory
        """
        task_dir = tempfile.mkdtemp(prefix="oh_ds_task_")
        done_path = os.path.join(task_dir, "done.json")

        tool_instances = [
            *DSExecuteCodeTool.create(self.sandbox, self._extra_binds()),
            *DSSubmitAnswerTool.create(record, done_path, reward_fn),
        ]

        agent = Agent(
            llm=self._llm,
            tools=[],
            include_default_tools=[],
            system_prompt_filename=self._system_prompt_file,
        )
        agent.__pydantic_private__["_tools"] = {t.name: t for t in tool_instances}
        agent.__pydantic_private__["_initialized"] = True

        conversation = Conversation(
            agent=agent,
            workspace=task_dir,
            max_iteration_per_run=self.max_steps,
        )

        user_message = _build_task_message(record)
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

        # Read submitted answer from done file
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

        return {
            "solved": reward_value >= 1.0,
            "answer": answer,
            "best_reward": reward_value,
            "steps_used": steps_used,
            "_trajectory": trajectory,
        }
