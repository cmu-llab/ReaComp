"""
SolverBuilder controller — runs a single OpenHands conversation that reads
reasoning traces in DEMOS.json and writes SOLVER.py + SOLVER_ALGORITHM.md.

The agent is given:
  - The task description from SOLVER_BUILDING_PROMPT.md (injected into the user message)
  - Read-only access to DEMOS.json and rewards/pbebench.py inside the sandbox
    (both bind-mounted at /workspace/... so the agent can inspect them with execute_code)
  - execute_code    — run Python snippets to explore DEMOS.json or test logic
  - write_file      — write SOLVER.py or SOLVER_ALGORITHM.md to output_dir
  - finish          — signal completion

Output files land in output_dir on the host.
"""

import logging
import os
import tempfile

from pydantic import SecretStr

from openhands.sdk import LLM, Agent, Conversation

from .tools import SBExecuteCodeTool, SBWriteFileTool, SBFinishTool

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT_PBE = """\
You are an expert in symbolic program induction and Python.

Your task is to write a symbolic solver for Programming-by-Example (PBE) tasks.
You will be given a detailed specification in the user message.

You have three tools:

  execute_code(code)
      Run Python code in a sandbox.
      DEMOS.json is at /workspace/DEMOS.json.
      The verifier is directly importable (sys.path already includes /workspace):
          from rewards.pbebench import reward
      Use this to explore DEMOS.json, understand task structure, and test
      small solver logic snippets before writing the final files.

  write_file(filename, content)
      Write a file to the output directory.
      Call once with filename='SOLVER.py' and once with filename='SOLVER_ALGORITHM.md'.
      You MUST call this for both files before finishing.

  finish(summary)
      Signal that you are done. Call after both files have been written.

Workflow:
  1. Read the specification in the user message carefully.
  2. Use execute_code to inspect DEMOS.json (structure, patterns, DSL constraints).
  3. Design your solver algorithm based on what you observe.
  4. Write SOLVER.py with the complete implementation.
  5. Write SOLVER_ALGORITHM.md with a clear algorithm description.
  6. Call finish with a brief summary.

Requirements:
  - The solver must implement solve_pbe(examples) where examples is a list of
    (input_string, output_string) pairs.
  - Use only the Python standard library (no numpy, sympy, etc.).
  - The solver must use the verifier (rewards/pbebench.py reward function) to
    score candidate programs.
  - If no fully correct program is found, return top-K highest scoring programs.
  - Follow the DSL: programs are ordered sequences of replace(A, B) calls where
    1<=len(A)<=3, 0<=len(B)<=3, max 5 programs (PBEBench-Lite).

Do NOT produce placeholder code. Write a complete, working implementation.
"""

_SYSTEM_PROMPT_SLR = """\
You are an expert in symbolic rule induction and Python.

Your task is to write a symbolic solver for SLR-Bench (Symbolic Logic Rule) tasks.
You will be given a detailed specification in the user message.

You have three tools:

  execute_code(code)
      Run Python code in a sandbox.
      DEMOS.json is at /workspace/DEMOS.json.
      The reward module is importable (sys.path already includes /workspace):
          from rewards.slr_bench import reward, parse_prompt_examples, rule_complexity
      SWI-Prolog is installed — you can also verify rules via the HuggingFace judge:
          from rewards.slr_bench import _get_judge
      Use this to explore DEMOS.json, understand task structure, and test
      small solver logic snippets before writing the final files.

  write_file(filename, content)
      Write a file to the output directory.
      Call once with filename='SOLVER_SLR.py' and once with filename='SOLVER_SLR_ALGORITHM.md'.
      You MUST call this for both files before finishing.

  finish(summary)
      Signal that you are done. Call after both files have been written.

Workflow:
  1. Read the specification in the user message carefully.
  2. Use execute_code to inspect DEMOS.json (structure, patterns, rule forms).
  3. Design your solver algorithm based on what you observe.
  4. Write SOLVER_SLR.py with the complete implementation.
  5. Write SOLVER_SLR_ALGORITHM.md with a clear algorithm description.
  6. Call finish with a brief summary.

Requirements:
  - The solver must implement solve_slr(examples) where examples is a list of
    (facts_string, label) pairs; label is "eastbound" or "westbound".
  - Use only the Python standard library (no numpy, sympy, etc.).
  - The solver must score candidates using a local Python evaluator (do NOT
    require SWI-Prolog at inference time — evaluate purely in Python).
  - Return a dict with at least: success (bool), program (str), top_k_programs (list[str]),
    score (float 0–1).
  - The output program must be a valid Prolog rule string ending with '.', e.g.:
    'eastbound(T) :- has_car(T, C), car_len(C, short).'

Do NOT produce placeholder code. Write a complete, working implementation.
"""

# Default to PBE for backwards compatibility; SLR selected via solver_type param.
_SYSTEM_PROMPT = _SYSTEM_PROMPT_PBE


class SolverBuilderController:
    """
    Runs a single conversation that writes SOLVER.py + SOLVER_ALGORITHM.md.

    Parameters
    ----------
    base_url : str
    model : str
    sandbox : ApptainerSandbox
    demos_path : str
        Host path to DEMOS.json.
    rewards_dir : str
        Host path to the rewards/ package directory (contains pbebench.py).
    output_dir : str
        Host directory where SOLVER.py and SOLVER_ALGORITHM.md will be written.
    api_key : str
    max_steps : int
    max_tokens : int
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        sandbox,
        demos_path: str,
        rewards_dir: str,
        output_dir: str,
        api_key: str = "EMPTY",
        max_steps: int = 200,
        max_tokens: int = 16384,
        solver_type: str = "pbe",
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.sandbox = sandbox
        self.demos_path = os.path.abspath(demos_path)
        self.rewards_dir = os.path.abspath(rewards_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.solver_type = solver_type

        self._llm = LLM(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            max_tokens=max_tokens,
        )

        system_prompt = _SYSTEM_PROMPT_SLR if solver_type == "slr" else _SYSTEM_PROMPT_PBE

        # Write system prompt to a temp .j2 file (SDK requires a file path).
        fd, self._system_prompt_file = tempfile.mkstemp(
            suffix=".j2", prefix="sb_system_prompt_"
        )
        with os.fdopen(fd, "w") as f:
            f.write(system_prompt)

    def _extra_binds(self) -> list:
        """
        Bind mounts for the sandbox:
          - DEMOS.json        → /workspace/DEMOS.json      (ro)
          - rewards/          → /workspace/rewards          (ro)
        """
        binds = [
            (self.demos_path, "/workspace/DEMOS.json", "ro"),
            (self.rewards_dir, "/workspace/rewards", "ro"),
        ]
        return binds

    def _make_agent(self, task_dir: str) -> Agent:
        done_path = os.path.join(task_dir, "done.txt")
        tool_instances = [
            *SBExecuteCodeTool.create(
                self.sandbox,
                workspace_dir="/workspace",
                extra_binds=self._extra_binds(),
            ),
            *SBWriteFileTool.create(self.output_dir),
            *SBFinishTool.create(done_path),
        ]
        agent = Agent(
            llm=self._llm,
            tools=[],
            include_default_tools=[],
            system_prompt_filename=self._system_prompt_file,
        )
        agent.__pydantic_private__["_tools"] = {t.name: t for t in tool_instances}
        agent.__pydantic_private__["_initialized"] = True
        logger.info("SolverBuilder agent tools: %s", list(agent.tools_map.keys()))
        return agent

    def build(self, building_prompt: str) -> dict:
        """
        Run the agent conversation to produce SOLVER.py + SOLVER_ALGORITHM.md.

        Parameters
        ----------
        building_prompt : str
            Content of SOLVER_BUILDING_PROMPT.md — injected as the user message.

        Returns
        -------
        dict with keys: solver_path, algorithm_path, success, summary
        """
        os.makedirs(self.output_dir, exist_ok=True)
        task_dir = tempfile.mkdtemp(prefix="oh_sb_task_")
        done_path = os.path.join(task_dir, "done.txt")

        # Prepend path resolution note so the agent knows where @-referenced files live.
        if self.solver_type == "slr":
            file_note = (
                "File path reference guide (for @-references in the spec below):\n"
                "  @DEMOS.json              → /workspace/DEMOS.json   (use execute_code to read)\n"
                "  @rewards/slr_bench.py    → importable as: from rewards.slr_bench import reward, parse_prompt_examples, rule_complexity\n"
                "  @SOLVER_SLR.py           → write via write_file(filename='SOLVER_SLR.py', content=...)\n"
                "  @SOLVER_SLR_ALGORITHM.md → write via write_file(filename='SOLVER_SLR_ALGORITHM.md', content=...)\n"
                "\n"
                "---\n\n"
            )
        else:
            file_note = (
                "File path reference guide (for @-references in the spec below):\n"
                "  @DEMOS.json         → /workspace/DEMOS.json   (use execute_code to read)\n"
                "  @rewards/pbebench.py → importable as: from rewards.pbebench import reward\n"
                "  @SOLVER.py          → write via write_file(filename='SOLVER.py', content=...)\n"
                "  @SOLVER_ALGORITHM.md → write via write_file(filename='SOLVER_ALGORITHM.md', content=...)\n"
                "\n"
                "---\n\n"
            )
        user_message = file_note + building_prompt

        agent = self._make_agent(task_dir)
        conversation = Conversation(
            agent=agent,
            workspace=task_dir,
            max_iteration_per_run=self.max_steps,
        )
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
            logger.warning("solver_builder conversation error: %s\n%s",
                           exc, traceback.format_exc())

        if self.solver_type == "slr":
            solver_fname, algo_fname = "SOLVER_SLR.py", "SOLVER_SLR_ALGORITHM.md"
        else:
            solver_fname, algo_fname = "SOLVER.py", "SOLVER_ALGORITHM.md"
        solver_path = os.path.join(self.output_dir, solver_fname)
        algorithm_path = os.path.join(self.output_dir, algo_fname)
        summary = ""
        if os.path.exists(done_path):
            summary = open(done_path).read().strip()

        success = os.path.exists(solver_path) and os.path.exists(algorithm_path)
        logger.info(
            "solver_builder: success=%s  solver=%s  algorithm=%s",
            success, os.path.exists(solver_path), os.path.exists(algorithm_path),
        )
        return {
            "success": success,
            "solver_path": solver_path if os.path.exists(solver_path) else None,
            "algorithm_path": algorithm_path if os.path.exists(algorithm_path) else None,
            "summary": summary,
            "_trajectory": trajectory,
        }
