"""List Functions SolverBuilder controller (new-files-only, additive).

Subclasses ``SolverBuilderController`` and overrides only the domain-specific pieces
(system prompt, @-reference file note, and output filenames) so the base
``solver_type in {pbe, slr}`` branches are left untouched. Reuses the base
``_make_agent`` / ``_extra_binds`` / sandbox wiring verbatim.

Induces ``SOLVER_LISTFN.py`` + ``SOLVER_LISTFN_ALGORITHM.md`` implementing
``solve_listfn(examples)`` for integer-list -> integer-list tasks, scored by
``rewards/list_functions.py``.
"""
import logging
import os
import tempfile

from openhands.sdk import Conversation

from .controller import SolverBuilderController

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_LISTFN = """\
You are an expert in symbolic program induction and Python.

Your task is to write a symbolic solver for List Functions tasks: inferring a function
that maps a list of natural numbers to another list of natural numbers, from a set of
input/output examples. You will be given a detailed specification in the user message.

You have three tools:

  execute_code(code)
      Run Python code in a sandbox.
      DEMOS_LISTFN.json is mounted at /workspace/DEMOS.json.
      The verifier is directly importable (sys.path already includes /workspace):
          from rewards.list_functions import reward, score_program
      Use this to explore /workspace/DEMOS.json, understand task structure, and test
      small solver logic snippets before writing the final files.

  write_file(filename, content)
      Write a file to the output directory.
      Call once with filename='SOLVER_LISTFN.py' and once with
      filename='SOLVER_LISTFN_ALGORITHM.md'. You MUST call this for both files.

  finish(summary)
      Signal that you are done. Call after both files have been written.

Workflow:
  1. Read the specification in the user message carefully.
  2. Use execute_code to inspect DEMOS.json (structure, the kinds of list functions).
  3. Design a solver that SEARCHES over general list-transformation primitives and
     their compositions, rather than memorizing example outputs.
  4. Write SOLVER_LISTFN.py with the complete implementation.
  5. Write SOLVER_LISTFN_ALGORITHM.md with a clear algorithm description.
  6. Call finish with a brief summary.

Requirements:
  - The solver must implement solve_listfn(examples) where examples is a list of
    (input_list, output_list) pairs, each a list of ints.
  - Return a dict with at least {"success": bool, "program": <source string defining
    program(xs) or a callable>}.
  - Use only the Python standard library (no numpy, sympy, etc.).
  - The solver must use the verifier (rewards/list_functions.py) to score candidates.
  - Do NOT hardcode outputs for specific inputs: the program must generalize to unseen
    inputs of the same function.
  - If no fully correct program is found, return the top-K highest scoring programs.

Do NOT produce placeholder code. Write a complete, working implementation.
"""

_FILE_NOTE = (
    "File path reference guide (for @-references in the spec below):\n"
    "  @DEMOS_LISTFN.json         → /workspace/DEMOS.json   (use execute_code to read)\n"
    "  @rewards/list_functions.py → importable as: from rewards.list_functions import reward, score_program\n"
    "  @SOLVER_LISTFN.py          → write via write_file(filename='SOLVER_LISTFN.py', content=...)\n"
    "  @SOLVER_LISTFN_ALGORITHM.md → write via write_file(filename='SOLVER_LISTFN_ALGORITHM.md', content=...)\n"
    "\n"
    "---\n\n"
)


class ListFnSolverBuilderController(SolverBuilderController):
    def __init__(self, *args, **kwargs):
        # Force solver_type to a sentinel the base won't special-case, then override
        # the system prompt file the base wrote in its __init__.
        kwargs["solver_type"] = "listfn"
        super().__init__(*args, **kwargs)
        # Base __init__ wrote a PBE prompt (its else branch); replace it.
        with open(self._system_prompt_file, "w") as f:
            f.write(_SYSTEM_PROMPT_LISTFN)

    def build(self, building_prompt: str) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)
        task_dir = tempfile.mkdtemp(prefix="oh_sb_listfn_task_")
        done_path = os.path.join(task_dir, "done.txt")

        user_message = _FILE_NOTE + building_prompt

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
            logger.warning("listfn solver_builder conversation error: %s\n%s",
                           exc, traceback.format_exc())

        solver_fname = "SOLVER_LISTFN.py"
        algo_fname = "SOLVER_LISTFN_ALGORITHM.md"
        solver_path = os.path.join(self.output_dir, solver_fname)
        algorithm_path = os.path.join(self.output_dir, algo_fname)
        summary = ""
        if os.path.exists(done_path):
            summary = open(done_path).read().strip()

        success = os.path.exists(solver_path) and os.path.exists(algorithm_path)
        logger.info("listfn solver_builder: success=%s solver=%s algorithm=%s",
                    success, os.path.exists(solver_path), os.path.exists(algorithm_path))
        return {
            "success": success,
            "solver_path": solver_path if os.path.exists(solver_path) else None,
            "algorithm_path": algorithm_path if os.path.exists(algorithm_path) else None,
            "summary": summary,
            "_trajectory": trajectory,
        }
