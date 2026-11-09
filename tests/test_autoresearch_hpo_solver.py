"""Tests for the solver side of the autoresearch HPO domain: running an induced
SOLVER_HPO.py as a policy, the evaluation script, the OpenHands SolverBuilder (tools,
controller, entry point), the results summary and the launch scripts.

CPU only. The SolverBuilder tests need the OpenHands SDK (they are skipped without it)
but neither a model server nor Apptainer: the conversation is a scripted stand-in that
plays tool calls through the agent's real tools, and the sandbox runs the code with
the local interpreter.

Run from the project root:
    python -m unittest discover -s tests -t .
"""
import importlib.util
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest import mock

from tests.hpo_fixtures import (
    PROJECT_ROOT, REFERENCE_SOLVER, FakeLLMServer, FakeSandbox, exit_status, load_script, quiet,
    read_jsonl, sampled_answer, synthetic_entry, write_solver, write_tasks,
)

from rewards.autoresearch_hpo import (
    HP_NAMES, STARTING_CONFIG, UPSTREAM, EpisodeLogMismatch, normalize_config,
    public_task_view, run_episode,
)
from openhands_agents.hpo_agent.solver_policy import SolverError, SolverPolicy, load_solver

eval_hpo_solver = load_script("scripts/eval_hpo_solver.py", "eval_hpo_solver")
summarizer = load_script("scripts/summarize_hpo_results.py", "summarize_hpo_results")

HAVE_OPTUNA = importlib.util.find_spec("optuna") is not None
HAVE_OPENAI = importlib.util.find_spec("openai") is not None
HAVE_SDK = importlib.util.find_spec("openhands") is not None
if HAVE_SDK:
    os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
    from openhands_agents import build_solver_hpo
    from openhands_agents.solver_builder import hpo_controller, hpo_tools
    from openhands_agents.solver_builder.tools import SBFinishAction

SYNTHETIC_TASKS = os.path.join(PROJECT_ROOT, "data", "autoresearch_hpo", "synthetic_tasks.jsonl")
BUILDING_PROMPT = os.path.join(PROJECT_ROOT, "building_prompts", "SOLVER_BUILDING_PROMPT_HPO.md")
REWARDS_DIR = os.path.join(PROJECT_ROOT, "rewards")


def solver_source(body, preamble=""):
    """A solver file whose ``solve_hpo`` has the given body."""
    return preamble + "def solve_hpo(task, history):\n" + textwrap.indent(textwrap.dedent(body), "    ")


DEFAULT_SOLVER = solver_source('return {"config": dict(task["starting_config"])}\n')
GARBAGE_SOLVER = solver_source('return {"config": {"DEPTH": "deep"}}\n')
CRASHING_SOLVER = solver_source('raise ValueError("bad idea")\n')
HANGING_SOLVER = solver_source("while True:\n    pass\n")


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hpo_test_")
        self.addCleanup(shutil.rmtree, self.tmp)

    def path(self, *parts):
        return os.path.join(self.tmp, *parts)


# ──────────────────────────────────────────────────────────────────────────────
# SolverPolicy
# ──────────────────────────────────────────────────────────────────────────────

class SolverPolicyTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.entry = synthetic_entry()
        self.task = public_task_view(self.entry)

    def policy(self, source=REFERENCE_SOLVER, **kwargs):
        return SolverPolicy(write_solver(self.path("solver"), source), **kwargs)

    def decision_error(self, source, **kwargs):
        with self.assertRaises(SolverError) as caught:
            self.policy(source, **kwargs)(self.task, [])
        return str(caught.exception)

    def test_decision(self):
        suggestion = self.policy()(self.task, [])
        # what the solver prints is not mistaken for its answer; what it returns besides
        # the configuration comes back under "solver_meta"
        self.assertEqual(sorted(suggestion), ["config", "solver_meta", "source"])
        self.assertEqual(suggestion["source"], "solver")
        self.assertEqual(suggestion["solver_meta"], {"strategy": "local perturbation"})
        self.assertEqual(normalize_config(suggestion["config"]), suggestion["config"])

    def test_a_bare_configuration_is_an_answer(self):
        policy = self.policy(solver_source('return dict(task["starting_config"])\n'))
        self.assertEqual(policy(self.task, []), {"config": STARTING_CONFIG, "source": "solver"})
        # ... and so is anything else without a "config" key: the episode loop rejects it
        policy = self.policy(solver_source('return [1, 2]\n'))
        self.assertEqual(policy(self.task, []), {"config": [1, 2], "source": "solver"})
        episode = run_episode(policy, self.entry, max_consecutive_invalid=2)
        self.assertEqual([t["status"] for t in episode["trials"]], ["ok", "invalid", "invalid"])

    def test_the_solver_cannot_write_the_fields_of_the_harness(self):
        policy = self.policy(solver_source(
            'return {"config": dict(task["starting_config"]), "source": "tpe",\n'
            '        "token_usage": {"input": "many", "output": -5}, "solver_meta": 1}\n'))
        self.assertEqual(policy(self.task, []), {
            "config": STARTING_CONFIG, "source": "solver",
            "solver_meta": {"source": "tpe", "token_usage": {"input": "many", "output": -5},
                            "solver_meta": 1}})

    def test_oversized_extras_are_truncated(self):
        policy = self.policy(solver_source(
            'return {"config": dict(task["starting_config"]), "notes": "x" * 100000}\n'))
        suggestion = policy(self.task, [])
        self.assertEqual(sorted(suggestion), ["config", "solver_meta", "source"])
        self.assertEqual(list(suggestion["solver_meta"]), ["truncated"])
        self.assertEqual(len(suggestion["solver_meta"]["truncated"]), 20000)
        self.assertTrue(suggestion["solver_meta"]["truncated"].startswith('{"notes": "xxx'))

    def test_decisions_are_reproducible(self):
        policy = self.policy()
        self.assertEqual(policy(self.task, []), policy(self.task, []))
        other_seed = public_task_view(synthetic_entry(seed=1))
        self.assertNotEqual(policy(self.task, []), policy(other_seed, []))

    def test_name_identifies_the_file_content(self):
        first = self.policy()
        self.assertRegex(first.name, r"^solver:SOLVER_HPO\.py@[0-9a-f]{12}$")
        self.assertEqual(SolverPolicy(first.solver_path).name, first.name)
        self.assertNotEqual(self.policy(DEFAULT_SOLVER).name, first.name)

    def test_missing_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "solver file not found"):
            SolverPolicy(self.path("nowhere", "SOLVER_HPO.py"))
        with self.assertRaisesRegex(FileNotFoundError, "solver file not found"):
            SolverPolicy(self.tmp)                                   # a directory

    def test_exception_in_the_solver(self):
        error = self.decision_error(CRASHING_SOLVER)
        self.assertTrue(error.startswith("ValueError: bad idea\n"), error)
        self.assertIn("Traceback", error)
        self.assertIn("in solve_hpo", error)

    def test_solver_that_exits(self):
        self.assertTrue(self.decision_error(solver_source("raise SystemExit(4)\n"))
                        .startswith("SystemExit: 4\n"))
        error = self.decision_error(solver_source(
            'import os, sys\nsys.stderr.write("last words")\nsys.stderr.flush()\nos._exit(3)\n'))
        self.assertEqual(error, "solver process exited with status 3 and no answer: last words")

    def test_solver_that_hangs(self):
        error = self.decision_error(HANGING_SOLVER, decision_timeout_s=1)
        self.assertEqual(error, "solver decision timed out after 1 s")

    def test_unserialisable_suggestion(self):
        error = self.decision_error(solver_source('return {"config": {1, 2}}\n'))
        self.assertTrue(error.startswith("TypeError: "), error)

    def test_file_that_is_not_a_solver(self):
        self.assertIn("does not define solve_hpo(task, history)",
                      self.decision_error("def solve(task, history):\n    return {}\n"))
        self.assertIn("does not define solve_hpo(task, history)",
                      self.decision_error("solve_hpo = 3\n"))
        self.assertTrue(self.decision_error("def solve_hpo(task, history)\n")
                        .startswith("SyntaxError: "))

    def test_no_state_is_carried_between_decisions(self):
        policy = self.policy(solver_source(
            'global CALLS\nCALLS += 1\nreturn {"config": dict(task["starting_config"]), "calls": CALLS}\n',
            preamble="CALLS = 0\n\n"))
        self.assertEqual([policy(self.task, [])["solver_meta"]["calls"] for _ in range(3)],
                         [1, 1, 1])

    def test_a_changed_file_is_refused(self):
        policy = self.policy()
        write_solver(self.path("solver"), DEFAULT_SOLVER)
        with self.assertRaisesRegex(SolverError, "changed during the episode"):
            policy(self.task, [])

    def test_solver_can_import_its_neighbours_and_the_benchmark_module(self):
        write_solver(self.path("solver"), "DEPTH = 12\n", name="solver_helpers.py")
        policy = self.policy(solver_source(
            'return {"config": normalize_config(dict(task["starting_config"], DEPTH=DEPTH))}\n',
            preamble="from rewards.autoresearch_hpo import normalize_config\n"
                     "from solver_helpers import DEPTH\n\n"))
        self.assertEqual(policy(self.task, [])["config"]["DEPTH"], 12)

    def test_solver_directory_cannot_shadow_the_benchmark_module(self):
        os.makedirs(self.path("solver", "rewards"))
        with open(self.path("solver", "rewards", "__init__.py"), "w", encoding="utf-8") as f:
            f.write("raise RuntimeError('shadowed')\n")
        self.assertEqual(self.policy()(self.task, [])["solver_meta"],
                         {"strategy": "local perturbation"})

    def test_no_bytecode_is_left_next_to_the_solver(self):
        write_solver(self.path("solver"), "DEPTH = 12\n", name="solver_helpers.py")
        self.policy(solver_source('return {"config": dict(task["starting_config"])}\n',
                                  preamble="import solver_helpers\n\n"))(self.task, [])
        self.assertEqual(sorted(os.listdir(self.path("solver"))),
                         ["SOLVER_HPO.py", "solver_helpers.py"])

    def test_the_solver_sees_task_and_history_as_the_harness_holds_them(self):
        policy = self.policy(solver_source(
            'return {"config": dict(task["starting_config"]), "task": task, "history": history}\n'))
        history = [{"trial_id": 0, "config": STARTING_CONFIG, "status": "ok", "val_bpb": 0.991,
                    "error": None, "wall_time_s": 300.0, "peak_vram_mb": None, "metrics": {}}]
        suggestion = policy(self.task, history)
        self.assertEqual(suggestion["solver_meta"], {"task": self.task, "history": history})

    def test_load_solver_in_process(self):
        solve = load_solver(write_solver(self.path("solver"), DEFAULT_SOLVER))
        self.addCleanup(sys.modules.pop, "SOLVER_HPO", None)
        self.addCleanup(sys.path.remove, self.path("solver"))
        self.assertEqual(solve(self.task, []), {"config": STARTING_CONFIG})

    def test_episode(self):
        policy = self.policy()
        log = self.path("full.jsonl")
        episode = run_episode(policy, self.entry, log_path=log, policy_name=policy.name)
        score = episode["score"]
        self.assertEqual(episode["stop_reason"], "time_budget")
        self.assertGreaterEqual(score["n_trials"], 6)
        self.assertEqual(score["status_counts"]["invalid"], 0)
        self.assertEqual(episode["trials"][0]["policy_meta"], {"source": "starting_config"})
        for trial in episode["trials"][1:]:
            self.assertEqual(trial["policy_meta"], {
                "source": "solver", "solver_meta": {"strategy": "local perturbation"}})

        # An episode resumed from the first rows of the log continues exactly as the
        # uninterrupted one did: the decisions depend on (task, history) alone.
        rows = read_jsonl(log)
        self.assertEqual(rows[0]["policy"], policy.name)
        partial = self.path("partial.jsonl")
        with open(partial, "w", encoding="utf-8") as f:
            for row in rows[:4]:
                f.write(json.dumps(row) + "\n")
        resumed = run_episode(policy, self.entry, log_path=partial, policy_name=policy.name)
        key = lambda trial: (trial["trial_id"], trial["config"], trial["status"], trial["val_bpb"])
        self.assertEqual([key(t) for t in resumed["trials"]], [key(t) for t in episode["trials"]])

        # A different solver file is a different policy and does not continue the log.
        other = SolverPolicy(write_solver(self.path("other"), DEFAULT_SOLVER))
        with self.assertRaisesRegex(EpisodeLogMismatch, "refusing to resume"):
            run_episode(other, self.entry, log_path=log, policy_name=other.name)

    def test_a_solver_that_never_answers_ends_the_episode(self):
        policy = self.policy(CRASHING_SOLVER)
        episode = run_episode(policy, self.entry, max_consecutive_invalid=3)
        self.assertEqual(episode["stop_reason"], "policy_invalid")
        self.assertEqual([t["status"] for t in episode["trials"]], ["ok"] + ["invalid"] * 3)
        self.assertTrue(episode["trials"][1]["error"]
                        .startswith("policy raised SolverError: ValueError: bad idea"))
        self.assertEqual(episode["score"]["best_trial_id"], 0)


# ──────────────────────────────────────────────────────────────────────────────
# scripts/eval_hpo_solver.py
# ──────────────────────────────────────────────────────────────────────────────

class EvalScriptTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.entries = [synthetic_entry(seed=0), synthetic_entry(seed=1)]
        self.tasks = write_tasks(self.path("tasks.jsonl"), self.entries)
        self.out = self.path("out")

    def argv(self, *args):
        return [*args, "--tasks-file", self.tasks, "--output-dir", self.out]

    def result(self, entry):
        with open(os.path.join(self.out, f"{entry['task_id']}.result.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_solver(self):
        solver = write_solver(self.path("solver"))
        status, out, err = quiet(eval_hpo_solver.main, self.argv("--solver", solver))
        self.assertEqual((status, err), (0, ""))
        self.assertIn("NOTE: synthetic objective", out)
        self.assertEqual(sorted(os.listdir(self.out)), sorted(
            f"{entry['task_id']}{suffix}" for entry in self.entries
            for suffix in (".jsonl", ".result.json")))
        for entry in self.entries:
            result = self.result(entry)
            self.assertRegex(result["policy"], r"^solver:SOLVER_HPO\.py@[0-9a-f]{12}$")
            self.assertEqual(result["objective"]["kind"], "synthetic")
            self.assertEqual(result["stop_reason"], "time_budget")
            self.assertEqual(result["status_counts"]["invalid"], 0)
            self.assertEqual(result["token_usage"], {"input": 0, "output": 0, "reasoning": 0})
            self.assertLessEqual(result["best_val_bpb"], result["baseline_val_bpb"])
            self.assertEqual(result["n_trials"], len(read_jsonl(result["episode_log"])) - 1)

        # the same command again finds the finished episodes and decides nothing new
        before = {entry["task_id"]: self.result(entry) for entry in self.entries}
        with mock.patch.object(SolverPolicy, "__call__", side_effect=AssertionError("asked again")):
            status, _, _ = quiet(eval_hpo_solver.main, self.argv("--solver", solver))
        self.assertEqual(status, 0)
        for entry in self.entries:
            after = self.result(entry)
            for field in ("best_val_bpb", "n_trials", "best_config", "policy"):
                self.assertEqual(after[field], before[entry["task_id"]][field])

    def test_what_a_solver_reports_about_itself_is_not_counted(self):
        # Self-reported token counts and sources, well-formed or not, stay in the log
        # as solver_meta: the result has the harness's own figures.
        for name, body in (
                ("spoof", '"source": "tpe", "token_usage": {"input": 7, "output": 7, "reasoning": 7}'),
                ("malformed", '"source": ["x"], "token_usage": "lots"'),
                ("unhashable", '"source": {"a": 1}, "token_usage": {"input": None}')):
            with self.subTest(name):
                solver = write_solver(self.path(name), solver_source(
                    'return {"config": dict(task["starting_config"]), %s}\n' % body))
                out_dir = self.path(f"out_{name}")
                status, _, err = quiet(eval_hpo_solver.main, [
                    "--solver", solver, "--tasks-file", self.tasks, "--output-dir", out_dir,
                    "--task-id", self.entries[0]["task_id"]])
                self.assertEqual((status, err), (0, ""))
                with open(os.path.join(out_dir, f"{self.entries[0]['task_id']}.result.json"),
                          encoding="utf-8") as f:
                    result = json.load(f)
                self.assertEqual(result["token_usage"], {"input": 0, "output": 0, "reasoning": 0})
                self.assertEqual(result["suggestion_sources"],
                                 {"starting_config": 1, "solver": result["n_trials"] - 1})
                trial = read_jsonl(result["episode_log"])[2]
                self.assertEqual(trial["policy_meta"]["source"], "solver")
                self.assertIn("token_usage", trial["policy_meta"]["solver_meta"])
                self.assertNotIn("token_usage", trial["policy_meta"])

    def test_a_changed_solver_does_not_continue_the_logs_of_the_old_one(self):
        solver = write_solver(self.path("solver"))
        self.assertEqual(quiet(eval_hpo_solver.main, self.argv("--solver", solver))[0], 0)
        write_solver(self.path("solver"), DEFAULT_SOLVER)
        status, _, err = quiet(eval_hpo_solver.main, self.argv("--solver", solver))
        self.assertEqual(status, 1)
        self.assertIn("refusing to resume", err)
        self.assertNotIn("Traceback", err)

    def test_a_solver_without_usable_answers_scores_the_default(self):
        solver = write_solver(self.path("solver"), GARBAGE_SOLVER)
        status, _, _ = quiet(eval_hpo_solver.main, self.argv(
            "--solver", solver, "--task-id", self.entries[0]["task_id"]))
        self.assertEqual(status, 0)
        result = self.result(self.entries[0])
        self.assertEqual(result["stop_reason"], "policy_invalid")
        self.assertEqual(result["best_trial_id"], 0)
        self.assertEqual(result["best_val_bpb"], result["baseline_val_bpb"])
        self.assertFalse(os.path.exists(os.path.join(
            self.out, f"{self.entries[1]['task_id']}.result.json")))

    def test_random_baseline(self):
        status, out, _ = quiet(eval_hpo_solver.main, self.argv("--policy", "random"))
        self.assertEqual(status, 0)
        self.assertIn("policy: random", out)
        results = [self.result(entry) for entry in self.entries]
        self.assertEqual({r["policy"] for r in results}, {"random"})
        self.assertEqual([r["suggestion_sources"].get("random") for r in results],
                         [r["n_trials"] - 1 for r in results])

    @unittest.skipUnless(HAVE_OPTUNA, "Optuna is not installed")
    def test_tpe_baseline(self):
        status, _, _ = quiet(eval_hpo_solver.main, self.argv("--policy", "tpe"))
        self.assertEqual(status, 0)
        self.assertEqual({self.result(entry)["policy"] for entry in self.entries}, {"tpe"})

    @unittest.skipIf(HAVE_OPTUNA, "Optuna is installed")
    def test_optuna_baselines_say_what_to_install(self):
        status, err = exit_status(eval_hpo_solver.main, self.argv("--policy", "tpe"))
        self.assertEqual(status, 2)
        self.assertIn("pip install optuna", err)
        self.assertFalse(os.path.exists(self.out))

    def test_bad_arguments(self):
        solver = write_solver(self.path("solver"))
        for argv, message in (
                (self.argv(), "one of the arguments --solver --policy is required"),
                (self.argv("--solver", solver, "--policy", "random"), "not allowed with"),
                (self.argv("--policy", "grid"), "invalid choice"),
                (self.argv("--solver", self.path("missing.py")), "solver file not found"),
                (self.argv("--policy", "random", "--task-id", "nope"), "not in"),
                (["--policy", "random", "--output-dir", self.out,
                  "--tasks-file", self.path("missing.jsonl")], "No such file"),
        ):
            status, err = exit_status(eval_hpo_solver.main, argv)
            self.assertEqual(status, 2, argv)
            self.assertIn(message, err)
        self.assertFalse(os.path.exists(self.out))


# ──────────────────────────────────────────────────────────────────────────────
# OpenHands SolverBuilder
# ──────────────────────────────────────────────────────────────────────────────

def scripted_conversation(script):
    """A stand-in for the SDK's ``Conversation`` that plays ``script``.

    ``script`` is a list of ``(tool name, action)`` pairs, run through the executors of
    the agent's own tools, or exceptions to raise at that point. The class keeps the
    instances it created in ``created``.
    """
    class ScriptedConversation:
        created = []

        def __init__(self, agent, workspace, max_iteration_per_run):
            self.agent = agent
            self.workspace = workspace
            self.max_iteration_per_run = max_iteration_per_run
            self.messages = []
            self.observations = []
            self.state = SimpleNamespace(events=[])
            type(self).created.append(self)

        def send_message(self, text):
            self.messages.append(text)

        def run(self):
            for step in script:
                if isinstance(step, BaseException):
                    raise step
                name, action = step
                self.observations.append(self.agent.tools_map[name].executor(action))
                self.state.events.append(SimpleNamespace(model_dump=lambda name=name: {"tool": name}))

    return ScriptedConversation


@unittest.skipUnless(HAVE_SDK, "the OpenHands SDK is not installed")
class SolverBuilderTestCase(TempDirTestCase):
    """Base: a controller on the fake sandbox, temporary files kept inside the test dir."""

    def setUp(self):
        super().setUp()
        # The controller and the sandbox create temporary files; keep them in self.tmp.
        patcher = mock.patch.object(tempfile, "tempdir", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

        self.demos = self.path("demos.json")
        with open(self.demos, "w", encoding="utf-8") as f:
            json.dump([{"task_id": "t", "step": 1, "cot": "thinking", "success": True}], f)
        self.out = self.path("out")
        self.sandbox = FakeSandbox(timeout=120)

    def controller(self, **kwargs):
        options = dict(base_url="http://127.0.0.1:9/v1", model="openai/test-model",
                       sandbox=self.sandbox, demos_path=self.demos, rewards_dir=REWARDS_DIR,
                       output_dir=self.out, smoke_tasks_path=SYNTHETIC_TASKS)
        options.update(kwargs)
        return hpo_controller.HPOSolverBuilderController(**options)

    @staticmethod
    def execute(code):
        return ("hpo_execute_code", hpo_tools.HPOExecuteCodeAction(code=textwrap.dedent(code)))

    @staticmethod
    def write(filename, content):
        return ("hpo_write_file", hpo_tools.HPOWriteFileAction(filename=filename, content=content))

    @staticmethod
    def finish(summary):
        return ("sb_finish", SBFinishAction(summary=summary))

    def good_script(self, solver=REFERENCE_SOLVER):
        return [
            self.execute("""
                import json
                print(len(json.load(open("/workspace/DEMOS.json"))))
            """),
            self.write("SOLVER_HPO.py", solver),
            self.execute("""
                import json
                from rewards.autoresearch_hpo import run_episode
                from SOLVER_HPO import solve_hpo
                entry = json.loads(open("/workspace/SYNTHETIC_TASKS.jsonl").readline())
                print(run_episode(solve_hpo, entry)["score"]["status_counts"]["invalid"])
            """),
            self.write("SOLVER_HPO_ALGORITHM.md", "# Algorithm\n\nLocal search.\n"),
            self.finish("local search around the incumbent"),
        ]


class HPOToolsTests(SolverBuilderTestCase):
    def tools(self):
        return self.controller()._make_agent(self.path("task")).tools_map

    def text(self, observation):
        return "".join(part.text for part in observation.to_llm_content)

    def test_tool_names_match_the_prompts(self):
        tools = self.tools()
        self.assertEqual(sorted(tools), ["hpo_execute_code", "hpo_write_file", "sb_finish"])
        prompt = hpo_controller.system_prompt(120)
        for name in tools:
            self.assertIn(f"  {name}(", prompt)
            self.assertNotIn(name.replace("hpo_", "").replace("sb_", "") + "(",
                             prompt.replace(f"{name}(", ""))
        self.assertIn("limited to 120 s", tools["hpo_execute_code"].description)
        for filename in hpo_tools.ALLOWED_FILENAMES:
            self.assertIn(filename, tools["hpo_write_file"].description)
            self.assertIn(filename, prompt)
            self.assertIn(filename, hpo_controller._FILE_NOTE)

    def test_write_tool_accepts_only_the_two_files(self):
        write = self.tools()["hpo_write_file"].executor
        for filename in ("SOLVER.py", "solver_hpo.py", "../SOLVER_HPO.py", "sub/SOLVER_HPO.py",
                         os.path.join(self.out, "SOLVER_HPO.py"), "", "notes.md"):
            observation = write(hpo_tools.HPOWriteFileAction(filename=filename, content="x"))
            self.assertFalse(observation.ok, filename)
            self.assertIn("Only ['SOLVER_HPO.py', 'SOLVER_HPO_ALGORITHM.md']", self.text(observation))
        self.assertFalse(os.path.exists(self.out) and os.listdir(self.out))

        observation = write(hpo_tools.HPOWriteFileAction(filename=" SOLVER_HPO.py\n", content="A = 1\n"))
        self.assertTrue(observation.ok)
        self.assertIn("/workspace/solver/SOLVER_HPO.py", self.text(observation))
        write(hpo_tools.HPOWriteFileAction(filename="SOLVER_HPO.py", content="A = 2  # ü\n"))
        write(hpo_tools.HPOWriteFileAction(filename="SOLVER_HPO_ALGORITHM.md", content="# doc\n"))
        self.assertEqual(sorted(os.listdir(self.out)), ["SOLVER_HPO.py", "SOLVER_HPO_ALGORITHM.md"])
        with open(os.path.join(self.out, "SOLVER_HPO.py"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "A = 2  # ü\n")

    def test_execute_tool_sees_demos_tasks_benchmark_module_and_written_solver(self):
        tools = self.tools()
        tools["hpo_write_file"].executor(
            hpo_tools.HPOWriteFileAction(filename="SOLVER_HPO.py", content=REFERENCE_SOLVER))
        observation = tools["hpo_execute_code"].executor(hpo_tools.HPOExecuteCodeAction(code=textwrap.dedent("""
            import json, os
            from rewards.autoresearch_hpo import SEARCH_SPACE
            from SOLVER_HPO import solve_hpo
            demos = json.load(open("/workspace/DEMOS.json"))
            tasks = [json.loads(line) for line in open("/workspace/SYNTHETIC_TASKS.jsonl")]
            print(len(SEARCH_SPACE), len(demos), len(tasks), callable(solve_hpo),
                  sorted(os.listdir("/workspace/solver")))
        """)))
        self.assertTrue(observation.ok, observation.stderr)
        self.assertEqual(self.text(observation), "stdout:\n14 1 5 True ['SOLVER_HPO.py']")
        call = self.sandbox.calls[-1]
        self.assertIsNone(call["timeout"])               # the sandbox's own per-call limit
        self.assertEqual(
            [(container, mode) for _, container, mode in call["extra_binds"]],
            [("/workspace/DEMOS.json", "ro"), ("/workspace/rewards", "ro"),
             ("/workspace/SYNTHETIC_TASKS.jsonl", "ro"), ("/workspace/solver", "ro")])

    def test_execute_tool_reports_failures(self):
        execute = self.tools()["hpo_execute_code"].executor
        observation = execute(hpo_tools.HPOExecuteCodeAction(code="print('before')\nraise KeyError('DEPTH')"))
        self.assertFalse(observation.ok)
        text = self.text(observation)
        self.assertTrue(text.startswith("Execution failed.\nstderr:\nTraceback"), text)
        self.assertIn("KeyError: 'DEPTH'\nstdout:\nbefore", text)
        self.assertEqual(self.text(execute(hpo_tools.HPOExecuteCodeAction(code="pass"))),
                         "stdout:\n(no output)")

    def test_long_output_is_clipped_where_it_matters_least(self):
        observation = hpo_tools.HPOExecuteCodeObservation(
            ok=False, stdout="o" * (hpo_tools.MAX_STDOUT_CHARS + 7),
            stderr="e" * hpo_tools.MAX_STDERR_CHARS + "\nValueError: the actual error")
        text = self.text(observation)
        self.assertIn("ValueError: the actual error\nstdout:", text)      # the end of stderr
        self.assertIn(" earlier characters not shown] ...\n", text)
        self.assertTrue(text.endswith("\n... [7 more characters not shown]"))  # the start of stdout
        self.assertLess(len(text), hpo_tools.MAX_STDOUT_CHARS + hpo_tools.MAX_STDERR_CHARS + 200)

    def test_finish_tool_writes_the_summary(self):
        observation = self.tools()["sb_finish"].executor(SBFinishAction(summary="done and tested"))
        self.assertIn("done and tested", self.text(observation))
        with open(self.path("task", "done.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "done and tested")


class HPOControllerTests(SolverBuilderTestCase):
    def test_system_prompt_is_a_literal_template(self):
        import jinja2
        prompt = hpo_controller.system_prompt(120)
        for delimiter in ("{{", "}}", "{%", "%}", "{#", "#}"):
            self.assertNotIn(delimiter, prompt)
        self.assertEqual(jinja2.Template(prompt, keep_trailing_newline=True).render(), prompt)
        self.assertIn("120 s per call", prompt)
        controller = self.controller()
        self.assertEqual(controller.solver_type, "hpo")
        with open(controller._system_prompt_file, encoding="utf-8") as f:
            self.assertEqual(f.read(), prompt)

    def test_the_workflow_example_of_the_system_prompt_runs(self):
        lines = hpo_controller.system_prompt(120).splitlines()
        start = next(i for i, line in enumerate(lines) if line.strip() == "import json")
        indent = len(lines[start]) - len(lines[start].lstrip())
        block = []
        for line in lines[start:]:
            if line.strip() and len(line) - len(line.lstrip()) < indent:
                break
            block.append(line)
        tools = self.controller()._make_agent(self.path("task")).tools_map
        tools["hpo_write_file"].executor(
            hpo_tools.HPOWriteFileAction(filename="SOLVER_HPO.py", content=REFERENCE_SOLVER))
        observation = tools["hpo_execute_code"].executor(
            hpo_tools.HPOExecuteCodeAction(code=textwrap.dedent("\n".join(block))))
        self.assertTrue(observation.ok, observation.stderr)
        rows = [line for line in observation.stdout.splitlines()
                if line.startswith("autoresearch_hpo_synthetic_")]      # not the solver's prints
        self.assertEqual(len(rows), 5)
        for row, entry in zip(rows, read_jsonl(SYNTHETIC_TASKS)):
            self.assertTrue(row.startswith(f"{entry['task_id']} time_budget 72 "), row)
            self.assertIn("'invalid': 0", row)

    def test_building_prompt_and_file_note_agree_with_the_tools(self):
        with open(BUILDING_PROMPT, encoding="utf-8") as f:
            building_prompt = f.read()
        references = set(re.findall(r"@([\w./*-]+\w)", building_prompt))
        self.assertIn("rewards/autoresearch_hpo.py", references)
        self.assertIn("SOLVER_HPO.py", references)
        self.assertIn("SOLVER_HPO_ALGORITHM.md", references)
        for reference in references:
            if reference.startswith("demos/"):
                self.assertRegex(reference, r"^demos/DEMOS_AUTORESEARCH_HPO_.*\.json$")
            else:
                self.assertIn(f"@{reference} →", hpo_controller._FILE_NOTE)
                if "/" in reference:
                    self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, reference)), reference)
        self.assertIn("def solve_hpo(task, history)", building_prompt)
        for name in HP_NAMES:
            self.assertIn(name, building_prompt)

    def test_smoke_test(self):
        controller = self.controller()
        write_solver(self.out)
        report = controller.smoke_test()
        self.assertTrue(report["ok"])
        self.assertIsNone(report["error"])
        self.assertEqual([task["task_id"] for task in report["tasks"]],
                         [entry["task_id"] for entry in read_jsonl(SYNTHETIC_TASKS)])
        for task in report["tasks"]:
            self.assertEqual((task["stop_reason"], task["n_trials"]), ("time_budget", 72))
            self.assertLess(task["best_val_bpb"], task["baseline_val_bpb"])
            self.assertIsNone(task["first_invalid_error"])
        self.assertEqual(self.sandbox.calls[-1]["timeout"], hpo_controller.SMOKE_TEST_TIMEOUT_S)

    def test_smoke_test_catches_unusable_solvers(self):
        controller = self.controller()

        write_solver(self.out, GARBAGE_SOLVER)
        report = controller.smoke_test()
        self.assertFalse(report["ok"])
        self.assertEqual({task["stop_reason"] for task in report["tasks"]}, {"policy_invalid"})
        self.assertEqual(report["tasks"][0]["first_invalid_error"],
                         "invalid config: DEPTH='deep' is not a number")

        write_solver(self.out, CRASHING_SOLVER)
        report = controller.smoke_test()
        self.assertFalse(report["ok"])
        self.assertEqual(report["tasks"][0]["first_invalid_error"], "policy raised ValueError: bad idea")

        write_solver(self.out, "def solve_hpo(task, history)\n")
        report = controller.smoke_test()
        self.assertEqual((report["ok"], report["tasks"]), (False, []))
        self.assertIn("SyntaxError", report["error"])

        write_solver(self.out, "def solve(task, history):\n    return {}\n")
        self.assertIn("ImportError", controller.smoke_test()["error"])

        write_solver(self.out, solver_source("raise SystemExit(0)\n"))
        report = controller.smoke_test()
        self.assertFalse(report["ok"])
        self.assertIn("SystemExit", report["error"])

        write_solver(self.out, solver_source("import os\nos._exit(0)\n"))
        self.assertEqual(controller.smoke_test(),
                         {"ok": False, "tasks": [], "error": "no smoke-test report in the output"})

        # a solver cannot pass by printing a report of its own
        fake = json.dumps({"ok": True, "error": None, "tasks": []})
        write_solver(self.out, solver_source(
            f"print('SMOKE_TEST_0123:' + {fake!r})\nraise ValueError('bad idea')\n"))
        self.assertFalse(controller.smoke_test()["ok"])

    def test_smoke_test_time_limit(self):
        controller = self.controller()
        write_solver(self.out, HANGING_SOLVER)
        with mock.patch.object(hpo_controller, "SMOKE_TEST_TIMEOUT_S", 1):
            report = controller.smoke_test()
        self.assertEqual(report, {"ok": False, "tasks": [], "error": "Timeout after 1s"})

    def build(self, script, **kwargs):
        conversation = scripted_conversation(script)
        with mock.patch.object(hpo_controller, "Conversation", conversation):
            result = self.controller(**kwargs).build("THE SPECIFICATION")
        return result, conversation.created[0]

    def test_build(self):
        result, conversation = self.build(self.good_script(), max_steps=17)
        self.assertTrue(all(observation.ok for observation in conversation.observations[:4]),
                        [getattr(o, "stderr", "") for o in conversation.observations[:4]])
        self.assertEqual(conversation.observations[0].stdout.strip(), "1")
        # the solver's own prints come first; the last line is the practice episode's count
        self.assertEqual(conversation.observations[2].stdout.splitlines()[-1], "0")
        self.assertEqual(conversation.max_iteration_per_run, 17)
        self.assertEqual(conversation.messages, [hpo_controller._FILE_NOTE + "THE SPECIFICATION"])

        self.assertTrue(result["success"])
        self.assertEqual(result["solver_path"], os.path.join(self.out, "SOLVER_HPO.py"))
        self.assertEqual(result["algorithm_path"], os.path.join(self.out, "SOLVER_HPO_ALGORITHM.md"))
        self.assertEqual(result["summary"], "local search around the incumbent")
        self.assertTrue(result["smoke_test"]["ok"])
        self.assertEqual(len(result["smoke_test"]["tasks"]), 5)
        self.assertEqual([event["tool"] for event in result["_trajectory"]],
                         ["hpo_execute_code", "hpo_write_file", "hpo_execute_code",
                          "hpo_write_file", "sb_finish"])
        with open(result["solver_path"], encoding="utf-8") as f:
            self.assertEqual(f.read(), REFERENCE_SOLVER)

    def test_build_with_a_conversation_that_breaks_off(self):
        script = self.good_script()[:2] + [RuntimeError("context window exceeded")]
        result, _ = self.build(script)
        self.assertFalse(result["success"])
        self.assertEqual(result["solver_path"], os.path.join(self.out, "SOLVER_HPO.py"))
        self.assertIsNone(result["algorithm_path"])
        self.assertEqual(result["summary"], "")
        self.assertTrue(result["smoke_test"]["ok"])           # the solver file itself runs
        self.assertEqual(len(result["_trajectory"]), 2)

    def test_build_without_any_file(self):
        result, _ = self.build([self.finish("nothing to add")])
        self.assertFalse(result["success"])
        self.assertIsNone(result["solver_path"])
        self.assertIsNone(result["smoke_test"])
        self.assertEqual(result["summary"], "nothing to add")
        self.assertEqual(os.listdir(self.out), [])            # exists: it is bind-mounted

    def test_build_reports_a_solver_that_does_not_work(self):
        result, _ = self.build(self.good_script(solver=GARBAGE_SOLVER))
        self.assertTrue(result["success"])                    # both files were written
        self.assertFalse(result["smoke_test"]["ok"])


class BuildSolverCommandTests(SolverBuilderTestCase):
    def argv(self, *extra):
        return ["--building-prompt", BUILDING_PROMPT, "--demos-path", self.demos,
                "--rewards-dir", REWARDS_DIR, "--smoke-tasks", SYNTHETIC_TASKS,
                "--output-dir", self.out, "--sif-path", self.path("sandbox.sif"),
                "--sandbox-timeout", "77", "--debug-dir", self.path("debug"), *extra]

    def run_main(self, script, argv):
        """Exit status of the entry point with a scripted conversation and the fake sandbox."""
        conversation = scripted_conversation(script)
        sandboxes = []

        def make_sandbox(sif_path, timeout):
            sandboxes.append((sif_path, timeout))
            return FakeSandbox(timeout=timeout)

        with mock.patch.object(hpo_controller, "Conversation", conversation), \
                mock.patch("openhands_agents.sandbox.ApptainerSandbox", make_sandbox):
            status, err = exit_status(build_solver_hpo.main, argv)
        return status, err, conversation, sandboxes

    def test_successful_build(self):
        status, _, conversation, sandboxes = self.run_main(self.good_script(), self.argv())
        self.assertEqual(status, 0)
        self.assertEqual(sandboxes, [(self.path("sandbox.sif"), 77)])
        self.assertEqual(sorted(os.listdir(self.out)), ["SOLVER_HPO.py", "SOLVER_HPO_ALGORITHM.md"])
        with open(BUILDING_PROMPT, encoding="utf-8") as f:
            self.assertEqual(conversation.created[0].messages,
                             [hpo_controller._FILE_NOTE + f.read()])
        self.assertIn("77 s per call", open_text(
            conversation.created[0].agent.system_prompt_filename))

        with open(self.path("debug", "solver_builder_hpo_run.json"), encoding="utf-8") as f:
            run = json.load(f)
        self.assertTrue(run["success"] and run["smoke_test"]["ok"])
        self.assertNotIn("_trajectory", run)
        with open(self.path("debug", "solver_builder_hpo_trajectory.json"), encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)), 5)

        # the solver that was built is one the evaluation script accepts
        tasks = write_tasks(self.path("tasks.jsonl"), [synthetic_entry()])
        status, _, _ = quiet(eval_hpo_solver.main, [
            "--solver", os.path.join(self.out, "SOLVER_HPO.py"), "--tasks-file", tasks,
            "--output-dir", self.path("eval")])
        self.assertEqual(status, 0)

    def test_incomplete_or_broken_builds_fail(self):
        for name, script in (("no_algorithm", self.good_script()[:3]),
                             ("nothing", [RuntimeError("server went away")]),
                             ("garbage", self.good_script(solver=GARBAGE_SOLVER))):
            self.out = self.path(name)
            status, _, _, _ = self.run_main(script, self.argv())
            self.assertEqual(status, 1, name)

    def test_bad_arguments(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("SANDBOX_SIF", None)
            argv = self.argv()
            del argv[argv.index("--sif-path"):argv.index("--sif-path") + 2]
            status, err, _, _ = self.run_main([], argv)
            self.assertEqual(status, 2)
            self.assertIn("--sif-path required", err)

        status, err, _, _ = self.run_main([], self.argv("--demos-path", self.path("missing.json")))
        self.assertEqual(status, 2)
        self.assertIn("--demos-path path does not exist", err)

        # leftovers of an earlier run would be mistaken for this run's output
        write_solver(self.out)
        status, err, conversation, _ = self.run_main(self.good_script(), self.argv())
        self.assertEqual(status, 2)
        self.assertIn("already contains ['SOLVER_HPO.py']", err)
        self.assertEqual(conversation.created, [])

    def test_sif_path_from_the_environment(self):
        argv = self.argv()
        del argv[argv.index("--sif-path"):argv.index("--sif-path") + 2]
        with mock.patch.dict(os.environ, {"SANDBOX_SIF": "/images/sandbox.sif"}):
            status, _, _, sandboxes = self.run_main(self.good_script(), argv)
        self.assertEqual(status, 0)
        self.assertEqual(sandboxes, [("/images/sandbox.sif", 77)])


def open_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ──────────────────────────────────────────────────────────────────────────────
# scripts/summarize_hpo_results.py
# ──────────────────────────────────────────────────────────────────────────────

NANOCHAT = {"kind": "nanochat", "pinned": True, "cuda_mem_fraction": 0.54,
            "device": {"gpu": "NVIDIA H200", "total_vram_gb": 139.8}, "hardware_matches": True}
SYNTHETIC = {"kind": "synthetic", "landscape_seed": 0}
AGENT_POLICY = "hpo_agent[Qwen/Qwen3.6-35B-A3B|cot|T=0.7|max_tokens=8192|retries=2]"


def result_row(policy, seed, best, objective=NANOCHAT, hours=24, stop_reason="time_budget",
               n_trials=280, failure_rate=0.25, tokens=0):
    return {"task_id": f"autoresearch_hpo_{hours}h_seed{seed}", "task": "autoresearch_hpo",
            "seed": seed, "policy": policy, "objective": objective,
            "time_budget_s": hours * 3600, "stop_reason": stop_reason, "best_val_bpb": best,
            "baseline_val_bpb": 0.991, "n_trials": n_trials, "failure_rate": failure_rate,
            "token_usage": {"input": tokens, "output": 0, "reasoning": 0}}


class SummaryTests(TempDirTestCase):
    def write_rows(self, directory, rows):
        os.makedirs(self.path(directory), exist_ok=True)
        paths = []
        for row in rows:
            paths.append(self.path(directory, f"{row['task_id']}.result.json"))
            with open(paths[-1], "w", encoding="utf-8") as f:
                json.dump(row, f)
        return paths

    def test_find_results(self):
        solver = self.write_rows("evals/solver", [result_row("solver", s, 0.97) for s in range(3)])
        tpe = self.write_rows("outputs/tpe", [result_row("tpe", 0, 0.98)])
        with open(self.path("outputs", "tpe", "autoresearch_hpo_24h_seed0.jsonl"), "w") as f:
            f.write("{}\n")                                         # episode logs are not results
        self.assertEqual(summarizer.find_results([self.tmp]), solver + tpe)
        # a directory and a file inside it: no duplicates
        self.assertEqual(summarizer.find_results([self.path("outputs"), tpe[0], solver[1]]),
                         tpe + [solver[1]])
        self.assertEqual(summarizer.find_results([self.path("evals", "solver", "..", "..", "outputs")]),
                         [self.path("evals", "solver", "..", "..", "outputs", "tpe",
                                    "autoresearch_hpo_24h_seed0.result.json")])
        with self.assertRaisesRegex(FileNotFoundError, "no such file or directory"):
            summarizer.find_results([self.path("missing")])

    def test_labels_never_pass_other_runs_off_as_the_benchmark(self):
        label = summarizer.objective_label
        self.assertEqual(label(NANOCHAT), "nanochat")
        self.assertEqual(label(SYNTHETIC), "synthetic (NOT benchmark)")
        self.assertEqual(label(dict(NANOCHAT, pinned=False)), "nanochat (unpinned sources)")
        self.assertEqual(label(dict(NANOCHAT, hardware_matches=False)),
                         "nanochat (hardware not verified)")
        self.assertEqual(label(dict(NANOCHAT, hardware_matches=None, pinned=False)),
                         "nanochat (unpinned sources, hardware not verified)")
        self.assertEqual(label({"kind": "nanochat"}),
                         "nanochat (unpinned sources, hardware not verified)")
        self.assertEqual(label({}), "None (NOT benchmark)")

        self.assertEqual(summarizer.device_label(NANOCHAT), "NVIDIA H200 139.8 GB x 0.54")
        self.assertEqual(summarizer.device_label(dict(NANOCHAT, cuda_mem_fraction=None)),
                         "NVIDIA H200 139.8 GB")
        self.assertEqual(summarizer.device_label(dict(NANOCHAT, device=None)), "unknown GPU")
        self.assertEqual(summarizer.device_label(SYNTHETIC), "-")

    def test_mean_and_sample_standard_deviation(self):
        self.assertEqual(summarizer.mean_std([0.5]), (0.5, None))
        mean, std = summarizer.mean_std([1.0, 2.0, 3.0])
        self.assertEqual((mean, std), (2.0, 1.0))
        mean, std = summarizer.mean_std([0.9739, 0.9751, 0.9727])
        self.assertAlmostEqual(mean, 0.9739)
        self.assertAlmostEqual(std, 0.0012)

    def rows(self):
        return ([result_row("tpe", s, v) for s, v in enumerate((0.9768, 0.9787, 0.9749))]
                + [result_row("solver:SOLVER_HPO.py@0123456789ab", s, v, tokens=0)
                   for s, v in enumerate((0.9739, 0.9751, 0.9727))]
                + [result_row(AGENT_POLICY, 0, 0.9904, tokens=1_200_000),
                   result_row(AGENT_POLICY, 1, None, tokens=800_000,
                              stop_reason="policy_invalid", n_trials=11, failure_rate=1.0)]
                + [result_row("tpe", 0, 0.9801, hours=1, n_trials=12)]
                + [result_row("tpe", 0, 0.9600, objective=SYNTHETIC),
                   result_row("tpe", 1, 0.9500, objective=dict(NANOCHAT, pinned=False))])

    def test_summarize(self):
        summary = summarizer.summarize(self.rows())
        self.assertEqual(
            [(g["objective"], g["time_budget_h"], g["policy"], g["n_runs"]) for g in summary],
            [("nanochat", 1.0, "tpe", 1),
             ("nanochat", 24.0, "solver:SOLVER_HPO.py@0123456789ab", 3),
             ("nanochat", 24.0, "tpe", 3),
             ("nanochat", 24.0, AGENT_POLICY, 2),
             ("nanochat (unpinned sources)", 24.0, "tpe", 1),
             ("synthetic (NOT benchmark)", 24.0, "tpe", 1)])
        solver, tpe, agent = summary[1:4]
        self.assertAlmostEqual(solver["best_val_bpb_mean"], 0.9739)
        self.assertAlmostEqual(solver["best_val_bpb_std"], 0.0012)
        self.assertEqual(solver["best_val_bpb_min"], 0.9727)
        self.assertEqual(solver["task_ids"], [f"autoresearch_hpo_24h_seed{s}" for s in range(3)])
        self.assertAlmostEqual(tpe["best_val_bpb_mean"], 0.9768)
        self.assertEqual(tpe["baseline_val_bpb_mean"], 0.991)
        # an episode without a finished trial counts as a run but not in the mean
        self.assertEqual((agent["best_val_bpb_mean"], agent["best_val_bpb_std"]), (0.9904, None))
        self.assertEqual(agent["stop_reasons"], ["policy_invalid", "time_budget"])
        self.assertEqual(agent["tokens_mean"], 1_000_000)
        self.assertEqual(agent["n_trials_mean"], 145.5)
        self.assertEqual(agent["failure_rate_mean"], 0.625)

        nothing = summarizer.summarize([result_row("tpe", 0, None)])[0]
        self.assertIsNone(nothing["best_val_bpb_mean"])
        self.assertIsNone(nothing["best_val_bpb_min"])

    def test_two_results_for_one_run_are_an_error(self):
        rows = [dict(result_row("tpe", 0, 0.97), _file="a/x.result.json"),
                dict(result_row("tpe", 0, 0.98), _file="b/x.result.json")]
        with self.assertRaisesRegex(ValueError, "two results for policy 'tpe' on task "
                                                "'autoresearch_hpo_24h_seed0': a/x.result.json and b/x"):
            summarizer.summarize(rows)

    def test_table(self):
        table = summarizer.format_table(summarizer.summarize(self.rows()))
        blocks = table.split("\n\n")
        self.assertEqual(len(blocks), 4)
        self.assertEqual(blocks[1].splitlines()[0],
                         "## autoresearch_hpo | objective: nanochat | "
                         "device: NVIDIA H200 139.8 GB x 0.54 | budget: 24 h")
        header, solver, tpe, agent = blocks[1].splitlines()[1:]
        self.assertEqual(header.split(), ["policy", "runs", "best", "val_bpb", "min", "default",
                                          "trials", "failed", "tokens"])
        self.assertEqual(solver.split(), ["solver:SOLVER_HPO.py@0123456789ab", "3", "0.9739", "±",
                                          "0.0012", "0.9727", "0.9910", "280.0", "25.0%", "0"])
        self.assertEqual(tpe.split()[:5], ["tpe", "3", "0.9768", "±", "0.0019"])
        self.assertTrue(agent.endswith("1000000  [stopped: policy_invalid]"), agent)
        # the columns line up whatever the length of the longest policy name
        self.assertEqual(agent.split()[:3], [AGENT_POLICY, "2", "0.9904"])
        self.assertEqual({len(line) for line in (header, solver, tpe)}, {len(header)})
        self.assertEqual(len(agent) - len("  [stopped: policy_invalid]"), len(header))
        for line in (header, solver, tpe, agent):
            self.assertEqual(line[len(AGENT_POLICY)], " ", line)
            self.assertNotEqual(line[len(AGENT_POLICY) + 2:].lstrip(), "", line)
        self.assertIn("synthetic (NOT benchmark) | device: - |", blocks[3])
        self.assertIn("no finished trial",
                      summarizer.format_table(summarizer.summarize([result_row("tpe", 0, None)])))

    def test_command_line(self):
        self.write_rows("solver", [r for r in self.rows() if r["policy"].startswith("solver")])
        self.write_rows("tpe", [r for r in self.rows()
                                if r["policy"] == "tpe" and r["objective"] is NANOCHAT])
        summary_path = self.path("reports", "summary.json")
        status, out, err = quiet(summarizer.main, [self.path("solver"), self.path("tpe"),
                                                   "--json", summary_path])
        self.assertEqual((status, err), (0, ""))
        self.assertIn("0.9739 ± 0.0012", out)
        self.assertIn("7 episode(s).", out)
        self.assertNotIn("not benchmark results", out)
        with open(summary_path, encoding="utf-8") as f:
            self.assertEqual(json.load(f), summarizer.summarize(summarizer.load_results(
                summarizer.find_results([self.path("solver"), self.path("tpe")]))))

        self.write_rows("synthetic", [result_row("tpe", 0, 0.96, objective=SYNTHETIC)])
        _, out, _ = quiet(summarizer.main, [self.tmp])
        self.assertIn("are not benchmark results.", out)

    def test_command_line_errors(self):
        os.makedirs(self.path("empty"))
        for argv, message in (([self.path("empty")], "no *.result.json files under"),
                              ([self.path("missing")], "no such file or directory"),
                              ([], "the following arguments are required")):
            status, err = exit_status(summarizer.main, argv)
            self.assertEqual(status, 2)
            self.assertIn(message, err)
        self.write_rows("a", [result_row("tpe", 0, 0.97)])
        self.write_rows("b", [result_row("tpe", 0, 0.98)])
        status, err = exit_status(summarizer.main, [self.tmp])
        self.assertEqual(status, 2)
        self.assertIn("two results for policy 'tpe'", err)

    def test_results_of_the_runners(self):
        tasks = write_tasks(self.path("tasks.jsonl"), [synthetic_entry(seed=s) for s in range(3)])
        for policy in ("--policy=random", "--solver=" + write_solver(self.path("solver"))):
            status, _, _ = quiet(eval_hpo_solver.main, [
                policy, "--tasks-file", tasks, "--output-dir", self.path("runs", policy[2:8])])
            self.assertEqual(status, 0)
        status, out, _ = quiet(summarizer.main, [self.path("runs")])
        self.assertEqual(status, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "## autoresearch_hpo_synthetic | objective: synthetic (NOT benchmark) | "
                                   "device: - | budget: 0.5 h")
        self.assertEqual(sorted(line.split()[0].split("@")[0] for line in lines[2:4]),
                         ["random", "solver:SOLVER_HPO.py"])
        self.assertEqual([line.split()[1] for line in lines[2:4]], ["3", "3"])
        self.assertIn("6 episode(s).", out)
        self.assertIn("are not benchmark results.", out)


# ──────────────────────────────────────────────────────────────────────────────
# Launch scripts
# ──────────────────────────────────────────────────────────────────────────────

LAUNCHERS = ("scripts/setup_autoresearch_hpo.sh", "scripts/run_hpo_traces_vllm.sh",
             "scripts/run_solver_builder_hpo_openhands.sh", "scripts/run_hpo_eval.sh")
# Variables the launchers read: a value in the caller's environment must not leak in.
LAUNCHER_VARIABLES = (
    "TASKS", "SOLVER", "POLICY", "TRAIN_GPU", "OUTPUT_DIR", "RUN_DIR", "PORT", "MODEL",
    "MODEL_TAG", "PROMPT_STYLE", "MAX_TOKENS", "TEMPERATURE", "DEMOS_PATH", "SIF_PATH",
    "SANDBOX_SIF", "AUTORESEARCH_DIR", "AUTORESEARCH_TRAIN_COMMAND", "PREPARE_ARGS",
    "CUDA_VISIBLE_DEVICES")

REAL_PYTHON = f"#!/bin/sh\nexec {shlex.quote(sys.executable)} \"$@\"\n"
# Prints its environment of interest and its arguments instead of running anything.
ECHO_PYTHON = '#!/bin/sh\necho "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"\nprintf \'%s\\n\' "$@"\n'


@unittest.skipUnless(shutil.which("bash"), "bash is not available")
class LauncherTests(TempDirTestCase):
    def stub(self, name, body):
        """Put an executable ``name`` first on the launchers' PATH."""
        os.makedirs(self.path("bin"), exist_ok=True)
        path = self.path("bin", name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        os.chmod(path, 0o755)

    def launch(self, script, *args, **variables):
        env = {k: v for k, v in os.environ.items() if k not in LAUNCHER_VARIABLES}
        env.update(variables, PATH=self.path("bin") + os.pathsep + env["PATH"])
        return subprocess.run(["bash", os.path.join(PROJECT_ROOT, script), *args], cwd=PROJECT_ROOT,
                              env=env, capture_output=True, text=True, timeout=600)

    def launched_args(self, script, *args, **variables):
        """The arguments ``script`` hands to python (and the training GPU it exports)."""
        self.stub("python", ECHO_PYTHON)
        proc = self.launch(script, *args, **variables)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("CUDA_VISIBLE_DEVICES="))
        return lines[start + 1:], lines[start].split("=", 1)[1]

    @staticmethod
    def option(args, name):
        return args[args.index(name) + 1]

    def test_scripts_are_executable_and_parse(self):
        for script in LAUNCHERS:
            path = os.path.join(PROJECT_ROOT, script)
            self.assertTrue(os.stat(path).st_mode & stat.S_IXUSR, script)
            with open(path, encoding="utf-8") as f:
                source = f.read()
            self.assertTrue(source.startswith("#!/usr/bin/env bash\n"), script)
            self.assertIn("\nset -euo pipefail\n", source)
            self.assertIn("# ── Config ──", source)
            proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
            self.assertEqual((proc.returncode, proc.stderr), (0, ""), script)

    def test_eval_launcher_wants_exactly_one_policy(self):
        self.stub("python", ECHO_PYTHON)
        for variables in ({}, {"SOLVER": "x/SOLVER_HPO.py", "POLICY": "tpe"}):
            proc = self.launch("scripts/run_hpo_eval.sh", **variables)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("Set exactly one of SOLVER=", proc.stderr)
            self.assertEqual(proc.stdout, "")

    def test_eval_launcher_arguments(self):
        solver = write_solver(self.path("built_solvers", "claude_code", "Mon_Nov_9_101_PM"))
        args, gpu = self.launched_args("scripts/run_hpo_eval.sh", "--task-id", "some task",
                                       SOLVER=solver, TRAIN_GPU="3")
        self.assertEqual(args, [
            "scripts/eval_hpo_solver.py", "--solver", solver,
            "--tasks-file", "data/autoresearch_hpo/pilot_tasks.jsonl",
            "--output-dir", "evals/solver_results/claude_code/Mon_Nov_9_101_PM/autoresearch_hpo_pilot_tasks",
            "--task-id", "some task"])
        self.assertEqual(gpu, "3")

        args, gpu = self.launched_args("scripts/run_hpo_eval.sh", POLICY="tpe",
                                       TASKS="data/autoresearch_hpo/tasks_full.jsonl")
        self.assertEqual(args[1:], ["--policy", "tpe",
                                    "--tasks-file", "data/autoresearch_hpo/tasks_full.jsonl",
                                    "--output-dir", "outputs/autoresearch_hpo_tasks_full_tpe"])
        self.assertEqual(gpu, "")
        args, _ = self.launched_args("scripts/run_hpo_eval.sh", POLICY="tpe", OUTPUT_DIR="elsewhere")
        self.assertEqual(self.option(args, "--output-dir"), "elsewhere")

    def test_eval_launcher_dry_run(self):
        self.stub("python", REAL_PYTHON)
        entries = [synthetic_entry(seed=0), synthetic_entry(seed=1)]
        tasks = write_tasks(self.path("tasks.jsonl"), entries)
        proc = self.launch("scripts/run_hpo_eval.sh", "--task-id", entries[1]["task_id"],
                           POLICY="random", TASKS=tasks, OUTPUT_DIR=self.path("out"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("NOTE: synthetic objective", proc.stdout)
        self.assertEqual(sorted(os.listdir(self.path("out"))),
                         [f"{entries[1]['task_id']}.jsonl", f"{entries[1]['task_id']}.result.json"])

        solver = write_solver(self.path("built_solvers", "claude_code", "run"))
        proc = self.launch("scripts/run_hpo_eval.sh", SOLVER=solver, TASKS=tasks,
                           OUTPUT_DIR=self.path("solver_out"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(len(os.listdir(self.path("solver_out"))), 4)

    def test_trace_launcher_arguments(self):
        args, gpu = self.launched_args("scripts/run_hpo_traces_vllm.sh", "--parse-retries", "0")
        self.assertEqual(args, [
            "-m", "openhands_agents.run_hpo_traces",
            "--tasks-file", "data/autoresearch_hpo/pilot_tasks.jsonl",
            "--run-dir", "outputs/autoresearch_hpo_pilot_tasks_hpo_agent_qwen3.6_35b_a3b",
            "--base-url", "http://localhost:8000/v1", "--model", "openai/Qwen/Qwen3.6-35B-A3B",
            "--prompt-style", "cot", "--max-tokens", "8192", "--temperature", "0.7",
            "--parse-retries", "0"])
        self.assertEqual(gpu, "")
        args, gpu = self.launched_args(
            "scripts/run_hpo_traces_vllm.sh", PROMPT_STYLE="paper", MAX_TOKENS="2048", PORT="8123",
            TRAIN_GPU="1", TASKS="data/autoresearch_hpo/tasks_full.jsonl")
        self.assertEqual(self.option(args, "--run-dir"),
                         "outputs/autoresearch_hpo_tasks_full_hpo_agent_qwen3.6_35b_a3b_paper")
        self.assertEqual(self.option(args, "--base-url"), "http://localhost:8123/v1")
        self.assertEqual((self.option(args, "--max-tokens"), gpu), ("2048", "1"))

    @unittest.skipUnless(HAVE_OPENAI, "the openai package is not installed")
    def test_trace_launcher_dry_run(self):
        self.stub("python", REAL_PYTHON)
        entry = synthetic_entry()
        tasks = write_tasks(self.path("tasks.jsonl"), [entry])
        with FakeLLMServer(lambda prompt, seed: (sampled_answer(prompt, seed), "thinking")) as server:
            port = server.base_url.rsplit(":", 1)[1].split("/")[0]
            proc = self.launch("scripts/run_hpo_traces_vllm.sh", TASKS=tasks, PORT=port,
                               RUN_DIR=self.path("run"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(sorted(os.listdir(self.path("run"))),
                         [f"{entry['task_id']}.jsonl", f"{entry['task_id']}.result.json"])
        self.assertEqual({request["path"] for request in server.requests}, {"/v1/chat/completions"})
        self.assertEqual({request["body"]["model"] for request in server.requests},
                         {"Qwen/Qwen3.6-35B-A3B"})                 # the name vLLM serves
        self.assertEqual({request["body"]["max_tokens"] for request in server.requests}, {8192})

    @unittest.skipUnless(HAVE_SDK, "the OpenHands SDK is not installed")
    def test_solver_builder_launcher_arguments_fit_the_entry_point(self):
        demos = self.path("DEMOS_variant.json")
        with open(demos, "w", encoding="utf-8") as f:
            json.dump([], f)
        args, _ = self.launched_args("scripts/run_solver_builder_hpo_openhands.sh", "8123",
                                     SIF_PATH=self.path("sandbox.sif"), DEMOS_PATH=demos)
        self.assertEqual(args[:2], ["-m", "openhands_agents.build_solver_hpo"])
        self.assertEqual(self.option(args, "--base-url"), "http://localhost:8123/v1")
        self.assertEqual(self.option(args, "--model"), "openai/Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(self.option(args, "--building-prompt"),
                         "building_prompts/SOLVER_BUILDING_PROMPT_HPO.md")
        self.assertEqual(self.option(args, "--smoke-tasks"), "data/autoresearch_hpo/synthetic_tasks.jsonl")
        self.assertRegex(self.option(args, "--output-dir"),
                         r"^built_solvers/qwen3\.6_35b_a3b/[A-Z][a-z]{2}_[A-Z][a-z]{2}_\d{1,2}_\d{3,4}_[AP]M"
                         r"_hpo_DEMOS_variant$")
        self.assertEqual(self.option(args, "--debug-dir").split("/", 1)[1],
                         self.option(args, "--output-dir").split("/", 2)[2])
        default, _ = self.launched_args("scripts/run_solver_builder_hpo_openhands.sh",
                                        SIF_PATH=self.path("sandbox.sif"))
        self.assertEqual(self.option(default, "--demos-path"),
                         "demos/DEMOS_AUTORESEARCH_HPO_seed_42_100_examples_with_CoT.json")
        self.assertTrue(self.option(default, "--output-dir").endswith("M_hpo"))
        self.assertEqual(self.option(default, "--base-url"), "http://localhost:8000/v1")

        # The entry point accepts exactly these options: run it with them (project
        # paths made absolute, outputs redirected, scripted conversation, fake sandbox).
        argv = args[2:]
        for name in ("--building-prompt", "--rewards-dir", "--smoke-tasks"):
            argv[argv.index(name) + 1] = os.path.join(PROJECT_ROOT, self.option(argv, name))
        argv[argv.index("--output-dir") + 1] = self.path("built")
        argv[argv.index("--debug-dir") + 1] = self.path("debug")
        case = BuildSolverCommandTests("test_successful_build")
        case.setUp()
        self.addCleanup(case.doCleanups)
        status, err, _, sandboxes = case.run_main(case.good_script(), argv)
        self.assertEqual(status, 0, err)
        self.assertEqual(sandboxes, [(self.path("sandbox.sif"), 120)])
        self.assertEqual(sorted(os.listdir(self.path("built"))),
                         ["SOLVER_HPO.py", "SOLVER_HPO_ALGORITHM.md"])

    def test_setup_refuses_sources_other_than_the_pinned_ones(self):
        self.stub("python", REAL_PYTHON)
        log = self.path("commands.log")
        self.stub("git", textwrap.dedent("""\
            #!/bin/sh
            echo "git $*" >> "$STUB_LOG"
            if [ "$1" = clone ]; then
                mkdir -p "$3/.git"
                echo "# altered" > "$3/train.py"
                echo "# altered" > "$3/prepare.py"
            fi
            """))
        self.stub("uv", '#!/bin/sh\necho "uv $*" >> "$STUB_LOG"\n')
        checkout = self.path("repos", "autoresearch")
        proc = self.launch("scripts/setup_autoresearch_hpo.sh", AUTORESEARCH_DIR=checkout,
                           STUB_LOG=log)
        self.assertEqual(proc.returncode, 1)
        self.assertIn(f"ERROR: {checkout}/train.py has sha256", proc.stderr)
        self.assertIn(f"expected {UPSTREAM['train_py_sha256']}", proc.stderr)
        self.assertNotIn("match the pinned sources", proc.stdout)
        with open(log, encoding="utf-8") as f:
            commands = f.read().splitlines()
        self.assertEqual(commands, [
            f"git clone https://github.com/{UPSTREAM['train_repo']}.git {checkout}",
            f"git -C {checkout} cat-file -e {UPSTREAM['train_commit']}^{{commit}}",
            f"git -C {checkout} checkout --detach {UPSTREAM['train_commit']}"])   # uv never ran


if __name__ == "__main__":
    unittest.main()
