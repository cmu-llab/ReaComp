"""Tests for rewards/autoresearch_hpo.py: search space, objectives, episodes, scoring.

CPU only. The real objective is exercised through a stand-in checkout and launcher
(tests/hpo_fixtures.py); nothing here trains nanochat.

Run from the project root:
    python -m unittest discover -s tests -t .
"""
import ast
import copy
import json
import math
import os
import random
import tempfile
import time
import unittest
from unittest import mock

from tests.hpo_fixtures import (
    A100_80GB, H200, PROJECT_ROOT, SMALL_GPU, STUB_PREPARE, STUB_TRAIN, device_env, generator,
    make_stub_repo, nanochat_entry, read_jsonl, stub_launcher, synthetic_entry,
)

from rewards import load_reward
from rewards.autoresearch_hpo import (
    BENCHMARK, FAILURE_PENALTY, HISTORY_KEYS, HP_NAMES, MAX_SEQ_LEN, PUBLIC_TASK_KEYS,
    SEARCH_SPACE, STARTING_CONFIG, STATUS_DIVERGED, STATUS_ERROR, STATUS_INVALID, STATUS_OK,
    STATUS_OOM, STATUS_TIMEOUT, UPSTREAM, BaselineFailed, EpisodeInterrupted,
    EpisodeLogMismatch, InjectionError,
    InvalidConfig, NanochatObjective, ObjectiveUnavailable, SyntheticObjective, check_entry,
    derive_architecture, describe_search_space, inject_config, make_objective,
    normalize_config, normalized_score, parse_training_summary, public_task_view, reward,
    run_episode, sample_config, score_episode, search_space_spec,
)
from openhands_agents.hpo_agent.baselines import random_search

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "autoresearch_hpo")


def with_config(**changes):
    return dict(STARTING_CONFIG, **changes)


class SearchSpaceTests(unittest.TestCase):
    def test_paper_search_space(self):
        self.assertEqual(len(SEARCH_SPACE), 14)
        self.assertEqual(HP_NAMES, tuple(STARTING_CONFIG))
        by_name = {hp["name"]: hp for hp in SEARCH_SPACE}
        self.assertEqual((by_name["DEPTH"]["low"], by_name["DEPTH"]["high"]), (4, 24))
        self.assertEqual((by_name["TOTAL_BATCH_SIZE"]["low"], by_name["TOTAL_BATCH_SIZE"]["high"]),
                         (65536, 2097152))
        self.assertEqual(by_name["WINDOW_PATTERN"]["choices"],
                         ["SSSL", "SSLL", "SLSL", "LLLL", "SSSS", "LSSL"])
        self.assertEqual({n for n, hp in by_name.items() if hp.get("power_of_2")},
                         {"HEAD_DIM", "DEVICE_BATCH_SIZE", "TOTAL_BATCH_SIZE"})
        self.assertEqual((FAILURE_PENALTY, MAX_SEQ_LEN), (100.0, 2048))

    def test_starting_config_is_a_fixed_point(self):
        self.assertEqual(normalize_config(STARTING_CONFIG), STARTING_CONFIG)
        self.assertEqual(normalize_config({}), STARTING_CONFIG)
        for hp in search_space_spec():
            self.assertEqual(hp["default"], STARTING_CONFIG[hp["name"]])

    def test_clamping_and_casting(self):
        config = normalize_config({"DEPTH": 100, "ASPECT_RATIO": 7.9, "EMBEDDING_LR": 5,
                                   "WEIGHT_DECAY": -1.0, "MATRIX_LR": 0.04})
        self.assertEqual(config["DEPTH"], 24)
        self.assertEqual(config["ASPECT_RATIO"], 32)
        self.assertEqual(config["EMBEDDING_LR"], 2.0)
        self.assertIsInstance(config["EMBEDDING_LR"], float)
        self.assertEqual(config["WEIGHT_DECAY"], 0.0)
        self.assertEqual(normalize_config({"DEPTH": 11.9})["DEPTH"], 11)     # truncates
        self.assertIsInstance(normalize_config({"DEPTH": 11.9})["DEPTH"], int)

    def test_power_of_two_snapping(self):
        self.assertEqual(normalize_config({"HEAD_DIM": 96})["HEAD_DIM"], 64)     # tie goes down
        self.assertEqual(normalize_config({"HEAD_DIM": 97})["HEAD_DIM"], 128)
        self.assertEqual(normalize_config({"HEAD_DIM": 1})["HEAD_DIM"], 64)
        self.assertEqual(normalize_config({"DEVICE_BATCH_SIZE": 200})["DEVICE_BATCH_SIZE"], 256)
        self.assertEqual(normalize_config({"TOTAL_BATCH_SIZE": 3e5})["TOTAL_BATCH_SIZE"], 262144)

    def test_total_batch_size_is_raised_to_one_device_batch(self):
        config = normalize_config({"DEVICE_BATCH_SIZE": 256, "TOTAL_BATCH_SIZE": 65536})
        self.assertEqual(config["TOTAL_BATCH_SIZE"], 256 * MAX_SEQ_LEN)
        self.assertEqual(derive_architecture(config)["grad_accum_steps"], 1)

    def test_unknown_keys_are_dropped(self):
        config = normalize_config({"DEPTH": 6, "LEARNING_RATE": 1.0, "val_bpb": 0.1})
        self.assertEqual(config, with_config(DEPTH=6))

    def test_unreadable_values_are_rejected(self):
        for raw in (None, [], "DEPTH=8", 3,
                    {"DEPTH": "8"}, {"DEPTH": True}, {"DEPTH": None}, {"DEPTH": [8]},
                    {"MATRIX_LR": float("nan")}, {"MATRIX_LR": float("inf")},
                    {"WINDOW_PATTERN": "LLSS"}, {"WINDOW_PATTERN": 3}):
            with self.subTest(raw=raw), self.assertRaises(InvalidConfig):
                normalize_config(raw)

    def test_sampled_configs_are_valid_and_normalised(self):
        rng = random.Random(0)
        seen_patterns = set()
        for _ in range(500):
            config = sample_config(rng)
            self.assertEqual(normalize_config(config), config)
            self.assertEqual(tuple(config), HP_NAMES)
            for hp in SEARCH_SPACE:
                value = config[hp["name"]]
                if hp["type"] == "cat":
                    self.assertIn(value, hp["choices"])
                    seen_patterns.add(value)
                    continue
                self.assertIsInstance(value, int if hp["type"] == "int" else float)
                self.assertGreaterEqual(value, hp["low"])
                self.assertLessEqual(value, hp["high"])
                if hp["power_of_2"]:
                    self.assertEqual(value & (value - 1), 0)
            # train.py asserts this divisibility
            self.assertEqual(config["TOTAL_BATCH_SIZE"]
                             % (config["DEVICE_BATCH_SIZE"] * MAX_SEQ_LEN), 0)
            self.assertGreaterEqual(derive_architecture(config)["grad_accum_steps"], 1)
        self.assertEqual(len(seen_patterns), 6)

    def test_derive_architecture(self):
        self.assertEqual(derive_architecture(STARTING_CONFIG),
                         {"model_dim": 512, "num_heads": 4, "grad_accum_steps": 2})
        # 9 * 64 = 576 is rounded up to a multiple of the head dimension
        self.assertEqual(derive_architecture(with_config(DEPTH=9))["model_dim"], 640)

    def test_description_names_every_hyperparameter(self):
        text = describe_search_space()
        for name in HP_NAMES:
            self.assertIn(f"  {name}: ", text)
        self.assertIn("snapped to a power of 2", text)


class InjectionTests(unittest.TestCase):
    def test_round_trip_keeps_layout(self):
        config = normalize_config({"DEPTH": 12, "MATRIX_LR": 0.012345678901234567,
                                   "TOTAL_BATCH_SIZE": 1048576, "WINDOW_PATTERN": "LSSL",
                                   "WARMUP_RATIO": 0.1})
        injected = inject_config(STUB_TRAIN, config)
        values = {}
        for node in ast.parse(injected).body:
            if isinstance(node, ast.Assign) and node.targets[0].id in config:
                values[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(values, config)
        self.assertIn("DEPTH = 12               # number of transformer layers", injected)
        self.assertIn("TOTAL_BATCH_SIZE = 1048576 # ~524K tokens per optimizer step", injected)
        self.assertEqual(len(injected.splitlines()), len(STUB_TRAIN.splitlines()))
        self.assertEqual(inject_config(STUB_TRAIN, STARTING_CONFIG).count("\n"),
                         STUB_TRAIN.count("\n"))

    def test_multiline_and_non_ascii_sources(self):
        source = ('# größe\nDEPTH = (\n    4 +\n    4\n)  # tiefe ü\n'
                  'NAME = "ü"; MATRIX_LR = 0.04  # µ\n')
        injected = inject_config(source, {"DEPTH": 6, "MATRIX_LR": 0.5})
        # only the value expression is replaced; the parentheses around it stay
        self.assertEqual(injected, '# größe\nDEPTH = (\n    6\n)  # tiefe ü\n'
                                   'NAME = "ü"; MATRIX_LR = 0.5  # µ\n')

    def test_missing_or_ambiguous_assignments_are_errors(self):
        with self.assertRaises(InjectionError):
            inject_config("DEPTH = 8\n", {"DEPTH": 6, "MATRIX_LR": 0.1})
        with self.assertRaises(InjectionError):
            inject_config("DEPTH = 8\nDEPTH = 9\n", {"DEPTH": 6})
        with self.assertRaises(InjectionError):      # only module-level assignments count
            inject_config("def f():\n    DEPTH = 8\n", {"DEPTH": 6})

    def test_summary_parsing_uses_the_last_block(self):
        stdout = ("val_bpb: 9.0\n---\nval_bpb: 5.0\nstep 3\n---\n"
                  "val_bpb:          0.987654\ntraining_seconds: 300.1\nnote: text\n")
        self.assertEqual(parse_training_summary(stdout),
                         {"val_bpb": 0.987654, "training_seconds": 300.1})
        self.assertEqual(parse_training_summary("val_bpb: 0.9\n"), {})


class SyntheticObjectiveTests(unittest.TestCase):
    def test_deterministic_and_seed_dependent(self):
        a, b, other = (SyntheticObjective(3, 76.0), SyntheticObjective(3, 76.0),
                       SyntheticObjective(4, 76.0))
        rng = random.Random(1)
        configs = [sample_config(rng) for _ in range(50)]
        self.assertEqual([a.evaluate(c) for c in configs], [b.evaluate(c) for c in configs])
        self.assertNotEqual([a.evaluate(c) for c in configs], [other.evaluate(c) for c in configs])

    def test_starting_config_trains_and_optimum_is_the_floor(self):
        rng = random.Random(2)
        for seed in range(5):
            objective = SyntheticObjective(seed, 76.0)
            start = objective.evaluate(STARTING_CONFIG)
            self.assertEqual(start["status"], STATUS_OK)
            self.assertEqual(start["wall_time_s"], 300.0)
            self.assertLess(start["peak_vram_mb"], 76.0 * 1024)
            best = objective.evaluate(objective._optimum)
            self.assertEqual(best["val_bpb"], objective.optimum_val_bpb)
            self.assertLess(best["val_bpb"], start["val_bpb"])
            for _ in range(200):
                result = objective.evaluate(sample_config(rng))
                if result["status"] == STATUS_OK:
                    self.assertGreaterEqual(result["val_bpb"], objective.optimum_val_bpb)

    def test_failure_modes(self):
        objective = SyntheticObjective(0, 76.0)
        oom = objective.evaluate(with_config(DEPTH=24, ASPECT_RATIO=128, DEVICE_BATCH_SIZE=256))
        self.assertEqual((oom["status"], oom["val_bpb"], oom["wall_time_s"]),
                         (STATUS_OOM, None, SyntheticObjective.OOM_WALL_TIME_S))
        hot = objective.evaluate(with_config(MATRIX_LR=0.2, EMBEDDING_LR=2.0, WARMUP_RATIO=0.0))
        self.assertEqual((hot["status"], hot["val_bpb"]), (STATUS_DIVERGED, None))
        self.assertLess(hot["wall_time_s"], 300.0)

    def test_memory_limit_comes_from_the_task(self):
        big = with_config(DEPTH=16, DEVICE_BATCH_SIZE=256)
        self.assertEqual(SyntheticObjective(0, 76.0).evaluate(big)["status"], STATUS_OOM)
        self.assertEqual(SyntheticObjective(0, 1000.0).evaluate(big)["status"], STATUS_OK)
        with self.assertRaises(ValueError):      # the starting config has to fit
            SyntheticObjective(0, 8.0)


class NanochatObjectiveTests(unittest.TestCase):
    """The subprocess path of the real objective, against the stand-in checkout."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo = os.path.join(self._tmp.name, "autoresearch")
        self.real_pin = dict(UPSTREAM)
        self.hashes = make_stub_repo(self.repo)
        self.pin(self.hashes)                # the stand-in counts as the pinned sources
        self.launcher = stub_launcher(os.path.join(self._tmp.name, "bin"))

    def pin(self, values):
        patcher = mock.patch.dict(UPSTREAM, values)
        patcher.start()
        self.addCleanup(patcher.stop)

    def checkout_files(self):
        """Files in the checkout; the interpreter's bytecode cache is not ours to clean."""
        return sorted(name for name in os.listdir(self.repo) if name != "__pycache__")

    def objective(self, device=H200, mode="ok", **kwargs):
        kwargs.setdefault("cuda_mem_fraction", 0.54)
        kwargs.setdefault("expected_total_vram_gb", 141.0)
        kwargs.setdefault("training_vram_gb", 76.0)
        extra_env = device_env(device, STUB_MODE=mode, **kwargs.pop("env", {}))
        return NanochatObjective(self.repo, run_command=self.launcher, extra_env=extra_env,
                                 **kwargs)

    def test_successful_trial(self):
        objective = self.objective()
        config = with_config(DEPTH=12, MATRIX_LR=0.02, WINDOW_PATTERN="LLLL",
                             DEVICE_BATCH_SIZE=64)
        result = objective.evaluate(config)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertEqual(result["val_bpb"], 0.914)            # 0.9 + 12/1000 + 0.02/10
        self.assertEqual(result["wall_time_s"], 300.1)        # the reported training time
        self.assertEqual(result["peak_vram_mb"], 64 * 350.5)
        self.assertIsNone(result["error"])
        metrics = result["metrics"]
        self.assertEqual(metrics["depth"], 12.0)
        self.assertEqual(metrics["window_long"], 4.0)
        self.assertEqual(metrics["mfu_percent"], 0.54)        # the stub echoes CUDA_MEM_FRACTION
        self.assertFalse({"val_bpb", "peak_vram_mb", "training_seconds"} & set(metrics))
        # the per-job copy is gone and the checkout is unchanged
        self.assertEqual(self.checkout_files(), ["prepare.py", "train.py"])
        with open(os.path.join(self.repo, "train.py"), encoding="utf-8") as f:
            self.assertEqual(f.read(), STUB_TRAIN)

    def test_no_memory_cap_without_a_fraction(self):
        result = self.objective(cuda_mem_fraction=None).evaluate(STARTING_CONFIG)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertNotIn("mfu_percent", result["metrics"])    # stub printed -1: not parsed

    def test_failure_classification(self):
        expected = {
            "oom": (STATUS_OOM, "GPU out-of-memory"),
            "diverged": (STATUS_DIVERGED, "Training diverged (loss NaN or > 100)"),
            "assert": (STATUS_ERROR, "AssertionError"),
            "crash": (STATUS_ERROR, "train.py exited with code 3"),
            "silent": (STATUS_ERROR, "train.py finished without reporting val_bpb"),
        }
        for mode, (status, error) in expected.items():
            with self.subTest(mode=mode):
                result = self.objective(mode=mode).evaluate(STARTING_CONFIG)
                self.assertEqual((result["status"], result["error"]), (status, error))
                self.assertIsNone(result["val_bpb"])
                self.assertLess(result["wall_time_s"], 60.0)   # measured, not the budget
                self.assertEqual(self.checkout_files(), ["prepare.py", "train.py"])
        self.assertIn("something else", self.objective(mode="crash").evaluate(STARTING_CONFIG)["detail"])

    def test_timeout_kills_the_whole_process_group(self):
        pid_file = os.path.join(self._tmp.name, "child.pid")
        objective = self.objective(mode="hang", env={"STUB_PID_FILE": pid_file})
        objective._TRIAL_GRACE_S = objective._MIN_TRIAL_TIMEOUT_S = 0.0
        objective.trial_time_budget_s = 2.0
        result = objective.evaluate(STARTING_CONFIG)
        self.assertEqual((result["status"], result["error"]), (STATUS_TIMEOUT, "Timeout after 2s"))
        self.assertEqual(self.checkout_files(), ["prepare.py", "train.py"])
        with open(pid_file) as f:
            child = int(f.read())

        def gone():
            try:
                with open(f"/proc/{child}/stat") as f:
                    return f.read().rsplit(")", 1)[1].split()[0] == "Z"
            except FileNotFoundError:
                return True
        deadline = time.monotonic() + 10
        while not gone() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(gone(), "the process started by the launcher survived the timeout")

    def test_sources_must_match_the_pin(self):
        self.assertTrue(self.objective().describe()["pinned"])
        self.pin(self.real_pin)              # the stand-in is not the real train.py
        with self.assertRaisesRegex(ObjectiveUnavailable, "differ from the pinned upstream"):
            NanochatObjective(self.repo, run_command=self.launcher, extra_env=device_env(H200))
        unpinned = NanochatObjective(self.repo, run_command=self.launcher, verify_pin=False,
                                     extra_env=device_env(H200), expected_total_vram_gb=141.0)
        self.assertFalse(unpinned.describe()["pinned"])
        self.assertTrue(unpinned.describe()["hardware_matches"])
        self.assertEqual(unpinned.evaluate(STARTING_CONFIG)["status"], STATUS_OK)

    def test_sources_must_not_change_during_a_run(self):
        for name in ("train.py", "prepare.py"):
            for verify_pin in (True, False):
                with self.subTest(name=name, verify_pin=verify_pin):
                    make_stub_repo(self.repo)
                    objective = self.objective(verify_pin=verify_pin)
                    self.assertEqual(objective.evaluate(STARTING_CONFIG)["status"], STATUS_OK)
                    with open(os.path.join(self.repo, name), "a", encoding="utf-8") as f:
                        f.write("\nTIME_BUDGET = 3000\n")
                    with self.assertRaisesRegex(ObjectiveUnavailable,
                                                f"{name} changed while the run was in progress"):
                        objective.evaluate(STARTING_CONFIG)
                    self.assertEqual(self.checkout_files(), ["prepare.py", "train.py"])
        os.remove(os.path.join(self.repo, "prepare.py"))
        with self.assertRaisesRegex(ObjectiveUnavailable, "cannot read"):
            objective.evaluate(STARTING_CONFIG)

    def test_an_episode_stops_when_the_sources_change(self):
        entry = nanochat_entry(hours=0.5)
        objective = self.objective()

        def tampering_policy(task, history):
            if len(history) == 2:
                with open(os.path.join(self.repo, "prepare.py"), "a", encoding="utf-8") as f:
                    f.write("\nTIME_BUDGET = 3000\n")
            return random_search(task, history)

        log = os.path.join(self._tmp.name, "episode.jsonl")
        with self.assertRaisesRegex(ObjectiveUnavailable, "prepare.py changed"):
            run_episode(tampering_policy, entry, objective=objective, log_path=log)
        self.assertEqual(len(read_jsonl(log)), 3)          # header + the two honest trials

    def test_missing_checkout_and_wrong_time_budget(self):
        with self.assertRaisesRegex(ObjectiveUnavailable, "setup_autoresearch_hpo.sh"):
            NanochatObjective(os.path.join(self._tmp.name, "nowhere"), run_command=self.launcher)
        other = os.path.join(self._tmp.name, "other")
        self.pin(make_stub_repo(other, prepare_source=STUB_PREPARE.replace("300", "600")))
        with self.assertRaisesRegex(ObjectiveUnavailable, "TIME_BUDGET is 600.0"):
            NanochatObjective(other, run_command=self.launcher, extra_env=device_env(H200))

    def test_train_py_without_the_hyperparameters_is_rejected_up_front(self):
        other = os.path.join(self._tmp.name, "other")
        self.pin(make_stub_repo(other, train_source=STUB_TRAIN.replace("WARMUP_RATIO = 0.0", "pass")))
        with self.assertRaises(InjectionError):
            NanochatObjective(other, run_command=self.launcher, extra_env=device_env(H200))

    def test_the_task_gpu_runs_without_a_warning(self):
        objective = self.objective()
        described = objective.describe()
        self.assertEqual(described["device"], H200)
        self.assertIs(described["hardware_matches"], True)
        self.assertEqual(described["cuda_mem_fraction"], 0.54)     # the task's own fraction
        self.assertEqual(described["kind"], "nanochat")
        self.assertIsNone(objective.hardware_warning)

        # a task record that states another GPU is the benchmark of that GPU
        own = self.objective(device=SMALL_GPU, expected_total_vram_gb=24.0,
                             training_vram_gb=24.0, cuda_mem_fraction=None)
        self.assertIs(own.describe()["hardware_matches"], True)
        self.assertIsNone(own.hardware_warning)

        # no claim in the task record: nothing to compare, the fraction is used as given
        no_claim = self.objective(device=SMALL_GPU, expected_total_vram_gb=None)
        self.assertIsNone(no_claim.describe()["hardware_matches"])
        self.assertEqual(no_claim.describe()["cuda_mem_fraction"], 0.54)
        self.assertIsNone(no_claim.hardware_warning)

    def test_another_gpu_is_accepted_and_recorded(self):
        # more memory than the task trains with: the cap stays at the task's 76 GB
        objective = self.objective(device=A100_80GB)
        described = objective.describe()
        self.assertEqual((described["device"], described["hardware_matches"], described["pinned"]),
                         (A100_80GB, False, True))
        self.assertEqual(described["cuda_mem_fraction"], 0.96)     # 76 / 79.25
        self.assertRegex(objective.hardware_warning,
                         r"defined for a GPU with 141 GB; training runs on NVIDIA "
                         r"A100-SXM4-80GB \(79.25 GB\)\. Results are labelled 'hardware differs "
                         r"from the task': a trial is 300 s of whatever this GPU computes.*"
                         r"capped to the 76 GB the task states \(CUDA_MEM_FRACTION=0.96\)\. "
                         r"Task records that state this GPU can be generated with "
                         r"data_generation/autoresearch_hpo_generator.py "
                         r"\(README, 'Running on another GPU'\)\.$")
        result = objective.evaluate(STARTING_CONFIG)
        self.assertEqual(result["status"], STATUS_OK)
        self.assertEqual(result["metrics"]["mfu_percent"], 0.96)   # what training was given

        # less memory than the task trains with: no cap, and the warning says what follows
        objective = self.objective(device=SMALL_GPU)
        described = objective.describe()
        self.assertEqual((described["device"], described["hardware_matches"]), (SMALL_GPU, False))
        self.assertIsNone(described["cuda_mem_fraction"])
        self.assertRegex(objective.hardware_warning,
                         r"NVIDIA RTX 4090 \(23.99 GB\).*less than the 76 GB of training memory "
                         r"the task states: training uses the whole device, and policies, which "
                         r"are told 76 GB, will propose configurations that run out of memory\. "
                         r"Task records that state this GPU can be generated with "
                         r"data_generation/autoresearch_hpo_generator.py")
        result = objective.evaluate(STARTING_CONFIG)
        self.assertNotIn("mfu_percent", result["metrics"])         # CUDA_MEM_FRACTION not set

        # a device with the task's training memory and nothing to spare needs no cap
        for total in (76.0, 76.3):
            objective = self.objective(device={"gpu": "other", "total_vram_gb": total})
            self.assertIsNone(objective.describe()["cuda_mem_fraction"])
            self.assertRegex(objective.hardware_warning, "whole device, which is about the 76 GB")

        # a larger device than the task's gets the same amount of memory, not the same share
        big = self.objective(device={"gpu": "other", "total_vram_gb": 190.0})
        self.assertEqual(big.describe()["cuda_mem_fraction"], 0.4)

        # a task without a statement about training memory: the fraction is used as given
        unstated = self.objective(device=SMALL_GPU, training_vram_gb=None)
        self.assertEqual(unstated.describe()["cuda_mem_fraction"], 0.54)
        self.assertRegex(unstated.hardware_warning, r"other hardware\. Task records that state")

    def test_a_gpu_is_required(self):
        with self.assertRaisesRegex(ObjectiveUnavailable, "no CUDA device visible.*no NVIDIA driver"):
            self.objective(device=None)
        # --allow-unpinned runs on whatever is there; the result says the GPU is unknown
        objective = self.objective(device=None, verify_pin=False)
        described = objective.describe()
        self.assertEqual((described["device"], described["hardware_matches"]), (None, False))
        self.assertEqual(described["cuda_mem_fraction"], 0.54)
        self.assertRegex(objective.hardware_warning,
                         r"could not be identified \(.*no NVIDIA driver.*\); the task is defined "
                         r"for one with 141 GB\. Results are labelled 'hardware not verified'\.")

    def test_launcher_that_cannot_start(self):
        with self.assertRaisesRegex(ObjectiveUnavailable, "no CUDA device visible"):
            NanochatObjective(self.repo, run_command=["/nonexistent/launcher"])
        objective = NanochatObjective(self.repo, run_command=["/nonexistent/launcher"],
                                      verify_pin=False)
        with self.assertRaisesRegex(ObjectiveUnavailable, "cannot start"):
            objective.evaluate(STARTING_CONFIG)
        self.assertEqual(self.checkout_files(), ["prepare.py", "train.py"])

    def test_make_objective_reads_the_task_record(self):
        entry = nanochat_entry()
        os.environ["STUB_DEVICE"] = json.dumps(H200)
        self.addCleanup(os.environ.pop, "STUB_DEVICE", None)
        objective = make_objective(entry, autoresearch_dir=self.repo, run_command=self.launcher)
        self.assertEqual(objective.describe(), {
            "kind": "nanochat", "pinned": True, **self.hashes, "cuda_mem_fraction": 0.54,
            "trial_time_budget_s": 300.0, "device": H200, "hardware_matches": True})
        self.assertIsNone(objective.hardware_warning)
        # another GPU: the task's training memory decides the cap, not the task's fraction
        os.environ["STUB_DEVICE"] = json.dumps(A100_80GB)
        objective = make_objective(entry, autoresearch_dir=self.repo, run_command=self.launcher)
        self.assertEqual(objective.describe(), {
            "kind": "nanochat", "pinned": True, **self.hashes, "cuda_mem_fraction": 0.96,
            "trial_time_budget_s": 300.0, "device": A100_80GB, "hardware_matches": False})
        self.assertIn("hardware differs from the task", objective.hardware_warning)

    def test_the_pin_is_not_taken_from_the_task_record(self):
        os.environ["STUB_DEVICE"] = json.dumps(H200)
        self.addCleanup(os.environ.pop, "STUB_DEVICE", None)
        self.pin(self.real_pin)
        # the shipped task records name the real upstream files, which the stand-in is not
        shipped = read_jsonl(os.path.join(DATA_DIR, "pilot_tasks.jsonl"))[0]
        with self.assertRaisesRegex(ObjectiveUnavailable, "differ from the pinned"):
            make_objective(shipped, autoresearch_dir=self.repo, run_command=self.launcher)
        # ... and a record edited to name the stand-in's hashes is not a task of this benchmark
        for key, value in (("train_py_sha256", self.hashes["train_py_sha256"]),
                           ("prepare_py_sha256", self.hashes["prepare_py_sha256"]),
                           ("commit", "0" * 40), ("repo", "someone/else")):
            edited = copy.deepcopy(shipped)
            edited["objective"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "other training sources"):
                make_objective(edited, autoresearch_dir=self.repo, run_command=self.launcher,
                               verify_pin=False)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "other training sources"):
                check_entry(edited)

    def test_episode_on_the_stub_checkout(self):
        entry = nanochat_entry(hours=0.25)                  # 900 s: three 300.1 s trials
        objective = self.objective()
        episode = run_episode(random_search, entry, objective=objective)
        self.assertEqual(episode["stop_reason"], "time_budget")
        self.assertEqual(len(episode["trials"]), 3)
        self.assertEqual(episode["trials"][0]["config"], STARTING_CONFIG)
        self.assertAlmostEqual(episode["trials"][-1]["cumulative_time_s"], 900.3)
        self.assertEqual(episode["score"]["baseline_val_bpb"], 0.912)


class StubObjective:
    """Scripted results, for the episode-loop tests."""
    kind = "scripted"

    def __init__(self, results=None, default=None):
        self.results = list(results or [])
        self.default = default or self.ok(0.99)
        self.evaluated = []

    @staticmethod
    def ok(val_bpb, wall_time_s=300.0):
        return {"status": STATUS_OK, "val_bpb": val_bpb, "error": None, "wall_time_s": wall_time_s,
                "peak_vram_mb": 45000.0, "metrics": {"num_steps": 100.0}, "detail": None}

    @staticmethod
    def failed(status, wall_time_s=40.0):
        return {"status": status, "val_bpb": None, "error": status.upper(), "wall_time_s": wall_time_s,
                "peak_vram_mb": None, "metrics": {}, "detail": "secret stderr text"}

    def describe(self):
        return {"kind": self.kind}

    def evaluate(self, config):
        self.evaluated.append(config)
        return self.results.pop(0) if self.results else dict(self.default)


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log = os.path.join(self._tmp.name, "run", "episode.jsonl")

    def test_protocol_on_a_synthetic_task(self):
        entry = synthetic_entry(hours=2.0)
        seen = []

        def policy(task, history):
            seen.append((task, history))
            return random_search(task, history)

        episode = run_episode(policy, entry)
        trials = episode["trials"]
        self.assertEqual(episode["task_id"], entry["task_id"])
        self.assertEqual(episode["stop_reason"], "time_budget")
        self.assertEqual(trials[0]["config"], STARTING_CONFIG)
        self.assertEqual(trials[0]["policy_meta"], {"source": "starting_config"})
        self.assertEqual([t["trial_id"] for t in trials], list(range(len(trials))))
        # the budget is cumulative trial time: the last trial started below it
        self.assertGreaterEqual(trials[-1]["cumulative_time_s"], 7200.0)
        self.assertLess(trials[-2]["cumulative_time_s"], 7200.0)
        self.assertAlmostEqual(trials[-1]["cumulative_time_s"], sum(t["wall_time_s"] for t in trials))
        self.assertGreater(len(trials), 24)       # failures end early, so more than 2 h / 300 s
        best = None
        for trial in trials:
            self.assertEqual(normalize_config(trial["config"]), trial["config"])
            if trial["status"] == STATUS_OK and (best is None or trial["val_bpb"] < best):
                best = trial["val_bpb"]
            self.assertEqual(trial["best_val_bpb"], best)
        # what the policy saw
        self.assertEqual(len(seen), len(trials) - 1)
        task, history = seen[-1]
        self.assertEqual(set(task), set(PUBLIC_TASK_KEYS))
        self.assertEqual(len(history), len(trials) - 1)
        self.assertEqual(tuple(history[0]), HISTORY_KEYS)
        self.assertEqual(history[3], {key: trials[3][key] for key in HISTORY_KEYS})

    def test_policies_never_see_hidden_fields(self):
        entry = synthetic_entry()
        view = public_task_view(entry)
        self.assertNotIn("objective", view)
        self.assertNotIn("reference", view)
        self.assertNotIn("landscape_seed", json.dumps(view))
        objective = StubObjective([StubObjective.ok(0.99), StubObjective.failed(STATUS_OOM)])
        captured = {}

        def policy(task, history):
            captured["history"] = history
            return {"config": {"DEPTH": 6}}

        run_episode(policy, entry, objective=objective)
        self.assertNotIn("secret stderr text", json.dumps(captured["history"]))
        self.assertEqual(captured["history"][1]["error"], "OOM")

    def test_policy_cannot_alter_the_record(self):
        entry = synthetic_entry()
        pristine = copy.deepcopy(entry)

        def policy(task, history):
            history[0]["val_bpb"] = 0.0
            history[0]["config"]["DEPTH"] = 4
            task["budget"]["time_budget_s"] = 1e9
            task["starting_config"]["DEPTH"] = 4
            return {"config": {"DEPTH": 6}}

        episode = run_episode(policy, entry, objective=StubObjective())
        self.assertEqual(entry, pristine)
        self.assertEqual(episode["stop_reason"], "time_budget")
        self.assertEqual(len(episode["trials"]), 6)          # 1800 s / 300 s
        self.assertEqual(episode["trials"][0]["config"], STARTING_CONFIG)
        self.assertEqual(episode["score"]["best_val_bpb"], 0.99)

    def test_failed_trials_cost_their_wall_time(self):
        entry = synthetic_entry(hours=0.25)                  # 900 s
        objective = StubObjective([StubObjective.ok(0.99), StubObjective.failed(STATUS_OOM, 40.0),
                                   StubObjective.failed(STATUS_DIVERGED, 60.0),
                                   StubObjective.ok(0.98), StubObjective.ok(0.985, 200.0),
                                   StubObjective.ok(0.97)])
        episode = run_episode(lambda task, history: {"DEPTH": 6}, entry, objective=objective)
        self.assertEqual([t["cumulative_time_s"] for t in episode["trials"]],
                         [300.0, 340.0, 400.0, 700.0, 900.0])
        score = episode["score"]
        self.assertEqual(score["status_counts"],
                         {STATUS_OK: 3, STATUS_OOM: 1, STATUS_DIVERGED: 1, STATUS_ERROR: 0,
                          STATUS_TIMEOUT: 0, STATUS_INVALID: 0})
        self.assertEqual((score["n_trials"], score["failure_rate"], score["train_time_s"]),
                         (5, 0.4, 900.0))
        self.assertEqual((score["baseline_val_bpb"], score["best_val_bpb"], score["best_trial_id"]),
                         (0.99, 0.98, 3))
        self.assertAlmostEqual(score["improvement"], 0.01)
        self.assertEqual(score["curve"], [
            {"trial_id": 0, "cumulative_time_s": 300.0, "best_val_bpb": 0.99},
            {"trial_id": 3, "cumulative_time_s": 700.0, "best_val_bpb": 0.98}])
        self.assertEqual(score_episode(episode["trials"], entry), score)

    def test_max_trials_stops_a_run_of_instant_failures(self):
        entry = synthetic_entry(hours=0.25)
        entry["budget"]["max_trials"] = 4
        objective = StubObjective([StubObjective.ok(0.99)], default=StubObjective.failed(STATUS_OOM, 1.0))
        episode = run_episode(lambda task, history: {"DEPTH": 6}, entry, objective=objective)
        self.assertEqual((episode["stop_reason"], len(episode["trials"])), ("max_trials", 4))

    def test_invalid_suggestions(self):
        entry = synthetic_entry(hours=0.25)
        answers = iter([{"config": {"DEPTH": "deep"}, "note": "kept"}, "not a config",
                        RuntimeError("boom"), {"config": None, "DEPTH": 9}, {"DEPTH": 6},
                        {"DEPTH": 7}])

        def policy(task, history):
            answer = next(answers)
            if isinstance(answer, Exception):
                raise answer
            return answer

        objective = StubObjective()
        episode = run_episode(policy, entry, objective=objective)
        trials = episode["trials"]
        self.assertEqual([t["status"] for t in trials],
                         [STATUS_OK] + [STATUS_INVALID] * 4 + [STATUS_OK, STATUS_OK])
        self.assertEqual(len(objective.evaluated), 3)                # nothing invalid was trained
        self.assertEqual([t["wall_time_s"] for t in trials[1:5]], [0.0] * 4)
        self.assertEqual(trials[1]["error"], "invalid config: DEPTH='deep' is not a number")
        self.assertEqual(trials[1]["policy_meta"], {"note": "kept"})
        self.assertEqual(trials[1]["raw_suggestion"], {"DEPTH": "deep"})
        self.assertIsNone(trials[1]["config"])
        self.assertEqual(trials[2]["raw_suggestion"], "not a config")
        self.assertEqual(trials[3]["error"], "policy raised RuntimeError: boom")
        # a "config" key is the configuration, whatever else the mapping holds
        self.assertEqual(trials[4]["error"], "invalid config: config must be a JSON object, got NoneType")
        self.assertEqual(trials[4]["policy_meta"], {"DEPTH": 9})
        self.assertEqual(trials[5]["config"], dict(STARTING_CONFIG, DEPTH=6))
        self.assertEqual(episode["stop_reason"], "time_budget")

    def test_a_policy_that_only_fails_ends_the_episode(self):
        entry = synthetic_entry(hours=6.0)
        calls = []

        def policy(task, history):
            calls.append(len(history))
            if len(history) == 3:
                return {"DEPTH": 6}              # one valid answer resets the count
            raise ValueError("no idea")

        episode = run_episode(policy, entry, objective=StubObjective(), max_consecutive_invalid=3)
        self.assertEqual(episode["stop_reason"], "policy_invalid")
        self.assertEqual([t["status"] for t in episode["trials"]],
                         [STATUS_OK] + [STATUS_INVALID] * 2 + [STATUS_OK] + [STATUS_INVALID] * 3)
        self.assertEqual(calls, [1, 2, 3, 4, 5, 6])
        self.assertEqual(episode["score"]["n_trials"], 7)

    def test_baseline_failure_is_an_environment_error(self):
        entry = synthetic_entry()
        objective = StubObjective([StubObjective.failed(STATUS_OOM)])
        with self.assertRaisesRegex(BaselineFailed, "starting configuration failed \\(oom"):
            run_episode(random_search, entry, objective=objective, log_path=self.log)
        self.assertEqual(len(read_jsonl(self.log)), 1)       # the header only: no trial recorded

    def test_log_and_resume(self):
        entry = synthetic_entry(hours=1.0)
        rows = []
        full = run_episode(random_search, entry, log_path=self.log, policy_name="random",
                           on_trial=rows.append)
        logged = read_jsonl(self.log)
        self.assertEqual(logged[0]["type"], "header")
        self.assertEqual((logged[0]["benchmark"], logged[0]["task_id"], logged[0]["policy"]),
                         (BENCHMARK, entry["task_id"], "random"))
        self.assertRegex(logged[0]["task_sha256"], "^[0-9a-f]{64}$")
        self.assertEqual(logged[0]["objective"]["kind"], "synthetic")
        self.assertEqual(logged[1:], full["trials"])
        self.assertEqual(rows, full["trials"])

        # a finished episode is not continued
        again = run_episode(lambda task, history: self.fail("policy called"), entry,
                            log_path=self.log, policy_name="random")
        self.assertEqual(again["trials"], full["trials"])
        self.assertEqual(again["score"], full["score"])

        # cut the log after four trials, in the middle of the fifth row
        with open(self.log, "rb") as f:
            lines = f.read().splitlines(keepends=True)
        with open(self.log, "wb") as f:
            f.write(b"".join(lines[:5]) + lines[5][:40])
        new_rows = []
        resumed = run_episode(random_search, entry, log_path=self.log, policy_name="random",
                              on_trial=new_rows.append)
        strip = lambda trials: [{k: v for k, v in t.items() if k != "policy_time_s"} for t in trials]
        self.assertEqual(strip(resumed["trials"]), strip(full["trials"]))
        self.assertEqual(strip(read_jsonl(self.log)[1:]), strip(full["trials"]))
        self.assertEqual([r["trial_id"] for r in new_rows], list(range(4, len(full["trials"]))))

    def test_resume_refuses_another_run(self):
        entry = synthetic_entry()
        run_episode(random_search, entry, log_path=self.log, policy_name="random")
        with open(self.log, "rb") as f:
            before = f.read()
        with self.assertRaisesRegex(EpisodeLogMismatch,
                                    r'policy: log has "random", this run "tpe".*refusing to resume'):
            run_episode(random_search, entry, log_path=self.log, policy_name="tpe")
        with self.assertRaisesRegex(EpisodeLogMismatch, r"\(task_id: .*objective: .*refusing"):
            run_episode(random_search, synthetic_entry(landscape=1), log_path=self.log,
                        policy_name="random")
        with self.assertRaisesRegex(EpisodeLogMismatch, "task_id: "):
            run_episode(random_search, synthetic_entry(seed=5), log_path=self.log,
                        policy_name="random")
        # the same task id with another budget, reference or hardware is another task
        for change in ({"budget": dict(entry["budget"], time_budget_s=7200.0)},
                       {"reference": dict(entry["reference"], target_val_bpb=0.5)},
                       {"hardware": dict(entry["hardware"], gpu="another GPU")}):
            with self.subTest(change=list(change)), \
                    self.assertRaisesRegex(EpisodeLogMismatch, r"\(task_sha256: "):
                run_episode(random_search, dict(entry, **change), log_path=self.log,
                            policy_name="random")
        with open(self.log, "rb") as f:
            self.assertEqual(f.read(), before)              # a refused run writes nothing

    def test_a_log_without_a_complete_header_is_started_again(self):
        entry = synthetic_entry()
        reference = run_episode(random_search, entry, policy_name="random")
        os.makedirs(os.path.dirname(self.log))
        for content in (b"", b'{"type": "header", "benchmark": "autoresea'):
            with self.subTest(content=content):
                with open(self.log, "wb") as f:
                    f.write(content)
                episode = run_episode(random_search, entry, log_path=self.log, policy_name="random")
                self.assertEqual([t["config"] for t in episode["trials"]],
                                 [t["config"] for t in reference["trials"]])
                logged = read_jsonl(self.log)
                self.assertEqual(logged[0]["type"], "header")
                self.assertEqual(len(logged), 1 + len(reference["trials"]))

    def test_a_policy_can_refuse_a_log(self):
        def policy(task, history):
            raise EpisodeLogMismatch("not my log")

        with self.assertRaisesRegex(EpisodeLogMismatch, "not my log"):
            run_episode(policy, synthetic_entry(), log_path=self.log, policy_name="p")
        self.assertEqual(len(read_jsonl(self.log)), 2)       # header + trial 0, no invalid trial

    def test_interruption_leaves_a_resumable_log(self):
        entry = synthetic_entry()

        def flaky(task, history):
            if len(history) == 3:
                raise EpisodeInterrupted("server down")
            return random_search(task, history)

        with self.assertRaises(EpisodeInterrupted):
            run_episode(flaky, entry, log_path=self.log, policy_name="random")
        self.assertEqual(len(read_jsonl(self.log)), 4)       # header + three trials
        resumed = run_episode(random_search, entry, log_path=self.log, policy_name="random")
        reference = run_episode(random_search, entry)
        self.assertEqual([t["config"] for t in resumed["trials"]],
                         [t["config"] for t in reference["trials"]])

    def test_task_records_of_another_benchmark_version_are_rejected(self):
        entry = synthetic_entry()
        for change in ({"reward": "pbebench"},
                       {"starting_config": dict(STARTING_CONFIG, DEPTH=12)},
                       {"search_space": search_space_spec()[:-1]}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                run_episode(random_search, dict(entry, **change))
        with self.assertRaisesRegex(ValueError, "unknown objective kind"):
            make_objective(dict(entry, objective={"kind": "surrogate"}))


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.entry = synthetic_entry()
        self.objective = make_objective(self.entry)

    def test_registry_contract(self):
        self.assertIs(load_reward(BENCHMARK), reward)
        self.assertIs(load_reward("autoresearch-hpo"), reward)
        result = reward({"DEPTH": 6}, True, self.entry)
        self.assertEqual(set(result), {"value", "message"})
        self.assertIsInstance(result["value"], float)

    def test_reference_points(self):
        self.assertEqual(reward(STARTING_CONFIG, True, self.entry)["value"], 0.0)
        best = reward(self.objective._optimum, True, self.entry)
        self.assertEqual(best["value"], 1.0)
        middle = dict(STARTING_CONFIG)
        middle["WINDOW_PATTERN"] = self.objective._optimum["WINDOW_PATTERN"]
        value = reward(middle, True, self.entry)["value"]
        self.assertTrue(0.0 <= value < 1.0)
        self.assertEqual(normalized_score(None, self.entry), 0.0)
        self.assertEqual(normalized_score(0.5, self.entry), 1.0)       # clipped
        self.assertEqual(normalized_score(2.0, self.entry), 0.0)

    def test_self_reported_results_are_ignored(self):
        config = dict(STARTING_CONFIG, WARMUP_RATIO=0.05)
        honest = reward({"config": config}, True, self.entry)
        boasting = reward({"config": config, "val_bpb": 0.5, "success": True, "value": 1.0,
                           "status": "ok", "best_val_bpb": 0.1}, True, self.entry)
        self.assertEqual(boasting, honest)
        self.assertEqual(reward(json.dumps({"config": config, "val_bpb": 0.5}), True, self.entry), honest)
        self.assertEqual(reward(json.dumps(config), True, self.entry), honest)
        # a failing configuration scores zero whatever it claims
        oom = dict(STARTING_CONFIG, DEPTH=24, ASPECT_RATIO=128, DEVICE_BATCH_SIZE=256)
        result = reward({"config": oom, "val_bpb": 0.5, "status": "ok"}, True, self.entry)
        self.assertEqual((result["value"], result["message"]), (0.0, "oom: GPU out-of-memory"))

    def test_unusable_results_score_zero(self):
        self.assertEqual(reward(STARTING_CONFIG, False, self.entry),
                         {"value": 0.0, "message": "execution failed"})
        for result in (None, "not json", "[1, 2]", 7, {"config": {"DEPTH": "x"}},
                       {"WINDOW_PATTERN": "nope"}):
            with self.subTest(result=result):
                scored = reward(result, True, self.entry)
                self.assertEqual(scored["value"], 0.0)
                self.assertTrue(scored["message"].startswith("invalid config"))

    def test_the_objective_can_be_reused(self):
        objective = StubObjective(default=StubObjective.ok(self.entry["reference"]["target_val_bpb"]))
        self.assertEqual(reward({"DEPTH": 6}, True, self.entry, objective=objective)["value"], 1.0)
        self.assertEqual(objective.evaluated, [dict(STARTING_CONFIG, DEPTH=6)])


class TaskFileTests(unittest.TestCase):
    """The shipped task files are what the generator writes."""

    def test_files_match_the_generator(self):
        for name, preset in generator.PRESETS.items():
            with self.subTest(preset=name):
                shipped = read_jsonl(os.path.join(DATA_DIR, preset["file"]))
                self.assertEqual(shipped, json.loads(json.dumps(generator.build_preset(name))))
                for entry in shipped:
                    check_entry(entry)

    def test_paper_protocol(self):
        full = read_jsonl(os.path.join(DATA_DIR, "tasks_full.jsonl"))
        self.assertEqual([e["task_id"] for e in full],
                         [f"autoresearch_hpo_24h_seed{s}" for s in (0, 1, 2)])
        for entry in full:
            self.assertEqual(entry["budget"], {"time_budget_s": 86400.0, "trial_time_budget_s": 300.0,
                                               "max_trials": 2880})
            self.assertEqual(entry["hardware"], {"gpu": "H200", "total_vram_gb": 141.0,
                                                 "training_vram_gb": 76.0, "cuda_mem_fraction": 0.54})
            self.assertEqual(entry["objective"]["kind"], "nanochat")
            self.assertEqual(entry["objective"]["commit"], UPSTREAM["train_commit"])
            self.assertEqual(entry["objective"]["train_py_sha256"], UPSTREAM["train_py_sha256"])
            self.assertEqual(entry["starting_config"], STARTING_CONFIG)
            self.assertEqual((entry["reference"]["start_val_bpb"], entry["reference"]["target_val_bpb"]),
                             (0.991, 0.9739))
            self.assertIn("reaches 24 h.", entry["prompt"])
            self.assertNotIn("0.9739", json.dumps(public_task_view(entry)))
        pilot = read_jsonl(os.path.join(DATA_DIR, "pilot_tasks.jsonl"))
        self.assertEqual([(e["task_id"], e["budget"]["time_budget_s"]) for e in pilot],
                         [("autoresearch_hpo_1h_seed0", 3600.0)])

    def test_synthetic_tasks_are_marked_and_solvable(self):
        synthetic = read_jsonl(os.path.join(DATA_DIR, "synthetic_tasks.jsonl"))
        self.assertEqual(len(synthetic), 5)
        for entry in synthetic:
            self.assertEqual(entry["task"], "autoresearch_hpo_synthetic")
            self.assertIn("synthetic", entry["task_id"])
            objective = make_objective(entry)
            self.assertIsInstance(objective, SyntheticObjective)
            self.assertEqual(objective.evaluate(STARTING_CONFIG)["val_bpb"],
                             entry["reference"]["start_val_bpb"])
            self.assertEqual(objective.optimum_val_bpb, entry["reference"]["target_val_bpb"])
            self.assertLess(entry["reference"]["target_val_bpb"], entry["reference"]["start_val_bpb"])

    def test_other_hardware_gets_its_own_task_ids(self):
        entry = generator.build_task("nanochat", 0, 24.0, gpu_name="A6000", total_vram_gb=48.0,
                                     training_vram_gb=48.0)
        self.assertEqual(entry["task_id"], "autoresearch_hpo_24h_seed0_a6000_48gb")
        self.assertIsNone(entry["hardware"]["cuda_mem_fraction"])
        self.assertIn("not comparable to the paper", entry["reference"]["source"])
        self.assertIn("A6000", entry["prompt"])
        with self.assertRaises(ValueError):
            generator.build_task("nanochat", 0, 24.0, total_vram_gb=24.0, training_vram_gb=48.0)

    def test_random_search_improves_on_the_default(self):
        # not a claim about quality, only that the loop and the landscape work together
        for entry in read_jsonl(os.path.join(DATA_DIR, "synthetic_tasks.jsonl")):
            score = run_episode(random_search, entry)["score"]
            self.assertLess(score["best_val_bpb"], score["baseline_val_bpb"])
            self.assertTrue(0.0 < score["value"] <= 1.0)
            self.assertFalse(math.isnan(score["failure_rate"]))


if __name__ == "__main__":
    unittest.main()
