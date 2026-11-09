"""Tests for the LLM HPO agent, the classical baselines, the shared runner, the trace
runner and the demos builder.

CPU only, no LLM server: the model is a scripted client, or a local HTTP endpoint that
speaks the OpenAI chat-completions protocol (so the real OpenAI client is exercised).

Run from the project root:
    python -m unittest discover -s tests -t .
"""
import importlib.util
import json
import os
import socket
import tempfile
import unittest

from tests.hpo_fixtures import (
    FakeLLMClient, FakeLLMServer, cot_answer, exit_status, quiet, read_jsonl, sampled_answer,
    synthetic_entry, write_tasks,
)

from rewards.autoresearch_hpo import (
    HISTORY_KEYS, HP_NAMES, STARTING_CONFIG, STATUS_INVALID, STATUS_OK, STATUS_OOM,
    BaselineFailed, EpisodeInterrupted, EpisodeLogMismatch, InvalidConfig,
    ObjectiveUnavailable, make_objective,
    normalize_config, public_task_view, run_episode,
)
from openhands_agents import build_hpo_balanced_demos as demos_builder
from openhands_agents import run_hpo_traces
from openhands_agents.hpo_agent import runner
from openhands_agents.hpo_agent.baselines import BASELINES, OptunaPolicy, make_baseline, random_search
from openhands_agents.hpo_agent.controller import (
    HISTORY_WINDOW, HPOAgentController, agent_policy_name, build_prompt, format_history,
    incumbent, parse_cot_answer, parse_paper_answer,
)

HAVE_OPTUNA = importlib.util.find_spec("optuna") is not None
HAVE_CMAES = HAVE_OPTUNA and importlib.util.find_spec("cmaes") is not None

OOM_CONFIG = dict(STARTING_CONFIG, DEPTH=24, ASPECT_RATIO=128, DEVICE_BATCH_SIZE=256)


def trial(trial_id, status=STATUS_OK, val_bpb=None, config=STARTING_CONFIG, error=None,
          wall_time_s=300.0):
    return {"trial_id": trial_id, "config": config, "status": status, "val_bpb": val_bpb,
            "error": error, "wall_time_s": wall_time_s, "peak_vram_mb": None, "metrics": {}}


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.task = public_task_view(synthetic_entry(hours=2.0))

    def test_first_decision(self):
        history = [trial(0, val_bpb=0.991)]
        prompt = build_prompt(self.task, history, "cot")
        self.assertIn("a single H200 GPU (141GB VRAM, ~76GB available for training)", prompt)
        for name in HP_NAMES:
            self.assertIn(f"  {name}: ", prompt)
        self.assertIn(f"  {json.dumps(STARTING_CONFIG)} → val_bpb=0.991", prompt)
        self.assertIn("Current best config (val_bpb=0.991):", prompt)
        self.assertIn("Trials so far: 1. Training time used: 0.08 h of 2 h "
                      "(a full trial takes 5 min).", prompt)
        self.assertIn("```json code block at the very end", prompt)

    def test_paper_style_asks_for_json_only(self):
        prompt = build_prompt(self.task, [trial(0, val_bpb=0.991)], "paper")
        self.assertTrue(prompt.endswith("Respond ONLY with a valid JSON object, no explanation."))
        self.assertNotIn("Trials so far", prompt)
        self.assertNotIn("step by step", prompt)
        # the context block is the same in both styles
        cot = build_prompt(self.task, [trial(0, val_bpb=0.991)], "cot")
        marker = "Current best config"
        self.assertEqual(prompt[:prompt.index(marker)], cot[:cot.index(marker)])

    def test_history_block(self):
        history = [trial(0, val_bpb=0.991),
                   trial(1, STATUS_OOM, config=OOM_CONFIG, error="GPU out-of-memory", wall_time_s=37.0),
                   trial(2, STATUS_INVALID, config=None, error="invalid config: DEPTH='x' is not a number",
                         wall_time_s=0.0),
                   trial(3, "error", error=None),
                   trial(4, val_bpb=0.985, config=dict(STARTING_CONFIG, DEPTH=10))]
        lines = format_history(history).splitlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[1], f"  {json.dumps(OOM_CONFIG)} → CRASHED (GPU out-of-memory)")
        self.assertEqual(lines[2], "  (no valid configuration) → CRASHED "
                                   "(invalid config: DEPTH='x' is not a number)")
        self.assertTrue(lines[3].endswith("→ FAILED"))
        self.assertEqual(incumbent(history)["trial_id"], 4)
        self.assertIn('Current best config (val_bpb=0.985):\n{\n  "DEPTH": 10,',
                      build_prompt(self.task, history))

    def test_only_the_last_evaluations_are_shown(self):
        history = [trial(i, val_bpb=1.0 - i / 1000) for i in range(HISTORY_WINDOW + 5)]
        history[2]["val_bpb"] = 0.5        # the incumbent has scrolled out of the window
        prompt = build_prompt(self.task, history)
        self.assertEqual(prompt.count("→ val_bpb="), HISTORY_WINDOW)
        self.assertNotIn("→ val_bpb=1.0\n", prompt)
        self.assertIn("Current best config (val_bpb=0.5):", prompt)

    def test_no_history(self):
        prompt = build_prompt(self.task, [])
        self.assertIn("  (no evaluations yet)", prompt)
        self.assertNotIn("Current best config", prompt)
        self.assertIsNone(incumbent([trial(0, STATUS_OOM, error="GPU out-of-memory")]))

    def test_earliest_trial_wins_ties(self):
        history = [trial(0, val_bpb=0.99), trial(1, val_bpb=0.98), trial(2, val_bpb=0.98)]
        self.assertEqual(incumbent(history)["trial_id"], 1)


class CotAnswerTests(unittest.TestCase):
    def test_plain_and_fenced_answers(self):
        self.assertEqual(parse_cot_answer('{"DEPTH": 10, "MATRIX_LR": 0.03}'),
                         {"DEPTH": 10, "MATRIX_LR": 0.03})
        self.assertEqual(parse_cot_answer(cot_answer({"DEPTH": 12})), {"DEPTH": 12})
        self.assertEqual(parse_cot_answer('Answer: {"config": {"DEPTH": 6}, "why": "smaller"}'),
                         {"DEPTH": 6})

    def test_the_last_configuration_is_the_answer(self):
        text = ('The best so far was {"DEPTH": 8, "MATRIX_LR": 0.04} and {"DEPTH": 24} ran '
                'out of memory. Sets like {x} are not JSON.\n```json\n'
                '{"DEPTH": 10, "WINDOW_PATTERN": "SSLL"}\n```\n')
        self.assertEqual(parse_cot_answer(text), {"DEPTH": 10, "WINDOW_PATTERN": "SSLL"})
        # trailing objects that are not configurations are skipped
        self.assertEqual(parse_cot_answer('{"DEPTH": 10}\n{"confidence": 0.7}'), {"DEPTH": 10})

    def test_a_quoted_configuration_never_stands_in_for_the_answer(self):
        quoted = 'The incumbent is {"DEPTH": 8, "MATRIX_LR": 0.04}. '
        cases = {
            # the final answer is not a valid configuration
            quoted + 'I propose {"DEPTH": "many"}': "DEPTH='many' is not a number",
            quoted + 'I propose {"WINDOW_PATTERN": "LLSS"}': "WINDOW_PATTERN",
            # the final answer is malformed or was cut off
            quoted + 'I propose\n```json\n{"DEPTH": 12, "MATRIX_LR": 0.0':
                "the answer ends with an incomplete JSON object",
            quoted + 'I propose {"DEPTH": 12, MATRIX_LR: 0.05}':
                "the answer ends with an incomplete JSON object",
            quoted + 'I propose {"config": {"DEPTH": 12, "MATRIX_LR"':
                "the answer ends with an incomplete JSON object",
        }
        for text, reason in cases.items():
            with self.subTest(text=text), self.assertRaisesRegex(InvalidConfig, reason):
                parse_cot_answer(text)

    def test_values_are_left_for_the_episode_loop_to_normalise(self):
        self.assertEqual(parse_cot_answer('{"DEPTH": 99, "HEAD_DIM": 100, "EXTRA": 1}'),
                         {"DEPTH": 99, "HEAD_DIM": 100, "EXTRA": 1})

    def test_unusable_answers(self):
        cases = {
            None: "no configuration in the answer",
            "": "no configuration in the answer",
            "I would increase the depth.": "no configuration in the answer",
            '{"DEPTH": 10': "incomplete JSON object in the answer",
            '{"learning_rate": 0.1}': "no configuration in the answer",
            '{"WINDOW_PATTERN": "LLSS"}': "WINDOW_PATTERN",
        }
        for text, reason in cases.items():
            with self.subTest(text=text), self.assertRaisesRegex(InvalidConfig, reason):
                parse_cot_answer(text)


class PaperAnswerTests(unittest.TestCase):
    """Upstream's rule: the first fenced block, or the whole text, is one JSON object."""

    def test_accepted_answers(self):
        config = {"DEPTH": 10, "MATRIX_LR": 0.03}
        for text in (json.dumps(config), f"  {json.dumps(config)}\n",
                     f"```json\n{json.dumps(config)}\n```",
                     f"```\n{json.dumps(config)}\n```",
                     f"Here it is:\n```json\n{json.dumps(config)}\n```\nGood luck.",
                     f"```json\n{json.dumps(config)}\n```\n```json\n{{\"DEPTH\": 20}}\n```"):
            with self.subTest(text=text):
                self.assertEqual(parse_paper_answer(text), config)
        # as upstream: values are clamped later, unknown keys dropped, and an object
        # without hyperparameters is the default configuration
        self.assertEqual(parse_paper_answer('{"DEPTH": 99, "EXTRA": 1}'), {"DEPTH": 99, "EXTRA": 1})
        self.assertEqual(parse_paper_answer('{}'), {})

    def test_rejected_answers(self):
        cases = {
            None: "not a JSON object",
            "": "not a JSON object",
            "I would increase the depth.": "not a JSON object",
            'I propose {"DEPTH": 10}': "not a JSON object",          # prose around the object
            '{"DEPTH": 10} {"DEPTH": 12}': "not a JSON object",
            '{"DEPTH": 10': "not a JSON object",
            '```json\n{"DEPTH": 10': "not a JSON object",
            "[1, 2]": "config must be a JSON object, got list",
            '{"WINDOW_PATTERN": "LLSS"}': "WINDOW_PATTERN",
            '{"DEPTH": "ten"}': "DEPTH='ten' is not a number",
        }
        for text, reason in cases.items():
            with self.subTest(text=text), self.assertRaisesRegex(InvalidConfig, reason):
                parse_paper_answer(text)


class PolicyNameTests(unittest.TestCase):
    def test_every_setting_that_changes_the_proposals_is_in_the_name(self):
        base = dict(model="openai/Qwen/Test", prompt_style="cot", temperature=0.7,
                    max_tokens=8192, parse_retries=2)
        self.assertEqual(agent_policy_name(**base),
                         "hpo_agent[Qwen/Test|cot|T=0.7|max_tokens=8192|retries=2]")
        names = {agent_policy_name(**dict(base, **change)) for change in (
            {}, {"model": "other"}, {"prompt_style": "paper"}, {"temperature": 0.0},
            {"max_tokens": 2048}, {"parse_retries": 0})}
        self.assertEqual(len(names), 6)
        agent = HPOAgentController(base_url="http://unused", client=FakeLLMClient(["x"]), **base)
        self.assertEqual(agent.name, agent_policy_name(**base))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.task = public_task_view(synthetic_entry(seed=3))
        self.history = [trial(0, val_bpb=0.991)]

    def agent(self, answers, **kwargs):
        client = FakeLLMClient(answers, **{k: kwargs.pop(k) for k in ("reasoning_key", "usage")
                                           if k in kwargs})
        kwargs.setdefault("model", "openai/Qwen/Test-Model")
        return HPOAgentController(base_url="http://unused", client=client, **kwargs), client

    def test_one_request_per_decision(self):
        agent, client = self.agent([(cot_answer({"DEPTH": 10}), "hidden chain of thought")],
                                   temperature=0.3, max_tokens=4096)
        answer = agent(self.task, self.history)
        self.assertEqual(answer["config"], {"DEPTH": 10})
        self.assertEqual(answer["source"], "llm")
        self.assertEqual(answer["reasoning"], "hidden chain of thought")
        self.assertEqual(answer["response"], cot_answer({"DEPTH": 10}))
        self.assertEqual(answer["prompt"], build_prompt(self.task, self.history, "cot"))
        self.assertEqual((answer["prompt_style"], answer["model"], answer["attempts"],
                          answer["parse_error"], answer["finish_reason"]),
                         ("cot", "Qwen/Test-Model", 1, None, "stop"))
        self.assertEqual(answer["token_usage"], {"input": 100, "output": 20, "reasoning": 5})
        json.dumps(answer)                      # the episode log stores it as JSON
        self.assertEqual(len(client.requests), 1)
        request = client.requests[0]
        self.assertEqual(request["model"], "Qwen/Test-Model")     # litellm prefix removed
        self.assertEqual(request["messages"], [{"role": "user", "content": answer["prompt"]}])
        self.assertEqual((request["temperature"], request["max_tokens"]), (0.3, 4096))

    def test_reasoning_field_names(self):
        for key in ("reasoning", "reasoning_content"):
            agent, _ = self.agent([('{"DEPTH": 10}', "thoughts")], reasoning_key=key)
            self.assertEqual(agent(self.task, self.history)["reasoning"], "thoughts")

    def test_unusable_answers_are_retried(self):
        agent, client = self.agent(["I would go deeper.", '{"WINDOW_PATTERN": "LLSS"}',
                                    cot_answer({"DEPTH": 12})])
        answer = agent(self.task, self.history)
        self.assertEqual((answer["config"], answer["source"], answer["attempts"]),
                         ({"DEPTH": 12}, "llm", 3))
        self.assertIsNone(answer["parse_error"])
        self.assertEqual(answer["token_usage"], {"input": 300, "output": 60, "reasoning": 15})
        seeds = [request["seed"] for request in client.requests]
        self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(len({request["messages"][0]["content"] for request in client.requests}), 1)

    def test_seeds_differ_between_tasks_and_trials(self):
        seeds = set()
        for seed in (0, 1, 2):
            task = public_task_view(synthetic_entry(seed=seed))
            for length in (1, 2, 3):
                agent, client = self.agent(["nothing"], parse_retries=2)
                agent(task, [trial(i, val_bpb=0.99) for i in range(length)])
                seeds.update(request["seed"] for request in client.requests)
        self.assertEqual(len(seeds), 27)

    def test_random_fallback(self):
        agent, client = self.agent(["no configuration here"], parse_retries=1)
        answer = agent(self.task, self.history)
        self.assertEqual((answer["source"], answer["attempts"]), ("random_fallback", 2))
        self.assertEqual(answer["parse_error"], "no configuration in the answer")
        self.assertEqual(normalize_config(answer["config"]), answer["config"])
        self.assertEqual(answer["token_usage"]["input"], 200)
        self.assertEqual(len(client.requests), 2)
        # reproducible, and a different draw for another trial
        again, _ = self.agent(["still nothing"], parse_retries=0)
        self.assertEqual(again(self.task, self.history)["config"], answer["config"])
        later = again(self.task, self.history + [trial(1, val_bpb=0.99)])
        self.assertNotEqual(later["config"], answer["config"])

    def test_upstream_settings_make_a_single_request(self):
        agent, client = self.agent(["nothing"], prompt_style="paper", max_tokens=2048,
                                   parse_retries=0)
        answer = agent(self.task, self.history)
        self.assertEqual((answer["source"], answer["attempts"], len(client.requests)),
                         ("random_fallback", 1, 1))
        self.assertTrue(client.requests[0]["messages"][0]["content"].endswith("no explanation."))

    def test_each_style_reads_answers_by_its_own_rule(self):
        text = 'The incumbent is {"DEPTH": 8}; next:\n```json\n{"DEPTH": 12}\n```\n{"DEPTH": 14}'
        cot, _ = self.agent([text], prompt_style="cot")
        self.assertEqual(cot(self.task, self.history)["config"], {"DEPTH": 14})
        paper, _ = self.agent([text], prompt_style="paper")
        self.assertEqual(paper(self.task, self.history)["config"], {"DEPTH": 12})
        prose = 'I propose {"DEPTH": 12}'
        cot, _ = self.agent([prose], prompt_style="cot")
        self.assertEqual(cot(self.task, self.history)["source"], "llm")
        paper, _ = self.agent([prose], prompt_style="paper", parse_retries=0)
        answer = paper(self.task, self.history)
        self.assertEqual(answer["source"], "random_fallback")
        self.assertIn("not a JSON object", answer["parse_error"])

    def test_an_answer_cut_off_by_the_token_limit_is_not_used(self):
        # the reasoning quotes the incumbent; the answer itself never arrived
        cut = 'The incumbent is {"DEPTH": 8, "MATRIX_LR": 0.04}. A deeper model should'
        agent, client = self.agent([(cut, "", "length"), (cot_answer({"DEPTH": 12}), "", "stop")])
        answer = agent(self.task, self.history)
        self.assertEqual((answer["config"], answer["source"], answer["attempts"],
                          answer["finish_reason"]), ({"DEPTH": 12}, "llm", 2, "stop"))
        agent, client = self.agent([(cut, "", "length")], parse_retries=1)
        answer = agent(self.task, self.history)
        self.assertEqual((answer["source"], answer["attempts"], answer["finish_reason"]),
                         ("random_fallback", 2, "length"))
        self.assertEqual(answer["parse_error"], "the answer was cut off by the token limit")
        # ... also when the limit was spent on the reasoning channel
        agent, _ = self.agent([("", 'so far {"DEPTH": 8} is best and', "length")], parse_retries=0)
        self.assertEqual(agent(self.task, self.history)["source"], "random_fallback")
        # upstream's rule has no such check: a complete object is taken as it is
        agent, _ = self.agent([('{"DEPTH": 12}', "", "length")], prompt_style="paper",
                              parse_retries=0)
        self.assertEqual(agent(self.task, self.history)["config"], {"DEPTH": 12})

    def test_empty_answer_falls_back_to_the_reasoning_text(self):
        agent, _ = self.agent([("", 'ran out of tokens, but I had settled on {"DEPTH": 14}'),
                               ("   \n", "")])
        answer = agent(self.task, self.history)
        self.assertEqual((answer["config"], answer["source"], answer["response"]),
                         ({"DEPTH": 14}, "llm", ""))
        # a visible answer is never overridden by the reasoning text
        agent, _ = self.agent([('{"DEPTH": 6}', 'maybe {"DEPTH": 20}?')])
        self.assertEqual(agent(self.task, self.history)["config"], {"DEPTH": 6})

    def test_missing_usage_is_counted_as_zero(self):
        agent, client = self.agent(['{"DEPTH": 6}'])
        original = client.chat.completions.create

        def without_usage(**request):
            response = original(**request)
            response.usage = None
            return response

        client.chat.completions.create = without_usage
        self.assertEqual(agent(self.task, self.history)["token_usage"],
                         {"input": 0, "output": 0, "reasoning": 0})

    def test_transport_errors_interrupt_the_episode(self):
        for error in (ConnectionError("connection refused"), TimeoutError("timed out")):
            agent, _ = self.agent([error])
            with self.subTest(error=type(error).__name__), \
                    self.assertRaisesRegex(EpisodeInterrupted, type(error).__name__):
                agent(self.task, self.history)
        # anything else is a bug and must not be swallowed
        agent, _ = self.agent([KeyError("bug")])
        with self.assertRaises(KeyError):
            agent(self.task, self.history)

    def test_unreachable_server_through_the_real_client(self):
        with socket.socket() as probe:           # a port nothing listens on
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        agent = HPOAgentController(base_url=f"http://127.0.0.1:{port}/v1", model="m",
                                   request_timeout=5.0)
        agent._client = agent._client.with_options(max_retries=0)
        with self.assertRaisesRegex(EpisodeInterrupted, "LLM request failed: APIConnectionError"):
            agent(self.task, self.history)

    def test_unknown_prompt_style(self):
        with self.assertRaises(ValueError):
            HPOAgentController(base_url="http://unused", model="m", prompt_style="terse",
                               client=FakeLLMClient(["x"]))

    def test_episode_records_the_traces(self):
        entry = synthetic_entry(hours=1.0)
        agent, client = self.agent([sampled_answer, "no config this time", sampled_answer],
                                   parse_retries=0)
        episode = run_episode(agent, entry)
        trials = episode["trials"]
        self.assertEqual(len(client.requests), len(trials) - 1)
        self.assertEqual(trials[0]["policy_meta"], {"source": "starting_config"})
        first = trials[1]
        self.assertEqual(first["policy_meta"]["source"], "llm")
        self.assertEqual(first["raw_suggestion"], parse_cot_answer(first["policy_meta"]["response"]))
        self.assertEqual(first["config"], normalize_config(first["raw_suggestion"]))
        self.assertNotIn("config", first["policy_meta"])
        self.assertEqual(trials[2]["policy_meta"]["source"], "random_fallback")
        self.assertNotEqual(trials[2]["status"], STATUS_INVALID)      # the fallback was trained
        # the prompt of a later decision shows the earlier trials, not their traces
        prompt = trials[3]["policy_meta"]["prompt"]
        self.assertIn(json.dumps(trials[2]["config"]), prompt)
        self.assertNotIn("no config this time", prompt)


class BaselineTests(unittest.TestCase):
    def test_random_search_is_stateless_and_seeded(self):
        task = public_task_view(synthetic_entry(seed=0))
        other = public_task_view(synthetic_entry(seed=1))
        history = [trial(0, val_bpb=0.99)]
        first = random_search(task, history)
        self.assertEqual(first, random_search(task, history))
        self.assertEqual(first["source"], "random")
        self.assertEqual(normalize_config(first["config"]), first["config"])
        self.assertNotEqual(first["config"], random_search(other, history)["config"])
        self.assertNotEqual(first["config"], random_search(task, history * 2)["config"])

    def test_make_baseline(self):
        entry = synthetic_entry()
        self.assertEqual(BASELINES, ("random", "tpe", "cmaes"))
        self.assertIs(make_baseline("random", entry), random_search)
        with self.assertRaisesRegex(ValueError, "unknown baseline"):
            make_baseline("smac", entry)

    @unittest.skipIf(HAVE_OPTUNA, "Optuna is installed")
    def test_missing_optuna_is_reported(self):
        with self.assertRaisesRegex(ImportError, "pip install optuna cmaes"):
            make_baseline("tpe", synthetic_entry())

    def _check_sampler(self, name):
        entry = synthetic_entry(hours=3.0)
        policy = make_baseline(name, entry)
        self.assertEqual(policy.name, name)
        episode = run_episode(policy, entry)
        trials = episode["trials"]
        self.assertGreaterEqual(len(trials), 36)
        self.assertEqual(episode["score"]["status_counts"][STATUS_INVALID], 0)
        self.assertTrue(all(t["policy_meta"]["source"] == name for t in trials[1:]))
        # Every trial, the starting configuration included, is in the study. A result
        # reaches the sampler with the next decision, so only the last one is not told.
        self.assertEqual(len(policy._study.trials), len(trials))
        self.assertEqual(policy._study.trials[0].params, STARTING_CONFIG)
        values = [t.value for t in policy._study.trials]
        self.assertEqual(values[:-1], [t["val_bpb"] if t["status"] == STATUS_OK else 100.0
                                       for t in trials[:-1]])
        self.assertIsNone(values[-1])
        # same seed, same episode
        repeat = run_episode(make_baseline(name, entry), entry)
        self.assertEqual([t["config"] for t in repeat["trials"]], [t["config"] for t in trials])
        other = run_episode(make_baseline(name, synthetic_entry(hours=3.0, seed=1)),
                            synthetic_entry(hours=3.0, seed=1))
        self.assertNotEqual([t["config"] for t in other["trials"]], [t["config"] for t in trials])
        return episode

    @unittest.skipUnless(HAVE_OPTUNA, "needs optuna")
    def test_tpe(self):
        self._check_sampler("tpe")

    @unittest.skipUnless(HAVE_CMAES, "needs optuna and cmaes")
    def test_cmaes(self):
        self._check_sampler("cmaes")

    def _check_resume(self, name):
        """A run that is stopped and resumed is the run that was never stopped."""
        entry = synthetic_entry(hours=3.0)
        strip = lambda trials: [{k: v for k, v in t.items() if k != "policy_time_s"} for t in trials]
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "episode.jsonl")
            full = run_episode(make_baseline(name, entry), entry, log_path=log, policy_name=name)
            self.assertGreaterEqual(len(full["trials"]), 36)
            with open(log, "rb") as f:
                lines = f.read().splitlines(keepends=True)
            for kept in (1, 2, 9, 20):                  # trials left in the log
                with self.subTest(kept=kept):
                    with open(log, "wb") as f:
                        f.write(b"".join(lines[:1 + kept]))
                    policy = make_baseline(name, entry)
                    resumed = run_episode(policy, entry, log_path=log, policy_name=name)
                    self.assertEqual(strip(resumed["trials"]), strip(full["trials"]))
                    self.assertEqual(len(policy._study.trials), len(resumed["trials"]))

            # a log this sampler would not have written is refused, and left as it was
            rows = [json.loads(line) for line in lines[:10]]
            rows[5]["config"] = dict(rows[5]["config"], DEPTH=rows[5]["config"]["DEPTH"] % 24 + 1)
            content = "".join(json.dumps(row) + "\n" for row in rows)
            with open(log, "w", encoding="utf-8") as f:
                f.write(content)
            with self.assertRaisesRegex(EpisodeLogMismatch, f"trial 4 .* not what the {name} sampler"):
                run_episode(make_baseline(name, entry), entry, log_path=log, policy_name=name)
            with open(log, encoding="utf-8") as f:
                self.assertEqual(f.read(), content)

    @unittest.skipUnless(HAVE_OPTUNA, "needs optuna")
    def test_tpe_resumes_exactly(self):
        self._check_resume("tpe")

    @unittest.skipUnless(HAVE_CMAES, "needs optuna and cmaes")
    def test_cmaes_resumes_exactly(self):
        self._check_resume("cmaes")

    @unittest.skipUnless(HAVE_OPTUNA, "needs optuna")
    def test_unknown_sampler(self):
        with self.assertRaises(ValueError):
            OptunaPolicy("gp", 0)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = os.path.join(self._tmp.name, "run")
        self.entries = [synthetic_entry(seed=0), synthetic_entry(seed=1)]
        self.tasks_file = write_tasks(os.path.join(self._tmp.name, "tasks.jsonl"), self.entries)

    def test_load_tasks(self):
        self.assertEqual(runner.load_tasks(self.tasks_file), self.entries)
        second = self.entries[1]["task_id"]
        self.assertEqual(runner.load_tasks(self.tasks_file, [second]), [self.entries[1]])
        with self.assertRaisesRegex(ValueError, r"task id\(s\) \['nope'\] not in"):
            runner.load_tasks(self.tasks_file, [second, "nope"])
        empty = write_tasks(os.path.join(self._tmp.name, "empty.jsonl"), [])
        with self.assertRaisesRegex(ValueError, "no tasks in"):
            runner.load_tasks(empty)
        with self.assertRaises(FileNotFoundError):
            runner.load_tasks(os.path.join(self._tmp.name, "missing.jsonl"))

    def test_run_tasks_writes_one_log_and_result_per_task(self):
        rows, out, _ = quiet(runner.run_tasks, self.entries, lambda entry: random_search,
                             self.run_dir, "random")
        self.assertEqual(sorted(os.listdir(self.run_dir)), sorted(
            f"{e['task_id']}{suffix}" for e in self.entries for suffix in (".jsonl", ".result.json")))
        self.assertEqual(out.count(runner.SYNTHETIC_BANNER), 2)
        for entry, row in zip(self.entries, rows):
            with open(runner.result_path(self.run_dir, entry["task_id"]), encoding="utf-8") as f:
                self.assertEqual(json.load(f), row)
            log = read_jsonl(row["episode_log"])
            self.assertEqual(row["n_trials"], len(log) - 1)
            self.assertEqual((row["task_id"], row["task"], row["seed"], row["policy"]),
                             (entry["task_id"], "autoresearch_hpo_synthetic", entry["seed"], "random"))
            self.assertEqual(row["objective"]["kind"], "synthetic")
            self.assertEqual((row["time_budget_s"], row["stop_reason"]), (1800.0, "time_budget"))
            self.assertEqual(row["suggestion_sources"],
                             {"starting_config": 1, "random": row["n_trials"] - 1})
            self.assertEqual(row["token_usage"], {"input": 0, "output": 0, "reasoning": 0})
            self.assertNotIn("curve", row)
            self.assertEqual(row["best_val_bpb"], min(t["val_bpb"] for t in log[1:]
                                                      if t["status"] == STATUS_OK))
        # the two seeds are different episodes on the same landscape
        self.assertEqual(rows[0]["baseline_val_bpb"], rows[1]["baseline_val_bpb"])
        first_suggestions = [read_jsonl(row["episode_log"])[2]["config"] for row in rows]
        self.assertNotEqual(*first_suggestions)

    def test_rerun_resumes_instead_of_repeating(self):
        first, _, _ = quiet(runner.run_tasks, self.entries, lambda entry: random_search,
                            self.run_dir, "random")
        os.remove(runner.result_path(self.run_dir, self.entries[0]["task_id"]))

        def no_policy(entry):
            return lambda task, history: self.fail("a finished episode asked for a suggestion")

        second, out, _ = quiet(runner.run_tasks, self.entries, no_policy, self.run_dir, "random")
        strip = lambda rows: [{k: v for k, v in r.items() if k != "policy_time_s"} for r in rows]
        self.assertEqual(strip(second), strip(first))
        self.assertTrue(os.path.exists(runner.result_path(self.run_dir, self.entries[0]["task_id"])))
        self.assertNotIn("trial ", out)             # nothing was trained again
        with self.assertRaisesRegex(ValueError, "refusing to resume"):
            quiet(runner.run_tasks, self.entries, lambda entry: random_search, self.run_dir, "tpe")

    def test_token_usage_and_sources_are_summed(self):
        client = FakeLLMClient([sampled_answer, "nothing", sampled_answer])
        agent = HPOAgentController(base_url="http://unused", model="m", client=client,
                                   parse_retries=1)
        rows, _, _ = quiet(runner.run_tasks, self.entries[:1], lambda entry: agent,
                           self.run_dir, "agent")
        row = rows[0]
        self.assertEqual(row["token_usage"], {"input": 100 * len(client.requests),
                                              "output": 20 * len(client.requests),
                                              "reasoning": 5 * len(client.requests)})
        self.assertEqual(sum(row["suggestion_sources"].values()), row["n_trials"])
        self.assertEqual(row["suggestion_sources"]["starting_config"], 1)

    def test_malformed_policy_metadata_is_not_counted(self):
        junk = iter(["n/a", None, 7, ["input"], {"input": "many", "output": -5, "reasoning": True},
                     {"input": 3, "output": 2.5}])

        def policy(task, history):
            return dict(random_search(task, history), token_usage=next(junk, {"input": 1}),
                        source=["not", "a", "name"] if len(history) % 2 else "odd")

        rows, _, _ = quiet(runner.run_tasks, self.entries[:1], lambda entry: policy,
                           self.run_dir, "junk")
        row = rows[0]
        self.assertGreater(row["n_trials"], 7)
        self.assertEqual(row["token_usage"], {"input": 3 + (row["n_trials"] - 7), "output": 0,
                                              "reasoning": 0})
        self.assertEqual(row["suggestion_sources"],
                         {"starting_config": 1, "odd": (row["n_trials"] - 1) // 2})

    def test_cli_exit_status(self):
        make = lambda entry: random_search
        status, _, err = quiet(runner.run_tasks_cli, self.entries, make, self.run_dir, "random")
        self.assertEqual((status, err), (0, ""))

        def interrupted(entry):
            def policy(task, history):
                raise EpisodeInterrupted("LLM request failed: server down")
            return policy

        status, _, err = quiet(runner.run_tasks_cli, self.entries, interrupted,
                               os.path.join(self._tmp.name, "other"), "agent")
        self.assertEqual(status, 1)
        self.assertEqual(err, "interrupted: LLM request failed: server down\n"
                              "Re-run the same command to resume.\n")

        for error in (ObjectiveUnavailable("no checkout"), BaselineFailed("default failed")):
            def failing(entry, error=error):
                raise error
            status, _, err = quiet(runner.run_tasks_cli, self.entries, failing,
                                   os.path.join(self._tmp.name, "third"), "x")
            self.assertEqual((status, err), (1, f"ERROR: {error}\n"))

    def test_real_tasks_need_the_checkout(self):
        shipped = runner.load_tasks("data/autoresearch_hpo/pilot_tasks.jsonl")
        options = {"autoresearch_dir": os.path.join(self._tmp.name, "no_checkout"),
                   "run_command": None, "verify_pin": True}
        status, _, err = quiet(runner.run_tasks_cli, shipped, lambda entry: random_search,
                               self.run_dir, "random", options)
        self.assertEqual(status, 1)
        self.assertIn("setup_autoresearch_hpo.sh", err)
        self.assertEqual(os.listdir(self.run_dir), [])       # no episode was started


def scripted_server(entry):
    """Answers of the fake LLM server: the optimum of the landscape first, then
    configurations that run out of memory or change nothing."""
    optimum = make_objective(entry)._optimum
    answers = [optimum, OOM_CONFIG, STARTING_CONFIG]
    count = {"n": 0}

    def answer(prompt, seed):
        config = answers[min(count["n"], 2) if count["n"] < 3 else 1 + count["n"] % 2]
        count["n"] += 1
        return cot_answer(config, thought=f"decision {count['n']}"), f"private reasoning {seed}"

    return answer


class TraceRunnerTests(unittest.TestCase):
    """run_hpo_traces end to end against a local OpenAI-compatible endpoint, then the
    demos builder on the logs it wrote."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.entry = synthetic_entry(hours=1.0)
        self.tasks_file = write_tasks(os.path.join(self._tmp.name, "tasks.jsonl"), [self.entry])
        self.run_dir = os.path.join(self._tmp.name, "traces")

    def argv(self, server, *extra):
        return ["--tasks-file", self.tasks_file, "--run-dir", self.run_dir,
                "--base-url", server.base_url, "--model", "openai/test-model",
                "--request-timeout", "30", *extra]

    def test_traces_then_demos(self):
        with FakeLLMServer(scripted_server(self.entry)) as server:
            status, out, err = quiet(run_hpo_traces.main, self.argv(server))
            self.assertEqual((status, err), (0, ""))
            n_requests = len(server.requests)
            # a second run finds the finished episode and asks the model nothing
            status, _, _ = quiet(run_hpo_traces.main, self.argv(server))
            self.assertEqual((status, len(server.requests)), (0, n_requests))

        self.assertIn("synthetic objective", out)
        request = server.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["body"]["model"], "test-model")
        self.assertEqual((request["body"]["temperature"], request["body"]["max_tokens"]), (0.7, 8192))
        self.assertIsInstance(request["body"]["seed"], int)

        task_id = self.entry["task_id"]
        with open(runner.result_path(self.run_dir, task_id), encoding="utf-8") as f:
            result = json.load(f)
        self.assertEqual(result["policy"], "hpo_agent[test-model|cot|T=0.7|max_tokens=8192|retries=2]")
        self.assertEqual(result["suggestion_sources"], {"starting_config": 1, "llm": n_requests})
        self.assertEqual(result["token_usage"], {"input": 50 * n_requests, "output": 30 * n_requests,
                                                 "reasoning": 10 * n_requests})
        self.assertEqual(result["best_val_bpb"], self.entry["reference"]["target_val_bpb"])
        self.assertEqual(result["value"], 1.0)
        self.assertGreater(result["status_counts"][STATUS_OOM], 0)

        log = read_jsonl(os.path.join(self.run_dir, f"{task_id}.jsonl"))
        self.assertEqual(log[0]["policy"], "hpo_agent[test-model|cot|T=0.7|max_tokens=8192|retries=2]")
        first = log[2]
        self.assertEqual(first["policy_meta"]["reasoning"],
                         f"private reasoning {server.requests[0]['body']['seed']}")
        self.assertTrue(first["policy_meta"]["response"].startswith("decision 1"))
        self.assertEqual(first["policy_meta"]["prompt"], request["body"]["messages"][0]["content"])

        # ── demos from these logs ──
        out_file = os.path.join(self._tmp.name, "demos", "DEMOS.json")
        status, out, err = quiet(demos_builder.main, ["--episodes", self.run_dir, "--n", "6",
                                                      "--seed", "42", "--out", out_file])
        self.assertEqual(status, 0)
        self.assertIn("synthetic objective", err)
        with open(out_file, encoding="utf-8") as f:
            demos = json.load(f)
        # the only improvement is the first decision, so one demo per side
        self.assertEqual([d["success"] for d in demos], [True, False])
        good, bad = demos
        self.assertEqual((good["task_id"], good["step"], good["outcome"], good["phase"]),
                         (task_id, 1, "improved", "early"))
        self.assertEqual(good["prompt"], first["policy_meta"]["prompt"])
        self.assertEqual(good["cot"], first["policy_meta"]["reasoning"] + "\n\n"
                         + first["policy_meta"]["response"])
        self.assertEqual(json.loads(good["final_response"]), first["raw_suggestion"])
        self.assertEqual((good["val_bpb"], good["incumbent_val_bpb"]),
                         (first["val_bpb"], log[1]["val_bpb"]))
        self.assertEqual((good["model"], good["objective"]), ("test-model", "synthetic"))
        self.assertIn(bad["outcome"], ("oom", "no_improvement"))
        self.assertEqual(set(good), {"task_id", "step", "prompt", "cot", "final_response", "success",
                                     "outcome", "config", "val_bpb", "incumbent_val_bpb", "error",
                                     "wall_time_s", "peak_vram_mb", "phase", "model", "objective"})

    def test_other_settings_do_not_continue_a_run(self):
        # Temperature, prompt style, token limit and retries are part of what was
        # measured: a run directory holds the episodes of one setting.
        log_path = os.path.join(self.run_dir, f"{self.entry['task_id']}.jsonl")
        with FakeLLMServer(scripted_server(self.entry)) as server:
            self.assertEqual(quiet(run_hpo_traces.main, self.argv(server))[0], 0)
            n_requests = len(server.requests)
            with open(log_path, encoding="utf-8") as f:
                before = f.read()
            for extra in (("--temperature", "0.2"), ("--prompt-style", "paper"),
                          ("--max-tokens", "2048"), ("--parse-retries", "0"),
                          ("--model", "openai/other-model")):
                with self.subTest(extra):
                    status, err = exit_status(run_hpo_traces.main, self.argv(server, *extra))
                    self.assertEqual(status, 1)
                    self.assertRegex(err, r"^ERROR: .* was written by a different run \(policy: ")
                    self.assertIn("refusing to resume", err)
                    self.assertNotIn("Traceback", err)
            self.assertEqual(len(server.requests), n_requests)
        with open(log_path, encoding="utf-8") as f:
            self.assertEqual(f.read(), before)

    def test_server_going_away_stops_the_run(self):
        with FakeLLMServer(scripted_server(self.entry)) as server:
            argv = self.argv(server)
        status, err = exit_status(run_hpo_traces.main, argv + ["--request-timeout", "5"])
        self.assertEqual(status, 1)
        self.assertIn("interrupted: LLM request failed", err)
        # the default configuration was trained and is kept for the resume
        log = read_jsonl(os.path.join(self.run_dir, f"{self.entry['task_id']}.jsonl"))
        self.assertEqual([row["type"] for row in log], ["header", "trial"])
        self.assertFalse(os.path.exists(runner.result_path(self.run_dir, self.entry["task_id"])))

    def test_bad_arguments(self):
        self.assertEqual(exit_status(run_hpo_traces.main, ["--run-dir", self.run_dir, "--tasks-file",
                                                          self.tasks_file, "--task-id", "nope"])[0], 2)
        self.assertEqual(exit_status(run_hpo_traces.main, ["--run-dir", self.run_dir, "--tasks-file",
                                                          "/nonexistent/tasks.jsonl"])[0], 2)


class ScriptedObjective:
    kind = "scripted"

    def __init__(self, values):
        self.values = list(values)

    def describe(self):
        return {"kind": self.kind}

    def evaluate(self, config):
        value = self.values.pop(0)
        ok = isinstance(value, float)
        return {"status": STATUS_OK if ok else value, "val_bpb": value if ok else None,
                "error": None if ok else f"{value} happened", "wall_time_s": 300.0,
                "peak_vram_mb": 40000.0 if ok else None, "metrics": {}, "detail": None}


class DemosBuilderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = os.path.join(self._tmp.name, "run")

    def episode(self, name, values, sources, hours=1.0, kind="scripted"):
        """Write an episode log whose trials have the given outcomes and sources.

        Every trial costs 300 s, so twelve values fill the one-hour budget exactly.
        """
        self.assertEqual((len(values), len(sources)), (hours * 12, hours * 12 - 1))
        entry = synthetic_entry(hours=hours)
        entry["task_id"] = name
        objective = ScriptedObjective(values)
        objective.kind = kind
        answers = iter(sources)

        def policy(task, history):
            source = next(answers)
            return {"config": {"DEPTH": 6 + len(history) % 10}, "source": source,
                    "prompt": f"prompt {len(history)}", "reasoning": f"reasoning {len(history)}",
                    "response": f"response {len(history)}", "model": "m"}

        path = os.path.join(self.run_dir, f"{name}.jsonl")
        run_episode(policy, entry, objective=objective, log_path=path, policy_name="agent")
        return path

    def test_outcomes_and_exclusions(self):
        path = self.episode(
            "a", [0.99, 0.98, 0.985, "oom", 0.97, "diverged", 0.97, 0.96, 0.99, 0.95, 0.95, 0.95],
            ["llm", "llm", "llm", "random_fallback", "llm", "llm", "llm", "llm", "random_fallback",
             "llm", "llm"])
        header, trials = demos_builder.load_episode(path)
        demos = demos_builder.episode_demos(header, trials)
        by_step = {d["step"]: d for d in demos}
        self.assertNotIn(0, by_step)                       # the starting configuration
        self.assertNotIn(4, by_step)                       # random fallbacks are not the model's
        self.assertNotIn(9, by_step)
        self.assertEqual({step: d["outcome"] for step, d in by_step.items()},
                         {1: "improved", 2: "no_improvement", 3: "oom", 5: "diverged",
                          6: "no_improvement", 7: "improved", 8: "no_improvement",
                          10: "no_improvement", 11: "no_improvement"})
        self.assertEqual([d["success"] for d in demos], [d["outcome"] == "improved" for d in demos])
        self.assertEqual(by_step[6]["incumbent_val_bpb"], 0.97)   # equal is not an improvement
        self.assertEqual(by_step[3]["error"], "oom happened")
        self.assertIsNone(by_step[3]["val_bpb"])
        self.assertEqual(by_step[1]["cot"], "reasoning 1\n\nresponse 1")
        self.assertEqual({d["phase"] for d in demos}, {"early", "mid", "late"})
        self.assertEqual(by_step[1]["phase"], "early")
        self.assertEqual((by_step[5]["phase"], by_step[8]["phase"]), ("mid", "late"))

    def test_selection_is_balanced_stratified_and_reproducible(self):
        candidates = []
        for index in range(60):
            outcome = ("improved", "no_improvement", "oom", "diverged")[index % 4]
            if index >= 40 and outcome == "improved":
                outcome = "no_improvement"
            candidates.append({"task_id": f"t{index % 3}", "step": index, "outcome": outcome,
                               "success": outcome == "improved",
                               "phase": ("early", "mid", "late")[index % 3]})
        demos = demos_builder.build_demos(candidates, 12, 42)
        self.assertEqual(len(demos), 12)
        self.assertEqual(sum(d["success"] for d in demos), 6)
        failures = sorted(d["outcome"] for d in demos if not d["success"])
        self.assertEqual(failures, ["diverged"] * 2 + ["no_improvement"] * 2 + ["oom"] * 2)
        for outcome in ("diverged", "no_improvement", "oom"):       # two phases per outcome
            self.assertEqual(len({d["phase"] for d in demos if d["outcome"] == outcome}), 2)
        self.assertEqual({d["phase"] for d in demos if d["success"]}, {"early", "mid", "late"})
        self.assertEqual(demos, sorted(demos, key=lambda d: (d["task_id"], d["step"])))
        self.assertEqual(len({(d["task_id"], d["step"]) for d in demos}), 12)
        self.assertEqual(demos, demos_builder.build_demos(list(reversed(candidates)), 12, 42))
        # fewer demos than strata: which outcomes are left out depends on the seed
        seen = set()
        for seed in range(20):
            seen.update(d["outcome"] for d in demos_builder.build_demos(candidates, 4, seed))
        self.assertEqual(seen, {"improved", "no_improvement", "oom", "diverged"})
        self.assertNotEqual(demos, demos_builder.build_demos(candidates, 12, 43))
        # limited by the rarer side: 10 improvements -> 10 + 10
        self.assertEqual(len(demos_builder.build_demos(candidates, 1000, 42)), 20)
        self.assertEqual(len(demos_builder.build_demos(candidates, 7, 42)), 6)
        self.assertEqual(demos_builder.build_demos([c for c in candidates if not c["success"]], 10, 42), [])

    def test_command_line(self):
        self.episode("a", [0.99, 0.98, 0.985, "oom", 0.97, 0.97, 0.96, 0.99, 0.95, 0.95, 0.95, 0.95],
                     ["llm"] * 11)
        self.episode("b", [0.99, 0.98, "oom", 0.97, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
                     ["llm"] * 11)
        with open(os.path.join(self.run_dir, "notes.jsonl"), "w") as f:
            f.write('{"type": "something else"}\n')
        out_file = os.path.join(self._tmp.name, "out", "demos.json")
        status, out, err = quiet(demos_builder.main, ["--episodes", self.run_dir, "--n", "8",
                                                      "--out", out_file])
        self.assertEqual((status, err), (0, ""))
        with open(out_file, encoding="utf-8") as f:
            demos = json.load(f)
        self.assertEqual(len(demos), 8)
        self.assertEqual(sum(d["success"] for d in demos), 4)
        self.assertEqual({d["task_id"] for d in demos}, {"a", "b"})
        self.assertIn("-> 8 demos", out)
        # a single log file works as well as a directory
        status, out, _ = quiet(demos_builder.main, ["--episodes", os.path.join(self.run_dir, "a.jsonl"),
                                                    "--n", "2", "--out", out_file])
        self.assertEqual(status, 0)
        self.assertIn("-> 2 demos", out)

    def test_default_output_name(self):
        self.episode("a", [0.99, 0.98, 0.985, 0.97, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
                     ["llm"] * 11)
        previous = os.getcwd()
        os.chdir(self._tmp.name)
        self.addCleanup(os.chdir, previous)
        status, _, _ = quiet(demos_builder.main, ["--episodes", self.run_dir, "--n", "4", "--seed", "7"])
        self.assertEqual(status, 0)
        self.assertEqual(os.listdir(os.path.join(self._tmp.name, "demos")),
                         ["DEMOS_AUTORESEARCH_HPO_seed_7_4_examples_with_CoT.json"])

    def test_errors(self):
        self.episode("a", [0.99] * 12, ["llm"] * 11)       # nothing ever improves
        self.assertEqual(exit_status(demos_builder.main, ["--episodes", self.run_dir])[0], 2)
        self.episode("b", [0.99, 0.98] + [0.99] * 10, ["llm"] * 11, kind="synthetic")
        self.assertEqual(exit_status(demos_builder.main, ["--episodes", self.run_dir])[0], 2)

    def test_demos_without_reasoning_are_reported(self):
        entry = synthetic_entry(hours=0.5)
        path = os.path.join(self.run_dir, "bare.jsonl")
        values = iter([{"DEPTH": 6}] * 10)
        run_episode(lambda task, history: {"config": next(values), "source": "llm"}, entry,
                    objective=ScriptedObjective([0.99, 0.98, 0.99, 0.99, 0.99, 0.99]),
                    log_path=path, policy_name="agent")
        status, _, err = quiet(demos_builder.main, ["--episodes", path, "--out",
                                                    os.path.join(self._tmp.name, "d.json")])
        self.assertEqual(status, 0)
        self.assertIn("WARNING: 2 demo(s) have no reasoning text", err)


if __name__ == "__main__":
    unittest.main()
