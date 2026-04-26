"""Unit tests for TroVE candidate selection."""

from symbolic_agent.baselines.trove.controller import TroVEController


def _reward(output, is_success, entry):
    return {"value": 1.0 if is_success else 0.0, "message": ""}


def _controller():
    controller = object.__new__(TroVEController)
    controller.selection = "reward"
    return controller


def test_reward_tie_prefers_candidate_that_adds_reusable_functions():
    candidates = [
        {
            "solution_code": "programs = [\"replace('a','b')\"]\nprint(programs)",
            "exec_output": "[\"replace('a','b')\"]",
            "is_success": True,
            "functions": [],
        },
        {
            "solution_code": (
                "programs = infer_programs(['a'], ['b'])\n"
                "print(programs)\n"
                "def helper_for_ast_size():\n"
                "    return 1\n"
            ),
            "exec_output": "[\"replace('a','b')\"]",
            "is_success": True,
            "functions": [{"name": "infer_programs"}],
        },
    ]

    idx, score = _controller()._select_best_by_reward(candidates, _reward, {})

    assert idx == 1
    assert score == (1.0, "")


def test_reward_tie_prefers_candidate_that_called_import_tools():
    candidates = [
        {
            "solution_code": "programs = [\"replace('a','b')\"]\nprint(programs)",
            "exec_output": "[\"replace('a','b')\"]",
            "is_success": True,
            "functions": [],
            "tool_calls": [],
        },
        {
            "solution_code": "programs = infer_programs(['a'], ['b'])\nprint(programs)",
            "exec_output": "[\"replace('a','b')\"]",
            "is_success": True,
            "functions": [],
            "tool_calls": [{"name": "infer_programs"}],
        },
    ]

    idx, score = _controller()._select_best_by_reward(candidates, _reward, {})

    assert idx == 1
    assert score == (1.0, "")


def test_reward_tie_uses_smallest_ast_when_reuse_signal_matches():
    candidates = [
        {
            "solution_code": "x = 1\ny = 2\nprograms = [\"replace('a','b')\"]\nprint(programs)",
            "exec_output": "[\"replace('a','b')\"]",
            "is_success": True,
            "functions": [],
        },
        {
            "solution_code": "programs = [\"replace('a','b')\"]\nprint(programs)",
            "exec_output": "[\"replace('a','b')\"]",
            "is_success": True,
            "functions": [],
        },
    ]

    idx, score = _controller()._select_best_by_reward(candidates, _reward, {})

    assert idx == 1
    assert score == (1.0, "")
