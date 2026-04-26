"""Regression tests for PBEBench-shaped TroVE prompts."""

from symbolic_agent.baselines.trove.prompts import (
    build_create_prompt,
    build_import_with_tools_prompt,
    build_skip_prompt,
)


def _assert_pbebench_prompt_prints_program_sequence(prompt: str) -> None:
    assert "print(programs)" in prompt
    assert "\"replace(' ', '_')\"" in prompt
    assert "\"replace('h', 'H')\"" in prompt
    assert "print(result)" not in prompt
    assert "print(s)" not in prompt


def test_pbebench_create_prompt_models_replace_program_list_stdout():
    prompt = build_create_prompt("Task", task_family="pbebench")

    _assert_pbebench_prompt_prints_program_sequence(prompt)
    assert "must define at least one reusable helper function" in prompt
    assert "**Tools**" in prompt


def test_pbebench_skip_prompt_models_replace_program_list_stdout():
    prompt = build_skip_prompt("Task", task_family="pbebench")

    _assert_pbebench_prompt_prints_program_sequence(prompt)


def test_pbebench_import_with_tools_prompt_models_replace_program_list_stdout():
    prompt = build_import_with_tools_prompt("Task", task_family="pbebench")

    _assert_pbebench_prompt_prints_program_sequence(prompt)
