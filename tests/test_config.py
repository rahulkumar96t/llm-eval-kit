from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval_kit.config import ConfigError, load_cases, load_suite, render

BASE = """
name: demo
dataset: data.jsonl
prompt: "Q: {question}"
scorers:
  - type: contains
    values_field: expected
"""


def test_load_suite_and_cases(write_suite):
    path = write_suite(BASE, ['{"id": "a", "question": "1+1?", "expected": "2"}', ""])
    suite = load_suite(path)
    cases = load_cases(suite)
    assert suite.name == "demo"
    assert suite.dataset_path == path.parent / "data.jsonl"
    assert [c.id for c in cases] == ["a"]
    assert cases[0].vars == {"question": "1+1?", "expected": "2"}


def test_ids_default_to_line_numbers(write_suite):
    path = write_suite(BASE, ['{"question": "a"}', '{"question": "b"}'])
    assert [c.id for c in load_cases(load_suite(path))] == ["case-0001", "case-0002"]


def test_case_level_scorers_are_parsed(write_suite):
    row = '{"id": "a", "question": "q", "scorers": [{"type": "exact", "expected": "x"}]}'
    case = load_cases(load_suite(write_suite(BASE, [row])))[0]
    assert case.scorers[0].type == "exact"
    assert "scorers" not in case.vars


@pytest.mark.parametrize("param", ["temperature", "top_p", "top_k"])
def test_sampling_parameters_are_rejected(write_suite, param):
    path = write_suite(BASE + f"params:\n  {param}: 0.2\n", ['{"question": "q"}'])
    with pytest.raises(ConfigError, match="2026-10-21"):
        load_suite(path)


def test_unknown_suite_key_rejected(write_suite):
    path = write_suite(BASE + "modle: x\n", ['{"question": "q"}'])
    with pytest.raises(ConfigError, match="modle"):
        load_suite(path)


def test_invalid_effort(write_suite):
    path = write_suite(BASE + "params:\n  effort: extreme\n", ['{"question": "q"}'])
    with pytest.raises(ConfigError, match="effort"):
        load_suite(path)


def test_effort_none_means_omit(write_suite):
    path = write_suite(BASE + "params:\n  effort: none\n", ['{"question": "q"}'])
    assert load_suite(path).params.effort is None


def test_prompt_required_for_live_provider(write_suite):
    path = write_suite("name: x\ndataset: data.jsonl\n", ['{"q": 1}'])
    with pytest.raises(ConfigError, match="'prompt' is required"):
        load_suite(path)


def test_dataset_errors_point_at_the_line(write_suite):
    path = write_suite(BASE, ['{"id": "a"}', "{not json"])
    with pytest.raises(ConfigError, match=r"data.jsonl:2: invalid JSON"):
        load_cases(load_suite(path))


def test_duplicate_ids(write_suite):
    path = write_suite(BASE, ['{"id": "a"}', '{"id": "a"}'])
    with pytest.raises(ConfigError, match="duplicate case id 'a'"):
        load_cases(load_suite(path))


def test_missing_files(tmp_path: Path, write_suite):
    with pytest.raises(ConfigError, match="suite file not found"):
        load_suite(tmp_path / "nope.yaml")
    path = write_suite(BASE.replace("data.jsonl", "other.jsonl"), ["{}"])
    with pytest.raises(ConfigError, match="dataset not found"):
        load_cases(load_suite(path))


def test_render_fills_and_escapes():
    assert render("{a} and {{literal}}", {"a": 1}) == "1 and {literal}"


def test_render_reports_missing_fields():
    with pytest.raises(ConfigError, match=r"missing field.*b"):
        render("{a} {b}", {"a": 1})
