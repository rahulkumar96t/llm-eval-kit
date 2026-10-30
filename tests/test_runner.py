from __future__ import annotations

import asyncio

import pytest
from fakes import FakeModelClient, answers

from llm_eval_kit.config import ConfigError, load_cases, load_suite
from llm_eval_kit.models import ModelResponse
from llm_eval_kit.ratelimit import RetryPolicy
from llm_eval_kit.runner import ModelGateway, Runner

SUITE = """
name: arithmetic
dataset: data.jsonl
system: "Answer with a number. Topic: {topic}"
prompt: "What is {question}?"
params:
  max_tokens: 64
  effort: low
scorers:
  - type: numeric
    expected_field: answer
"""
ROWS = [
    '{"id": "add", "topic": "sums", "question": "2+2", "answer": 4}',
    '{"id": "mul", "topic": "products", "question": "6*7", "answer": 42}',
    '{"id": "sub", "topic": "differences", "question": "9-3", "answer": 6}',
]


def run_suite(path, client, **kwargs):
    suite = load_suite(path)
    gateway = ModelGateway(client, retry=RetryPolicy(sleep=_no_sleep))
    runner = Runner(suite, model_gateway=gateway, judge_gateway=gateway, **kwargs)
    return asyncio.run(runner.run(load_cases(suite)))


async def _no_sleep(_: float) -> None:
    return None


def test_runs_cases_in_dataset_order_and_scores(write_suite):
    client = FakeModelClient(answers({"2+2": "4", "6*7": "It is 42.", "9-3": "7"}))
    run = run_suite(write_suite(SUITE, ROWS), client)
    assert [c.id for c in run.cases] == ["add", "mul", "sub"]
    assert [c.status for c in run.cases] == ["passed", "passed", "failed"]
    assert run.summary.pass_rate == pytest.approx(2 / 3)
    assert run.model == "claude-sonnet-5-5"
    request = client.calls[0]
    assert request.system == "Answer with a number. Topic: sums"
    assert request.effort == "low" and request.max_tokens == 64
    assert not run.thresholds[0].ok  # default pass_rate threshold is 1.0


def test_concurrency_is_bounded(write_suite):
    rows = [f'{{"id": "c{i}", "topic": "t", "question": "1+1", "answer": 2}}' for i in range(12)]
    client = FakeModelClient(answers({"1+1": "2"}), delay=0.01)
    run = run_suite(write_suite(SUITE, rows), client, concurrency=3)
    assert run.summary.passed == 12
    assert client.max_in_flight == 3


def test_model_override(write_suite):
    client = FakeModelClient(answers({}))
    run = run_suite(write_suite(SUITE, ROWS[:1]), client, model_override="claude-opus-5-5")
    assert client.calls[0].model == "claude-opus-5-5" and run.model == "claude-opus-5-5"


def test_refusal_becomes_an_error_case(write_suite):
    refusal = ModelResponse(text="", model="m", stop_reason="refusal", refusal_category="bio")
    run = run_suite(write_suite(SUITE, ROWS[:1]), FakeModelClient(lambda r: refusal))
    case = run.cases[0]
    assert case.status == "error" and "refused (category: bio)" in case.error
    assert case.scores == []


def test_transient_failures_are_retried_and_counted(write_suite):
    client = FakeModelClient(answers({"2+2": "4"}), fail_first=2)
    run = run_suite(write_suite(SUITE, ROWS[:1]), client)
    assert run.cases[0].status == "passed" and run.cases[0].attempts == 3


def test_exhausted_retries_mark_only_that_case(write_suite):
    client = FakeModelClient(answers({}), fail_first=100)
    run = run_suite(write_suite(SUITE, ROWS[:1]), client)
    assert run.cases[0].status == "error" and "TransientModelError" in run.cases[0].error
    assert run.summary.errors == 1


def test_judge_and_case_scorers_combine(write_suite):
    suite = SUITE + "  - type: llm_judge\n    name: judge\n    rubric: Correct arithmetic.\n"
    row = (
        '{"id": "add", "topic": "t", "question": "2+2", "answer": 4,'
        ' "scorers": [{"type": "contains", "values": ["4"]}]}'
    )
    client = FakeModelClient(answers({"2+2": "4"}, judge_score=3))
    run = run_suite(write_suite(suite, [row]), client)
    names = [s.name for s in run.cases[0].scores]
    assert names == ["numeric", "judge", "contains"]
    assert run.cases[0].status == "failed"  # judge gave 3 < pass_score 4
    assert len(client.calls) == 2


def test_duplicate_scorer_names_are_a_config_error(write_suite):
    row = (
        '{"id": "a", "topic": "t", "question": "q", "answer": 1,'
        ' "scorers": [{"type": "numeric", "expected": 1}]}'
    )
    with pytest.raises(ConfigError, match="unique"):
        run_suite(write_suite(SUITE, [row]), FakeModelClient(answers({})))


def test_missing_template_field_fails_before_any_call(write_suite):
    client = FakeModelClient(answers({}))
    with pytest.raises(ConfigError, match="missing field"):
        run_suite(write_suite(SUITE, ['{"id": "a", "question": "1", "answer": 1}']), client)
    assert client.calls == []
