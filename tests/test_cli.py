from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fakes import FakeModelClient, answers

from llm_eval_kit.cli import (
    EXIT_CONFIG,
    EXIT_OK,
    EXIT_REGRESSION,
    EXIT_RUNTIME,
    EXIT_THRESHOLD,
    main,
)
from llm_eval_kit.results import RunResult

QA_ANSWERS = {
    "return a book?": "You have 30 days from delivery to return a book.",
    "shipping free?": "Orders over 40 EUR ship free.",
    "standard shipping": "Standard shipping takes 3-5 business days.",
    "Reply only with JSON": '{"free_shipping_threshold_eur": 40}',
    "after 90 days?": "No, returns are accepted within 30 days of delivery.",
}

SUITE = """
name: tiny
dataset: data.jsonl
prompt: "{question}"
scorers:
  - type: exact
    expected_field: answer
thresholds:
  pass_rate: {pass_rate}
"""


def factory(client):
    return lambda: client


def run_cli(args: list[str], client=None) -> int:
    return main(args, client_factory=factory(client) if client else None)


def test_recorded_example_runs_offline_and_writes_reports(examples_dir: Path, tmp_path: Path):
    out = tmp_path / "out"
    code = run_cli(
        [
            "run",
            str(examples_dir / "recorded" / "suite.yaml"),
            "--no-cache",
            "--quiet",
            "--output",
            str(out / "run.json"),
            "--report-md",
            str(out / "report.md"),
            "--report-html",
            str(out / "report.html"),
            "--junit",
            str(out / "junit.xml"),
        ]
    )
    assert code == EXIT_OK
    run = RunResult.load(out / "run.json")
    assert run.summary.total == 6 and run.summary.passed == 5
    assert [c.id for c in run.cases if not c.passed] == ["order-status-bad-enum"]
    assert "support-replies-recorded" in (out / "report.md").read_text()
    assert (out / "report.html").read_text().startswith("<!doctype html>")
    ET.parse(out / "junit.xml")


def test_live_example_with_fake_client_and_cache(examples_dir: Path, tmp_path: Path, capsys):
    suite = str(examples_dir / "qa" / "suite.yaml")
    args = ["run", suite, "--cache-dir", str(tmp_path / "cache"), "--rpm", "6000"]
    first = FakeModelClient(answers(QA_ANSWERS, judge_score=5))
    assert run_cli([*args, "--output", str(tmp_path / "a.json")], first) == EXIT_OK
    assert len(first.calls) == 10  # 5 answers + 5 judge verdicts

    second = FakeModelClient(answers(QA_ANSWERS, judge_score=5))
    assert run_cli([*args, "--output", str(tmp_path / "b.json")], second) == EXIT_OK
    assert second.calls == []  # everything came from the cache
    assert RunResult.load(tmp_path / "b.json").summary.cache_hits == 5
    assert "(cached)" in capsys.readouterr().err

    third = FakeModelClient(answers(QA_ANSWERS, judge_score=5))
    assert run_cli([*args, "--no-cache", "-q"], third) == EXIT_OK
    assert len(third.calls) == 10


def test_threshold_failure_exit_code(write_suite, tmp_path: Path):
    suite = write_suite(SUITE.replace("{pass_rate}", "1.0"), ['{"question": "q", "answer": "a"}'])
    client = FakeModelClient(lambda r: "wrong")
    assert run_cli(["run", str(suite), "--no-cache", "-q"], client) == EXIT_THRESHOLD


def _baseline_flow(write_suite, tmp_path: Path, pass_rate: str, second_answer: str) -> int:
    rows = [
        '{"id": "a", "question": "qa", "answer": "A"}',
        '{"id": "b", "question": "qb", "answer": "B"}',
    ]
    suite = str(write_suite(SUITE.replace("{pass_rate}", pass_rate), rows))
    baseline = tmp_path / "baseline.json"
    good = FakeModelClient(lambda r: {"qa": "A", "qb": "B"}[r.prompt])
    assert run_cli(["run", suite, "--no-cache", "-q", "--save-baseline", str(baseline)], good) == 0
    worse = FakeModelClient(lambda r: {"qa": "A", "qb": second_answer}[r.prompt])
    return run_cli(["run", suite, "--no-cache", "-q", "--baseline", str(baseline)], worse)


def test_regression_exit_code(write_suite, tmp_path: Path, capsys):
    assert _baseline_flow(write_suite, tmp_path, "0.5", "nope") == EXIT_REGRESSION
    assert "REGRESSION b" in capsys.readouterr().out


def test_threshold_and_regression_combine(write_suite, tmp_path: Path):
    code = _baseline_flow(write_suite, tmp_path, "1.0", "nope")
    assert code == EXIT_THRESHOLD | EXIT_REGRESSION == 3


def test_no_regression_when_unchanged(write_suite, tmp_path: Path):
    assert _baseline_flow(write_suite, tmp_path, "1.0", "B") == EXIT_OK


def test_missing_baseline(write_suite, tmp_path: Path):
    row = '{"question": "q", "answer": "a"}'
    suite = str(write_suite(SUITE.replace("{pass_rate}", "0"), [row]))
    client = FakeModelClient(lambda r: "a")
    missing = str(tmp_path / "none.json")
    assert run_cli(["run", suite, "--no-cache", "-q", "--baseline", missing], client) == EXIT_CONFIG
    args = ["run", suite, "--no-cache", "-q", "--baseline", missing, "--allow-missing-baseline"]
    assert run_cli(args, client) == EXIT_OK


def test_every_case_erroring_is_a_runtime_failure(write_suite):
    suite = write_suite(SUITE.replace("{pass_rate}", "0"), ['{"question": "q", "answer": "a"}'])
    client = FakeModelClient(lambda r: "a", fail_first=100)
    args = ["run", str(suite), "--no-cache", "-q", "--max-attempts", "1"]
    assert run_cli(args, client) == EXIT_RUNTIME


def test_config_and_usage_errors(tmp_path: Path, capsys):
    assert run_cli(["run", str(tmp_path / "missing.yaml")]) == EXIT_CONFIG
    assert "config error" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["run"])
    assert exc.value.code == EXIT_CONFIG
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == EXIT_CONFIG


def test_diff_command(tmp_path: Path, capsys):
    from fakes import make_run

    base = make_run([("a", "passed", {"exact": 1.0})])
    current = make_run([("a", "failed", {"exact": 0.0})])
    base.save(tmp_path / "base.json")
    current.save(tmp_path / "cur.json")
    args = ["diff", str(tmp_path / "base.json"), str(tmp_path / "cur.json")]
    assert main(args) == EXIT_REGRESSION
    assert main([*args, "--max-regressions", "1"]) == EXIT_OK
    (tmp_path / "junk.json").write_text(json.dumps({"nope": 1}))
    assert main(["diff", str(tmp_path / "junk.json"), str(tmp_path / "cur.json")]) == EXIT_CONFIG


def test_validate_command(examples_dir: Path, capsys):
    assert main(["validate", str(examples_dir / "qa" / "suite.yaml")]) == EXIT_OK
    assert "5 cases" in capsys.readouterr().out
    assert main(["validate", str(examples_dir / "recorded" / "suite.yaml")]) == EXIT_OK
