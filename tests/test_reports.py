from __future__ import annotations

import xml.etree.ElementTree as ET

from fakes import make_run

from llm_eval_kit.baseline import diff_runs
from llm_eval_kit.config import Thresholds
from llm_eval_kit.reports import render_html, render_junit, render_markdown
from llm_eval_kit.thresholds import evaluate_thresholds

RUN = make_run(
    [
        ("ok", "passed", {"exact": 1.0}),
        ("bad", "failed", {"exact": 0.0}),
        ("err", "error", {}),
    ]
)
RUN.cases[1].output = "<script>alert('x')</script> | ```code```"
RUN.cases[0].output = "bell \x07 and null \x00 bytes"
RUN.cases[1].scores[0].detail = "expected 'a|b'"
RUN.thresholds = evaluate_thresholds(RUN, Thresholds(pass_rate=0.9))
BASELINE = make_run([("ok", "passed", {"exact": 1.0}), ("bad", "passed", {"exact": 1.0})])


def test_markdown_has_summary_thresholds_and_failures():
    md = render_markdown(RUN, diff_runs(BASELINE, RUN))
    assert "# Eval report: s" in md
    assert "| pass_rate | 0.333 | >= 0.900 | **FAILED** |" in md
    assert "## Baseline comparison" in md and "`bad`: passed -> failed" in md
    assert "## Failing outputs" in md
    # A fence longer than the backticks inside the output keeps the block intact.
    assert "````\n<script>" in md
    assert "\\|" in md  # pipes in table cells are escaped


def test_html_is_self_contained_and_escaped():
    page = render_html(RUN, diff_runs(BASELINE, RUN))
    assert page.startswith("<!doctype html>")
    assert "<script>" not in page
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in page
    assert "<link" not in page and "src=" not in page and "http" not in page
    assert "regressions vs baseline" in page
    assert "prefers-color-scheme: dark" in page


def test_junit_is_valid_xml_with_failures_and_errors():
    xml = render_junit(RUN, diff_runs(BASELINE, RUN))
    root = ET.fromstring(xml)
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.get("tests") == "4"  # 3 cases + 1 threshold
    assert suite.get("errors") == "1"
    assert suite.get("failures") == "2"  # failed case + failed threshold
    cases = {tc.get("name"): tc for tc in suite.iter("testcase")}
    assert cases["bad"].find("failure") is not None
    assert cases["err"].find("error").get("message") == "boom"
    assert cases["pass_rate"].find("failure") is not None
    assert "\x00" not in xml and "\x07" not in xml


def test_junit_flags_score_only_regressions():
    base = make_run([("drift", "passed", {"judge": 1.0})])
    current = make_run([("drift", "passed", {"judge": 0.5})])
    root = ET.fromstring(render_junit(current, diff_runs(base, current)))
    failure = root.find("testsuite/testcase/failure")
    assert failure is not None and failure.get("message") == "regression vs baseline"
    assert root.find("testsuite").get("failures") == "1"
