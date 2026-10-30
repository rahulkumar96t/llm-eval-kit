from __future__ import annotations

from fakes import make_run

from llm_eval_kit.baseline import diff_runs

BASE = make_run(
    [
        ("stable", "passed", {"exact": 1.0}),
        ("breaks", "passed", {"exact": 1.0}),
        ("fixed", "failed", {"exact": 0.0}),
        ("drifts", "passed", {"judge": 1.0}),
        ("gone", "passed", {"exact": 1.0}),
    ]
)


def test_classifies_every_case():
    current = make_run(
        [
            ("stable", "passed", {"exact": 1.0}),
            ("breaks", "failed", {"exact": 0.0}),
            ("fixed", "passed", {"exact": 1.0}),
            ("drifts", "passed", {"judge": 0.5}),
            ("added", "passed", {"exact": 1.0}),
        ]
    )
    diff = diff_runs(BASE, current)
    assert [c.case_id for c in diff.regressions] == ["breaks", "drifts"]
    assert "passed -> failed (exact)" in diff.regressions[0].reason
    assert "judge 1.00 -> 0.50" in diff.regressions[1].reason
    assert [c.case_id for c in diff.improvements] == ["fixed"]
    assert diff.new_cases == ["added"] and diff.removed_cases == ["gone"]
    assert diff.unchanged == 1
    assert diff.has_regressions


def test_tolerance_absorbs_small_score_drops():
    current = make_run([("drifts", "passed", {"judge": 0.75})])
    assert diff_runs(BASE, current, tolerance=0.25).regressions == []
    assert len(diff_runs(BASE, current, tolerance=0.1).regressions) == 1


def test_errors_count_as_regressions():
    current = make_run([("stable", "error", {})])
    diff = diff_runs(BASE, current)
    assert diff.regressions[0].reason == "passed -> error (boom)"


def test_identical_runs_have_no_changes():
    diff = diff_runs(BASE, BASE)
    assert not diff.regressions and not diff.improvements
    assert diff.unchanged == len(BASE.cases)
