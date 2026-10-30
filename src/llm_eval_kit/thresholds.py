"""Turn a run summary into threshold checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .results import ThresholdCheck

if TYPE_CHECKING:
    from .config import Thresholds
    from .results import RunResult


def evaluate_thresholds(run: RunResult, thresholds: Thresholds) -> list[ThresholdCheck]:
    checks = [
        ThresholdCheck(
            name="pass_rate",
            actual=run.summary.pass_rate,
            required=thresholds.pass_rate,
            ok=run.summary.pass_rate >= thresholds.pass_rate,
        )
    ]
    by_name = {s.name: s for s in run.summary.scorers}
    for scorer_name, limits in thresholds.scorers.items():
        summary = by_name.get(scorer_name)
        for metric in ("pass_rate", "mean_score"):
            required = getattr(limits, metric)
            if required is None:
                continue
            # A threshold on a scorer that never ran is a failure, not a silent pass.
            actual = getattr(summary, metric) if summary else 0.0
            checks.append(
                ThresholdCheck(
                    name=f"{scorer_name}.{metric}",
                    actual=actual,
                    required=required,
                    ok=summary is not None and actual >= required,
                )
            )
    return checks
