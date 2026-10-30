"""Compare a run against a stored baseline run."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .results import RunResult

ChangeKind = Literal["regression", "improvement"]


class CaseChange(BaseModel):
    case_id: str
    kind: ChangeKind
    reason: str


class BaselineDiff(BaseModel):
    baseline_suite: str
    current_suite: str
    regressions: list[CaseChange] = Field(default_factory=list)
    improvements: list[CaseChange] = Field(default_factory=list)
    new_cases: list[str] = Field(default_factory=list)
    removed_cases: list[str] = Field(default_factory=list)
    unchanged: int = 0
    pass_rate_before: float = 0.0
    pass_rate_after: float = 0.0

    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)


def diff_runs(baseline: RunResult, current: RunResult, *, tolerance: float = 0.0) -> BaselineDiff:
    """Classify each case shared by both runs.

    A case regresses when it passed in the baseline and does not pass now, or
    when any scorer's score dropped by more than ``tolerance``. Improvements
    are the mirror image. A case can only be in one bucket; status changes win
    over score changes.
    """
    before, after = baseline.case_map(), current.case_map()
    diff = BaselineDiff(
        baseline_suite=baseline.suite,
        current_suite=current.suite,
        new_cases=sorted(after.keys() - before.keys()),
        removed_cases=sorted(before.keys() - after.keys()),
        pass_rate_before=baseline.summary.pass_rate,
        pass_rate_after=current.summary.pass_rate,
    )
    for case_id in sorted(before.keys() & after.keys()):
        old, new = before[case_id], after[case_id]
        if old.passed and not new.passed:
            reason = f"{old.status} -> {new.status}"
            failing = new.error or ", ".join(s.name for s in new.scores if not s.passed)
            if failing:
                reason = f"{reason} ({failing})"
            diff.regressions.append(CaseChange(case_id=case_id, kind="regression", reason=reason))
            continue
        if not old.passed and new.passed:
            diff.improvements.append(
                CaseChange(case_id=case_id, kind="improvement", reason=f"{old.status} -> passed")
            )
            continue
        drops, gains = _score_changes(old.score_by_name(), new.score_by_name(), tolerance)
        if drops:
            diff.regressions.append(CaseChange(case_id=case_id, kind="regression", reason=drops))
        elif gains:
            diff.improvements.append(CaseChange(case_id=case_id, kind="improvement", reason=gains))
        else:
            diff.unchanged += 1
    return diff


def _score_changes(old: dict, new: dict, tolerance: float) -> tuple[str, str]:
    drops, gains = [], []
    for name in sorted(old.keys() & new.keys()):
        delta = new[name].score - old[name].score
        if delta < -tolerance - 1e-9:
            drops.append(f"{name} {old[name].score:.2f} -> {new[name].score:.2f}")
        elif delta > tolerance + 1e-9:
            gains.append(f"{name} {old[name].score:.2f} -> {new[name].score:.2f}")
    return "; ".join(drops), "; ".join(gains)
