"""Run results: what a run produced, how it is summarised and stored as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

RESULT_SCHEMA_VERSION = 1

CaseStatus = Literal["passed", "failed", "error"]


class ScoreResult(BaseModel):
    name: str
    passed: bool
    score: float = Field(ge=0, le=1)
    detail: str = ""


class CaseResult(BaseModel):
    id: str
    status: CaseStatus
    prompt: str = ""
    output: str = ""
    scores: list[ScoreResult] = Field(default_factory=list)
    error: str | None = None
    cached: bool = False
    attempts: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def score_by_name(self) -> dict[str, ScoreResult]:
        return {s.name: s for s in self.scores}


class ScorerSummary(BaseModel):
    name: str
    count: int
    passed: int
    pass_rate: float
    mean_score: float


class Summary(BaseModel):
    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    scorers: list[ScorerSummary] = Field(default_factory=list)


class ThresholdCheck(BaseModel):
    name: str
    actual: float
    required: float
    ok: bool
    comparison: Literal[">=", "<="] = ">="


class RunResult(BaseModel):
    schema_version: int = RESULT_SCHEMA_VERSION
    suite: str
    provider: str
    model: str
    started_at: str
    duration_s: float
    cases: list[CaseResult]
    summary: Summary
    thresholds: list[ThresholdCheck] = Field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def case_map(self) -> dict[str, CaseResult]:
        return {c.id: c for c in self.cases}


def summarise(cases: list[CaseResult], cache_hits: int = 0) -> Summary:
    total = len(cases)
    passed = sum(c.status == "passed" for c in cases)
    errors = sum(c.status == "error" for c in cases)
    by_scorer: dict[str, list[ScoreResult]] = {}
    for case in cases:
        for score in case.scores:
            by_scorer.setdefault(score.name, []).append(score)
    scorers = [
        ScorerSummary(
            name=name,
            count=len(results),
            passed=sum(r.passed for r in results),
            pass_rate=sum(r.passed for r in results) / len(results),
            mean_score=sum(r.score for r in results) / len(results),
        )
        for name, results in by_scorer.items()
    ]
    return Summary(
        total=total,
        passed=passed,
        failed=total - passed - errors,
        errors=errors,
        pass_rate=passed / total if total else 0.0,
        cache_hits=cache_hits,
        input_tokens=sum(c.input_tokens for c in cases),
        output_tokens=sum(c.output_tokens for c in cases),
        scorers=scorers,
    )
