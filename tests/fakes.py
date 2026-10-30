"""Test doubles: a scripted model client and a virtual clock."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from llm_eval_kit.models import ModelRequest, ModelResponse, TransientModelError

Responder = Callable[[ModelRequest], "str | ModelResponse"]


class FakeModelClient:
    """Answers from a responder function and records every request.

    ``fail_first`` makes the first N calls raise a transient error, to exercise retries.
    """

    name = "fake"

    def __init__(
        self,
        responder: Responder,
        *,
        fail_first: int = 0,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.responder = responder
        self.fail_first = fail_first
        self.error = error
        self.delay = delay
        self.calls: list[ModelRequest] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            if self.fail_first > 0:
                self.fail_first -= 1
                raise TransientModelError("simulated 429", retry_after=0.0)
            result = self.responder(request)
        finally:
            self.in_flight -= 1
        if isinstance(result, ModelResponse):
            return result
        return ModelResponse(
            text=result,
            model=request.model,
            stop_reason="end_turn",
            input_tokens=len(request.prompt.split()),
            output_tokens=len(result.split()),
        )


def verdict(score: int, reasoning: str = "ok") -> str:
    return json.dumps({"reasoning": reasoning, "score": score})


def answers(mapping: dict[str, str], judge_score: int = 5) -> Responder:
    """Answer test prompts by substring match; judge requests get a fixed verdict."""

    def respond(request: ModelRequest) -> str:
        if request.output_schema is not None:
            return verdict(judge_score)
        for needle, answer in mapping.items():
            if needle in request.prompt:
                return answer
        return "I don't know."

    return respond


class VirtualClock:
    """A monotonic clock that only moves when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_run(rows: list[tuple[str, str, dict[str, float]]], suite: str = "s"):
    """Build a RunResult from (case_id, status, {scorer: score}) rows."""
    from llm_eval_kit.results import CaseResult, RunResult, ScoreResult, summarise

    cases = [
        CaseResult(
            id=case_id,
            status=status,  # type: ignore[arg-type]
            output=f"output of {case_id}",
            scores=[
                ScoreResult(name=n, passed=v >= 0.5, score=v, detail="" if v >= 0.5 else "low")
                for n, v in scores.items()
            ],
            error="boom" if status == "error" else None,
        )
        for case_id, status, scores in rows
    ]
    return RunResult(
        suite=suite,
        provider="fake",
        model="m",
        started_at="2026-10-30T00:00:00+00:00",
        duration_s=1.0,
        cases=cases,
        summary=summarise(cases),
    )
