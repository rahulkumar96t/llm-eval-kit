"""Runs a suite: render prompts, call the model (cached, rate limited, retried), score."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from .cache import ResponseCache, cache_key
from .config import Case, ConfigError, Suite, load_cases, render
from .models import ModelClient, ModelRequest, ModelResponse, default_model
from .ratelimit import RetryPolicy, TokenBucket
from .results import CaseResult, RunResult, ScoreResult, summarise
from .scorers import ScoreContext, Scorer, build_scorer
from .thresholds import evaluate_thresholds


@dataclass
class _CallStats:
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class ModelGateway:
    """Cache -> rate limiter -> retries -> client, shared by the model under test and judges."""

    def __init__(
        self,
        client: ModelClient,
        *,
        cache: ResponseCache | None = None,
        limiter: TokenBucket | None = None,
        retry: RetryPolicy | None = None,
        cache_extra: dict[str, object] | None = None,
    ) -> None:
        self.client = client
        self.cache = cache
        self.limiter = limiter
        self.retry = retry or RetryPolicy()
        self.cache_extra = cache_extra or {}

    async def call(self, request: ModelRequest, stats: _CallStats | None = None) -> ModelResponse:
        key = cache_key(self.client.name, request, self.cache_extra)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        async def attempt() -> ModelResponse:
            if self.limiter is not None:
                await self.limiter.acquire()
            return await self.client.complete(request)

        response, attempts = await self.retry.run(attempt)
        if stats is not None:
            stats.attempts += attempts
            stats.input_tokens += response.input_tokens
            stats.output_tokens += response.output_tokens
        # Refusals are not cached: a later run (or a fallback) may answer.
        if self.cache is not None and not response.refused:
            self.cache.put(key, request, response)
        return response


class Runner:
    def __init__(
        self,
        suite: Suite,
        *,
        model_gateway: ModelGateway | None,
        judge_gateway: ModelGateway | None = None,
        concurrency: int = 4,
        model_override: str | None = None,
        on_case_done: Callable[[CaseResult], None] | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.suite = suite
        self.model_gateway = model_gateway
        self.judge_gateway = judge_gateway
        self.concurrency = concurrency
        self.model = model_override or suite.model or default_model()
        self.on_case_done = on_case_done
        self.suite_scorers = [build_scorer(spec, suite) for spec in suite.scorers]

    def request_for(self, case: Case) -> ModelRequest:
        params = self.suite.params
        system = self.suite.system
        return ModelRequest(
            model=self.model,
            prompt=render(self.suite.prompt, case.vars, where=f"prompt for case {case.id!r}"),
            system=render(system, case.vars, where="system prompt") if system else None,
            max_tokens=params.max_tokens,
            effort=params.effort,
            stop_sequences=tuple(params.stop_sequences),
        )

    def plan(self, cases: list[Case]) -> list[tuple[Case, list[Scorer], str]]:
        """Build every scorer and render every prompt without calling a model.

        Running this before any request makes config mistakes fail fast.
        """
        return [(case, self._scorers_for(case), self._prompt_for(case)) for case in cases]

    async def run(self, cases: list[Case] | None = None) -> RunResult:
        cases = cases if cases is not None else load_cases(self.suite)
        if self.suite.provider == "anthropic" and self.model_gateway is None:
            raise ConfigError("provider 'anthropic' needs a model client")
        plans = self.plan(cases)
        started = datetime.now(UTC)
        clock = time.perf_counter()
        semaphore = asyncio.Semaphore(self.concurrency)

        async def guarded(case: Case, scorers: list[Scorer], prompt: str) -> CaseResult:
            async with semaphore:
                result = await self._run_case(case, scorers, prompt)
            if self.on_case_done:
                self.on_case_done(result)
            return result

        results = await asyncio.gather(*(guarded(*plan) for plan in plans))
        cache_hits = sum(r.cached for r in results)
        run = RunResult(
            suite=self.suite.name,
            provider=self.suite.provider,
            model=self.model if self.suite.provider == "anthropic" else "recorded",
            started_at=started.isoformat(timespec="seconds"),
            duration_s=round(time.perf_counter() - clock, 3),
            cases=list(results),
            summary=summarise(list(results), cache_hits=cache_hits),
        )
        run.thresholds = evaluate_thresholds(run, self.suite.thresholds)
        return run

    def _scorers_for(self, case: Case) -> list[Scorer]:
        scorers = self.suite_scorers + [build_scorer(spec, self.suite) for spec in case.scorers]
        if not scorers:
            raise ConfigError(f"case {case.id!r} has no scorers")
        names = [s.name for s in scorers]
        if len(set(names)) != len(names):
            raise ConfigError(f"case {case.id!r}: scorer names must be unique, got {names}")
        return scorers

    def _prompt_for(self, case: Case) -> str:
        if self.suite.provider == "recorded":
            if self.suite.output_field not in case.vars:
                raise ConfigError(f"case {case.id!r} has no {self.suite.output_field!r} field")
            return render(self.suite.prompt, case.vars) if self.suite.prompt else ""
        return self.request_for(case).prompt

    async def _run_case(self, case: Case, scorers: list[Scorer], prompt: str) -> CaseResult:
        stats = _CallStats()
        started = time.perf_counter()
        cached = False
        try:
            if self.suite.provider == "recorded":
                output = str(case.vars[self.suite.output_field])
            else:
                assert self.model_gateway is not None
                response = await self.model_gateway.call(self.request_for(case), stats)
                cached = response.cached
                if response.refused:
                    error = f"model refused (category: {response.refusal_category})"
                    return self._finish(case, prompt, "", [], stats, started, cached, error)
                output = response.text
        except Exception as exc:
            return self._finish(case, prompt, "", [], stats, started, cached, error=_describe(exc))

        call_judge = None
        if self.judge_gateway is not None:
            gateway = self.judge_gateway

            async def call_judge(request: ModelRequest) -> ModelResponse:
                return await gateway.call(request, stats)

        scores: list[ScoreResult] = []
        ctx = ScoreContext(case=case, prompt=prompt, output=output, call_judge=call_judge)
        for scorer in scorers:
            try:
                scores.append(await scorer.score(ctx))
            except Exception as exc:
                error = f"scorer {scorer.name!r} failed: {_describe(exc)}"
                return self._finish(case, prompt, output, scores, stats, started, cached, error)
        return self._finish(case, prompt, output, scores, stats, started, cached)

    def _finish(
        self,
        case: Case,
        prompt: str,
        output: str,
        scores: list[ScoreResult],
        stats: _CallStats,
        started: float,
        cached: bool,
        error: str | None = None,
    ) -> CaseResult:
        if error is not None:
            status = "error"
        else:
            status = "passed" if all(s.passed for s in scores) else "failed"
        return CaseResult(
            id=case.id,
            status=status,
            prompt=prompt,
            output=output,
            scores=scores,
            error=error,
            cached=cached,
            attempts=stats.attempts,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            input_tokens=stats.input_tokens,
            output_tokens=stats.output_tokens,
        )


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
