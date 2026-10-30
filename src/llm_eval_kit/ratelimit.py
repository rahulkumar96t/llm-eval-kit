"""Async rate limiting and retry with exponential backoff."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .models import is_retryable, retry_after_seconds

T = TypeVar("T")

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


class TokenBucket:
    """Token bucket: ``rate`` tokens per second, holding at most ``capacity``.

    ``acquire`` waits until a token is available. Waiters are served in order
    because acquisition is serialised by a lock.
    """

    def __init__(
        self,
        rate: float,
        capacity: float = 1.0,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if rate <= 0 or capacity < 1:
            raise ValueError("rate must be > 0 and capacity >= 1")
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = asyncio.Lock()

    @classmethod
    def per_minute(cls, requests_per_minute: float, **kwargs: object) -> TokenBucket:
        return cls(requests_per_minute / 60.0, **kwargs)  # type: ignore[arg-type]

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.rate)
        self._updated = now

    async def acquire(self) -> None:
        async with self._lock:
            self._refill()
            while self._tokens < 1:
                await self._sleep((1 - self._tokens) / self.rate)
                self._refill()
            self._tokens -= 1


class RetryPolicy:
    def __init__(
        self,
        max_attempts: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        *,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
        retryable: Callable[[BaseException], bool] = is_retryable,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._retryable = retryable

    def delay(self, attempt: int, exc: BaseException) -> float:
        """Full-jitter exponential backoff, never shorter than a server Retry-After."""
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        delay = self._rng.uniform(0, ceiling)
        hint = retry_after_seconds(exc)
        if hint is not None:
            delay = max(delay, min(hint, self.max_delay))
        return delay

    async def run(self, call: Callable[[], Awaitable[T]]) -> tuple[T, int]:
        """Run ``call`` until it succeeds; returns (result, attempts used)."""
        attempt = 0
        while True:
            attempt += 1
            try:
                return await call(), attempt
            except Exception as exc:
                if attempt >= self.max_attempts or not self._retryable(exc):
                    raise
                await self._sleep(self.delay(attempt, exc))
