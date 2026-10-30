from __future__ import annotations

import asyncio
import random

import anthropic
import httpx2
import pytest
from fakes import FakeModelClient, VirtualClock

from llm_eval_kit.models import ModelRequest, TransientModelError, is_retryable
from llm_eval_kit.ratelimit import RetryPolicy, TokenBucket
from llm_eval_kit.runner import ModelGateway

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls, status: int, headers: dict | None = None):
    response = httpx2.Response(status, request=REQUEST, headers=headers or {})
    return cls("error", response=response, body=None)


def test_token_bucket_paces_requests():
    clock = VirtualClock()
    bucket = TokenBucket.per_minute(60, capacity=2, clock=clock, sleep=clock.sleep)

    async def take(n: int) -> None:
        for _ in range(n):
            await bucket.acquire()

    asyncio.run(take(5))
    # Two tokens are available immediately, then one per second.
    assert clock.now == pytest.approx(3.0)


def test_token_bucket_refills_over_time():
    clock = VirtualClock()
    bucket = TokenBucket(rate=1.0, capacity=1, clock=clock, sleep=clock.sleep)
    asyncio.run(bucket.acquire())
    clock.now += 5  # idle time refills, but only up to capacity
    asyncio.run(bucket.acquire())
    asyncio.run(bucket.acquire())
    assert clock.sleeps == [pytest.approx(1.0)]


def test_token_bucket_validates_arguments():
    with pytest.raises(ValueError):
        TokenBucket(rate=0)


@pytest.mark.parametrize(
    ("error", "retry"),
    [
        (TransientModelError("x"), True),
        (_status_error(anthropic.RateLimitError, 429), True),
        (_status_error(anthropic.OverloadedError, 529), True),
        (_status_error(anthropic.InternalServerError, 500), True),
        (anthropic.APIConnectionError(request=REQUEST), True),
        (anthropic.APITimeoutError(request=REQUEST), True),
        (_status_error(anthropic.BadRequestError, 400), False),
        (_status_error(anthropic.AuthenticationError, 401), False),
        (ValueError("bug"), False),
    ],
)
def test_is_retryable(error, retry):
    assert is_retryable(error) is retry


def test_retry_until_success_with_backoff():
    clock = VirtualClock()
    policy = RetryPolicy(max_attempts=4, base_delay=1.0, sleep=clock.sleep, rng=random.Random(1))
    client = FakeModelClient(lambda r: "ok", fail_first=2)
    gateway = ModelGateway(client, retry=policy)
    response = asyncio.run(gateway.call(ModelRequest(model="m", prompt="p")))
    assert response.text == "ok"
    assert len(client.calls) == 3
    assert len(clock.sleeps) == 2
    assert clock.sleeps[0] <= 1.0 and clock.sleeps[1] <= 2.0


def test_retry_gives_up_after_max_attempts():
    clock = VirtualClock()
    policy = RetryPolicy(max_attempts=3, sleep=clock.sleep)
    client = FakeModelClient(lambda r: "ok", fail_first=10)
    with pytest.raises(TransientModelError):
        asyncio.run(ModelGateway(client, retry=policy).call(ModelRequest(model="m", prompt="p")))
    assert len(client.calls) == 3


def test_non_retryable_errors_are_raised_immediately():
    clock = VirtualClock()
    client = FakeModelClient(lambda r: "ok", error=_status_error(anthropic.BadRequestError, 400))
    gateway = ModelGateway(client, retry=RetryPolicy(sleep=clock.sleep))
    with pytest.raises(anthropic.BadRequestError):
        asyncio.run(gateway.call(ModelRequest(model="m", prompt="p")))
    assert len(client.calls) == 1 and clock.sleeps == []


def test_retry_after_header_sets_a_floor():
    policy = RetryPolicy(base_delay=0.001, max_delay=30, rng=random.Random(0))
    error = _status_error(anthropic.RateLimitError, 429, {"retry-after": "7"})
    assert policy.delay(1, error) == pytest.approx(7.0)
    capped = _status_error(anthropic.RateLimitError, 429, {"retry-after": "600"})
    assert policy.delay(1, capped) == pytest.approx(30.0)
