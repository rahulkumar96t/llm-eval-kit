from __future__ import annotations

import asyncio
from pathlib import Path

from fakes import FakeModelClient

from llm_eval_kit.cache import ResponseCache, cache_key
from llm_eval_kit.models import ModelRequest, ModelResponse
from llm_eval_kit.runner import ModelGateway

REQ = ModelRequest(model="m", prompt="hello", system="sys", max_tokens=100, effort="low")


def test_key_is_stable_and_sensitive_to_every_param():
    assert cache_key("p", REQ) == cache_key("p", ModelRequest(**REQ.cache_payload()))
    variants = [
        ModelRequest(model="m2", prompt="hello", system="sys", max_tokens=100, effort="low"),
        ModelRequest(model="m", prompt="hello!", system="sys", max_tokens=100, effort="low"),
        ModelRequest(model="m", prompt="hello", system="sys2", max_tokens=100, effort="low"),
        ModelRequest(model="m", prompt="hello", system="sys", max_tokens=101, effort="low"),
        ModelRequest(model="m", prompt="hello", system="sys", max_tokens=100, effort="high"),
    ]
    keys = {cache_key("p", r) for r in variants} | {cache_key("p", REQ)}
    assert len(keys) == len(variants) + 1
    assert cache_key("p", REQ, {"fallbacks": True}) != cache_key("p", REQ, {"fallbacks": False})
    assert cache_key("p", REQ, {"a": 1, "b": 2}) == cache_key("p", REQ, {"b": 2, "a": 1})


def test_round_trip(tmp_path: Path):
    cache = ResponseCache(tmp_path)
    key = cache_key("p", REQ)
    assert cache.get(key) is None
    cache.put(key, REQ, ModelResponse(text="hi", model="m", stop_reason="end_turn"))
    hit = cache.get(key)
    assert hit is not None and hit.text == "hi" and hit.cached
    assert (cache.hits, cache.misses) == (1, 1)
    assert not list(tmp_path.rglob(".tmp-*"))


def test_corrupt_entry_is_a_miss(tmp_path: Path):
    cache = ResponseCache(tmp_path)
    key = cache_key("p", REQ)
    path = tmp_path / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{truncated", encoding="utf-8")
    assert cache.get(key) is None


def test_gateway_hits_skip_the_model(tmp_path: Path):
    client = FakeModelClient(lambda r: "answer")
    gateway = ModelGateway(client, cache=ResponseCache(tmp_path))
    first = asyncio.run(gateway.call(REQ))
    second = asyncio.run(gateway.call(REQ))
    assert (first.text, first.cached) == ("answer", False)
    assert (second.text, second.cached) == ("answer", True)
    assert len(client.calls) == 1


def test_refusals_are_not_cached(tmp_path: Path):
    refusal = ModelResponse(text="", model="m", stop_reason="refusal")
    client = FakeModelClient(lambda r: refusal)
    gateway = ModelGateway(client, cache=ResponseCache(tmp_path))
    asyncio.run(gateway.call(REQ))
    asyncio.run(gateway.call(REQ))
    assert len(client.calls) == 2
