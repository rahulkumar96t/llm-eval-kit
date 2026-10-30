from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from anthropic.types.beta import BetaMessage

from llm_eval_kit.models import (
    FALLBACK_BETA,
    AnthropicModelClient,
    ModelRequest,
    default_model,
    parse_json_text,
    response_from_message,
)


def message(content: list[dict], stop_reason: str = "end_turn", **extra) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 3},
            **extra,
        }
    )


class RecordingSDK:
    """Stands in for AsyncAnthropic; records which endpoint was called with what."""

    def __init__(self, reply: BetaMessage) -> None:
        self.calls: list[tuple[str, dict]] = []

        async def create(kind: str, **params):
            self.calls.append((kind, params))
            return reply

        self.messages = SimpleNamespace(create=lambda **p: create("messages", **p))
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **p: create("beta", **p))
        )


def test_default_model_and_env_override(monkeypatch: pytest.MonkeyPatch):
    assert default_model() == "claude-sonnet-5-5"
    monkeypatch.setenv("LLM_EVAL_MODEL", "claude-haiku-5-5")
    assert default_model() == "claude-haiku-5-5"


def test_build_params_never_sends_sampling_or_thinking():
    request = ModelRequest(
        model="m",
        prompt="p",
        system="s",
        max_tokens=50,
        effort="low",
        output_schema={"type": "object"},
        stop_sequences=("END",),
    )
    params = AnthropicModelClient(fallbacks=False).build_params(request)
    assert params == {
        "model": "m",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "p"}],
        "system": "s",
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": {"type": "object"}},
        },
        "stop_sequences": ["END"],
    }
    client = AnthropicModelClient(fallbacks=False)
    minimal = client.build_params(ModelRequest(model="m", prompt="p"))
    assert set(minimal) == {"model", "max_tokens", "messages"}


def test_fallbacks_on_by_default_use_the_beta_endpoint():
    sdk = RecordingSDK(message([{"type": "text", "text": "hi"}]))
    client = AnthropicModelClient(client=sdk)
    response = asyncio.run(client.complete(ModelRequest(model="m", prompt="p")))
    kind, params = sdk.calls[0]
    assert kind == "beta"
    assert params["betas"] == [FALLBACK_BETA] and params["fallbacks"] == "default"
    assert response.text == "hi" and response.input_tokens == 12


def test_fallbacks_can_be_switched_off_by_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLM_EVAL_FALLBACKS", "off")
    sdk = RecordingSDK(message([{"type": "text", "text": "hi"}]))
    asyncio.run(AnthropicModelClient(client=sdk).complete(ModelRequest(model="m", prompt="p")))
    kind, params = sdk.calls[0]
    assert kind == "messages" and "fallbacks" not in params


def test_only_text_blocks_are_read():
    reply = message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Hello "},
            {
                "type": "fallback",
                "from": {"model": "a"},
                "to": {"model": "b"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {"type": "text", "text": "world"},
        ]
    )
    assert response_from_message(reply).text == "Hello world"


def test_refusal_is_checked_before_content():
    reply = message(
        [{"type": "text", "text": "partial answ"}],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "cyber", "explanation": None},
    )
    response = response_from_message(reply)
    assert response.refused and response.text == "" and response.refusal_category == "cyber"


def test_parse_json_text_variants():
    assert parse_json_text('{"a": 1}') == {"a": 1}
    assert parse_json_text('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_text('Verdict: {"a": 3} done') == {"a": 3}
