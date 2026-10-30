"""Optional smoke test against the real API. Skipped unless explicitly enabled."""

from __future__ import annotations

import asyncio
import os

import pytest

from llm_eval_kit.models import AnthropicModelClient, ModelRequest, default_model

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("RUN_LIVE_TESTS") == "1"),
    reason="set ANTHROPIC_API_KEY and RUN_LIVE_TESTS=1 to call the real API",
)


def test_one_real_request():
    request = ModelRequest(
        model=default_model(), prompt="Reply with the single word: pong", max_tokens=1024
    )
    response = asyncio.run(AnthropicModelClient().complete(request))
    assert response.refused or "pong" in response.text.lower()
