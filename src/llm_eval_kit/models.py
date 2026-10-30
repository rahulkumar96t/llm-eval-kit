"""Model clients: the protocol the runner depends on, plus the Claude implementation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def default_model() -> str:
    return os.environ.get("LLM_EVAL_MODEL") or DEFAULT_MODEL


def default_judge_model() -> str:
    return os.environ.get("LLM_EVAL_JUDGE_MODEL") or default_model()


def fallbacks_enabled() -> bool:
    """Server-side refusal fallback is on unless LLM_EVAL_FALLBACKS is off/0/false."""
    value = os.environ.get("LLM_EVAL_FALLBACKS", "default").strip().lower()
    return value not in {"off", "0", "false", "no", "none"}


@dataclass(frozen=True)
class ModelRequest:
    """Everything that determines a model response, and therefore the cache key."""

    model: str
    prompt: str
    system: str | None = None
    max_tokens: int = 16000
    effort: str | None = None
    output_schema: dict[str, Any] | None = None
    stop_sequences: tuple[str, ...] = ()

    def cache_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stop_sequences"] = list(self.stop_sequences)
        return payload


@dataclass
class ModelResponse:
    text: str
    model: str
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    refusal_category: str | None = None
    cached: bool = field(default=False, compare=False)

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("cached")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelResponse:
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data and k != "cached"}
        return cls(**known)


class ModelClient(Protocol):
    """Anything that turns a ModelRequest into a ModelResponse.

    Implementations raise ``TransientModelError`` (or a provider error that
    ``is_retryable`` recognises) for failures worth retrying.
    """

    name: str

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


class TransientModelError(Exception):
    """A failure that may succeed on retry (rate limit, overload, network)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, TransientModelError):
        return True
    import anthropic

    # APITimeoutError is a subclass of APIConnectionError.
    if isinstance(exc, anthropic.RateLimitError | anthropic.APIConnectionError):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code >= 500
    return False


def retry_after_seconds(exc: BaseException) -> float | None:
    """Server-suggested delay, if the error carries one."""
    if isinstance(exc, TransientModelError):
        return exc.retry_after
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


class AnthropicModelClient:
    """Calls the Claude Messages API.

    Credentials come only from the environment (the SDK's default resolution).
    The SDK's own retries are disabled because the runner owns retry policy.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        fallbacks: bool | None = None,
        timeout: float = 300.0,
        client: Any | None = None,
    ) -> None:
        self.fallbacks = fallbacks_enabled() if fallbacks is None else fallbacks
        self._timeout = timeout
        self._client = client

    def _sdk(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(max_retries=0, timeout=self._timeout)
        return self._client

    def build_params(self, request: ModelRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            params["system"] = request.system
        output_config: dict[str, Any] = {}
        if request.effort:
            output_config["effort"] = request.effort
        if request.output_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.output_schema}
        if output_config:
            params["output_config"] = output_config
        if request.stop_sequences:
            params["stop_sequences"] = list(request.stop_sequences)
        return params

    async def complete(self, request: ModelRequest) -> ModelResponse:
        sdk = self._sdk()
        params = self.build_params(request)
        if self.fallbacks:
            message = await sdk.beta.messages.create(
                **params, betas=[FALLBACK_BETA], fallbacks="default"
            )
        else:
            message = await sdk.messages.create(**params)
        return response_from_message(message)


def response_from_message(message: Any) -> ModelResponse:
    """Convert an SDK message (beta or not) into a ModelResponse.

    The stop reason is checked before any content is read: a refusal can arrive
    with empty or partial content, which must not be scored as an answer.
    """
    usage = getattr(message, "usage", None)
    common = {
        "model": getattr(message, "model", ""),
        "stop_reason": message.stop_reason,
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }
    if message.stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        return ModelResponse(text="", refusal_category=getattr(details, "category", None), **common)
    text = "".join(block.text for block in message.content if block.type == "text")
    return ModelResponse(text=text, **common)


def parse_json_text(text: str) -> Any:
    """Parse JSON from a model reply, tolerating a Markdown code fence around it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1])
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise
