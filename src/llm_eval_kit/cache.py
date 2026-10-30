"""On-disk response cache keyed by everything that determines a response."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import ModelRequest, ModelResponse

CACHE_VERSION = 1


def cache_key(provider: str, request: ModelRequest, extra: dict[str, Any] | None = None) -> str:
    """Stable SHA-256 over prompt + model + params (canonical, key-sorted JSON)."""
    payload = {
        "v": CACHE_VERSION,
        "provider": provider,
        "request": request.cache_payload(),
        "extra": extra or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """One JSON file per entry, sharded by key prefix; writes are atomic renames."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> ModelResponse | None:
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            response = ModelResponse.from_dict(data["response"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
            self.misses += 1
            return None
        self.hits += 1
        response.cached = True
        return response

    def put(self, key: str, request: ModelRequest, response: ModelResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"key": key, "request": request.cache_payload(), "response": response.to_dict()}
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(entry, handle, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
